import {
  applyMove,
  decodeAction,
  encodeAction,
  generateLegalMoves,
  isLegalMove,
  scoreState,
} from "../../packages/core/src/index.js";
import { evaluateMoveQuick, evaluateState } from "../../apps/web/src/ai/evaluation.js";
import { decideTrainingMove, normalizeTrainingAiSpec } from "../ai_runtime.mjs";
import { normalizeScoreDiff } from "./branch_dataset_schema.mjs";

function scoreDiffForPlayer(state, player) {
  const [blackScore, whiteScore] = scoreState(state);
  return player === 0 ? blackScore - whiteScore : whiteScore - blackScore;
}

function actionToLegalMove(state, player, action) {
  const move = decodeAction(action, player);
  return isLegalMove(state, move) ? move : null;
}

export function selectReplayActions(state, player, legalActions, chosenAction, config = {}) {
  const topK = Math.max(1, Number(config.topKActions ?? 8));
  const scored = [];
  for (const action of legalActions) {
    const move = actionToLegalMove(state, player, action);
    if (!move) continue;
    scored.push({
      action,
      score: evaluateMoveQuick(state, player, move),
      move,
    });
  }
  scored.sort((a, b) => b.score - a.score);
  const selected = scored.slice(0, topK);
  if (!selected.some((entry) => entry.action === chosenAction)) {
    const chosenMove = actionToLegalMove(state, player, chosenAction);
    if (chosenMove) {
      selected.push({
        action: chosenAction,
        score: evaluateMoveQuick(state, player, chosenMove),
        move: chosenMove,
      });
    }
  }
  return selected;
}

async function chooseReplayMove(state, spec, rng = Math.random) {
  const normalized = normalizeTrainingAiSpec(spec);
  const epsilon = Math.max(0, Math.min(1, Number(spec.epsilon ?? 0.08)));
  const legalMoves = generateLegalMoves(state);
  if (legalMoves.length === 1) return legalMoves[0];
  if (rng() < epsilon) {
    const scored = legalMoves
      .map((move) => ({ move, score: evaluateMoveQuick(state, state.currentPlayer, move) }))
      .sort((a, b) => b.score - a.score);
    const pool = scored.slice(0, Math.max(1, Math.min(scored.length, Number(spec.noiseTopK ?? 12))));
    return pool[Math.floor(rng() * pool.length)].move;
  }
  const decision = await decideTrainingMove(state, normalized);
  return decision.move;
}

export async function playToEndFromState(state, config = {}) {
  let current = state;
  const maxPlies = Number(config.maxPlies ?? 120);
  const replaySpec = {
    difficulty: config.teacher ?? config.difficulty ?? "expert",
    timeLimitMs: config.teacherMs ?? 25,
    modelPath: config.teacherModel ?? config.modelPath ?? null,
    epsilon: config.epsilon ?? 0.08,
  };

  for (let ply = 0; ply < maxPlies && current.status === "playing"; ply += 1) {
    const move = await chooseReplayMove(current, replaySpec);
    current = applyMove(current, move);
  }
  return current;
}

export async function replayCriticalState(criticalState, config = {}) {
  const selected = selectReplayActions(
    criticalState.state,
    criticalState.player,
    criticalState.legal_actions,
    criticalState.chosen_action,
    config,
  );
  const playoutsPerAction = Math.max(1, Number(config.playoutsPerAction ?? 2));
  const evaluatedActions = [];
  const qValues = [];
  const rawScoreDiffs = [];

  for (const entry of selected) {
    const scores = [];
    for (let index = 0; index < playoutsPerAction; index += 1) {
      const next = applyMove(criticalState.state, entry.move);
      const terminal = await playToEndFromState(next, config);
      scores.push(scoreDiffForPlayer(terminal, criticalState.player));
    }
    const averageScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    evaluatedActions.push(entry.action);
    rawScoreDiffs.push(averageScore);
    qValues.push(normalizeScoreDiff(averageScore));
  }

  if (evaluatedActions.length === 0) {
    const fallbackValue = normalizeScoreDiff(scoreDiffForPlayer(criticalState.state, criticalState.player));
    return {
      evaluatedActions: [encodeAction({ kind: "pass", player: criticalState.player })],
      qValues: [fallbackValue],
      rawScoreDiffs: [fallbackValue],
    };
  }

  const staticValue = normalizeScoreDiff(evaluateState(criticalState.state, criticalState.player));
  return { evaluatedActions, qValues, rawScoreDiffs, staticValue, playoutsPerAction };
}

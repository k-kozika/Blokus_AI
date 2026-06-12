import { applyMove, createInitialState, isLegalMove, scoreState } from "../packages/core/src/index.js";
import { decideTrainingMove, normalizeTrainingAiSpec } from "./ai_runtime.mjs";

function marginFor(contestantName, seats, scores) {
  if (seats.black.name === contestantName) {
    return scores[0] - scores[1];
  }
  return scores[1] - scores[0];
}

function winnerFor(scores) {
  if (scores[0] === scores[1]) return null;
  return scores[0] > scores[1] ? 0 : 1;
}

export async function playArenaGame(config = {}) {
  const seats = {
    black: {
      name: config.black?.name ?? "black",
      spec: normalizeTrainingAiSpec(config.black?.spec ?? config.black ?? {}),
    },
    white: {
      name: config.white?.name ?? "white",
      spec: normalizeTrainingAiSpec(config.white?.spec ?? config.white ?? {}),
    },
  };
  const startPolicy = config.startPolicy ?? "fixedStart";
  const collectTrace = Boolean(config.collectTrace);
  const trace = [];
  let state = createInitialState(startPolicy);
  let plies = 0;

  while (state.status === "playing") {
    const seat = state.currentPlayer === 0 ? seats.black : seats.white;
    const decision = await decideTrainingMove(state, seat.spec);
    if (!isLegalMove(state, decision.move)) {
      throw new Error(`Illegal move emitted by ${seat.name}.`);
    }
    if (collectTrace) {
      trace.push({
        ply: plies,
        player: state.currentPlayer,
        seat: seat.name,
        engine: decision.stats?.engine ?? seat.spec.difficulty,
        strategy: decision.stats?.strategy ?? null,
        move: decision.move,
      });
    }
    state = applyMove(state, decision.move);
    plies += 1;
  }

  const scores = scoreState(state);
  const winner = winnerFor(scores);
  return {
    finalState: state,
    scores,
    winner,
    winnerName: winner === null ? null : winner === 0 ? seats.black.name : seats.white.name,
    black: seats.black.name,
    white: seats.white.name,
    plies,
    trace,
  };
}

export async function runArena(config = {}) {
  const games = Math.max(1, Number(config.games ?? 2));
  const indexOffset = Math.max(0, Number(config.indexOffset ?? 0));
  const swapColors = config.swapColors !== false;
  const alphaName = config.alpha?.name ?? "alpha";
  const betaName = config.beta?.name ?? "beta";
  const alphaSpec = normalizeTrainingAiSpec(config.alpha?.spec ?? config.alpha ?? {});
  const betaSpec = normalizeTrainingAiSpec(config.beta?.spec ?? config.beta ?? {});
  const results = [];
  const record = {
    [alphaName]: { wins: 0, losses: 0, draws: 0, totalMargin: 0 },
    [betaName]: { wins: 0, losses: 0, draws: 0, totalMargin: 0 },
  };

  for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
    const absoluteGameIndex = indexOffset + gameIndex;
    const swapped = swapColors && absoluteGameIndex % 2 === 1;
    const seats = swapped
      ? {
        black: { name: betaName, spec: betaSpec },
        white: { name: alphaName, spec: alphaSpec },
      }
      : {
        black: { name: alphaName, spec: alphaSpec },
        white: { name: betaName, spec: betaSpec },
      };

    const game = await playArenaGame({
      black: seats.black,
      white: seats.white,
      startPolicy: config.startPolicy ?? "fixedStart",
      collectTrace: Boolean(config.collectTrace),
    });

    const alphaMargin = marginFor(alphaName, seats, game.scores);
    const betaMargin = -alphaMargin;
    record[alphaName].totalMargin += alphaMargin;
    record[betaName].totalMargin += betaMargin;

    if (game.winnerName === alphaName) {
      record[alphaName].wins += 1;
      record[betaName].losses += 1;
    } else if (game.winnerName === betaName) {
      record[betaName].wins += 1;
      record[alphaName].losses += 1;
    } else {
      record[alphaName].draws += 1;
      record[betaName].draws += 1;
    }

    results.push({
      index: absoluteGameIndex,
      black: game.black,
      white: game.white,
      scores: game.scores,
      winnerName: game.winnerName,
      plies: game.plies,
      alphaMargin,
      trace: game.trace,
    });
  }

  return {
    totalGames: games,
    swapColors,
    startPolicy: config.startPolicy ?? "fixedStart",
    contestants: {
      [alphaName]: {
        difficulty: alphaSpec.difficulty,
        modelPath: alphaSpec.modelPath ?? null,
        ...record[alphaName],
        averageMargin: Number((record[alphaName].totalMargin / games).toFixed(3)),
      },
      [betaName]: {
        difficulty: betaSpec.difficulty,
        modelPath: betaSpec.modelPath ?? null,
        ...record[betaName],
        averageMargin: Number((record[betaName].totalMargin / games).toFixed(3)),
      },
    },
    games: results,
  };
}

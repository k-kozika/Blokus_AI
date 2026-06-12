import { createReadStream, createWriteStream } from "node:fs";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { pipeline } from "node:stream/promises";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import {
  applyMove,
  cloneState,
  createInitialState,
  encodeAction,
  encodeStateTensor,
  generateLegalMoves,
  scoreState,
} from "../../packages/core/src/index.js";
import {
  countCornerCandidates,
  countLegalPlacementsForPlayer,
  evaluateState,
} from "../../apps/web/src/ai/evaluation.js";
import { PIECE_SHAPES } from "../../packages/core/src/pieces.js";
import { decideTrainingMove, normalizeTrainingAiSpec } from "../ai_runtime.mjs";

const root = dirname(dirname(dirname(fileURLToPath(import.meta.url))));

function pieceCells(pieceId) {
  return PIECE_SHAPES[pieceId].length;
}

function remainingCells(state, player) {
  return state.remainingPieces[player].reduce((sum, pieceId) => sum + pieceCells(pieceId), 0);
}

function other(player) {
  return player === 0 ? 1 : 0;
}

export function hashStateForReplay(state) {
  return JSON.stringify({
    board: state.board,
    currentPlayer: state.currentPlayer,
    remainingPieces: state.remainingPieces,
    startAssignment: state.startAssignment,
    turn: state.turn,
  });
}

export function summarizeState(state, player) {
  const opponent = other(player);
  return {
    evaluation: evaluateState(state, player),
    myLegalMoves: countLegalPlacementsForPlayer(state, player),
    oppLegalMoves: countLegalPlacementsForPlayer(state, opponent),
    myCornerCandidates: countCornerCandidates(state, player),
    oppCornerCandidates: countCornerCandidates(state, opponent),
    myRemainingCells: remainingCells(state, player),
    oppRemainingCells: remainingCells(state, opponent),
  };
}

export function parseTrajectoryArgs(argv) {
  const args = {
    games: 100,
    out: join(root, "training", "dataset", "trajectories", "smoke"),
    teacherMs: 50,
    startPolicy: "fixedStart",
    blackAi: "expert",
    whiteAi: "expert",
    blackModel: null,
    whiteModel: null,
    parallel: 1,
    indexOffset: 0,
    workerMode: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--games") args.games = Number(argv[++index]);
    if (value === "--out") args.out = argv[++index];
    if (value === "--teacher-ms") args.teacherMs = Number(argv[++index]);
    if (value === "--start-policy") args.startPolicy = argv[++index];
    if (value === "--ai") {
      args.blackAi = argv[++index];
      args.whiteAi = args.blackAi;
    }
    if (value === "--black-ai") args.blackAi = argv[++index];
    if (value === "--white-ai") args.whiteAi = argv[++index];
    if (value === "--black-model") args.blackModel = argv[++index];
    if (value === "--white-model") args.whiteModel = argv[++index];
    if (value === "--parallel") args.parallel = Math.max(1, Number(argv[++index]));
    if (value === "--index-offset") args.indexOffset = Math.max(0, Number(argv[++index]));
    if (value === "--worker-mode") args.workerMode = true;
  }

  return args;
}

export async function collectTrajectoryGame(config = {}) {
  const black = normalizeTrainingAiSpec({
    difficulty: config.blackAi ?? "expert",
    modelPath: config.blackModel,
    timeLimitMs: config.teacherMs ?? 50,
    label: "black",
  });
  const white = normalizeTrainingAiSpec({
    difficulty: config.whiteAi ?? "expert",
    modelPath: config.whiteModel,
    timeLimitMs: config.teacherMs ?? 50,
    label: "white",
  });
  let state = createInitialState(config.startPolicy ?? "fixedStart");
  const steps = [];

  while (state.status === "playing") {
    const player = state.currentPlayer;
    const legalMoves = generateLegalMoves(state);
    const legalActions = legalMoves.map(encodeAction);
    const before = summarizeState(state, player);
    const stateBefore = cloneState(state);
    const decision = await decideTrainingMove(state, player === 0 ? black : white);
    const action = encodeAction(decision.move);
    const next = applyMove(state, decision.move);
    const after = summarizeState(next, player);
    const stats = decision.stats ?? {};

    steps.push({
      turn: state.turn,
      player,
      actor_difficulty: player === 0 ? black.difficulty : white.difficulty,
      state: stateBefore,
      state_hash: hashStateForReplay(stateBefore),
      encoded_state: Array.from(encodeStateTensor(stateBefore, player)),
      legal_actions: legalActions,
      chosen_action: action,
      selected_action: action,
      policy_target_actions: stats.policyTargetActions ?? [action],
      policy_target_probs: stats.policyTargetProbs ?? [1],
      before,
      after,
      deltas: {
        evaluation: after.evaluation - before.evaluation,
        myLegalMoves: after.myLegalMoves - before.myLegalMoves,
        myCornerCandidates: after.myCornerCandidates - before.myCornerCandidates,
        oppLegalMoves: after.oppLegalMoves - before.oppLegalMoves,
      },
    });

    state = next;
  }

  const score = scoreState(state);
  return { steps, score, finalState: state };
}

export async function generateTrajectories(config = {}) {
  const outDir = resolve(config.out);
  await mkdir(outDir, { recursive: true });
  const trajectoryPath = join(outDir, "trajectories.jsonl");
  const stream = createWriteStream(trajectoryPath, { encoding: "utf-8" });
  let totalSteps = 0;

  try {
    for (let gameIndex = 0; gameIndex < config.games; gameIndex += 1) {
      const game = await collectTrajectoryGame(config);
      totalSteps += game.steps.length;
      const absoluteGameIndex = (config.indexOffset ?? 0) + gameIndex;
      const record = {
        schema_version: 1,
        game_id: `game-${String(absoluteGameIndex).padStart(6, "0")}`,
        score: game.score,
        steps: game.steps,
      };
      if (!stream.write(`${JSON.stringify(record)}\n`)) {
        await new Promise((resolveDrain) => stream.once("drain", resolveDrain));
      }
      console.log(`Trajectory ${absoluteGameIndex + 1}: ${game.steps.length} steps, score ${game.score[0]}-${game.score[1]}`);
    }
  } finally {
    await new Promise((resolveEnd, reject) => {
      stream.end((error) => {
        if (error) reject(error);
        else resolveEnd();
      });
    });
  }

  const summary = {
    games: config.games,
    totalSteps,
    trajectoryPath,
    blackAi: config.blackAi,
    whiteAi: config.whiteAi,
    teacherMs: config.teacherMs,
    startPolicy: config.startPolicy,
    parallel: config.parallel ?? 1,
    indexOffset: config.indexOffset ?? 0,
  };
  await writeFile(join(outDir, "metadata.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf-8");
  return summary;
}

async function main() {
  const config = parseTrajectoryArgs(process.argv.slice(2));
  if (config.parallel > 1 && !config.workerMode) {
    await generateTrajectoriesParallel(config);
  } else {
    await generateTrajectories(config);
  }
}

function splitGames(totalGames, workers) {
  const actualWorkers = Math.max(1, Math.min(workers, totalGames));
  const base = Math.floor(totalGames / actualWorkers);
  const remainder = totalGames % actualWorkers;
  const chunks = [];
  let offset = 0;
  for (let index = 0; index < actualWorkers; index += 1) {
    const games = base + (index < remainder ? 1 : 0);
    chunks.push({ games, offset });
    offset += games;
  }
  return chunks;
}

function runTrajectoryWorker(config, chunk, outDir) {
  const args = [
    fileURLToPath(import.meta.url),
    "--games", String(chunk.games),
    "--index-offset", String(chunk.offset),
    "--out", outDir,
    "--teacher-ms", String(config.teacherMs),
    "--start-policy", config.startPolicy,
    "--black-ai", config.blackAi,
    "--white-ai", config.whiteAi,
    "--worker-mode",
    ...(config.blackModel ? ["--black-model", config.blackModel] : []),
    ...(config.whiteModel ? ["--white-model", config.whiteModel] : []),
  ];

  return new Promise((resolveWorker, reject) => {
    const child = spawn("node", args, {
      cwd: root,
      stdio: "inherit",
      shell: false,
      env: {
        ...process.env,
        BLOKUS_ORT_THREADS: process.env.BLOKUS_ORT_THREADS ?? "1",
      },
    });
    child.on("exit", (code) => {
      if ((code ?? 1) === 0) resolveWorker();
      else reject(new Error(`trajectory worker exited with ${code}`));
    });
  });
}

async function appendFileToStream(path, stream) {
  await pipeline(createReadStream(path, { encoding: "utf-8" }), stream, { end: false });
}

export async function generateTrajectoriesParallel(config) {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-trajectories-"));
  const outDir = resolve(config.out);
  const chunks = splitGames(config.games, config.parallel);
  await mkdir(outDir, { recursive: true });

  try {
    const shardDirs = chunks.map((_, index) => join(tempDir, `worker-${String(index + 1).padStart(3, "0")}`));
    await Promise.all(chunks.map((chunk, index) => runTrajectoryWorker(config, chunk, shardDirs[index])));

    let totalSteps = 0;
    const trajectoryPath = join(outDir, "trajectories.jsonl");
    const output = createWriteStream(trajectoryPath, { encoding: "utf-8" });
    try {
      for (const shardDir of shardDirs) {
        const meta = JSON.parse(await readFile(join(shardDir, "metadata.json"), "utf-8"));
        totalSteps += meta.totalSteps ?? 0;
        await appendFileToStream(join(shardDir, "trajectories.jsonl"), output);
      }
    } finally {
      await new Promise((resolveEnd, reject) => {
        output.end((error) => {
          if (error) reject(error);
          else resolveEnd();
        });
      });
    }

    const summary = {
      games: config.games,
      totalSteps,
      trajectoryPath,
      blackAi: config.blackAi,
      whiteAi: config.whiteAi,
      teacherMs: config.teacherMs,
      startPolicy: config.startPolicy,
      parallel: config.parallel,
    };
    await writeFile(join(outDir, "metadata.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf-8");
    return summary;
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await main();
}

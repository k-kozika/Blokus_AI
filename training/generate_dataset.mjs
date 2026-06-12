import { createReadStream, createWriteStream } from "node:fs";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { pipeline } from "node:stream/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import {
  applyMove,
  createInitialState,
  encodeAction,
  encodeStateTensor,
  generateLegalMoves,
  scoreState,
} from "../packages/core/src/index.js";
import { decideTrainingMove, normalizeTrainingAiSpec } from "./ai_runtime.mjs";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

export function parseArgs(argv) {
  const args = {
    games: 100,
    out: join(root, "training", "data", "smoke-expert-100.jsonl"),
    teacherMs: 25,
    startPolicy: "fixedStart",
    blackAi: "expert",
    whiteAi: "expert",
    blackModel: null,
    whiteModel: null,
    policyTargetSource: "auto",
    parallel: 1,
    workerMode: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--games") args.games = Number(argv[++index]);
    if (value === "--out") args.out = argv[++index];
    if (value === "--teacher-ms") args.teacherMs = Number(argv[++index]);
    if (value === "--start-policy") args.startPolicy = argv[++index];
    if (value === "--black-ai") args.blackAi = argv[++index];
    if (value === "--white-ai") args.whiteAi = argv[++index];
    if (value === "--black-model") args.blackModel = argv[++index];
    if (value === "--white-model") args.whiteModel = argv[++index];
    if (value === "--policy-target-source") args.policyTargetSource = argv[++index];
    if (value === "--parallel") args.parallel = Math.max(1, Number(argv[++index]));
    if (value === "--worker-mode") args.workerMode = true;
  }

  return args;
}

export async function playTrainingGame(config) {
  const black = normalizeTrainingAiSpec({
    difficulty: config.blackAi ?? "expert",
    modelPath: config.blackModel,
    timeLimitMs: config.teacherMs,
    label: "black",
  });
  const white = normalizeTrainingAiSpec({
    difficulty: config.whiteAi ?? "expert",
    modelPath: config.whiteModel,
    timeLimitMs: config.teacherMs,
    label: "white",
  });
  let state = createInitialState(config.startPolicy);
  const samples = [];
  const includeVisitTargets = config.policyTargetSource === "visit" || config.policyTargetSource === "auto";

  while (state.status === "playing") {
    const currentPlayer = state.currentPlayer;
    const legalMoves = generateLegalMoves(state);
    const decision = await decideTrainingMove(state, currentPlayer === 0 ? black : white);
    const legalActions = legalMoves.map(encodeAction);
    const stats = decision.stats ?? {};
    const useVisitTargets = includeVisitTargets
      && Array.isArray(stats.policyTargetActions)
      && Array.isArray(stats.policyTargetProbs)
      && stats.policyTargetActions.length > 0
      && stats.policyTargetActions.length === stats.policyTargetProbs.length;
    samples.push({
      player: currentPlayer,
      actor_difficulty: currentPlayer === 0 ? black.difficulty : white.difficulty,
      encoded_state: Array.from(encodeStateTensor(state, currentPlayer)),
      legal_actions: legalActions,
      selected_action: encodeAction(decision.move),
      expert_selected_action: encodeAction(decision.move),
      final_score_diff: 0,
      policy_target_actions: useVisitTargets ? stats.policyTargetActions : [encodeAction(decision.move)],
      policy_target_probs: useVisitTargets ? stats.policyTargetProbs : [1],
      policy_target_visits: useVisitTargets ? (stats.policyTargetVisits ?? null) : [1],
      policy_target_total_visits: useVisitTargets ? (stats.policyTargetTotalVisits ?? null) : 1,
      root_value: typeof stats.rootValue === "number" ? stats.rootValue : null,
      strategy: stats.strategy ?? null,
    });
    state = applyMove(state, decision.move);
  }

  const [blackScore, whiteScore] = scoreState(state);
  for (const sample of samples) {
    sample.final_score_diff = sample.player === 0
      ? blackScore - whiteScore
      : whiteScore - blackScore;
  }

  return { state, samples, score: [blackScore, whiteScore] };
}

export async function generateDataset(config) {
  let totalPositions = 0;
  await mkdir(dirname(config.out), { recursive: true });
  const stream = createWriteStream(config.out, { encoding: "utf-8" });

  try {
    for (let gameIndex = 0; gameIndex < config.games; gameIndex += 1) {
      const { samples, score } = await playTrainingGame(config);
      totalPositions += samples.length;

      for (const sample of samples) {
        if (!stream.write(`${JSON.stringify(sample)}\n`)) {
          await new Promise((resolve) => stream.once("drain", resolve));
        }
      }

      console.log(`Generated game ${gameIndex + 1}/${config.games} (${samples.length} positions, score ${score[0]}-${score[1]})`);
    }
  } finally {
    await new Promise((resolve, reject) => {
      stream.end((error) => {
        if (error) reject(error);
        else resolve();
      });
    });
  }
  await writeFile(
    `${config.out}.meta.json`,
    `${JSON.stringify({
      games: config.games,
      totalPositions,
      blackAi: config.blackAi ?? "expert",
      whiteAi: config.whiteAi ?? "expert",
      blackModel: config.blackModel ?? null,
      whiteModel: config.whiteModel ?? null,
      policyTargetSource: config.policyTargetSource ?? "auto",
      teacherMs: config.teacherMs,
      startPolicy: config.startPolicy,
    }, null, 2)}\n`,
  );

  console.log(`Saved ${totalPositions} samples to ${config.out}`);
  return {
    games: config.games,
    totalPositions,
    out: config.out,
    metaPath: `${config.out}.meta.json`,
  };
}

async function main() {
  const config = parseArgs(process.argv.slice(2));
  if (config.parallel > 1 && !config.workerMode) {
    await generateDatasetParallel(config);
  } else {
    await generateDataset(config);
  }
}

function splitGames(totalGames, workers) {
  const actualWorkers = Math.max(1, Math.min(workers, totalGames));
  const base = Math.floor(totalGames / actualWorkers);
  const remainder = totalGames % actualWorkers;
  return Array.from({ length: actualWorkers }, (_, index) => base + (index < remainder ? 1 : 0))
    .filter((games) => games > 0);
}

function runDatasetWorker(config, games, out) {
  const args = [
    fileURLToPath(import.meta.url),
    "--games", String(games),
    "--out", out,
    "--teacher-ms", String(config.teacherMs),
    "--start-policy", config.startPolicy,
    "--black-ai", config.blackAi,
    "--white-ai", config.whiteAi,
    "--policy-target-source", config.policyTargetSource,
    "--worker-mode",
    ...(config.blackModel ? ["--black-model", config.blackModel] : []),
    ...(config.whiteModel ? ["--white-model", config.whiteModel] : []),
  ];

  return new Promise((resolve, reject) => {
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
      if ((code ?? 1) === 0) resolve();
      else reject(new Error(`dataset worker exited with ${code}`));
    });
  });
}

async function appendFileToStream(path, stream) {
  await pipeline(createReadStream(path, { encoding: "utf-8" }), stream, { end: false });
}

export async function generateDatasetParallel(config) {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-dataset-"));
  const chunks = splitGames(config.games, config.parallel);
  await mkdir(dirname(config.out), { recursive: true });

  try {
    const shardPaths = chunks.map((_, index) => join(tempDir, `dataset-${String(index + 1).padStart(3, "0")}.jsonl`));
    await Promise.all(chunks.map((games, index) => runDatasetWorker(config, games, shardPaths[index])));

    let totalPositions = 0;
    const output = createWriteStream(config.out, { encoding: "utf-8" });
    try {
      for (const shardPath of shardPaths) {
        const meta = JSON.parse(await readFile(`${shardPath}.meta.json`, "utf-8"));
        totalPositions += meta.totalPositions ?? 0;
        await appendFileToStream(shardPath, output);
      }
    } finally {
      await new Promise((resolve, reject) => {
        output.end((error) => {
          if (error) reject(error);
          else resolve();
        });
      });
    }

    await writeFile(
      `${config.out}.meta.json`,
      `${JSON.stringify({
        games: config.games,
        totalPositions,
        blackAi: config.blackAi ?? "expert",
        whiteAi: config.whiteAi ?? "expert",
        blackModel: config.blackModel ?? null,
        whiteModel: config.whiteModel ?? null,
        policyTargetSource: config.policyTargetSource ?? "auto",
        teacherMs: config.teacherMs,
        startPolicy: config.startPolicy,
        parallel: config.parallel,
      }, null, 2)}\n`,
    );

    console.log(`Saved ${totalPositions} samples to ${config.out} using ${chunks.length} workers`);
    return {
      games: config.games,
      totalPositions,
      out: config.out,
      metaPath: `${config.out}.meta.json`,
      parallel: config.parallel,
    };
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await main();
}

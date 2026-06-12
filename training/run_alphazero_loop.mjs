import { access, copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { runArena } from "./arena_runtime.mjs";
import { runParallelArena } from "./run_arena.mjs";
import { eloGateDecision } from "./elo.mjs";
import { ensureModelRegistry, getActiveModel, promoteModel, registerModel } from "./model_registry.mjs";
import { runDistributedSelfPlay } from "./run_distributed_selfplay.mjs";
import { sampleReplayBufferToDataset } from "./replay_buffer.mjs";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const defaultBrowserModel = join(root, "apps", "web", "public", "models", "blokus_policy_value.onnx");

function parseArgs(argv) {
  const args = {
    iterations: 1,
    workers: 2,
    games: 20,
    teacherMs: 1000,
    sampleSize: 4096,
    epochs: 1,
    batchSize: 2048,
    cpu: false,
    evaluationGames: 6,
    arenaParallel: 1,
    evaluationOpponent: null,
    evaluationOpponentModel: null,
    candidateMs: 300,
    baselineMs: 300,
    minWinRate: 0.55,
    minAverageMargin: 0.5,
    minEloLowerBoundGain: 0,
    kFactor: 24,
    publishBest: false,
    replayBufferDir: join(root, "training", "replay_buffer"),
    registryDir: join(root, "training", "model_registry"),
    baseReportDir: join(root, "training", "reports", "alphazero"),
    maxBufferShards: 64,
    maxBufferSamples: 50000,
    startPolicy: "fixedStart",
    replaySampleStrategy: "priority",
    shardCompression: "gzip",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--iterations") args.iterations = Number(argv[++index]);
    if (value === "--workers") args.workers = Number(argv[++index]);
    if (value === "--games") args.games = Number(argv[++index]);
    if (value === "--teacher-ms") args.teacherMs = Number(argv[++index]);
    if (value === "--sample-size") args.sampleSize = Number(argv[++index]);
    if (value === "--epochs") args.epochs = Number(argv[++index]);
    if (value === "--batch-size") args.batchSize = Number(argv[++index]);
    if (value === "--cpu") {
      const next = argv[index + 1];
      if (next && !next.startsWith("--")) {
        args.cpu = next !== "false";
        index += 1;
      } else {
        args.cpu = true;
      }
    }
    if (value === "--evaluation-games") args.evaluationGames = Number(argv[++index]);
    if (value === "--arena-parallel") args.arenaParallel = Math.max(1, Number(argv[++index]));
    if (value === "--evaluation-opponent") args.evaluationOpponent = argv[++index];
    if (value === "--evaluation-opponent-model") args.evaluationOpponentModel = argv[++index];
    if (value === "--candidate-ms") args.candidateMs = Number(argv[++index]);
    if (value === "--baseline-ms") args.baselineMs = Number(argv[++index]);
    if (value === "--min-win-rate") args.minWinRate = Number(argv[++index]);
    if (value === "--min-average-margin") args.minAverageMargin = Number(argv[++index]);
    if (value === "--min-elo-lower-bound-gain") args.minEloLowerBoundGain = Number(argv[++index]);
    if (value === "--k-factor") args.kFactor = Number(argv[++index]);
    if (value === "--publish-best") {
      const next = argv[index + 1];
      if (next && !next.startsWith("--")) {
        args.publishBest = next !== "false";
        index += 1;
      } else {
        args.publishBest = true;
      }
    }
    if (value === "--replay-buffer-dir") args.replayBufferDir = argv[++index];
    if (value === "--registry-dir") args.registryDir = argv[++index];
    if (value === "--base-report-dir") args.baseReportDir = argv[++index];
    if (value === "--max-buffer-shards") args.maxBufferShards = Number(argv[++index]);
    if (value === "--max-buffer-samples") args.maxBufferSamples = Number(argv[++index]);
    if (value === "--start-policy") args.startPolicy = argv[++index];
    if (value === "--replay-sample-strategy") args.replaySampleStrategy = argv[++index];
    if (value === "--shard-compression") args.shardCompression = argv[++index];
  }
  return args;
}

async function exists(path) {
  try {
    await access(path, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function runNode(command, args) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: root,
      stdio: "inherit",
      shell: false,
    });
    child.on("exit", (code) => {
      if ((code ?? 1) === 0) resolve();
      else reject(new Error(`${command} ${args.join(" ")} exited with ${code}`));
    });
  });
}

async function runIteration(config, iterationIndex) {
  const iterationTag = `iter-${String(iterationIndex + 1).padStart(3, "0")}`;
  const iterationDir = join(config.baseReportDir, iterationTag);
  const workerDir = join(iterationDir, "workers");
  const datasetPath = join(iterationDir, "train_dataset.jsonl");
  const checkpointDir = join(iterationDir, "checkpoints");
  const candidatePath = join(iterationDir, "candidate_policy_value.onnx");
  const arenaSummaryPath = join(iterationDir, "arena-summary.json");
  const iterationSummaryPath = join(iterationDir, "iteration-summary.json");
  await mkdir(iterationDir, { recursive: true });

  const registryContext = await ensureModelRegistry(config.registryDir);
  const activeModel = await getActiveModel(config.registryDir);
  const activeModelPath = activeModel?.file ?? (await exists(defaultBrowserModel) ? defaultBrowserModel : null);

  const selfPlaySummary = await runDistributedSelfPlay({
    workers: config.workers,
    games: config.games,
    teacherMs: config.teacherMs,
    difficulty: "master",
    modelPath: activeModelPath,
    replayBufferDir: config.replayBufferDir,
    workerOutputDir: workerDir,
    maxBufferShards: config.maxBufferShards,
    maxBufferSamples: config.maxBufferSamples,
    startPolicy: config.startPolicy,
    policyTargetSource: "visit",
    shardCompression: config.shardCompression,
  });

  const sampled = await sampleReplayBufferToDataset(config.replayBufferDir, datasetPath, {
    maxSamples: config.sampleSize,
    seed: iterationIndex + 1,
    strategy: config.replaySampleStrategy,
  });

  await runNode("node", [
    join(root, "scripts", "run-python.mjs"),
    "training/train_policy_value.py",
    "--dataset",
    datasetPath,
    "--output-dir",
    checkpointDir,
    "--epochs",
    String(config.epochs),
    "--batch-size",
    String(config.batchSize),
    ...(config.cpu ? ["--cpu"] : []),
  ]);

  await runNode("node", [
    join(root, "scripts", "run-python.mjs"),
    "training/export_onnx.py",
    "--checkpoint",
    join(checkpointDir, "policy_value_latest.pt"),
    "--model-kind",
    "policy_value",
    "--out",
    candidatePath,
  ]);

  const opponentDifficulty = config.evaluationOpponent ?? (activeModelPath ? "master" : "expert");
  const opponentModelPath = config.evaluationOpponent
    ? (config.evaluationOpponentModel ?? null)
    : activeModelPath;
  const opponentName = config.evaluationOpponent ?? (activeModel ? activeModel.id : opponentDifficulty);
  const arenaArgs = {
    games: config.evaluationGames,
    parallel: config.arenaParallel,
    startPolicy: config.startPolicy,
    swapColors: true,
    alphaAi: "master",
    alphaName: "candidate",
    alphaModel: candidatePath,
    alphaMs: config.candidateMs,
    betaAi: opponentDifficulty,
    betaName: opponentName,
    betaModel: opponentModelPath,
    betaMs: config.baselineMs,
  };
  const arena = config.arenaParallel > 1
    ? await runParallelArena(arenaArgs)
    : await runArena({
      games: config.evaluationGames,
      startPolicy: config.startPolicy,
      swapColors: true,
      alpha: {
        name: "candidate",
        spec: {
          difficulty: "master",
          modelPath: candidatePath,
          timeLimitMs: config.candidateMs,
        },
      },
      beta: {
        name: opponentName,
        spec: {
          difficulty: opponentDifficulty,
          modelPath: opponentModelPath,
          timeLimitMs: config.baselineMs,
        },
      },
    });
  await writeFile(arenaSummaryPath, `${JSON.stringify(arena, null, 2)}\n`, "utf-8");

  const candidate = arena.contestants.candidate;
  const winRate = candidate.wins / Math.max(1, config.evaluationGames);
  const baselineName = opponentName;
  const useActiveBestEloGate = activeModel && !config.evaluationOpponent;
  const eloDecision = useActiveBestEloGate
    ? eloGateDecision({
      arena,
      candidateName: "candidate",
      opponentName: baselineName,
      candidateRating: activeModel.rating ?? 1200,
      opponentRating: activeModel.rating ?? 1200,
      kFactor: config.kFactor,
      minLowerBoundGain: config.minEloLowerBoundGain,
    })
    : null;
  const promote = useActiveBestEloGate
    ? eloDecision.promote
    : (winRate >= config.minWinRate && candidate.averageMargin >= config.minAverageMargin);

  const trainingSummary = JSON.parse(await readFile(join(checkpointDir, "train_summary.json"), "utf-8"));
  const registeredModel = await registerModel(config.registryDir, candidatePath, {
    kind: "policy_value",
    parentModelId: activeModel?.id ?? null,
    source: "alphazero-loop",
    status: "candidate",
    metrics: {
      winRate,
      averageMargin: candidate.averageMargin,
      evaluationGames: config.evaluationGames,
      arenaParallel: config.arenaParallel,
      evaluationOpponent: opponentDifficulty,
      evaluationOpponentModel: opponentModelPath,
      eloDecision,
    },
    arenaSummaryPath,
    trainingSummaryPath: join(checkpointDir, "train_summary.json"),
    rating: activeModel?.rating ?? 1200,
    ratingGames: activeModel?.ratingGames ?? 0,
  });

  let finalModelEntry = registeredModel;
  if (promote) {
    const opponentUpdatedRating = activeModel && eloDecision
      ? eloDecision.ratingUpdate.ratingB
      : activeModel?.rating;
    const opponentUpdatedGames = activeModel
      ? (activeModel.ratingGames ?? 0) + config.evaluationGames
      : activeModel?.ratingGames;
    finalModelEntry = await promoteModel(config.registryDir, registeredModel.id, {
      reason: "candidate passed arena gate",
      metrics: registeredModel.metrics,
      arenaSummaryPath,
      rating: activeModel && eloDecision ? eloDecision.ratingUpdate.ratingA : (registeredModel.rating ?? 1200),
      ratingGames: (registeredModel.ratingGames ?? 0) + config.evaluationGames,
      ratingHistoryEntry: eloDecision ? {
        at: new Date().toISOString(),
        arenaSummaryPath,
        rating: eloDecision.ratingUpdate.ratingA,
        scoreRate: eloDecision.score.scoreRate,
      } : null,
      opponentRating: opponentUpdatedRating,
      opponentRatingGames: opponentUpdatedGames,
      opponentRatingHistoryEntry: activeModel && eloDecision ? {
        at: new Date().toISOString(),
        arenaSummaryPath,
        rating: eloDecision.ratingUpdate.ratingB,
        scoreRate: 1 - eloDecision.score.scoreRate,
      } : null,
    });
    if (config.publishBest) {
      await copyFile(finalModelEntry.file, defaultBrowserModel);
    }
  }

  const summary = {
    generatedAt: new Date().toISOString(),
    iteration: iterationIndex + 1,
    activeBaselineModelId: activeModel?.id ?? null,
    activeBaselineModelPath: activeModelPath,
    evaluationOpponent: {
      name: opponentName,
      difficulty: opponentDifficulty,
      modelPath: opponentModelPath,
    },
    selfPlaySummary,
    sampledReplay: sampled,
    trainingSummary,
    candidateModel: finalModelEntry,
    arena,
    eloDecision,
    promote,
    publishBest: promote && config.publishBest,
    registryPath: registryContext.registryPath,
  };
  await writeFile(iterationSummaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf-8");
  return summary;
}

async function main() {
  const config = parseArgs(process.argv.slice(2));
  await mkdir(config.baseReportDir, { recursive: true });
  const summaries = [];
  for (let index = 0; index < config.iterations; index += 1) {
    summaries.push(await runIteration(config, index));
  }
  const loopSummary = {
    generatedAt: new Date().toISOString(),
    iterations: summaries,
    registryDir: config.registryDir,
    replayBufferDir: config.replayBufferDir,
  };
  await writeFile(join(config.baseReportDir, "loop-summary.json"), `${JSON.stringify(loopSummary, null, 2)}\n`, "utf-8");
  console.log(JSON.stringify(loopSummary, null, 2));
}

await main();

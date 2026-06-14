import { access, copyFile, mkdir, writeFile } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const binaryPath = join(root, "target", "release", "blokus_burn_rs");
const defaultWebPolicyValueModel = join(root, "apps", "web", "public", "models", "blokus_policy_value.onnx");

function parseArgs(argv) {
  const args = {
    iterations: 1,
    workers: 2,
    games: 20,
    teacherMs: 1000,
    sampleSize: 0,
    epochs: 1,
    batchSize: 2048,
    evaluationGames: 0,
    candidateMs: 300,
    baselineMs: 300,
    minEloLowerBoundGain: 0,
    publishBest: false,
    maxBufferShards: 0,
    maxBufferSamples: 0,
    replaySampleStrategy: "priority",
    startPolicy: "fixedStart",
    baseReportDir: join(root, "training", "reports", "alphazero-burn"),
    replayDir: join(root, "training", "replay_buffer_rs"),
    modelDir: join(root, "training", "models"),
    onnxOut: defaultWebPolicyValueModel,
    channels: 64,
    lr: 3e-4,
    valueWeight: 0.5,
    simulations: 256,
    candidateLimit: 120,
    explorationC: 1.5,
    seed: 7,
    cuda: null,
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
    if (value === "--evaluation-games") args.evaluationGames = Number(argv[++index]);
    if (value === "--candidate-ms") args.candidateMs = Number(argv[++index]);
    if (value === "--baseline-ms") args.baselineMs = Number(argv[++index]);
    if (value === "--min-elo-lower-bound-gain") args.minEloLowerBoundGain = Number(argv[++index]);
    if (value === "--max-buffer-shards") args.maxBufferShards = Number(argv[++index]);
    if (value === "--max-buffer-samples") args.maxBufferSamples = Number(argv[++index]);
    if (value === "--replay-sample-strategy") args.replaySampleStrategy = argv[++index];
    if (value === "--start-policy") args.startPolicy = argv[++index];
    if (value === "--base-report-dir") args.baseReportDir = argv[++index];
    if (value === "--replay-dir") args.replayDir = argv[++index];
    if (value === "--model-dir") args.modelDir = argv[++index];
    if (value === "--onnx-out") args.onnxOut = argv[++index];
    if (value === "--channels" || value === "--hidden") args.channels = Number(argv[++index]);
    if (value === "--lr") args.lr = Number(argv[++index]);
    if (value === "--value-weight") args.valueWeight = Number(argv[++index]);
    if (value === "--mcts-simulations" || value === "--simulations") args.simulations = Number(argv[++index]);
    if (value === "--candidate-limit") args.candidateLimit = Number(argv[++index]);
    if (value === "--exploration-c") args.explorationC = Number(argv[++index]);
    if (value === "--seed") args.seed = Number(argv[++index]);
    if (value === "--cuda") {
      const next = argv[index + 1];
      if (next && !next.startsWith("--")) {
        args.cuda = next !== "false";
        index += 1;
      } else {
        args.cuda = true;
      }
    }
    if (value === "--publish-best") {
      const next = argv[index + 1];
      if (next && !next.startsWith("--")) {
        args.publishBest = next !== "false";
        index += 1;
      } else {
        args.publishBest = true;
      }
    }
  }
  return args;
}

async function existsExecutable(path) {
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function existsFile(path) {
  try {
    await access(path, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function run(command, args, options = {}) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: root,
      stdio: "inherit",
      shell: false,
      ...options,
    });
    child.on("exit", (code) => {
      if ((code ?? 1) === 0) resolve();
      else reject(new Error(`${command} ${args.join(" ")} exited with ${code}`));
    });
  });
}

async function shouldUseCuda(config) {
  if (config.cuda !== null) return config.cuda;
  if (process.platform !== "linux") return false;
  try {
    await run("nvidia-smi", [], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function splitGames(totalGames, workers) {
  const base = Math.floor(totalGames / workers);
  const remainder = totalGames % workers;
  return Array.from({ length: workers }, (_, index) => base + (index < remainder ? 1 : 0))
    .filter((games) => games > 0);
}

async function ensureBinary(config) {
  const useCuda = await shouldUseCuda(config);
  const args = ["build", "--release", "-p", "blokus_burn_rs"];
  if (useCuda) args.push("--features", "cuda");
  await run("cargo", args);
  if (!(await existsExecutable(binaryPath))) {
    throw new Error(`Burn binary was not built: ${binaryPath}`);
  }
  return useCuda;
}

async function runSelfPlayWorkers(config, iterationDir, modelPath, iterationIndex) {
  const workerDir = join(iterationDir, "workers");
  await mkdir(workerDir, { recursive: true });
  const gameSplits = splitGames(config.games, Math.max(1, config.workers));
  const replayPaths = gameSplits.map((_, index) => join(workerDir, `worker-${String(index + 1).padStart(3, "0")}.bin`));

  await Promise.all(gameSplits.map((games, index) => run(binaryPath, [
    "selfplay",
    "--games", String(games),
    "--out", replayPaths[index],
    "--channels", String(config.channels),
    "--start-policy", config.startPolicy,
    "--simulations", String(config.simulations),
    "--time-ms", String(config.teacherMs),
    "--candidate-limit", String(config.candidateLimit),
    "--exploration-c", String(config.explorationC),
    "--seed", String(config.seed + iterationIndex * 1000 + index),
    ...(modelPath ? ["--model", modelPath] : []),
  ])));

  const mergedReplay = join(iterationDir, "replay.bin");
  await run(binaryPath, ["merge", "--out", mergedReplay, ...replayPaths]);
  return { mergedReplay, replayPaths };
}

async function exportBurnModelToOnnx(config, modelPath, iterationDir) {
  const weightsJson = join(iterationDir, "burn_torch_weights.json");
  await run(binaryPath, [
    "export-weights",
    "--model", modelPath,
    "--out", weightsJson,
    "--channels", String(config.channels),
  ]);
  await run("node", [
    join(root, "scripts", "run-python.mjs"),
    "training/export_burn_onnx.py",
    "--weights-json", weightsJson,
    "--out", config.onnxOut,
    "--channels", String(config.channels),
  ]);
  return { weightsJson, onnxOut: config.onnxOut };
}

export async function runBurnAlphaZeroLoop(config = {}) {
  const useCuda = await ensureBinary(config);
  await mkdir(config.baseReportDir, { recursive: true });
  await mkdir(config.replayDir, { recursive: true });
  await mkdir(config.modelDir, { recursive: true });

  const summaries = [];
  let activeModel = null;
  const publishedBurnModel = join(config.modelDir, "best_burn_policy_value.bin");
  if (await existsFile(publishedBurnModel)) activeModel = publishedBurnModel;

  for (let iterationIndex = 0; iterationIndex < config.iterations; iterationIndex += 1) {
    const tag = `iter-${String(iterationIndex + 1).padStart(3, "0")}`;
    const iterationDir = join(config.baseReportDir, tag);
    const checkpointDir = join(iterationDir, "checkpoint");
    await mkdir(iterationDir, { recursive: true });

    const { mergedReplay, replayPaths } = await runSelfPlayWorkers(config, iterationDir, activeModel, iterationIndex);
    const replaySnapshot = join(config.replayDir, `${tag}.bin`);
    await copyFile(mergedReplay, replaySnapshot);

    await run(binaryPath, [
      "train",
      "--replay", mergedReplay,
      "--output-dir", checkpointDir,
      "--epochs", String(config.epochs),
      "--batch-size", String(config.batchSize),
      "--channels", String(config.channels),
      "--lr", String(config.lr),
      "--value-weight", String(config.valueWeight),
    ]);

    activeModel = join(checkpointDir, "model.bin");
    const summary = {
      iteration: iterationIndex + 1,
      backend: useCuda ? "cuda" : "flex",
      replay: mergedReplay,
      replaySnapshot,
      workerReplays: replayPaths,
      model: activeModel,
      acceptedArgs: {
        games: config.games,
        workers: config.workers,
        teacherMs: config.teacherMs,
        sampleSize: config.sampleSize,
        epochs: config.epochs,
        batchSize: config.batchSize,
        evaluationGames: config.evaluationGames,
        candidateMs: config.candidateMs,
        baselineMs: config.baselineMs,
        replaySampleStrategy: config.replaySampleStrategy,
      },
    };
    summaries.push(summary);
    await writeFile(join(iterationDir, "iteration-summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf-8");
  }

  if (config.publishBest && activeModel) {
    await copyFile(activeModel, publishedBurnModel);
  }
  const exportedWebModel = config.publishBest && activeModel
    ? await exportBurnModelToOnnx(config, activeModel, config.baseReportDir)
    : null;

  const loopSummary = {
    backend: useCuda ? "cuda" : "flex",
    publishedBurnModel: config.publishBest ? publishedBurnModel : null,
    webModel: exportedWebModel?.onnxOut ?? null,
    webModelWeightsJson: exportedWebModel?.weightsJson ?? null,
    webModelNote: config.publishBest
      ? "Burn best model was converted through the Python PolicyValueNet exporter and published as ONNX for the web app."
      : "No web ONNX model was written because --publish-best was not set.",
    iterations: summaries,
  };
  await writeFile(join(config.baseReportDir, "loop-summary.json"), `${JSON.stringify(loopSummary, null, 2)}\n`, "utf-8");
  return loopSummary;
}

async function main() {
  const config = parseArgs(process.argv.slice(2));
  const summary = await runBurnAlphaZeroLoop(config);
  console.log(JSON.stringify(summary, null, 2));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await main();
}

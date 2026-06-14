import { access, mkdir, writeFile } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { addShardToReplayBuffer, pruneReplayBuffer, replayBufferStats } from "./replay_buffer.mjs";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

function parseArgs(argv) {
  const args = {
    workers: 2,
    games: 20,
    teacherMs: 1000,
    difficulty: "master",
    modelPath: null,
    replayBufferDir: join(root, "training", "replay_buffer"),
    workerOutputDir: join(root, "training", "reports", "distributed-selfplay"),
    maxBufferShards: 64,
    maxBufferSamples: 50000,
    startPolicy: "fixedStart",
    policyTargetSource: "visit",
    shardCompression: "gzip",
    selfplayBackend: "node",
  };

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--workers") args.workers = Math.max(1, Number(argv[++index]));
    if (value === "--games") args.games = Math.max(1, Number(argv[++index]));
    if (value === "--teacher-ms") args.teacherMs = Number(argv[++index]);
    if (value === "--difficulty") args.difficulty = argv[++index];
    if (value === "--model-path") args.modelPath = argv[++index];
    if (value === "--replay-buffer-dir") args.replayBufferDir = argv[++index];
    if (value === "--worker-output-dir") args.workerOutputDir = argv[++index];
    if (value === "--max-buffer-shards") args.maxBufferShards = Number(argv[++index]);
    if (value === "--max-buffer-samples") args.maxBufferSamples = Number(argv[++index]);
    if (value === "--start-policy") args.startPolicy = argv[++index];
    if (value === "--policy-target-source") args.policyTargetSource = argv[++index];
    if (value === "--shard-compression") args.shardCompression = argv[++index];
    if (value === "--selfplay-backend") args.selfplayBackend = argv[++index];
  }
  return args;
}

function splitGames(totalGames, workers) {
  const base = Math.floor(totalGames / workers);
  const remainder = totalGames % workers;
  return Array.from({ length: workers }, (_, index) => base + (index < remainder ? 1 : 0))
    .filter((games) => games > 0);
}

async function runWorker(workerSpec) {
  const useRustFast = workerSpec.selfplayBackend === "rust-fast";
  const command = useRustFast ? join(root, "target", "release", "blokus_train_rs") : "node";
  if (useRustFast) {
    await access(command, fsConstants.X_OK).catch(() => {
      throw new Error(`Rust self-play backend is not built: ${command}. Run "pnpm run build:rust" first.`);
    });
  }
  const args = useRustFast
    ? [
      "--out",
      workerSpec.out,
      "--games",
      String(workerSpec.games),
      "--start-policy",
      workerSpec.startPolicy,
      "--seed",
      String(workerSpec.seed),
      "--teacher-ms",
      String(workerSpec.teacherMs),
      "--difficulty",
      workerSpec.difficulty,
      "--policy-target-source",
      workerSpec.policyTargetSource,
      ...(workerSpec.modelPath ? ["--model-path", workerSpec.modelPath] : []),
    ]
    : [
      join(root, "training", "run_selfplay_worker.mjs"),
      "--worker-id",
      workerSpec.workerId,
      "--out",
      workerSpec.out,
      "--games",
      String(workerSpec.games),
      "--teacher-ms",
      String(workerSpec.teacherMs),
      "--difficulty",
      workerSpec.difficulty,
      "--start-policy",
      workerSpec.startPolicy,
      "--policy-target-source",
      workerSpec.policyTargetSource,
      ...(workerSpec.modelPath ? ["--model-path", workerSpec.modelPath] : []),
    ];
  await new Promise((resolve, reject) => {
    const child = spawn(
      command,
      args,
      {
        cwd: root,
        stdio: "inherit",
        shell: false,
      },
    );
    child.on("exit", (code) => {
      if ((code ?? 1) === 0) resolve();
      else reject(new Error(`worker ${workerSpec.workerId} exited with ${code}`));
    });
  });
}

export async function runDistributedSelfPlay(config = {}) {
  const workers = splitGames(config.games ?? 20, config.workers ?? 2);
  await mkdir(config.workerOutputDir, { recursive: true });
  const workerSpecs = workers.map((games, index) => ({
    workerId: `worker-${String(index + 1).padStart(3, "0")}`,
    out: join(config.workerOutputDir, `worker-${String(index + 1).padStart(3, "0")}.jsonl`),
    games,
    seed: (config.seed ?? 7) + index,
    teacherMs: config.teacherMs ?? 1000,
    difficulty: config.difficulty ?? "master",
    modelPath: config.modelPath ?? null,
    startPolicy: config.startPolicy ?? "fixedStart",
    policyTargetSource: config.policyTargetSource ?? "visit",
    selfplayBackend: config.selfplayBackend ?? "node",
  }));

  await Promise.all(workerSpecs.map(runWorker));
  const imported = [];
  for (const workerSpec of workerSpecs) {
    imported.push(await addShardToReplayBuffer(config.replayBufferDir, workerSpec.out, {
      source: "distributed-selfplay",
      gameCount: workerSpec.games,
      actorPair: [workerSpec.difficulty, workerSpec.difficulty],
      metadata: {
        workerId: workerSpec.workerId,
        modelPath: workerSpec.modelPath ?? null,
        teacherMs: workerSpec.teacherMs,
      },
      compression: config.shardCompression ?? "gzip",
    }));
  }

  const pruneSummary = await pruneReplayBuffer(config.replayBufferDir, {
    maxShards: config.maxBufferShards,
    maxSamples: config.maxBufferSamples,
  });
  const stats = await replayBufferStats(config.replayBufferDir);
  const summary = {
    generatedAt: new Date().toISOString(),
    workerSpecs,
    importedShardIds: imported.map((entry) => entry.id),
    pruneSummary,
    stats,
  };
  await writeFile(join(config.workerOutputDir, "distributed-summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf-8");
  return summary;
}

async function main() {
  const config = parseArgs(process.argv.slice(2));
  const summary = await runDistributedSelfPlay(config);
  console.log(JSON.stringify(summary, null, 2));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await main();
}

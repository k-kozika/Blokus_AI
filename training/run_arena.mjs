import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { runArena } from "./arena_runtime.mjs";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

function parseArgs(argv) {
  const args = {
    games: 8,
    startPolicy: "fixedStart",
    swapColors: true,
    alphaAi: "learned",
    alphaName: "candidate",
    alphaModel: null,
    alphaMs: 500,
    betaAi: "expert",
    betaName: "expert",
    betaModel: null,
    betaMs: 300,
    out: join(root, "training", "reports", "arena-summary.json"),
    parallel: 1,
    indexOffset: 0,
    workerMode: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--games") args.games = Number(argv[++index]);
    if (value === "--start-policy") args.startPolicy = argv[++index];
    if (value === "--swap-colors") args.swapColors = argv[++index] !== "false";
    if (value === "--alpha-ai") args.alphaAi = argv[++index];
    if (value === "--alpha-name") args.alphaName = argv[++index];
    if (value === "--alpha-model") args.alphaModel = argv[++index];
    if (value === "--alpha-ms") args.alphaMs = Number(argv[++index]);
    if (value === "--beta-ai") args.betaAi = argv[++index];
    if (value === "--beta-name") args.betaName = argv[++index];
    if (value === "--beta-model") args.betaModel = argv[++index];
    if (value === "--beta-ms") args.betaMs = Number(argv[++index]);
    if (value === "--out") args.out = argv[++index];
    if (value === "--parallel") args.parallel = Math.max(1, Number(argv[++index]));
    if (value === "--index-offset") args.indexOffset = Math.max(0, Number(argv[++index]));
    if (value === "--worker-mode") args.workerMode = true;
  }

  return args;
}

function arenaConfigFromArgs(args) {
  return {
    games: args.games,
    indexOffset: args.indexOffset,
    startPolicy: args.startPolicy,
    swapColors: args.swapColors,
    alpha: {
      name: args.alphaName,
      spec: {
        difficulty: args.alphaAi,
        modelPath: args.alphaModel,
        timeLimitMs: args.alphaMs,
      },
    },
    beta: {
      name: args.betaName,
      spec: {
        difficulty: args.betaAi,
        modelPath: args.betaModel,
        timeLimitMs: args.betaMs,
      },
    },
  };
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

function runChild(args, chunk, out) {
  const childArgs = [
    fileURLToPath(import.meta.url),
    "--games", String(chunk.games),
    "--index-offset", String(chunk.offset),
    "--start-policy", args.startPolicy,
    "--swap-colors", String(args.swapColors),
    "--alpha-ai", args.alphaAi,
    "--alpha-name", args.alphaName,
    "--alpha-ms", String(args.alphaMs),
    "--beta-ai", args.betaAi,
    "--beta-name", args.betaName,
    "--beta-ms", String(args.betaMs),
    "--out", out,
    "--worker-mode",
    ...(args.alphaModel ? ["--alpha-model", args.alphaModel] : []),
    ...(args.betaModel ? ["--beta-model", args.betaModel] : []),
  ];

  return new Promise((resolve, reject) => {
    const child = spawn("node", childArgs, {
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
      else reject(new Error(`arena worker exited with ${code}`));
    });
  });
}

function mergeSummaries(summaries, args) {
  const alphaName = args.alphaName;
  const betaName = args.betaName;
  const first = summaries[0];
  const games = summaries.flatMap((summary) => summary.games).sort((a, b) => a.index - b.index);
  const contestants = {};

  for (const name of [alphaName, betaName]) {
    const base = first.contestants[name];
    const merged = summaries.reduce((acc, summary) => {
      const current = summary.contestants[name];
      acc.wins += current.wins;
      acc.losses += current.losses;
      acc.draws += current.draws;
      acc.totalMargin += current.totalMargin;
      return acc;
    }, {
      difficulty: base.difficulty,
      modelPath: base.modelPath ?? null,
      wins: 0,
      losses: 0,
      draws: 0,
      totalMargin: 0,
    });
    merged.averageMargin = Number((merged.totalMargin / args.games).toFixed(3));
    contestants[name] = merged;
  }

  return {
    totalGames: args.games,
    swapColors: args.swapColors,
    startPolicy: args.startPolicy,
    parallel: args.parallel,
    contestants,
    games,
  };
}

export async function runParallelArena(args) {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-arena-"));
  try {
    const chunks = splitGames(args.games, args.parallel);
    await Promise.all(chunks.map((chunk, index) => {
      const out = join(tempDir, `arena-${String(index + 1).padStart(3, "0")}.json`);
      return runChild(args, chunk, out);
    }));
    const summaries = [];
    for (let index = 0; index < chunks.length; index += 1) {
      const out = join(tempDir, `arena-${String(index + 1).padStart(3, "0")}.json`);
      summaries.push(JSON.parse(await readFile(out, "utf-8")));
    }
    return mergeSummaries(summaries, args);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const summary = args.parallel > 1 && !args.workerMode
    ? await runParallelArena(args)
    : await runArena(arenaConfigFromArgs(args));

  await mkdir(dirname(args.out), { recursive: true });
  await writeFile(args.out, `${JSON.stringify(summary, null, 2)}\n`, "utf-8");
  if (!args.workerMode) {
    console.log(JSON.stringify(summary, null, 2));
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await main();
}

import { createReadStream, createWriteStream } from "node:fs";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { pipeline } from "node:stream/promises";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { spawn } from "node:child_process";
import {
  CRITICAL_REPLAY_SCHEMA_VERSION,
  phaseForState,
  softmax,
  validateBranchRecord,
} from "./branch_dataset_schema.mjs";
import { selectCriticalStates } from "./select_critical_states.mjs";
import { replayCriticalState } from "./replay_from_state.mjs";

const root = dirname(dirname(dirname(fileURLToPath(import.meta.url))));

export function parseBranchArgs(argv) {
  const args = {
    trajectories: join(root, "training", "dataset", "trajectories", "smoke"),
    out: join(root, "training", "dataset", "critical_replay", "smoke"),
    criticalStatesPerGame: 8,
    topKActions: 8,
    playoutsPerAction: 2,
    teacher: "expert",
    teacherModel: null,
    teacherMs: 25,
    temperature: 0.7,
    epsilon: 0.08,
    parallel: 1,
    workerMode: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--trajectories") args.trajectories = argv[++index];
    if (value === "--out") args.out = argv[++index];
    if (value === "--critical-states-per-game") args.criticalStatesPerGame = Number(argv[++index]);
    if (value === "--top-k-actions") args.topKActions = Number(argv[++index]);
    if (value === "--playouts-per-action") args.playoutsPerAction = Number(argv[++index]);
    if (value === "--teacher") args.teacher = argv[++index];
    if (value === "--teacher-model") args.teacherModel = argv[++index];
    if (value === "--teacher-ms") args.teacherMs = Number(argv[++index]);
    if (value === "--temperature") args.temperature = Number(argv[++index]);
    if (value === "--epsilon") args.epsilon = Number(argv[++index]);
    if (value === "--parallel") args.parallel = Math.max(1, Number(argv[++index]));
    if (value === "--worker-mode") args.workerMode = true;
  }

  return args;
}

function normalizeTrajectoryPath(path) {
  const resolved = resolve(path);
  return resolved.endsWith(".jsonl") ? resolved : join(resolved, "trajectories.jsonl");
}

async function readTrajectories(path) {
  const trajectories = [];
  const input = createReadStream(normalizeTrajectoryPath(path), { encoding: "utf-8" });
  const lines = createInterface({ input, crlfDelay: Infinity });
  for await (const line of lines) {
    if (line.trim()) trajectories.push(JSON.parse(line));
  }
  return trajectories;
}

function buildRecord(criticalState, replayResult, config) {
  const policyProbs = softmax(replayResult.qValues, config.temperature);
  let bestIndex = 0;
  for (let index = 1; index < replayResult.qValues.length; index += 1) {
    if (replayResult.qValues[index] > replayResult.qValues[bestIndex]) bestIndex = index;
  }
  const bestAction = replayResult.evaluatedActions[bestIndex];
  const chosenIndex = replayResult.evaluatedActions.indexOf(criticalState.chosen_action);
  const chosenQ = chosenIndex >= 0 ? replayResult.qValues[chosenIndex] : Math.min(...replayResult.qValues);
  const blunderScore = Math.max(0, replayResult.qValues[bestIndex] - chosenQ);

  const record = {
    schema_version: CRITICAL_REPLAY_SCHEMA_VERSION,
    source: "critical_state_replay",
    state_id: criticalState.state_hash,
    game_id: criticalState.game_id,
    turn: criticalState.turn,
    player: criticalState.player,
    phase: criticalState.phase ?? phaseForState(criticalState.state),
    actor_difficulty: criticalState.actor_difficulty,
    teacher_ai: config.teacher,
    encoded_state: criticalState.encoded_state,
    legal_actions: criticalState.legal_actions,
    evaluated_actions: replayResult.evaluatedActions,
    q_values: replayResult.qValues,
    raw_score_diffs: replayResult.rawScoreDiffs,
    policy_probs: policyProbs,
    chosen_action: criticalState.chosen_action,
    best_action: bestAction,
    selected_action: bestAction,
    expert_selected_action: criticalState.chosen_action,
    policy_target_actions: replayResult.evaluatedActions,
    policy_target_probs: policyProbs,
    value_target: replayResult.qValues[bestIndex],
    final_score_diff: replayResult.rawScoreDiffs[bestIndex] ?? 0,
    blunder_score: blunderScore,
    sampling_priority: 1 + 12 * blunderScore + Math.max(0, criticalState.critical_priority ?? 0) / 10,
    playouts_per_action: replayResult.playoutsPerAction,
    critical_priority: criticalState.critical_priority,
    selection_metrics: {
      before: criticalState.before,
      after: criticalState.after,
      deltas: criticalState.deltas,
    },
  };
  return validateBranchRecord(record);
}

export async function buildBranchDataset(config = {}) {
  const trajectories = await readTrajectories(config.trajectories);
  const outDir = resolve(config.out);
  await mkdir(outDir, { recursive: true });
  const recordsPath = join(outDir, "records.jsonl");
  const stream = createWriteStream(recordsPath, { encoding: "utf-8" });
  let criticalStates = 0;
  let records = 0;

  try {
    for (const trajectory of trajectories) {
      const selected = selectCriticalStates(trajectory, config);
      criticalStates += selected.length;
      for (const criticalState of selected) {
        const replayResult = await replayCriticalState(criticalState, config);
        const record = buildRecord(criticalState, replayResult, config);
        records += 1;
        if (!stream.write(`${JSON.stringify(record)}\n`)) {
          await new Promise((resolveDrain) => stream.once("drain", resolveDrain));
        }
      }
      console.log(`Critical replay ${trajectory.game_id}: ${selected.length} states`);
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
    trajectories: trajectories.length,
    criticalStates,
    records,
    recordsPath,
    topKActions: config.topKActions,
    playoutsPerAction: config.playoutsPerAction,
    teacher: config.teacher,
    teacherModel: config.teacherModel ?? null,
    teacherMs: config.teacherMs,
  };
  await writeFile(join(outDir, "metadata.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf-8");
  return summary;
}

async function main() {
  const config = parseBranchArgs(process.argv.slice(2));
  if (config.parallel > 1 && !config.workerMode) {
    await buildBranchDatasetParallel(config);
  } else {
    await buildBranchDataset(config);
  }
}

function splitArray(items, workers) {
  const actualWorkers = Math.max(1, Math.min(workers, items.length));
  const chunks = Array.from({ length: actualWorkers }, () => []);
  for (let index = 0; index < items.length; index += 1) {
    chunks[index % actualWorkers].push(items[index]);
  }
  return chunks.filter((chunk) => chunk.length > 0);
}

function runCriticalReplayWorker(config, trajectoriesPath, outDir) {
  const args = [
    fileURLToPath(import.meta.url),
    "--trajectories", trajectoriesPath,
    "--out", outDir,
    "--critical-states-per-game", String(config.criticalStatesPerGame),
    "--top-k-actions", String(config.topKActions),
    "--playouts-per-action", String(config.playoutsPerAction),
    "--teacher", config.teacher,
    "--teacher-ms", String(config.teacherMs),
    "--temperature", String(config.temperature),
    "--epsilon", String(config.epsilon),
    "--worker-mode",
    ...(config.teacherModel ? ["--teacher-model", config.teacherModel] : []),
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
      else reject(new Error(`critical replay worker exited with ${code}`));
    });
  });
}

async function appendFileToStream(path, stream) {
  await pipeline(createReadStream(path, { encoding: "utf-8" }), stream, { end: false });
}

export async function buildBranchDatasetParallel(config) {
  const trajectories = await readTrajectories(config.trajectories);
  const outDir = resolve(config.out);
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-critical-replay-"));
  const chunks = splitArray(trajectories, config.parallel);
  await mkdir(outDir, { recursive: true });

  try {
    const workerSpecs = [];
    for (let index = 0; index < chunks.length; index += 1) {
      const workerDir = join(tempDir, `worker-${String(index + 1).padStart(3, "0")}`);
      await mkdir(workerDir, { recursive: true });
      const trajectoriesPath = join(workerDir, "trajectories.jsonl");
      await writeFile(
        trajectoriesPath,
        chunks[index].map((trajectory) => JSON.stringify(trajectory)).join("\n") + "\n",
        "utf-8",
      );
      workerSpecs.push({
        trajectoriesPath,
        outDir: join(workerDir, "critical_replay"),
      });
    }

    await Promise.all(workerSpecs.map((spec) => runCriticalReplayWorker(config, spec.trajectoriesPath, spec.outDir)));

    let criticalStates = 0;
    let records = 0;
    const recordsPath = join(outDir, "records.jsonl");
    const output = createWriteStream(recordsPath, { encoding: "utf-8" });
    try {
      for (const spec of workerSpecs) {
        const meta = JSON.parse(await readFile(join(spec.outDir, "metadata.json"), "utf-8"));
        criticalStates += meta.criticalStates ?? 0;
        records += meta.records ?? 0;
        await appendFileToStream(join(spec.outDir, "records.jsonl"), output);
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
      trajectories: trajectories.length,
      criticalStates,
      records,
      recordsPath,
      topKActions: config.topKActions,
      playoutsPerAction: config.playoutsPerAction,
      teacher: config.teacher,
      teacherModel: config.teacherModel ?? null,
      teacherMs: config.teacherMs,
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

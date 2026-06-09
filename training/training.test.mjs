import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createSuite } from "../tests/testHarness.mjs";
import { generateDataset } from "./generate_dataset.mjs";
import { runArena } from "./arena_runtime.mjs";
import { getDefaultLearnedModelPath } from "./ai_runtime.mjs";
import { addShardToReplayBuffer, replayBufferStats, sampleReplayBufferToDataset } from "./replay_buffer.mjs";
import { ensureModelRegistry, getActiveModel, promoteModel, registerModel } from "./model_registry.mjs";
import { claimNextJob, completeJob, enqueueSelfPlayJobs, queueStatus } from "./distributed_queue.mjs";
import { eloGateDecision, updateEloRatings } from "./elo.mjs";
import { generateTrajectories } from "./critical_state_replay/collect_trajectory.mjs";
import { selectCriticalStates } from "./critical_state_replay/select_critical_states.mjs";
import { buildBranchDataset } from "./critical_state_replay/build_branch_dataset.mjs";

const suite = createSuite("training-environment");

suite.test("dataset generator writes ai-vs-ai samples with selected actions", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-dataset-"));
  const out = join(tempDir, "dataset.jsonl");
  const summary = await generateDataset({
    games: 1,
    out,
    teacherMs: 10,
    startPolicy: "fixedStart",
    blackAi: "easy",
    whiteAi: "normal",
  });

  const lines = (await readFile(out, "utf-8")).trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(summary.games, 1);
  assert.ok(lines.length > 0);
  assert.ok(lines.every((sample) => Number.isInteger(sample.selected_action)));
  assert.ok(lines.every((sample) => Array.isArray(sample.legal_actions) && sample.legal_actions.length > 0));
});

suite.test("master self-play dataset records policy targets", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-master-dataset-"));
  const out = join(tempDir, "dataset.jsonl");
  await generateDataset({
    games: 1,
    out,
    teacherMs: 50,
    startPolicy: "fixedStart",
    blackAi: "master",
    whiteAi: "master",
    blackModel: join(process.cwd(), "apps", "web", "public", "models", "blokus_policy_value.onnx"),
    whiteModel: join(process.cwd(), "apps", "web", "public", "models", "blokus_policy_value.onnx"),
    policyTargetSource: "visit",
  });
  const samples = (await readFile(out, "utf-8")).trim().split("\n").map((line) => JSON.parse(line));
  assert.ok(samples.some((sample) => Array.isArray(sample.policy_target_actions) && sample.policy_target_actions.length > 0));
  assert.ok(samples.every((sample) => Array.isArray(sample.policy_target_probs) && sample.policy_target_probs.length > 0));
});

suite.test("arena can compare learned against expert in node", async () => {
  const summary = await runArena({
    games: 2,
    swapColors: true,
    startPolicy: "fixedStart",
    alpha: {
      name: "learned",
      spec: {
        difficulty: "learned",
        modelPath: getDefaultLearnedModelPath(),
        timeLimitMs: 200,
      },
    },
    beta: {
      name: "expert",
      spec: {
        difficulty: "expert",
        timeLimitMs: 60,
      },
    },
  });

  assert.equal(summary.totalGames, 2);
  assert.equal(summary.games.length, 2);
  assert.equal(
    summary.contestants.learned.wins
      + summary.contestants.learned.losses
      + summary.contestants.learned.draws,
    2,
  );
});

suite.test("replay buffer can ingest and resample shards", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-replay-"));
  const datasetPath = join(tempDir, "dataset.jsonl");
  const replayDir = join(tempDir, "replay");
  const sampledPath = join(tempDir, "sampled.jsonl");
  await generateDataset({
    games: 1,
    out: datasetPath,
    teacherMs: 10,
    startPolicy: "fixedStart",
    blackAi: "easy",
    whiteAi: "normal",
  });
  await addShardToReplayBuffer(replayDir, datasetPath, {
    source: "test",
    gameCount: 1,
    actorPair: ["easy", "normal"],
  });
  const sampled = await sampleReplayBufferToDataset(replayDir, sampledPath, { maxSamples: 10, seed: 1 });
  const stats = await replayBufferStats(replayDir);
  assert.ok(sampled.sampleCount > 0);
  assert.equal(stats.activeShardIds.length, 1);
  assert.ok(stats.activeSampleCount > 0);
});

suite.test("replay buffer sampling streams bounded output", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-replay-stream-"));
  const datasetPath = join(tempDir, "dataset.jsonl");
  const replayDir = join(tempDir, "replay");
  const sampledPath = join(tempDir, "sampled.jsonl");
  const lines = Array.from({ length: 200 }, (_, index) => JSON.stringify({
    selected_action: index,
    legal_actions: [index],
    final_score_diff: index % 7,
    sampling_priority: index + 1,
    payload: "x".repeat(4096),
  }));
  await writeFile(datasetPath, `${lines.join("\n")}\n`, "utf-8");
  await addShardToReplayBuffer(replayDir, datasetPath, { source: "large-test" });

  const sampled = await sampleReplayBufferToDataset(replayDir, sampledPath, {
    maxSamples: 7,
    seed: 3,
    strategy: "priority",
  });
  const records = (await readFile(sampledPath, "utf-8")).trim().split("\n").map((line) => JSON.parse(line));
  const meta = JSON.parse(await readFile(`${sampledPath}.meta.json`, "utf-8"));
  assert.equal(sampled.sampleCount, 7);
  assert.equal(records.length, 7);
  assert.equal(meta.sampleCount, 7);
  assert.equal(meta.strategy, "priority");
});

suite.test("replay buffer supports gzip shards and priority sampling", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-priority-replay-"));
  const replayDir = join(tempDir, "replay");
  const highPath = join(tempDir, "high.jsonl");
  const lowPath = join(tempDir, "low.jsonl");
  const sampledPath = join(tempDir, "priority-sampled.jsonl");
  const highLines = Array.from({ length: 5 }, (_, index) => JSON.stringify({
    selected_action: index,
    legal_actions: [index],
    final_score_diff: 50,
    sampling_priority: 20,
  })).join("\n");
  const lowLines = Array.from({ length: 5 }, (_, index) => JSON.stringify({
    selected_action: index + 10,
    legal_actions: [index + 10],
    final_score_diff: 1,
    sampling_priority: 0.1,
  })).join("\n");
  await writeFile(highPath, `${highLines}\n`, "utf-8");
  await writeFile(lowPath, `${lowLines}\n`, "utf-8");
  await addShardToReplayBuffer(replayDir, highPath, { compression: "gzip", priority: 10 });
  await addShardToReplayBuffer(replayDir, lowPath, { compression: "gzip", priority: 1 });
  const sampled = await sampleReplayBufferToDataset(replayDir, sampledPath, {
    maxSamples: 5,
    seed: 7,
    strategy: "priority",
  });
  const records = (await readFile(sampledPath, "utf-8")).trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(sampled.strategy, "priority");
  assert.ok(records.some((record) => record.final_score_diff === 50));
  const stats = await replayBufferStats(replayDir);
  assert.equal(stats.shards[0].compression, "gzip");
});

suite.test("model registry can register and promote best models", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-registry-"));
  const registryDir = join(tempDir, "registry");
  const modelPath = join(process.cwd(), "apps", "web", "public", "models", "blokus_policy_value.onnx");
  await ensureModelRegistry(registryDir);
  const candidate = await registerModel(registryDir, modelPath, {
    source: "test",
    status: "candidate",
  });
  await promoteModel(registryDir, candidate.id, {
    reason: "test promotion",
    metrics: { winRate: 1 },
  });
  const active = await getActiveModel(registryDir);
  assert.equal(active?.id, candidate.id);
});

suite.test("promotion records previous active model correctly", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-registry-history-"));
  const registryDir = join(tempDir, "registry");
  const modelPath = join(process.cwd(), "apps", "web", "public", "models", "blokus_policy_value.onnx");
  await ensureModelRegistry(registryDir);
  const first = await registerModel(registryDir, modelPath, {
    source: "test",
    status: "candidate",
  });
  await promoteModel(registryDir, first.id, { reason: "first" });
  const second = await registerModel(registryDir, modelPath, {
    source: "test",
    status: "candidate",
  });
  await promoteModel(registryDir, second.id, { reason: "second" });
  const registry = JSON.parse(await readFile(join(registryDir, "registry.json"), "utf-8"));
  const lastPromotion = registry.promotions.at(-1);
  assert.equal(lastPromotion.fromModelId, first.id);
  assert.equal(lastPromotion.toModelId, second.id);
});

suite.test("distributed queue can enqueue, claim, and complete jobs", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-queue-"));
  const queueDir = join(tempDir, "queue");
  await enqueueSelfPlayJobs(queueDir, [{
    payload: { out: join(tempDir, "worker.jsonl"), games: 1 },
  }]);
  const claimed = await claimNextJob(queueDir, "host-a");
  assert.ok(claimed);
  await completeJob(queueDir, claimed.id, { ok: true });
  const status = await queueStatus(queueDir);
  assert.equal(status.counts.done, 1);
});

suite.test("elo gating produces bounded promotion decisions", () => {
  const arena = {
    totalGames: 10,
    contestants: {
      candidate: { wins: 7, losses: 2, draws: 1, averageMargin: 6 },
      best: { wins: 2, losses: 7, draws: 1, averageMargin: -6 },
    },
  };
  const decision = eloGateDecision({
    arena,
    candidateName: "candidate",
    opponentName: "best",
    candidateRating: 1200,
    opponentRating: 1200,
    minLowerBoundGain: -20,
  });
  assert.ok(Number.isFinite(decision.estimatedGain));
  assert.ok(Number.isFinite(decision.ratingUpdate.ratingA));
  assert.equal(typeof decision.promote, "boolean");
  const updated = updateEloRatings(1200, 1200, 0.75, 24);
  assert.ok(updated.ratingA > 1200);
});

suite.test("critical state replay generates branch policy-value records", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "blokus-critical-replay-"));
  const trajectoriesDir = join(tempDir, "trajectories");
  const branchDir = join(tempDir, "branch");
  const trajectorySummary = await generateTrajectories({
    games: 1,
    out: trajectoriesDir,
    teacherMs: 10,
    startPolicy: "fixedStart",
    blackAi: "easy",
    whiteAi: "easy",
  });
  assert.equal(trajectorySummary.games, 1);
  assert.ok(trajectorySummary.totalSteps > 0);

  const trajectory = JSON.parse((await readFile(join(trajectoriesDir, "trajectories.jsonl"), "utf-8")).trim());
  const selected = selectCriticalStates(trajectory, { criticalStatesPerGame: 2 });
  assert.ok(selected.length > 0);

  const summary = await buildBranchDataset({
    trajectories: trajectoriesDir,
    out: branchDir,
    criticalStatesPerGame: 2,
    topKActions: 3,
    playoutsPerAction: 1,
    teacher: "easy",
    teacherMs: 10,
    epsilon: 0,
  });
  assert.ok(summary.records > 0);

  const records = (await readFile(join(branchDir, "records.jsonl"), "utf-8"))
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  assert.ok(records.every((record) => record.source === "critical_state_replay"));
  assert.ok(records.every((record) => record.encoded_state.length === 51 * 14 * 14));
  assert.ok(records.every((record) => record.evaluated_actions.length === record.q_values.length));
  assert.ok(records.every((record) => record.evaluated_actions.length === record.policy_probs.length));
  assert.ok(records.every((record) => Math.abs(record.policy_probs.reduce((sum, value) => sum + value, 0) - 1) < 1e-6));
  assert.ok(records.every((record) => record.legal_actions.includes(record.best_action)));
});

export default suite;

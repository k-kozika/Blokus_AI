import { copyFile, createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { dirname, extname, join, resolve } from "node:path";
import readline from "node:readline";
import { once } from "node:events";
import { createGunzip, createGzip } from "node:zlib";

function nowIso() {
  return new Date().toISOString();
}

function createSeededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

async function pipelinePromise(source, ...streams) {
  await new Promise((resolve, reject) => {
    let current = source;
    for (const stream of streams) {
      current = current.pipe(stream);
    }
    current.on("finish", resolve);
    current.on("error", reject);
    source.on("error", reject);
    for (const stream of streams) {
      stream.on("error", reject);
    }
  });
}

function shardCompressionFromOptions(options = {}) {
  const compression = String(options.compression ?? "none").toLowerCase();
  return compression === "gzip" ? "gzip" : "none";
}

function shardFileName(shardId, compression) {
  return compression === "gzip" ? `${shardId}.jsonl.gz` : `${shardId}.jsonl`;
}

function isCompressedFile(path) {
  return extname(path).toLowerCase() === ".gz";
}

function createShardReadStream(path) {
  const source = createReadStream(path, { encoding: isCompressedFile(path) ? undefined : "utf-8" });
  if (!isCompressedFile(path)) return source;
  return source.pipe(createGunzip());
}

async function copyShard(sourcePath, destinationPath, compression) {
  if (compression === "gzip") {
    await pipelinePromise(
      createReadStream(sourcePath),
      createGzip({ level: 6 }),
      createWriteStream(destinationPath),
    );
    return;
  }
  await new Promise((resolve, reject) => {
    copyFile(sourcePath, destinationPath, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });
}

function parseSamplePriority(sample) {
  const samplePriority = Number(sample.sampling_priority ?? sample.priority ?? (
    Math.abs(Number(sample.final_score_diff ?? 0)) / 10
    + Math.log2(1 + Number(sample.policy_target_total_visits ?? 1))
  ));
  return Number.isFinite(samplePriority) && samplePriority > 0 ? samplePriority : 1;
}

function computeShardPriority(summary, options = {}) {
  const base = Number(options.priority ?? summary.averagePriority ?? 1);
  const recencyBoost = Number(options.recencyBoost ?? 0);
  const total = base + recencyBoost;
  return Number.isFinite(total) && total > 0 ? total : 1;
}

export function createEmptyManifest(bufferDir) {
  return {
    version: 2,
    generatedAt: nowIso(),
    bufferDir,
    activeShardIds: [],
    activeSampleCount: 0,
    activeGameCount: 0,
    nextShardIndex: 1,
    shards: [],
  };
}

export async function ensureReplayBuffer(bufferDir) {
  const resolvedDir = resolve(bufferDir);
  const shardsDir = join(resolvedDir, "shards");
  const manifestPath = join(resolvedDir, "manifest.json");
  await mkdir(shardsDir, { recursive: true });
  try {
    const manifest = JSON.parse(await readFile(manifestPath, "utf-8"));
    if (!manifest.version) manifest.version = 1;
    return { bufferDir: resolvedDir, shardsDir, manifestPath, manifest };
  } catch {
    const manifest = createEmptyManifest(resolvedDir);
    await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf-8");
    return { bufferDir: resolvedDir, shardsDir, manifestPath, manifest };
  }
}

async function saveManifest(context) {
  context.manifest.generatedAt = nowIso();
  await writeFile(context.manifestPath, `${JSON.stringify(context.manifest, null, 2)}\n`, "utf-8");
}

async function summarizeSamples(path) {
  let samples = 0;
  let totalPriority = 0;
  const rl = readline.createInterface({
    input: createShardReadStream(path),
    crlfDelay: Infinity,
  });
  for await (const line of rl) {
    if (!line.trim()) continue;
    samples += 1;
    totalPriority += parseSamplePriority(JSON.parse(line));
  }
  return {
    sampleCount: samples,
    averagePriority: samples > 0 ? totalPriority / samples : 1,
  };
}

export async function addShardToReplayBuffer(bufferDir, sourcePath, options = {}) {
  const context = await ensureReplayBuffer(bufferDir);
  const shardId = options.shardId ?? `shard-${String(context.manifest.nextShardIndex).padStart(6, "0")}`;
  const compression = shardCompressionFromOptions(options);
  const destinationPath = join(context.shardsDir, shardFileName(shardId, compression));
  const summary = await summarizeSamples(sourcePath);
  const sampleCount = options.sampleCount ?? summary.sampleCount;
  const gameCount = options.gameCount ?? null;
  await copyShard(sourcePath, destinationPath, compression);

  const entry = {
    id: shardId,
    file: destinationPath,
    createdAt: nowIso(),
    sampleCount,
    gameCount,
    compression,
    priority: computeShardPriority(summary, options),
    averagePriority: summary.averagePriority,
    source: options.source ?? "unknown",
    actorPair: options.actorPair ?? null,
    metadata: options.metadata ?? {},
  };
  context.manifest.shards.push(entry);
  context.manifest.activeShardIds.push(shardId);
  context.manifest.activeSampleCount += sampleCount;
  context.manifest.activeGameCount += gameCount ?? 0;
  context.manifest.nextShardIndex += 1;
  await saveManifest(context);
  return entry;
}

export async function listReplayShards(bufferDir) {
  const { manifest } = await ensureReplayBuffer(bufferDir);
  const active = new Set(manifest.activeShardIds);
  return manifest.shards.map((entry) => ({
    ...entry,
    active: active.has(entry.id),
  }));
}

export async function pruneReplayBuffer(bufferDir, options = {}) {
  const context = await ensureReplayBuffer(bufferDir);
  const maxShards = options.maxShards ?? null;
  const maxSamples = options.maxSamples ?? null;
  const activeIds = [...context.manifest.activeShardIds];
  if (!maxShards && !maxSamples) {
    return { removed: [], remaining: activeIds.length };
  }

  const removed = [];
  const activeEntries = activeIds
    .map((id) => context.manifest.shards.find((entry) => entry.id === id))
    .filter(Boolean)
    .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime());

  let currentSamples = activeEntries.reduce((sum, entry) => sum + entry.sampleCount, 0);
  while (
    (maxShards && activeEntries.length > maxShards)
    || (maxSamples && currentSamples > maxSamples)
  ) {
    const oldest = activeEntries.shift();
    if (!oldest) break;
    removed.push(oldest.id);
    currentSamples -= oldest.sampleCount;
    try {
      await rm(oldest.file, { force: true });
    } catch {}
  }

  context.manifest.activeShardIds = activeEntries.map((entry) => entry.id);
  context.manifest.activeSampleCount = currentSamples;
  context.manifest.activeGameCount = activeEntries.reduce((sum, entry) => sum + (entry.gameCount ?? 0), 0);
  await saveManifest(context);
  return {
    removed,
    remaining: context.manifest.activeShardIds.length,
    activeSampleCount: context.manifest.activeSampleCount,
  };
}

function normalizedSampleLimit(value, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 1) return Math.max(1, Number(fallback) || 1);
  return Math.max(1, Math.floor(parsed));
}

function pushPrioritySample(samples, sample, limit) {
  if (samples.length < limit) {
    samples.push(sample);
    return;
  }
  let minIndex = 0;
  for (let index = 1; index < samples.length; index += 1) {
    if (samples[index].key < samples[minIndex].key) minIndex = index;
  }
  if (sample.key > samples[minIndex].key) samples[minIndex] = sample;
}

function pushUniformSample(samples, sample, seen, random, limit) {
  if (samples.length < limit) {
    samples.push(sample);
    return;
  }
  const replaceIndex = Math.floor(random() * seen);
  if (replaceIndex < limit) samples[replaceIndex] = sample;
}

async function writeJsonlLines(path, entries) {
  await mkdir(dirname(path), { recursive: true });
  const stream = createWriteStream(path, { encoding: "utf-8" });
  try {
    for (const entry of entries) {
      if (!stream.write(`${entry.line}\n`)) await once(stream, "drain");
    }
    stream.end();
    await once(stream, "finish");
  } catch (error) {
    stream.destroy();
    throw error;
  }
}

export async function sampleReplayBufferToDataset(bufferDir, outputPath, options = {}) {
  const context = await ensureReplayBuffer(bufferDir);
  const activeEntries = context.manifest.activeShardIds
    .map((id) => context.manifest.shards.find((entry) => entry.id === id))
    .filter(Boolean);
  if (activeEntries.length === 0) {
    throw new Error(`Replay buffer at ${bufferDir} is empty.`);
  }

  const maxSamples = normalizedSampleLimit(options.maxSamples ?? context.manifest.activeSampleCount, context.manifest.activeSampleCount);
  const includeShards = options.includeShards ? new Set(options.includeShards) : null;
  const strategy = String(options.strategy ?? "uniform").toLowerCase();
  const seed = Number(options.seed ?? Date.now());
  const rng = createSeededRandom(seed);

  const selectedEntries = activeEntries.filter((entry) => !includeShards || includeShards.has(entry.id));
  let seen = 0;
  const picked = [];
  for (const entry of selectedEntries) {
    const rl = readline.createInterface({
      input: createShardReadStream(entry.file),
      crlfDelay: Infinity,
    });
    for await (const line of rl) {
      if (!line.trim()) continue;
      const sample = JSON.parse(line);
      seen += 1;
      const priorityWeight = strategy === "priority"
        ? entry.priority * parseSamplePriority(sample)
        : 1;
      const candidate = {
        line,
        shardId: entry.id,
        weight: priorityWeight,
      };
      if (strategy === "priority") {
        const weight = Math.max(1e-6, Number(candidate.weight ?? 1));
        const u = Math.max(1e-9, rng());
        pushPrioritySample(picked, { ...candidate, key: Math.pow(u, 1 / weight) }, maxSamples);
      } else {
        pushUniformSample(picked, candidate, seen, rng, maxSamples);
      }
    }
  }

  if (strategy === "priority") picked.sort((a, b) => b.key - a.key);
  await writeJsonlLines(outputPath, picked);
  const meta = {
    createdAt: nowIso(),
    bufferDir: context.bufferDir,
    sourceShardIds: [...new Set(picked.map((entry) => entry.shardId))],
    sampleCount: picked.length,
    seed,
    strategy,
  };
  await writeFile(`${outputPath}.meta.json`, `${JSON.stringify(meta, null, 2)}\n`, "utf-8");
  return {
    outputPath,
    sampleCount: picked.length,
    sourceShardIds: meta.sourceShardIds,
    metaPath: `${outputPath}.meta.json`,
    strategy,
  };
}

export async function replayBufferStats(bufferDir) {
  const context = await ensureReplayBuffer(bufferDir);
  const statsByFile = await Promise.all(
    context.manifest.shards.map(async (entry) => ({
      id: entry.id,
      exists: await stat(entry.file).then(() => true).catch(() => false),
      sampleCount: entry.sampleCount,
      gameCount: entry.gameCount ?? 0,
      compression: entry.compression ?? "none",
      priority: entry.priority ?? 1,
    })),
  );
  return {
    bufferDir: context.bufferDir,
    manifestPath: context.manifestPath,
    activeShardIds: context.manifest.activeShardIds,
    activeSampleCount: context.manifest.activeSampleCount,
    activeGameCount: context.manifest.activeGameCount,
    totalShardEntries: context.manifest.shards.length,
    shards: statsByFile,
  };
}

export async function importExistingShardDirectory(bufferDir, shardDir, options = {}) {
  const entries = await readdir(shardDir, { withFileTypes: true });
  const imported = [];
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    if (!entry.name.endsWith(".jsonl") && !entry.name.endsWith(".jsonl.gz")) continue;
    const fullPath = join(shardDir, entry.name);
    imported.push(await addShardToReplayBuffer(bufferDir, fullPath, {
      source: options.source ?? "import",
      compression: entry.name.endsWith(".gz") ? "gzip" : "none",
      metadata: { importedFrom: shardDir },
    }));
  }
  return imported;
}

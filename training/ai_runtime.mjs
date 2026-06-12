import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";
import { decideDifficultyMove, normalizeAiConfig } from "../apps/web/src/ai/difficulty.js";
import { chooseLearnedMove } from "../apps/web/src/ai/learnedAi.js";
import { chooseMasterMove } from "../apps/web/src/ai/policyValueMctsAi.js";
import { isLegalMove } from "../packages/core/src/index.js";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const defaultModelPath = join(root, "apps", "web", "public", "models", "blokus_policy.onnx");
const sessionCache = new Map();

const configuredOrtThreads = Number(process.env.BLOKUS_ORT_THREADS ?? 0);
if (Number.isInteger(configuredOrtThreads) && configuredOrtThreads > 0 && ort.env?.wasm) {
  ort.env.wasm.numThreads = configuredOrtThreads;
}

export const TRAINING_AI_DIFFICULTIES = ["easy", "normal", "hard", "expert", "expert_plus", "learned", "master"];

function normalizeDifficulty(value) {
  if (!value) return "expert";
  const lower = String(value).trim().toLowerCase();
  return TRAINING_AI_DIFFICULTIES.includes(lower) ? lower : "expert";
}

function normalizeModelPath(modelPath) {
  if (!modelPath) return null;
  return isAbsolute(modelPath) ? modelPath : resolve(root, modelPath);
}

export function normalizeTrainingAiSpec(spec = {}) {
  const difficulty = normalizeDifficulty(spec.difficulty ?? spec.engine);
  const timeLimitMs = Math.max(1, Number(spec.timeLimitMs ?? spec.maxThinkingMs ?? (
    difficulty === "expert_plus" ? 4000
      : difficulty === "master" ? 2500
      : difficulty === "expert" ? 1500
      : difficulty === "hard" ? 800
      : difficulty === "learned" ? 500
      : 300
  )));

  return {
    label: spec.label ?? difficulty,
    difficulty,
    engine: difficulty,
    timeLimitMs,
    maxThinkingMs: timeLimitMs,
    modelPath: normalizeModelPath(spec.modelPath),
    ...spec,
    difficulty,
    engine: difficulty,
    timeLimitMs,
    maxThinkingMs: timeLimitMs,
    modelPath: normalizeModelPath(spec.modelPath),
  };
}

async function loadLearnedSession(modelPath) {
  const resolvedPath = normalizeModelPath(modelPath) ?? defaultModelPath;
  if (!sessionCache.has(resolvedPath)) {
    sessionCache.set(
      resolvedPath,
      ort.InferenceSession.create(resolvedPath, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
    );
  }
  return sessionCache.get(resolvedPath);
}

async function loadPolicyValueSession(modelPath) {
  const resolvedPath = normalizeModelPath(modelPath) ?? join(root, "apps", "web", "public", "models", "blokus_policy_value.onnx");
  if (!sessionCache.has(resolvedPath)) {
    sessionCache.set(
      resolvedPath,
      ort.InferenceSession.create(resolvedPath, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
    );
  }
  return sessionCache.get(resolvedPath);
}

export async function decideTrainingMove(state, spec = {}) {
  const normalized = normalizeTrainingAiSpec(spec);
  let decision;

  if (normalized.difficulty === "learned") {
    const session = await loadLearnedSession(normalized.modelPath);
    decision = await chooseLearnedMove(state, normalized, {
      ortModule: ort,
      sessionFactory: async () => session,
    });
  } else if (normalized.difficulty === "master") {
    const session = await loadPolicyValueSession(normalized.modelPath);
    decision = await chooseMasterMove(state, normalized, {
      ortModule: ort,
      sessionFactory: async () => session,
    });
  } else {
    decision = await decideDifficultyMove(state, normalizeAiConfig(normalized));
  }

  if (!isLegalMove(state, decision.move)) {
    throw new Error(`AI "${normalized.label}" returned an illegal move.`);
  }

  return decision;
}

export function clearTrainingSessionCache() {
  sessionCache.clear();
}

export function getDefaultLearnedModelPath() {
  return defaultModelPath;
}

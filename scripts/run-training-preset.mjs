import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const preset = process.argv[2];
const rest = process.argv.slice(3);
const useCpu = rest.includes("--cpu");

function run(command, args) {
  return new Promise((resolve, reject) => {
    console.log(`\n$ ${command} ${args.join(" ")}`);
    const child = spawn(command, args, {
      cwd: root,
      stdio: "inherit",
      shell: process.platform === "win32",
    });
    child.on("exit", (code) => {
      if ((code ?? 1) === 0) resolve();
      else reject(new Error(`${command} ${args.join(" ")} exited with ${code}`));
    });
  });
}

function npmScript(script, args = []) {
  return run(process.platform === "win32" ? "npm.cmd" : "npm", ["run", script, "--", ...args]);
}

function maybeCpu(args) {
  return useCpu ? [...args, "--cpu"] : args;
}

if (!preset) {
  throw new Error("Usage: npm run preset:<name>. Available presets: smoke:learning, smoke:critical");
}

if (preset === "smoke:learning") {
  await npmScript("generate:orientations");
  await npmScript("generate:dataset", [
    "--games", "100",
    "--out", "training/data/smoke-expert-100.jsonl",
    "--teacher-ms", "10",
    "--start-policy", "fixedStart",
  ]);
  await npmScript("train:policy", maybeCpu([
    "--dataset", "training/data/smoke-expert-100.jsonl",
    "--epochs", "1",
    "--output-dir", useCpu ? "training/checkpoints/policy-100-cpu" : "training/checkpoints/policy-100-gpu",
    "--batch-size", "2048",
  ]));
  await npmScript("train:policy-value", maybeCpu([
    "--dataset", "training/data/smoke-expert-100.jsonl",
    "--epochs", "1",
    "--output-dir", useCpu ? "training/checkpoints/policy-value-100-cpu" : "training/checkpoints/policy-value-100-gpu",
    "--batch-size", "2048",
  ]));
  await npmScript("export:onnx", [
    "--checkpoint",
    `${useCpu ? "training/checkpoints/policy-100-cpu" : "training/checkpoints/policy-100-gpu"}/policy_latest.pt`,
  ]);
  await npmScript("export:onnx:pv", [
    "--checkpoint",
    `${useCpu ? "training/checkpoints/policy-value-100-cpu" : "training/checkpoints/policy-value-100-gpu"}/policy_value_latest.pt`,
    "--out",
    "apps/web/public/models/blokus_policy_value.onnx",
  ]);
} else if (preset === "smoke:critical") {
  await npmScript("generate:trajectory", [
    "--games", "10",
    "--ai", "expert",
    "--teacher-ms", "50",
    "--out", "training/dataset/trajectories/smoke-expert-10",
  ]);
  await npmScript("generate:critical-replay", [
    "--trajectories", "training/dataset/trajectories/smoke-expert-10",
    "--out", "training/dataset/critical_replay/smoke-expert-10",
    "--critical-states-per-game", "2",
    "--top-k-actions", "4",
    "--playouts-per-action", "1",
    "--teacher", "master",
    "--teacher-ms", "300",
    "--teacher-model", "apps/web/public/models/blokus_policy_value.onnx",
  ]);
  await npmScript("train:policy-value", maybeCpu([
    "--dataset", "training/dataset/critical_replay/smoke-expert-10/records.jsonl",
    "--epochs", "1",
    "--output-dir", useCpu ? "training/checkpoints/critical-smoke-cpu" : "training/checkpoints/critical-smoke-gpu",
    "--batch-size", "512",
  ]));
} else {
  throw new Error(`Unknown preset: ${preset}`);
}

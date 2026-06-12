import { access } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

async function exists(path) {
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function resolvePython() {
  const candidates = [
    process.env.BLOKUS_PYTHON,
    process.platform === "win32"
      ? join(root, ".venv", "Scripts", "python.exe")
      : join(root, ".venv", "bin", "python"),
    "python",
    "python3",
    process.platform === "win32"
      ? join(process.env.USERPROFILE || "", ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies", "python", "python.exe")
      : null,
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (candidate === "python" || candidate === "python3") return candidate;
    if (await exists(candidate)) return candidate;
  }

  throw new Error("Python runtime not found. Set BLOKUS_PYTHON to a usable python executable.");
}

const [scriptPath, ...args] = process.argv.slice(2);
if (!scriptPath) {
  throw new Error("Usage: node scripts/run-python.mjs <script.py> [args...]");
}

const python = await resolvePython();
const child = spawn("uv", ["run", scriptPath, ...args], {
  cwd: root,
  stdio: "inherit",
  shell: false,
});

child.on("exit", (code) => {
  process.exit(code ?? 1);
});

import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const mode = process.argv[2] ?? "gpu-cu130";

function run(args) {
  return new Promise((resolve, reject) => {
    const child = spawn("node", [join(root, "scripts", "run-python.mjs"), "-m", "pip", ...args], {
      cwd: root,
      stdio: "inherit",
      shell: false,
    });
    child.on("exit", (code) => {
      if ((code ?? 1) === 0) resolve();
      else reject(new Error(`pip ${args.join(" ")} exited with ${code}`));
    });
  });
}

if (mode === "cpu") {
  await run(["install", "--upgrade", "torch", "onnx"]);
} else if (mode === "gpu-cu130") {
  await run(["install", "--upgrade", "--force-reinstall", "torch", "--index-url", "https://download.pytorch.org/whl/cu130"]);
  await run(["install", "--upgrade", "onnx"]);
} else {
  throw new Error(`Unknown Python dependency mode: ${mode}. Use cpu or gpu-cu130.`);
}

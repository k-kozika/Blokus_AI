import { spawn } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

const snippet = `
import json
import sys

info = {
  "python": sys.executable,
  "python_version": sys.version.split()[0],
}

try:
  import torch
  info["torch"] = torch.__version__
  info["cuda_available"] = bool(torch.cuda.is_available())
  info["cuda_version"] = torch.version.cuda
  info["cuda_device_count"] = int(torch.cuda.device_count())
  info["cuda_devices"] = [
    torch.cuda.get_device_name(index)
    for index in range(torch.cuda.device_count())
  ]
  if torch.cuda.is_available():
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    info["cuda_tensor_smoke"] = str(y.device)
except Exception as error:
  info["torch_error"] = f"{type(error).__name__}: {error}"

try:
  import onnx
  info["onnx"] = onnx.__version__
except Exception as error:
  info["onnx_error"] = f"{type(error).__name__}: {error}"

print(json.dumps(info, indent=2))
`;

const child = spawn("node", [join(root, "scripts", "run-python.mjs"), "-c", snippet], {
  cwd: root,
  stdio: "inherit",
  shell: false,
});

child.on("exit", (code) => {
  process.exit(code ?? 1);
});

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor_py"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import torch

from policy_model import PolicyValueNet


def load_state_dict(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_kind") != "policy_value":
        raise SystemExit(f"Unsupported Burn export model_kind: {payload.get('model_kind')}")
    state = {}
    for entry in payload.get("tensors", []):
        name = entry["name"]
        shape = tuple(int(dim) for dim in entry["shape"])
        values = entry["data"]
        tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
        state[name] = tensor
    return state


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-json", required=True)
    parser.add_argument("--out", default=str(ROOT.parent / "apps" / "web" / "public" / "models" / "blokus_policy_value.onnx"))
    parser.add_argument("--channels", type=int, default=64)
    args = parser.parse_args()

    model = PolicyValueNet(channels=args.channels)
    state = load_state_dict(Path(args.weights_json))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [name for name in missing if not name.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise SystemExit(f"State dict mismatch. missing={missing} unexpected={unexpected}")
    model.eval()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 51, 14, 14), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["policy_logits", "value"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "policy_logits": {0: "batch"}, "value": {0: "batch"}},
    )
    print(output_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor_py"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split

from blokus_shared import ACTION_SIZE, STATE_PLANES
from policy_model import PolicyValueNet


def _read_jsonl(path: Path) -> List[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def _parse_simple_yaml_dataset_config(text: str) -> List[dict]:
    datasets = []
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "datasets:":
            continue
        if line.startswith("- "):
            if current:
                datasets.append(current)
            current = {}
            line = line[2:].strip()
        if ":" in line and current is not None:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip().strip("'\"")
    if current:
        datasets.append(current)
    return datasets


def _load_dataset_config(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        config = json.loads(text)
        return config.get("datasets", [])
    return _parse_simple_yaml_dataset_config(text)


def _iter_configured_dataset_specs(path: Path) -> Iterable[tuple[Path, float]]:
    if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
        yield path, 1.0
        return
    for entry in _load_dataset_config(path):
        dataset_path = Path(entry["path"])
        if not dataset_path.is_absolute():
            dataset_path = path.parent / dataset_path
        yield dataset_path, float(entry.get("weight", 1.0))


def _resolve_state(sample: dict):
    state = sample.get("encoded_state")
    if state is None and isinstance(sample.get("state"), list):
        state = sample["state"]
    if state is None:
        raise KeyError("sample is missing encoded_state")
    return np.asarray(state, dtype=np.float32).reshape(STATE_PLANES, 14, 14)


def _resolve_policy_target(sample: dict):
    target = int(sample.get(
        "selected_action",
        sample.get("best_action", sample.get("chosen_action", sample.get("expert_selected_action", -1))),
    ))
    if target < 0:
        raise KeyError("sample is missing selected_action/best_action/chosen_action")

    actions = sample.get("policy_target_actions")
    probs = sample.get("policy_target_probs")
    if not actions or not probs:
        actions = sample.get("evaluated_actions")
        probs = sample.get("policy_probs")
    return target, actions, probs


def _resolve_value_target(sample: dict):
    if "value_target" in sample:
        return max(-1.0, min(1.0, float(sample["value_target"])))
    if isinstance(sample.get("q_values"), list) and sample["q_values"]:
        return max(-1.0, min(1.0, float(max(sample["q_values"]))))
    score_diff = float(sample.get("final_score_diff", 0.0))
    return max(-1.0, min(1.0, score_diff / 89.0))


class PolicyValueDataset(Dataset):
    def __init__(self, path: Path):
        self.samples = []
        for dataset_path, weight in _iter_configured_dataset_specs(path):
            for sample in _read_jsonl(dataset_path):
                sample["_sample_weight"] = max(0.0, weight)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        state = _resolve_state(sample)
        mask = np.zeros((ACTION_SIZE,), dtype=np.float32)
        mask[sample["legal_actions"]] = 1.0
        target, actions, probs = _resolve_policy_target(sample)
        policy = np.zeros((ACTION_SIZE,), dtype=np.float32)
        if actions and probs and len(actions) == len(probs):
            for action, prob in zip(actions, probs):
                policy[int(action)] = float(prob)
            total = float(policy.sum())
            if total > 0:
                policy /= total
        else:
            policy[target] = 1.0
        value_target = _resolve_value_target(sample)
        return (
            torch.from_numpy(state),
            torch.from_numpy(mask),
            torch.from_numpy(policy),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(value_target, dtype=torch.float32),
            torch.tensor(float(sample.get("_sample_weight", 1.0)), dtype=torch.float32),
        )


def weighted_mean(values, sample_weight):
    if sample_weight is None:
        return values.mean()
    return (values * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)


def mask_illegal_logits(logits, legal_mask):
    return logits.float().masked_fill(legal_mask <= 0, -1e9)


def masked_policy_loss(logits, legal_mask, target_distribution, target_action, sample_weight=None):
    masked_logits = mask_illegal_logits(logits, legal_mask)
    if target_distribution is not None:
        normalized = target_distribution * legal_mask
        denom = normalized.sum(dim=1, keepdim=True).clamp_min(1e-8)
        normalized = normalized / denom
        log_probs = nn.functional.log_softmax(masked_logits, dim=1)
        return weighted_mean(-(normalized * log_probs).sum(dim=1), sample_weight)
    return weighted_mean(nn.functional.cross_entropy(masked_logits, target_action, reduction="none"), sample_weight)


def collate_batch(batch):
    states, masks, policies, targets, values, weights = zip(*batch)
    return (
        torch.stack(states),
        torch.stack(masks),
        torch.stack(policies),
        torch.stack(targets),
        torch.stack(values),
        torch.stack(weights),
    )


def evaluate(model, loader, device, amp_enabled, value_weight):
    model.eval()
    losses: List[float] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []
    correct = 0
    total = 0
    mae_sum = 0.0

    with torch.no_grad():
      for states, masks, policy_targets, targets, value_targets, sample_weights in loader:
        states = states.to(device)
        masks = masks.to(device)
        policy_targets = policy_targets.to(device)
        targets = targets.to(device)
        value_targets = value_targets.to(device)
        sample_weights = sample_weights.to(device)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits, values = model(states)
            values = values.squeeze(1)
            policy_loss = masked_policy_loss(logits, masks, policy_targets, targets, sample_weights)
            value_loss = weighted_mean((values - value_targets) ** 2, sample_weights)
            loss = policy_loss + value_weight * value_loss
        losses.append(float(loss.item()))
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        predictions = mask_illegal_logits(logits, masks).argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += int(targets.numel())
        mae_sum += float(torch.abs(values - value_targets).sum().item())

    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "policy_loss": float(sum(policy_losses) / max(1, len(policy_losses))),
        "value_loss": float(sum(value_losses) / max(1, len(value_losses))),
        "accuracy": float(correct / max(1, total)),
        "value_mae": float(mae_sum / max(1, total)),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = PolicyValueDataset(Path(args.dataset))
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty.")

    val_size = max(1, int(len(dataset) * args.validation_split))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_enabled = device.type == "cuda"
    model = PolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=amp_enabled)

    batch_size = min(args.batch_size, len(train_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=min(args.batch_size, max(1, len(val_set))), shuffle=False, collate_fn=collate_batch)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "policy_value_latest.pt"
    best_path = output_dir / "policy_value_best.pt"
    history = []
    best_val = math.inf

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        for states, masks, policy_targets, targets, value_targets, sample_weights in train_loader:
            states = states.to(device)
            masks = masks.to(device)
            policy_targets = policy_targets.to(device)
            targets = targets.to(device)
            value_targets = value_targets.to(device)
            sample_weights = sample_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits, values = model(states)
                values = values.squeeze(1)
                policy_loss = masked_policy_loss(logits, masks, policy_targets, targets, sample_weights)
                value_loss = weighted_mean((values - value_targets) ** 2, sample_weights)
                loss = policy_loss + args.value_weight * value_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(float(loss.item()))

        metrics = evaluate(model, val_loader, device, amp_enabled, args.value_weight)
        train_loss = float(sum(batch_losses) / max(1, len(batch_losses)))
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_policy_loss": metrics["policy_loss"],
            "val_value_loss": metrics["value_loss"],
            "val_value_mae": metrics["value_mae"],
        }
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "epoch": epoch + 1,
            "model_kind": "policy_value",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        }
        torch.save(checkpoint, checkpoint_path)
        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            torch.save(checkpoint, best_path)

    summary = {
        "device": device.type,
        "epochs": args.epochs,
        "samples": len(dataset),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(checkpoint_path),
        "history": history,
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "policy_value"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())

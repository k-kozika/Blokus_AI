from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List

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
from policy_model import PolicyNet


class PolicyDataset(Dataset):
    def __init__(self, path: Path):
        self.samples = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        state = np.asarray(sample["encoded_state"], dtype=np.float32).reshape(STATE_PLANES, 14, 14)
        mask = np.zeros((ACTION_SIZE,), dtype=np.float32)
        mask[sample["legal_actions"]] = 1.0
        target = int(sample.get("selected_action", sample["expert_selected_action"]))
        return (
            torch.from_numpy(state),
            torch.from_numpy(mask),
            torch.tensor(target, dtype=torch.long),
        )


def mask_illegal_logits(logits, legal_mask):
    return logits.float().masked_fill(legal_mask <= 0, -1e9)


def masked_cross_entropy(logits, legal_mask, target):
    masked_logits = mask_illegal_logits(logits, legal_mask)
    return nn.functional.cross_entropy(masked_logits, target)


def collate_batch(batch):
    states, masks, targets = zip(*batch)
    return torch.stack(states), torch.stack(masks), torch.stack(targets)


def evaluate(model, loader, device, amp_enabled):
    model.eval()
    losses: List[float] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for states, masks, targets in loader:
            states = states.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(states)
                loss = masked_cross_entropy(logits, masks, targets)
            losses.append(float(loss.item()))
            predictions = mask_illegal_logits(logits, masks).argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "accuracy": float(correct / max(1, total)),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = PolicyDataset(Path(args.dataset))
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
    model = PolicyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=amp_enabled)

    batch_size = min(args.batch_size, len(train_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=min(args.batch_size, max(1, len(val_set))), shuffle=False, collate_fn=collate_batch)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "policy_latest.pt"
    best_path = output_dir / "policy_best.pt"
    history = []
    best_val = math.inf

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        for states, masks, targets in train_loader:
            states = states.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(states)
                loss = masked_cross_entropy(logits, masks, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(float(loss.item()))

        metrics = evaluate(model, val_loader, device, amp_enabled)
        train_loss = float(sum(batch_losses) / max(1, len(batch_losses)))
        record = {
          "epoch": epoch + 1,
          "train_loss": train_loss,
          "val_loss": metrics["loss"],
          "val_accuracy": metrics["accuracy"],
        }
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "epoch": epoch + 1,
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
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "policy"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())

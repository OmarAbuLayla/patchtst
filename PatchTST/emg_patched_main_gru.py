# ==============================================================
#  EMG PatchTST GRU Training Script
#  Author: Omar A. Layla
#  Purpose: Train GRU model on pre-patched EMG signals
# ==============================================================
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import amp

from emg_patched_dataset import build_patch_dataloaders
from emg_patched_model_gru import EMG_GRU_PatchTST


# --------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRU model on patched EMG data (PatchTST-ready)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root (Train/Val/Test folders)")
    parser.add_argument("--num-classes", type=int, default=101, help="Number of target classes")
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="runs/patch_gru")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint to resume from")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--in-channels", type=int, default=768, help="Features per patch (num_channels × patch_len)")
    parser.add_argument("--gru-hidden", type=int, default=192)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--gru-dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine LR schedule instead of ReduceLROnPlateau")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patch-dropout", type=float, default=0.2, help="Probability of dropping patches during training")
    parser.add_argument("--stride-test", type=int, default=96, choices=[64, 96, 128], help="Simulated stride for evaluation")
    return parser.parse_args()


# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: Dict, directory: Path, filename: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / filename)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0)


# --------------------------------------------------------------
# Epoch Runner
# --------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    max_grad_norm: float,
    desc: str,
    log_interval: int = 0,
    num_classes: int | None = None,
) -> Tuple[float, float, torch.Tensor | None, torch.Tensor | None]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = len(loader)
    start_time = time.perf_counter()

    per_class_correct = None
    per_class_count = None
    if not is_train and num_classes is not None:
        per_class_correct = torch.zeros(num_classes, dtype=torch.long, device=device)
        per_class_count = torch.zeros(num_classes, dtype=torch.long, device=device)

    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)   # (B, seq_len, feature_dim)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(inputs)  # (B, num_classes)
            loss = criterion(outputs, targets)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        if per_class_correct is not None and per_class_count is not None:
            per_class_count += torch.bincount(targets, minlength=num_classes)
            correct_mask = preds == targets
            if correct_mask.any():
                per_class_correct += torch.bincount(targets[correct_mask], minlength=num_classes)

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            elapsed = time.perf_counter() - start_time
            mean_loss = total_loss / max(total_samples, 1)
            accuracy = total_correct / max(total_samples, 1)
            pct = 100.0 * batch_idx / num_batches
            print(f"{desc} [{pct:5.1f}%] Loss: {mean_loss:.4f}  Acc: {accuracy:.4f}  Time: {elapsed:5.1f}s", flush=True)

    mean_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    if per_class_correct is not None and per_class_count is not None:
        per_class_correct = per_class_correct.cpu()
        per_class_count = per_class_count.cpu()
    return mean_loss, accuracy, per_class_correct, per_class_count


# --------------------------------------------------------------
# Train + Eval Routine
# --------------------------------------------------------------
def train_and_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    torch.backends.cudnn.benchmark = True

    loaders = build_patch_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        patch_dropout=args.patch_dropout,
        stride_test=args.stride_test,
    )

    feature_dim = loaders["train"].dataset.feature_dim
    if args.in_channels != feature_dim:
        print(
            f"Input dimension mismatch: CLI requested {args.in_channels} but data provides {feature_dim}. Using dataset value."
        )
    model = EMG_GRU_PatchTST(
        input_dim=feature_dim,
        hidden_dim=args.gru_hidden,
        num_layers=args.gru_layers,
        num_classes=args.num_classes,
        dropout=args.gru_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=args.lr_patience, factor=0.5, verbose=True
        )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None

    save_dir = Path(args.save_dir)
    best_val_acc = 0.0
    history = []

    if args.test_only:
        for split_name in ["val", "test"]:
            loss, acc, per_class_correct, per_class_count = run_epoch(
                model,
                loaders[split_name],
                criterion,
                device,
                max_grad_norm=0.0,
                desc=f"{split_name} eval",
                num_classes=args.num_classes,
            )
            print(f"{split_name.capitalize()} split: loss={loss:.4f}, acc={acc:.4f}")
            if per_class_correct is not None and per_class_count is not None:
                per_class_acc = (per_class_correct.float() / per_class_count.clamp_min(1)).cpu().numpy()
                print(f"{split_name.capitalize()} per-class accuracy:", np.round(per_class_acc, 3))
        return

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} | LR={optimizer.param_groups[0]['lr']:.6g}", flush=True)
        train_loss, train_acc, _, _ = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            desc=f"Train {epoch + 1}",
            log_interval=args.log_interval,
        )
        val_loss, val_acc, val_correct, val_count = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            max_grad_norm=0.0,
            desc=f"Val {epoch + 1}",
            num_classes=args.num_classes,
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        if val_correct is not None and val_count is not None:
            per_class_acc = (val_correct.float() / val_count.clamp_min(1)).cpu().numpy()
            print("Validation per-class accuracy:", np.round(per_class_acc, 3))
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save checkpoint
        checkpoint_state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
        }
        save_checkpoint(checkpoint_state, save_dir, f"epoch_{epoch + 1:03d}.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(checkpoint_state, save_dir, f"best_epoch_{epoch + 1:03d}.pt")

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} "
            f"| train {train_acc:.4f} val {val_acc:.4f} "
            f"| loss {train_loss:.4f}/{val_loss:.4f}"
        )

    # Load best model
    best_path = save_dir / f"best_epoch_{epoch + 1:03d}.pt"
    if best_path.exists():
        load_checkpoint(str(best_path), model)

    test_loss, test_acc, test_correct, test_count = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        max_grad_norm=0.0,
        desc="Test",
        log_interval=args.log_interval,
        num_classes=args.num_classes,
    )
    print(f"\nTest set: loss={test_loss:.4f}, acc={test_acc:.4f}")
    if test_correct is not None and test_count is not None:
        per_class_acc = (test_correct.float() / test_count.clamp_min(1)).cpu().numpy()
        print("Test per-class accuracy:", np.round(per_class_acc, 3))

    with open(Path(args.save_dir) / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# --------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------
def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()

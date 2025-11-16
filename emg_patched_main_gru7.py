# ==============================================================
#  EMG PatchTST GRU Training Script (v6)
#  - FIXED test-only mode
#  - Saves every checkpoint epoch_XXX.pt
# ==============================================================

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import amp

from emg_patched_dataset7 import build_patch_dataloaders
from emg_patched_model_gru7 import EMG_GRU_PatchTST


# --------------------------------------------------------------
# Result container
# --------------------------------------------------------------
@dataclass
class EpochResult:
    loss: float
    accuracy: float


# --------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU PatchTST Hybrid")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--epochs-ablation", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="runs/patch_gru")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--in-channels", type=int, default=768)
    parser.add_argument("--gru-hidden", type=int, default=256)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--gru-dropout", type=float, default=0.12)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--use-cosine", action="store_true")
    parser.add_argument("--patch-dropout", type=float, default=0.0)
    parser.add_argument("--subset-train", type=int, default=0)
    parser.add_argument("--subset-val", type=int, default=0)
    parser.add_argument("--subset-test", type=int, default=0)
    return parser.parse_args(argv)


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


def load_checkpoint(path: str, model: nn.Module, optimizer=None) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def create_scheduler(optimizer, *, use_cosine, total_epochs, warmup_epochs, lr_patience, min_lr):
    if not use_cosine:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=lr_patience, factor=0.5, verbose=True
        )

    cosine_epochs = max(1, total_epochs - max(0, warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=min_lr
    )

    if warmup_epochs <= 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# --------------------------------------------------------------
# Epoch Runner
# --------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device,
    *,
    optimizer=None,
    scaler=None,
    max_grad_norm=1.0,
    desc="",
    log_interval=50,
) -> EpochResult:

    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start = time.perf_counter()

    for batch_idx, (inputs, targets, *_) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)

        if is_train:
            optimizer.zero_grad()

        with amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(inputs)
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
        total_correct += (outputs.argmax(1) == targets).sum().item()
        total_samples += batch_size

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == len(loader)):
            elapsed = time.perf_counter() - start
            print(
                f"{desc} [{batch_idx}/{len(loader)}] "
                f"Loss={total_loss/total_samples:.4f} "
                f"Acc={total_correct/total_samples:.4f} "
                f"Time={elapsed:.1f}s"
            )

    return EpochResult(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


# --------------------------------------------------------------
# Train + Eval Routine
# --------------------------------------------------------------
def train_and_evaluate(args):
    set_seed(args.seed)
    device = select_device(args.device)
    torch.backends.cudnn.benchmark = True

    loaders = build_patch_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.workers,

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #  FIX: PASS PATCH_LEN AND STRIDE
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        patch_len=args.patch_len,
        stride=args.stride,
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        patch_dropout=args.patch_dropout,
        subset_train=args.subset_train,
        subset_val=args.subset_val,
        subset_test=args.subset_test,
        subset_seed=args.seed,
    )

    train_dataset = loaders["train"].dataset
    if hasattr(train_dataset, "dataset"):
        train_dataset = train_dataset.dataset

    model = EMG_GRU_PatchTST(
        input_dim=train_dataset.feature_dim,
        num_channels=train_dataset.num_channels,
        hidden_dim=args.gru_hidden,
        num_layers=args.gru_layers,
        num_classes=args.num_classes,
        dropout=args.gru_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_epochs = args.epochs_ablation if args.epochs_ablation > 0 else args.epochs
    scheduler = create_scheduler(
        optimizer,
        use_cosine=args.use_cosine,
        total_epochs=total_epochs,
        warmup_epochs=min(args.warmup_epochs, max(total_epochs - 1, 0)),
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # TEST-ONLY BLOCK
    # ----------------------------------------------------------
    if args.test_only:
        print("\nTEST-ONLY MODE — Evaluating checkpoint\n")
        if args.resume:
            load_checkpoint(args.resume, model)

        for split in ["val", "test"]:
            result = run_epoch(
                model,
                loaders[split],
                criterion,
                device,
                max_grad_norm=0.0,
                desc=f"{split.upper()} TEST",
            )
            print(
                f"{split.upper()} | loss={result.loss:.4f}  acc={result.accuracy:.4f}"
            )
        return

    # ----------------------------------------------------------
    # TRAINING LOOP
    # ----------------------------------------------------------
    best_val_acc = 0.0
    history = []

    for epoch_idx in range(start_epoch, total_epochs):
        epoch_num = epoch_idx + 1
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch_num}/{total_epochs} | LR={lr:.6f}")

        train_result = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            desc=f"Train {epoch_num}",
            log_interval=args.log_interval,
        )

        val_result = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            max_grad_norm=0.0,
            desc=f"Val {epoch_num}",
        )

        # scheduler update
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_result.loss)
        else:
            scheduler.step()

        # SAVE EVERY CHECKPOINT
        ckpt = {
            "epoch": epoch_num,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        save_checkpoint(ckpt, save_dir, f"epoch_{epoch_num:03d}.pt")

        # SAVE BEST
        if val_result.accuracy > best_val_acc:
            best_val_acc = val_result.accuracy
            save_checkpoint(ckpt, save_dir, "best.pt")

        print(
            f"Epoch {epoch_num:03d}: "
            f"train_acc={train_result.accuracy:.4f}, val_acc={val_result.accuracy:.4f}, "
            f"train_loss={train_result.loss:.4f}, val_loss={val_result.loss:.4f}"
        )

        history.append(
            {
                "epoch": epoch_num,
                "train_loss": train_result.loss,
                "train_acc": train_result.accuracy,
                "val_loss": val_result.loss,
                "val_acc": val_result.accuracy,
                "lr": lr,
            }
        )

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


# --------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    train_and_evaluate(args)


if __name__ == "__main__":
    main()

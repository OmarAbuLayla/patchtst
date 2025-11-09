"""Training entry point for the 15-channel EMG model."""
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

from emg15_dataset import MFSCConfig, build_dataloaders
from emg15_model import EMGNet15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 15-channel EMG recogniser")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the dataset root (Train/Val/Test folders)")
    parser.add_argument("--num-classes", type=int, default=101, help="Number of target classes")
    parser.add_argument("--batch-size", type=int, default=48, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--workers", type=int, default=0, help="Data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--every-frame", action="store_true", default=True, help="Average frame logits for the loss")
    parser.add_argument("--no-every-frame", dest="every_frame", action="store_false", help="Use only the last frame for loss")
    parser.add_argument("--log-interval", type=int, default=50, help="Batches between progress logs (0 disables)")
    parser.add_argument("--save-dir", type=str, default="runs/15vc", help="Directory to store checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint to resume from")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run evaluation on the test split")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping value (<=0 disables)")
    parser.add_argument("--lr-patience", type=int, default=8, help="ReduceLROnPlateau patience in epochs")
    parser.add_argument("--in-channels", type=int, default=15, help="Number of input channels")
    parser.add_argument("--proj-dim", type=int, default=256, help="Projection dimension before the GRU")
    parser.add_argument("--gru-hidden", type=int, default=512, help="GRU hidden size")
    parser.add_argument("--gru-layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--gru-dropout", type=float, default=0.3, help="Dropout inside the GRU stack")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Preferred device")
    return parser.parse_args()


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


def get_dataset_size(loader: torch.utils.data.DataLoader) -> int | None:
    try:
        return len(loader.dataset)  # type: ignore[arg-type]
    except TypeError:
        return None


def run_epoch(
    model: EMGNet15,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    every_frame: bool,
    max_grad_norm: float,
    desc: str,
    log_interval: int = 0,
    dataset_size: int | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    num_batches = len(loader)
    if dataset_size is None:
        try:
            dataset_size = len(loader.dataset)  # type: ignore[arg-type]
        except TypeError:
            dataset_size = None

    start_time = time.perf_counter()

    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(inputs)
            if every_frame and outputs.ndim == 3:
                outputs = outputs.mean(dim=1)
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

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            elapsed = time.perf_counter() - start_time
            mean_loss = total_loss / max(total_samples, 1)
            accuracy = total_correct / max(total_samples, 1)
            if dataset_size:
                pct = 100.0 * total_samples / dataset_size
                remaining = max(dataset_size - total_samples, 0)
                samples_per_sec = total_samples / max(elapsed, 1e-6)
                eta = remaining / max(samples_per_sec, 1e-6)
                progress = f"[{total_samples:6d}/{dataset_size:6d} ({pct:4.0f}%)]"
            else:
                pct_batches = 100.0 * batch_idx / max(num_batches, 1)
                remaining_batches = max(num_batches - batch_idx, 0)
                batches_per_sec = batch_idx / max(elapsed, 1e-6)
                eta = remaining_batches / max(batches_per_sec, 1e-6)
                progress = f"[batch {batch_idx:4d}/{num_batches:4d} ({pct_batches:4.0f}%)]"
            prefix = f"{desc} | " if desc else ""
            print(
                f"{prefix}Process: {progress}"
                f"     Loss: {mean_loss:.4f}"
                f"    Acc:{accuracy:.4f}"
                f"      Cost time:{elapsed:6.0f}s"
                f"        Estimated time: {eta:6.0f}s",
                flush=True,
            )

    mean_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return mean_loss, accuracy


def train_and_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    torch.backends.cudnn.benchmark = True

    cfg = MFSCConfig()
    loaders = build_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        cfg=cfg,
    )

    train_size = get_dataset_size(loaders["train"])
    val_size = get_dataset_size(loaders["val"])
    test_size = get_dataset_size(loaders["test"])

    model = EMGNet15(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        proj_dim=args.proj_dim,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        every_frame=args.every_frame,
        gru_dropout=args.gru_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.lr_patience, factor=0.5, verbose=True
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

        if args.epochs <= start_epoch:
            additional_epochs = args.epochs if args.epochs > 0 else 1
            target_epochs = start_epoch + additional_epochs
            print(
                "Requested epoch count does not exceed the checkpoint epoch. "
                f"Continuing for {additional_epochs} additional epoch(s) (target epoch {target_epochs}).",
                flush=True,
            )
            args.epochs = target_epochs

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None

    save_dir = Path(args.save_dir)
    history = []
    best_val_acc = 0.0
    best_path: Path | str = ""

    if args.test_only:
        print(
            "Statistics: "
            f"train: {train_size if train_size is not None else 'unknown'}, "
            f"val: {val_size if val_size is not None else 'unknown'}, "
            f"test: {test_size if test_size is not None else 'unknown'}",
            flush=True,
        )
        for split_name, loader, size in (
            ("val", loaders["val"], val_size),
            ("test", loaders["test"], test_size),
        ):
            loss, acc = run_epoch(
                model,
                loader,
                criterion,
                device,
                every_frame=args.every_frame,
                max_grad_norm=0.0,
                desc=f"{split_name} eval",
                log_interval=args.log_interval,
                dataset_size=size,
            )
            print(
                f"{split_name.capitalize()} split: loss={loss:.4f}, acc={acc:.4f}",
                flush=True,
            )
        return

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)
        print(f"Current Learning rate: [{optimizer.param_groups[0]['lr']:.6g}]", flush=True)
        train_loss, train_acc = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            every_frame=args.every_frame,
            max_grad_norm=args.max_grad_norm,
            desc=f"train {epoch + 1}/{args.epochs}",
            log_interval=args.log_interval,
            dataset_size=train_size,
        )
        val_loss, val_acc = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            every_frame=args.every_frame,
            max_grad_norm=0.0,
            desc=f"val {epoch + 1}/{args.epochs}",
            log_interval=0,
        )
        scheduler.step(val_acc)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}"
            f" | train loss {train_loss:.4f} acc {train_acc:.4f}"
            f" | val loss {val_loss:.4f} acc {val_acc:.4f}"
            f" | lr {optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )

        checkpoint_state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
        }

        save_checkpoint(
            checkpoint_state,
            save_dir,
            f"epoch_{epoch + 1:03d}.pt",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = save_dir / f"best_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                checkpoint_state,
                save_dir,
                Path(best_path).name,
            )

    if best_path:
        print(f"Loading best model from {best_path}", flush=True)
        load_checkpoint(str(best_path), model)

    test_loss, test_acc = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        every_frame=args.every_frame,
        max_grad_norm=0.0,
        desc="test",
        log_interval=args.log_interval,
        dataset_size=test_size,
    )
    print(f"Test set: loss={test_loss:.4f}, acc={test_acc:.4f}", flush=True)

    # Persist training history for later analysis
    history_path = Path(args.save_dir) / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
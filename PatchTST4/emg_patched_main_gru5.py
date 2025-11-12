# ==============================================================
#  EMG PatchTST GRU Training Script (v4)
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

from emg_patched_dataset import build_patch_dataloaders
from emg_patched_model_gru4 import EMG_GRU_PatchTST


# --------------------------------------------------------------
# Data structures
# --------------------------------------------------------------
@dataclass
class EpochResult:
    loss: float
    accuracy: float
    per_class_correct: Optional[torch.Tensor] = None
    per_class_count: Optional[torch.Tensor] = None
    confusion: Optional[torch.Tensor] = None
    topk: Optional[Dict[str, float]] = None
    patch_keep_mean: Optional[float] = None
    patch_keep_std: Optional[float] = None
    input_std_min: Optional[float] = None
    input_std_max: Optional[float] = None
    logit_std_min: Optional[float] = None
    logit_std_max: Optional[float] = None


# --------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GRU model on patched EMG data (PatchTST-ready)"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to dataset root (Train/Val/Test folders)",
    )
    parser.add_argument("--num-classes", type=int, default=101, help="Number of target classes")
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--epochs-ablation",
        type=int,
        default=0,
        help="Optional override for number of training epochs during quick ablations",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Lower bound for cosine LR schedule")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warm-up epochs before cosine decay")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="runs/patch_gru")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint to resume from")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--in-channels", type=int, default=768, help="Features per patch (num_channels × patch_len)")
    parser.add_argument("--gru-hidden", type=int, default=256)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--gru-dropout", type=float, default=0.12)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine LR schedule instead of ReduceLROnPlateau")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--patch-dropout",
        type=float,
        default=0.0,
        help="Probability of dropping patches during training",
    )
    parser.add_argument(
        "--stride-test", type=int, default=64, choices=[64, 96, 128], help="Simulated stride for evaluation"
    )
    parser.add_argument(
        "--subset-train",
        type=int,
        default=0,
        help="Optional cap on number of training samples (0 = full set)",
    )
    parser.add_argument(
        "--subset-val",
        type=int,
        default=0,
        help="Optional cap on number of validation samples (0 = full set)",
    )
    parser.add_argument(
        "--subset-test",
        type=int,
        default=0,
        help="Optional cap on number of test samples (0 = full set)",
    )
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


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    use_cosine: bool,
    total_epochs: int,
    warmup_epochs: int,
    lr_patience: int,
    min_lr: float,
):
    if not use_cosine:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=lr_patience,
            factor=0.5,
            verbose=True,
        )

    cosine_epochs = max(1, total_epochs - max(0, warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=min_lr,
    )
    if warmup_epochs <= 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs,
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
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: float,
    desc: str,
    log_interval: int = 0,
    num_classes: Optional[int] = None,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = len(loader)
    start_time = time.perf_counter()

    patch_token_sum = 0.0
    patch_token_sq_sum = 0.0
    patch_token_count = 0

    input_std_min = float("inf")
    input_std_max = float("-inf")
    logit_std_min = float("inf")
    logit_std_max = float("-inf")

    per_class_correct = None
    per_class_count = None
    confusion = None
    top1_correct = 0.0
    top5_correct = 0.0

    if not is_train and num_classes is not None:
        per_class_correct = torch.zeros(num_classes, dtype=torch.long)
        per_class_count = torch.zeros(num_classes, dtype=torch.long)
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch_idx, batch in enumerate(loader, start=1):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            inputs, targets, kept_counts = batch
            kept_counts = kept_counts.to(torch.float32)
        else:
            inputs, targets = batch
            kept_counts = torch.full((inputs.size(0),), inputs.size(1), dtype=torch.float32)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if torch.isnan(inputs).any():
            raise ValueError("Detected NaNs in input batch")

        if is_train:
            optimizer.zero_grad()

        with amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if torch.isnan(outputs).any() or torch.isnan(loss).any():
            raise ValueError("Detected NaNs in model outputs or loss")

        batch_input_std = inputs.float().std().item()
        batch_logit_std = outputs.float().std().item()
        input_std_min = min(input_std_min, batch_input_std)
        input_std_max = max(input_std_max, batch_input_std)
        logit_std_min = min(logit_std_min, batch_logit_std)
        logit_std_max = max(logit_std_max, batch_logit_std)

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

        patch_token_sum += kept_counts.sum().item()
        patch_token_sq_sum += torch.square(kept_counts).sum().item()
        patch_token_count += kept_counts.numel()

        if per_class_correct is not None and per_class_count is not None:
            targets_cpu = targets.detach().cpu()
            preds_cpu = preds.detach().cpu()
            per_class_count += torch.bincount(targets_cpu, minlength=num_classes)
            correct_mask = preds_cpu == targets_cpu
            if correct_mask.any():
                per_class_correct += torch.bincount(targets_cpu[correct_mask], minlength=num_classes)
            cm_flat = torch.bincount(
                targets_cpu * num_classes + preds_cpu,
                minlength=num_classes * num_classes,
            )
            confusion += cm_flat.view(num_classes, num_classes)

            max_k = min(5, outputs.size(1))
            topk_idx = outputs.topk(max_k, dim=1).indices.detach().cpu()
            matches = topk_idx.eq(targets_cpu.unsqueeze(1))
            top1_correct += matches[:, :1].sum().item()
            top5_correct += matches[:, :max_k].any(dim=1).sum().item()

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            elapsed = time.perf_counter() - start_time
            mean_loss = total_loss / max(total_samples, 1)
            accuracy = total_correct / max(total_samples, 1)
            patch_mean = patch_token_sum / max(patch_token_count, 1)
            pct = 100.0 * batch_idx / num_batches
            print(
                f"{desc} [{pct:5.1f}%] Loss: {mean_loss:.4f}  Acc: {accuracy:.4f}  "
                f"Patches: {patch_mean:.2f}  Inputσ: {batch_input_std:.4f}  "
                f"Logitσ: {batch_logit_std:.4f}  Time: {elapsed:5.1f}s",
                flush=True,
            )

    mean_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    patch_keep_mean = None
    patch_keep_std = None
    if patch_token_count > 0:
        patch_keep_mean = patch_token_sum / patch_token_count
        variance = patch_token_sq_sum / patch_token_count - (patch_keep_mean ** 2)
        patch_keep_std = math.sqrt(max(variance, 0.0))

    topk = None
    if not is_train and num_classes is not None and total_samples > 0:
        topk = {
            "top1": top1_correct / total_samples,
            "top5": top5_correct / total_samples,
        }
        per_class_correct = per_class_correct.cpu()
        per_class_count = per_class_count.cpu()
        confusion = confusion.cpu()

    return EpochResult(
        loss=mean_loss,
        accuracy=accuracy,
        per_class_correct=per_class_correct,
        per_class_count=per_class_count,
        confusion=confusion,
        topk=topk,
        patch_keep_mean=patch_keep_mean,
        patch_keep_std=patch_keep_std,
        input_std_min=None if input_std_min == float("inf") else input_std_min,
        input_std_max=None if input_std_max == float("-inf") else input_std_max,
        logit_std_min=None if logit_std_min == float("inf") else logit_std_min,
        logit_std_max=None if logit_std_max == float("-inf") else logit_std_max,
    )


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
        subset_train=args.subset_train,
        subset_val=args.subset_val,
        subset_test=args.subset_test,
        subset_seed=args.seed,
    )

    train_dataset = loaders["train"].dataset
    if hasattr(train_dataset, "dataset"):
        train_dataset = train_dataset.dataset

    feature_dim = train_dataset.feature_dim
    num_channels = getattr(train_dataset, "num_channels", None)
    if num_channels is None:
        raise AttributeError("EMGPatchDataset is expected to expose 'num_channels' for model wiring")
    if args.in_channels != feature_dim:
        print(
            f"Input dimension mismatch: CLI requested {args.in_channels} but data provides {feature_dim}. Using dataset value."
        )

    model = EMG_GRU_PatchTST(
        input_dim=feature_dim,
        num_channels=num_channels,
        hidden_dim=args.gru_hidden,
        num_layers=args.gru_layers,
        num_classes=args.num_classes,
        dropout=args.gru_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_epochs = args.epochs_ablation if args.epochs_ablation and args.epochs_ablation > 0 else args.epochs
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

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    if args.test_only:
        for split_name in ["val", "test"]:
            result = run_epoch(
                model,
                loaders[split_name],
                criterion,
                device,
                max_grad_norm=0.0,
                desc=f"{split_name} eval",
                num_classes=args.num_classes,
            )
            print(f"{split_name.capitalize()} split: loss={result.loss:.4f}, acc={result.accuracy:.4f}")
            if result.topk:
                print(
                    f"{split_name.capitalize()} top-1={result.topk['top1']:.4f} top-5={result.topk['top5']:.4f}"
                )
            if result.per_class_correct is not None and result.per_class_count is not None:
                per_class_acc = (
                    result.per_class_correct.float() / result.per_class_count.clamp_min(1)
                ).numpy()
                print(f"{split_name.capitalize()} per-class accuracy:", np.round(per_class_acc, 3))
            if result.confusion is not None:
                conf_path = save_dir / f"{split_name}_confusion_test_only.npy"
                np.save(conf_path, result.confusion.numpy())
                print(f"Saved {split_name} confusion matrix to {conf_path}")
        return

    for epoch_idx in range(start_epoch, total_epochs):
        epoch_num = epoch_idx + 1
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch_num}/{total_epochs} | LR={current_lr:.6g}", flush=True)

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
            num_classes=args.num_classes,
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_result.loss)
        elif scheduler is not None:
            scheduler.step()

        if train_result.patch_keep_mean is not None:
            print(
                f"Train patches kept: {train_result.patch_keep_mean:.2f} ± {train_result.patch_keep_std or 0:.2f}"
            )
        if val_result.patch_keep_mean is not None:
            print(
                f"Val patches kept: {val_result.patch_keep_mean:.2f} ± {val_result.patch_keep_std or 0:.2f}"
            )
        if val_result.topk:
            print(
                f"Validation top-1={val_result.topk['top1']:.4f} top-5={val_result.topk['top5']:.4f}"
            )
        if val_result.per_class_correct is not None and val_result.per_class_count is not None:
            per_class_acc = (
                val_result.per_class_correct.float() / val_result.per_class_count.clamp_min(1)
            ).numpy()
            print("Validation per-class accuracy:", np.round(per_class_acc, 3))
        if val_result.confusion is not None:
            conf_path = save_dir / f"val_confusion_epoch_{epoch_num:03d}.npy"
            np.save(conf_path, val_result.confusion.numpy())
            print(f"Saved validation confusion matrix to {conf_path}")

        history.append(
            {
                "epoch": epoch_num,
                "train_loss": train_result.loss,
                "train_acc": train_result.accuracy,
                "val_loss": val_result.loss,
                "val_acc": val_result.accuracy,
                "lr": optimizer.param_groups[0]["lr"],
                "train_patch_keep": train_result.patch_keep_mean,
                "val_patch_keep": val_result.patch_keep_mean,
                "val_top1": val_result.topk.get("top1") if val_result.topk else None,
                "val_top5": val_result.topk.get("top5") if val_result.topk else None,
            }
        )

        checkpoint_state = {
            "epoch": epoch_num,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
        }
        save_checkpoint(checkpoint_state, save_dir, f"epoch_{epoch_num:03d}.pt")

        if val_result.accuracy > best_val_acc:
            best_val_acc = val_result.accuracy
            best_epoch = epoch_num
            save_checkpoint(checkpoint_state, save_dir, "best.pt")

        print(
            f"Epoch {epoch_num:03d}/{total_epochs} | train {train_result.accuracy:.4f} val {val_result.accuracy:.4f} "
            f"| loss {train_result.loss:.4f}/{val_result.loss:.4f}"
        )

    best_path = save_dir / "best.pt"
    if best_path.exists():
        load_checkpoint(str(best_path), model)
        print(f"Loaded best checkpoint from epoch {best_epoch}")

    test_result = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        max_grad_norm=0.0,
        desc="Test",
        log_interval=args.log_interval,
        num_classes=args.num_classes,
    )
    print(f"\nTest set: loss={test_result.loss:.4f}, acc={test_result.accuracy:.4f}")
    if test_result.topk:
        print(f"Test top-1={test_result.topk['top1']:.4f} top-5={test_result.topk['top5']:.4f}")
    if test_result.per_class_correct is not None and test_result.per_class_count is not None:
        per_class_acc = (
            test_result.per_class_correct.float() / test_result.per_class_count.clamp_min(1)
        ).numpy()
        print("Test per-class accuracy:", np.round(per_class_acc, 3))
    if test_result.confusion is not None:
        conf_path = save_dir / "test_confusion.npy"
        np.save(conf_path, test_result.confusion.numpy())
        print(f"Saved test confusion matrix to {conf_path}")

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# --------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
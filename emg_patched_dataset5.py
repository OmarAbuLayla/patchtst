# ==============================================================
#  Dynamic EMG Dataset (On-the-Fly Patching + Filtering)
#  (real-time patching version, GRU-compatible)
# ==============================================================
import os
import math
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io as sio
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------
# Signal Processing Helpers
# --------------------------------------------------------------
def design_filters(fs: int = 1000):
    """Create a 50 Hz notch and 10–400 Hz band-pass filter pair."""
    notch = signal.iirnotch(50, 30, fs)
    nyquist = fs / 2.0
    band = signal.butter(4, [10 / nyquist, 400 / nyquist], btype="bandpass")
    return notch, band


def apply_filters(emg: np.ndarray, notch, band):
    """Apply zero-phase denoising per channel."""
    demeaned = emg - emg.mean(axis=1, keepdims=True)
    padlen = max(len(notch[0]), len(band[0])) * 3
    if emg.shape[1] <= padlen:
        return demeaned.astype(np.float32, copy=True)
    filtered = signal.filtfilt(*notch, demeaned, axis=1)
    filtered = signal.filtfilt(*band, filtered, axis=1)
    return filtered.astype(np.float32, copy=False)


def per_channel_zscore(emg: np.ndarray):
    """Normalise each channel independently across the full recording."""
    mean = emg.mean(axis=1, keepdims=True)
    std = emg.std(axis=1, keepdims=True) + 1e-6
    return (emg - mean) / std


def segment_emg(emg: np.ndarray, patch_len: int, stride: int):
    """Create overlapping patches from (channels, time) EMG."""
    channels, total_samples = emg.shape
    if total_samples < patch_len:
        raise ValueError(
            f"Recording shorter than patch length ({total_samples} < {patch_len})"
        )
    num_patches = 1 + (total_samples - patch_len) // stride
    patches = np.empty((num_patches, channels, patch_len), dtype=np.float32)
    for i in range(num_patches):
        start = i * stride
        end = start + patch_len
        patches[i] = emg[:, start:end]
    return patches


# --------------------------------------------------------------
# Dataset
# --------------------------------------------------------------
class EMGPatchDataset(Dataset):
    """Dynamically patches raw EMG signals during training (no pre-saved patches)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        *,
        patch_len: int = 128,
        stride: int = 96,
        patch_dropout: float = 0.0,
        max_items: Optional[int] = None,
        subset_seed: int = 42,
    ) -> None:
        self.root = os.path.abspath(root)
        self.split = split.capitalize()
        self._is_train = self.split == "Train"
        self.patch_len = patch_len
        self.stride = stride
        self.patch_dropout = (
            min(max(patch_dropout, 0.0), 0.99) if self._is_train else 0.0
        )
        self.max_items = max_items
        self._rng = np.random.default_rng(subset_seed)

        # Precompute filters once
        self.notch, self.band = design_filters(1000)

        split_dir = os.path.join(self.root, self.split, "EMG")
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"Missing split directory: {split_dir}")

        # Collect all raw .mat files
        self.file_list = []
        for subject in sorted(os.listdir(split_dir)):
            subj_dir = os.path.join(split_dir, subject)
            if not os.path.isdir(subj_dir):
                continue
            for session in sorted(os.listdir(subj_dir)):
                sess_dir = os.path.join(subj_dir, session)
                if not os.path.isdir(sess_dir):
                    continue
                for fname in sorted(os.listdir(sess_dir)):
                    if fname.endswith(".mat"):
                        self.file_list.append(os.path.join(sess_dir, fname))

        if not self.file_list:
            raise RuntimeError(f"No .mat files found under {split_dir}")

        # Optional subset
        if self.max_items and self.max_items < len(self.file_list):
            indices = self._rng.permutation(len(self.file_list))[: self.max_items]
            self.file_list = [self.file_list[i] for i in sorted(indices)]

        # Inspect one sample to infer shape
        first_mat = sio.loadmat(self.file_list[0])
        if "data" not in first_mat:
            raise KeyError(f"No 'data' field in {self.file_list[0]}")
        sample = np.asarray(first_mat["data"], dtype=np.float32)
        if sample.shape[0] != 6 and sample.shape[1] == 6:
            sample = sample.T
        self.num_channels = sample.shape[0]
        self.feature_dim = self.num_channels * self.patch_len

        print(
            f"Loaded {len(self.file_list)} {self.split.lower()} samples "
            f"(channels={self.num_channels}, patch_len={self.patch_len}, stride={self.stride})"
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        path = self.file_list[idx]
        mat = sio.loadmat(path)
        if "data" not in mat:
            raise KeyError(f"Missing 'data' in {path}")
        emg = np.asarray(mat["data"], dtype=np.float32)
        if emg.shape[0] != 6 and emg.shape[1] == 6:
            emg = emg.T
        if emg.shape[0] != self.num_channels:
            raise ValueError(f"{path} has {emg.shape[0]} channels, expected {self.num_channels}")

        # Filtering + normalization
        filtered = apply_filters(emg, self.notch, self.band)
        normalized = per_channel_zscore(filtered)

        # Patch segmentation
        patches = segment_emg(normalized, self.patch_len, self.stride)
        features = patches.reshape(patches.shape[0], self.feature_dim)

        # Optional patch dropout
        kept_tokens = features.shape[0]
        if self.patch_dropout > 0.0:
            total_tokens = features.shape[0]
            keep_prob = 1.0 - self.patch_dropout
            rng = np.random.default_rng(self._rng.integers(0, 2**32 - 1, dtype=np.uint32))
            keep_mask = rng.random(total_tokens) < keep_prob
            if keep_mask.sum() < 5:
                keep_mask[:5] = True
            features = features.copy()
            features[~keep_mask] = 0.0
            kept_tokens = int(keep_mask.sum())

        label = int(np.squeeze(mat.get("label", [[-1]])).astype(np.int64))
        return torch.from_numpy(features), label, kept_tokens


# --------------------------------------------------------------
# Dataloader Builder
# --------------------------------------------------------------
def build_patch_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    *,
    patch_len: int = 128,
    stride: int = 96,
    patch_dropout: float = 0.0,
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    subset_test: Optional[int] = None,
    subset_seed: int = 42,
) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders for dynamic EMG patching."""
    splits = ["train", "val", "test"]
    loaders: Dict[str, DataLoader] = {}
    subset_map = {
        "train": subset_train if subset_train and subset_train > 0 else None,
        "val": subset_val if subset_val and subset_val > 0 else None,
        "test": subset_test if subset_test and subset_test > 0 else None,
    }

    for split in splits:
        dataset = EMGPatchDataset(
            data_root,
            split,
            patch_len=patch_len,
            stride=stride,
            patch_dropout=patch_dropout,
            max_items=subset_map.get(split),
            subset_seed=subset_seed,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


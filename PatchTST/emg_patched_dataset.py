import os
from typing import Dict, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


class EMGPatchDataset(Dataset):
    """Dataset for GRU-based training on pre-patched EMG sequences."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        normalize: bool = True,
    ) -> None:
        self.root = os.path.abspath(root)
        self.split = split.capitalize()
        self.normalize = normalize

        split_dir = os.path.join(self.root, self.split, "EMG")
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"Missing split directory: {split_dir}")

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
                        label = int(os.path.splitext(fname)[0])
                        path = os.path.join(sess_dir, fname)
                        self.file_list.append((label, path))

        if not self.file_list:
            raise RuntimeError(f"No .mat files found under {split_dir}")

        # Inspect a representative sample to capture patch metadata.  The patched
        # arrays now follow (num_patches, num_channels, patch_len), so we retain
        # the channel axis until the final reshape.
        _, sample_path = self.file_list[0]
        sample = sio.loadmat(sample_path)["patched_data"]
        if sample.ndim != 3:
            raise ValueError(
                f"Expected 3-D patched data (num_patches, channels, patch_len); got {sample.shape}"
            )
        self.seq_len, self.num_channels, self.patch_len = sample.shape
        self.feature_dim = self.num_channels * self.patch_len

        print(
            f"Loaded {len(self.file_list)} {self.split.lower()} samples "
            f"(seq_len={self.seq_len}, channels={self.num_channels}, patch_len={self.patch_len})"
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def _normalise(self, patches: np.ndarray) -> np.ndarray:
        """Apply per-channel z-score over the entire recording."""

        # patches: (num_patches, channels, patch_len)
        channels_first = patches.transpose(1, 0, 2).reshape(self.num_channels, -1)
        mean = channels_first.mean(axis=1, keepdims=True)
        std = channels_first.std(axis=1, keepdims=True) + 1e-6
        normalised = (channels_first - mean) / std
        return normalised.reshape(self.num_channels, self.seq_len, self.patch_len).transpose(1, 0, 2)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label, path = self.file_list[idx]
        mat = sio.loadmat(path)
        patches = np.asarray(mat["patched_data"], dtype=np.float32)

        if patches.shape != (self.seq_len, self.num_channels, self.patch_len):
            raise ValueError(
                f"Unexpected patch shape {patches.shape} in {path}; "
                f"expected {(self.seq_len, self.num_channels, self.patch_len)}"
            )

        if self.normalize:
            patches = self._normalise(patches)

        # Flatten the channel/time axes so the GRU sees one token per patch while
        # preserving channel locality within each feature vector.
        features = patches.reshape(self.seq_len, self.feature_dim)
        return torch.from_numpy(features), label


def build_patch_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    splits = ["train", "val", "test"]
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        dataset = EMGPatchDataset(data_root, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders

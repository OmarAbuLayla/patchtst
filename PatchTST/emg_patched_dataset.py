import os
import math
from typing import Dict

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
        *,
        patch_dropout: float = 0.0,
        stride_test: int | None = None,
        max_items: int | None = None,
        subset_seed: int = 42,
    ) -> None:
        self.root = os.path.abspath(root)
        self.split = split.capitalize()
        self._is_train = self.split == "Train"
        clipped_dropout = min(max(patch_dropout, 0.0), 0.99)
        self.patch_dropout = clipped_dropout if self._is_train else 0.0
        self.target_stride = stride_test
        self.max_items = max_items
        self._rng = np.random.default_rng(subset_seed)

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
                        path = os.path.join(sess_dir, fname)
                        self.file_list.append(path)

        if not self.file_list:
            raise RuntimeError(f"No .mat files found under {split_dir}")

        if self.max_items is not None and self.max_items < len(self.file_list):
            indices = self._rng.permutation(len(self.file_list))[: self.max_items]
            indices.sort()
            self.file_list = [self.file_list[i] for i in indices]

        # Inspect a representative sample to capture patch metadata.  The patched
        # arrays now follow (num_patches, num_channels, patch_len), so we retain
        # the channel axis until the final reshape.
        sample_path = self.file_list[0]
        sample_dict = sio.loadmat(sample_path)
        sample = sample_dict["patched_data"]
        if sample.ndim != 3:
            raise ValueError(
                f"Expected 3-D patched data (num_patches, channels, patch_len); got {sample.shape}"
            )
        self.base_seq_len, self.num_channels, self.patch_len = sample.shape
        self.stored_stride = int(np.squeeze(sample_dict.get("stride", np.array([[0]])))) or None

        self.subsample_step = 1
        # When emulating a smaller stride, we upsample by reindexing into the
        # stored patch sequence; otherwise we optionally downsample via striding.
        self.upsample_indices: np.ndarray | None = None
        if self.target_stride is not None and self.stored_stride is not None:
            if self.target_stride < self.stored_stride:
                desired_len = int(math.ceil(self.base_seq_len * self.stored_stride / self.target_stride))
                interp = np.linspace(0, self.base_seq_len - 1, num=desired_len)
                self.upsample_indices = np.clip(np.round(interp).astype(int), 0, self.base_seq_len - 1)
            elif self.target_stride > self.stored_stride:
                ratio = max(1, int(math.ceil(self.target_stride / self.stored_stride)))
                self.subsample_step = ratio

        if self.upsample_indices is not None:
            self.seq_len = len(self.upsample_indices)
        else:
            self.seq_len = int(np.ceil(self.base_seq_len / self.subsample_step))
        self.feature_dim = self.num_channels * self.patch_len

        stride_msg_parts = []
        if self.stored_stride is not None:
            stride_msg_parts.append(f"stored_stride={self.stored_stride}")
        if self.target_stride is not None:
            stride_msg_parts.append(f"target_stride={self.target_stride}")
        stride_msg = ", ".join(stride_msg_parts) if stride_msg_parts else "stride=unspecified"
        print(
            f"Loaded {len(self.file_list)} {self.split.lower()} samples "
            f"(seq_len={self.seq_len}, channels={self.num_channels}, patch_len={self.patch_len}, {stride_msg})"
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        path = self.file_list[idx]
        mat = sio.loadmat(path)
        patches = np.asarray(mat["patched_data"], dtype=np.float32)

        if patches.shape != (self.base_seq_len, self.num_channels, self.patch_len):
            raise ValueError(
                f"Unexpected patch shape {patches.shape} in {path}; "
                f"expected {(self.base_seq_len, self.num_channels, self.patch_len)}"
            )

        if self.upsample_indices is not None:
            patches = patches[self.upsample_indices]
        elif self.subsample_step > 1:
            patches = patches[:: self.subsample_step]

        # Filtering + per-channel normalisation now occur during patch generation,
        # so the dataset avoids an additional z-score that previously erased
        # cross-session amplitude cues.
        label_arr = mat.get("label")
        if label_arr is None:
            raise KeyError(f"Missing 'label' in {path}")
        label = int(np.squeeze(label_arr).astype(np.int64))

        # Flatten the channel/time axes so the GRU sees one token per patch while
        # preserving channel locality within each feature vector.
        features = patches.reshape(patches.shape[0], self.feature_dim)

        kept_tokens = features.shape[0]
        if self.patch_dropout > 0.0:
            total_tokens = features.shape[0]
            keep_prob = 1.0 - self.patch_dropout
            proposed = int(round(total_tokens * keep_prob))
            min_keep = min(5, total_tokens)
            keep_count = max(min_keep, min(total_tokens, proposed))
            rng = np.random.default_rng(self._rng.integers(0, 2**32 - 1, dtype=np.uint32))
            keep_indices = rng.choice(total_tokens, size=keep_count, replace=False)
            keep_indices.sort()
            if keep_count < total_tokens:
                extra = rng.choice(keep_indices, size=total_tokens - keep_count, replace=True)
                gather_idx = np.concatenate([keep_indices, extra])
                rng.shuffle(gather_idx)
            else:
                gather_idx = keep_indices
            features = features[gather_idx]
            kept_tokens = keep_count

        return torch.from_numpy(features), label, int(kept_tokens)


def build_patch_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    *,
    patch_dropout: float = 0.0,
    stride_test: int | None = None,
    subset_train: int | None = None,
    subset_val: int | None = None,
    subset_test: int | None = None,
    subset_seed: int = 42,
) -> Dict[str, DataLoader]:
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
            patch_dropout=patch_dropout,
            stride_test=stride_test,
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

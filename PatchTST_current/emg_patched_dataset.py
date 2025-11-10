import os
import math
from typing import Dict, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional



class EMGPatchDataset(Dataset):
    """Dataset for GRU-based training on pre-patched EMG sequences."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        *,
        patch_dropout: float = 0.0,
        #stride_test: int | None = None,
        stride_test: Optional[int] = None
    ) -> None:
        self.root = os.path.abspath(root)
        self.split = split.capitalize()
        self._is_train = self.split == "Train"
        clipped_dropout = min(max(patch_dropout, 0.0), 0.99)
        self.patch_dropout = clipped_dropout if self._is_train else 0.0
        self.target_stride = stride_test

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
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

        if self.patch_dropout > 0.0:
            # Stochastically zero-out a subset of patches during training to
            # increase temporal diversity without altering tensor shapes.
            keep_prob = 1.0 - self.patch_dropout
            mask = np.random.rand(features.shape[0]) < keep_prob
            min_keep = min(5, features.shape[0])
            if mask.sum() < min_keep:
                ensure_idx = np.random.choice(features.shape[0], size=min_keep, replace=False)
                mask[ensure_idx] = True
            features[~mask] = 0.0

        return torch.from_numpy(features), label


def build_patch_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    *,
    patch_dropout: float = 0.0,
    #stride_test: int | None = None,
    
    stride_test: Optional[int] = None
) -> Dict[str, DataLoader]:
    splits = ["train", "val", "test"]
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        dataset = EMGPatchDataset(
            data_root,
            split,
            patch_dropout=patch_dropout,
            stride_test=stride_test,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
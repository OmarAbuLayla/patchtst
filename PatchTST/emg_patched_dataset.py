import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

class EMGPatchDataset(Dataset):
    def __init__(self, root: str, split: str = "train", normalize: bool = True):
        self.root = os.path.abspath(root)
        self.split = split.capitalize()
        self.normalize = normalize

        # Traverse structure
        split_dir = os.path.join(self.root, self.split, "EMG")
        self.file_list = []
        for subject in sorted(os.listdir(split_dir)):
            subj_dir = os.path.join(split_dir, subject)
            if not os.path.isdir(subj_dir):
                continue
            for session in sorted(os.listdir(subj_dir)):
                sess_dir = os.path.join(subj_dir, session)
                for fname in sorted(os.listdir(sess_dir)):
                    if fname.endswith(".mat"):
                        label = int(os.path.splitext(fname)[0])
                        path = os.path.join(sess_dir, fname)
                        self.file_list.append((label, path))

        if not self.file_list:
            raise RuntimeError(f"No .mat files found under {split_dir}")

        print(f"Loaded {len(self.file_list)} {split} samples from {split_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        label, path = self.file_list[idx]
        mat = sio.loadmat(path)
        emg = np.asarray(mat["patched_data"], dtype=np.float32)  # (366, 64)

        if self.normalize:
            mean = emg.mean()
            std = emg.std() + 1e-8
            emg = (emg - mean) / std

        return torch.from_numpy(emg), label


def build_patch_dataloaders(data_root, batch_size=32, num_workers=0):
    splits = ["train", "val", "test"]
    loaders = {}
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

import os
import numpy as np
import scipy.io as sio
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader


def _design_filters(fs: int):
    """Create the notch + band-pass filters reused for every sample."""

    # Recreate the denoising stack from the legacy pipeline so every channel is
    # filtered before we build patches; this restores the spectral cleaning that
    # was missing from the original loader.
    return {
        "notch_50": signal.iirnotch(50, 30, fs),
        "bandpass": signal.butter(4, [10 / (fs / 2), 400 / (fs / 2)], "bandpass"),
    }


def _apply_filters(emg: np.ndarray, filters) -> np.ndarray:
    """Apply zero-phase filtering along the temporal axis for every channel."""

    if emg.ndim != 2:
        raise ValueError(f"Expected 2-D patched data, got shape {emg.shape}")

    # Interpret each column as one EMG channel sampled over successive patches
    # (time steps).  Filtering along axis=0 mirrors the preprocessing done on
    # the raw waveforms before patch extraction.
    emg = emg.astype(np.float32, copy=False)
    channel_first = emg.T  # (channels, seq_len)

    # Remove the per-channel DC component so that notch/band-pass work as
    # intended and polarity is preserved.
    channel_first = channel_first - channel_first.mean(axis=1, keepdims=True)

    padlen = max(max(len(b), len(a)) for b, a in filters.values()) * 3
    if channel_first.shape[1] < padlen:
        # Signals shorter than the filter kernel are left untouched to avoid
        # filtfilt instability; we still return a copy so downstream ops stay
        # consistent.
        return channel_first.T.copy()

    filtered = channel_first
    for b, a in filters.values():
        filtered = signal.filtfilt(b, a, filtered, axis=1)
    return filtered.T.astype(np.float32, copy=False)

class EMGPatchDataset(Dataset):
    def __init__(self, root: str, split: str = "train", normalize: bool = True, sample_rate: int = 1000):
        self.root = os.path.abspath(root)
        self.split = split.capitalize()
        self.normalize = normalize
        self.sample_rate = sample_rate
        self._filters = _design_filters(sample_rate)

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

        # Run the same denoising pass used in the 15VC pipeline so each channel
        # regains the spectral cleaning that improves class separability.
        emg = _apply_filters(emg, self._filters)

        if self.normalize:
            # Normalise per channel instead of globally so that relative channel
            # amplitudes survive patching and the model sees discriminative
            # polarity cues.
            mean = emg.mean(axis=0, keepdims=True)
            std = emg.std(axis=0, keepdims=True) + 1e-6
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

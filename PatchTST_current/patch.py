"""Patch extraction script for the PatchTST GRU pipeline.

This file mirrors the directory-driven workflow of the legacy script while
incorporating the channel-preserving fixes discussed during the diagnostics:

* Raw EMG matrices are interpreted as (time, channels) and reshaped to keep the
  electrode axis explicit.
* Each channel is denoised with a 50 Hz notch + 10–400 Hz band-pass filter and
  z-score normalised before segmentation.
* Patches are emitted as (num_patches, num_channels, patch_len) so the current
  dataset/model stack can flatten tokens without losing polarity information.

Adjust ``src_root``/``dst_root`` below to match your local dataset paths and run
the script directly (no CLI arguments required).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as sio
from scipy import signal


# ---------------------------------------------------------------------------
# Configuration (mirror of the legacy script)
# ---------------------------------------------------------------------------
src_root = Path(r"C:\Users\ompis\Desktop\work GJU\Codes\AVE-Speech")
dst_root = Path(r"D:\Omar\AVE-Speech_PatchTST")
splits = ["Train", "Val", "Test"]

# Recommended PatchTST settings for 6-channel, 2000-sample EMG segments.
patch_len = 128
stride = 96  # ≈25% overlap to curb redundancy while preserving continuity

# EMG metadata
sample_rate = 1000  # Hz
expected_channels = 6


# ---------------------------------------------------------------------------
# Filtering / preprocessing helpers
# ---------------------------------------------------------------------------
def _design_filters(fs: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Create a 50 Hz notch and 10–400 Hz band-pass filter pair."""

    notch = signal.iirnotch(50, 30, fs)
    nyquist = fs / 2.0
    band = signal.butter(4, [10 / nyquist, 400 / nyquist], btype="bandpass")
    return notch, band


def _load_emg(mat_dict: dict, mat_path: Path) -> np.ndarray:
    """Load an EMG matrix and ensure channels occupy the first axis."""

    if "data" not in mat_dict:
        raise KeyError(f"No 'data' field found in {mat_path}")

    emg = np.asarray(mat_dict["data"], dtype=np.float32)

    if emg.ndim != 2:
        raise ValueError(f"Expected 2-D EMG array in {mat_path}, got {emg.shape}")

    # Convert (time, channels) to (channels, time) so downstream filtering runs
    # along the temporal axis per electrode.
    if emg.shape[0] != expected_channels and emg.shape[1] == expected_channels:
        emg = emg.T

    if emg.shape[0] != expected_channels:
        raise ValueError(
            f"{mat_path} contains {emg.shape[0]} channels; expected {expected_channels}"
        )

    return emg


def _apply_filters(emg: np.ndarray, notch, band) -> np.ndarray:
    """Apply zero-phase denoising per channel."""

    demeaned = emg - emg.mean(axis=1, keepdims=True)

    padlen = max(len(notch[0]), len(band[0])) * 3
    if emg.shape[1] <= padlen:
        # For extremely short clips, filtering would introduce artefacts; return
        # the demeaned signal so the pipeline remains functional.
        return demeaned.astype(np.float32, copy=True)

    filtered = signal.filtfilt(*notch, demeaned, axis=1)
    filtered = signal.filtfilt(*band, filtered, axis=1)
    return filtered.astype(np.float32, copy=False)


def _per_channel_zscore(emg: np.ndarray) -> np.ndarray:
    """Normalise each channel independently across the full recording."""

    mean = emg.mean(axis=1, keepdims=True)
    std = emg.std(axis=1, keepdims=True) + 1e-6
    return (emg - mean) / std


def _segment(emg: np.ndarray) -> np.ndarray:
    """Generate overlapping patches with the channel axis preserved."""

    channels, total_samples = emg.shape
    if total_samples < patch_len:
        raise ValueError(
            f"Recording shorter than patch length ({total_samples} < {patch_len})"
        )

    num_patches = 1 + (total_samples - patch_len) // stride
    if num_patches <= 0:
        raise ValueError("Stride/patch configuration produced zero patches")

    patches = np.empty((num_patches, channels, patch_len), dtype=np.float32)
    for i in range(num_patches):
        start = i * stride
        end = start + patch_len
        patches[i] = emg[:, start:end]
    return patches


# ---------------------------------------------------------------------------
# Main processing loop (mirrors legacy script structure)
# ---------------------------------------------------------------------------
def process_mat_file(src_path: Path, dst_path: Path, notch, band) -> None:
    try:
        mat = sio.loadmat(src_path)
        emg = _load_emg(mat, src_path)
        filtered = _apply_filters(emg, notch, band)
        normalised = _per_channel_zscore(filtered)
        patches = _segment(normalised)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(
            dst_path,
            {
                "patched_data": patches,
                "label": np.asarray(mat.get("label", [[-1]]), dtype=np.int32),
                "patch_len": np.asarray([[patch_len]], dtype=np.int32),
                "stride": np.asarray([[stride]], dtype=np.int32),
            },
        )
        print(f"✅ Saved {dst_path} {patches.shape}")

    except Exception as exc:  # noqa: BLE001 - print-friendly wrapper for batch jobs
        print(f"❌ Error processing {src_path}: {exc}")


def main() -> None:
    notch, band = _design_filters(sample_rate)

    for split in splits:
        src_split = src_root / split / "EMG"
        dst_split = dst_root / split / "EMG"

        if not src_split.exists():
            print(f"⚠️ Missing source directory: {src_split}")
            continue

        for root, _, files in os.walk(src_split):
            for file in files:
                if not file.endswith(".mat"):
                    continue

                src_file = Path(root) / file
                rel_path = Path(os.path.relpath(src_file, src_split))
                dst_file = dst_split / rel_path
                process_mat_file(src_file, dst_file, notch, band)


if __name__ == "__main__":
    main()
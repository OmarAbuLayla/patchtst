import os
import scipy.io as sio
import numpy as np

# --- Configuration ---
src_root = r"C:\Users\ompis\Desktop\work GJU\Codes\AVE-Speech"
dst_root = r"D:\Omar\AVE-Speech_PatchTST"
splits = ["Train", "Val", "Test"]

patch_len = 64
stride = 32

# --- Helper function to process one file ---
def process_mat_file(src_path, dst_path):
    try:
        data = sio.loadmat(src_path)
        if "data" not in data:
            print(f"⚠️ Skipping (no 'data'): {src_path}")
            return

        emg = data["data"]  # shape (T, C)
        label = data.get("label", None)

        T, C = emg.shape
        n_patches = (T - patch_len) // stride + 1

        patched = []
        for ch in range(C):
            signal = emg[:, ch]
            for i in range(n_patches):
                start = i * stride
                end = start + patch_len
                patched.append(signal[start:end])
        patched = np.stack(patched)  # shape (C * n_patches, patch_len)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        sio.savemat(dst_path, {
            "patched_data": patched,
            "label": label if label is not None else np.array([-1])
        })
        print(f"✅ Saved {dst_path} ({patched.shape})")

    except Exception as e:
        print(f"❌ Error processing {src_path}: {e}")

# --- Main loop ---
for split in splits:
    src_split = os.path.join(src_root, split, "EMG")
    dst_split = os.path.join(dst_root, split, "EMG")

    for root, _, files in os.walk(src_split):
        for file in files:
            if file.endswith(".mat"):
                src_file = os.path.join(root, file)
                # replicate folder structure relative to src_split
                rel_path = os.path.relpath(src_file, src_split)
                dst_file = os.path.join(dst_split, rel_path)
                process_mat_file(src_file, dst_file)

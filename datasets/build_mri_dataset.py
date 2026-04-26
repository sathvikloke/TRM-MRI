"""
dataset/build_mri_dataset.py

Converts raw fastMRI .h5 files into .npy arrays consumed by the MRI training pipeline.

Output layout (one directory per split, e.g. data/mri-knee/train/):
    all__inputs.npy   float32  (N, 2, H*W)   — real & imag of masked k-space (single-coil-equivalent)
    all__labels.npy   float32  (N, H*W)       — RSS magnitude image, normalised to [0, 1]
    all__masks.npy    float32  (N, W)          — 1-D Cartesian undersampling mask
    all__scales.npy   float32  (N,)            — per-slice RSS max used for label normalisation
    dataset.json                               — MRIDatasetMetadata

Convention notes
----------------
fastMRI k-space is "centred" — the DC component sits at index (H/2, W/2).
NumPy's `np.fft.ifft2` expects the DC at the corner, so we have to do:

        image = fftshift(ifft2(ifftshift(kspace)))

The previous version of this file applied `ifftshift` *after* `ifft2`, which
silently produced shifted (and therefore wrong) images.

Multi-coil handling
-------------------
For multi-coil data the per-coil iFFT is RSS-combined to give the magnitude
target.  Because the model emits a single (1-channel) magnitude image we
also build a "virtual single-coil" k-space, defined as `fft2(rss_image)`,
masked the same way the multi-coil acquisition was masked.  This lets the
model's data-consistency block blend its prediction against an observation
that lives in the same physical scale as the prediction itself.

Per-slice reproducibility
-------------------------
Each slice's undersampling mask is drawn from its own `np.random.Generator`,
seeded as `seed XOR <global slice index>`.  This keeps mask draws stable
under parallelism / re-orderings and is independent of how many other
slices were processed before this one.

Usage:
    python -m datasets.build_mri_dataset \\
        --input-dir data/fastmri_knee_raw \\
        --output-dir data/mri-knee \\
        --acceleration 4
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Metadata schema  (self-contained, no dependency on dataset.common)
# ──────────────────────────────────────────────────────────────

class MRIDatasetMetadata(BaseModel):
    """Written to dataset.json; read by MRIDataset in pretrain.py."""
    height: int
    width: int
    seq_len: int          # = height * width
    acceleration: int
    center_fraction: float
    is_multicoil: bool
    total_slices: int
    sets: List[str]


# ──────────────────────────────────────────────────────────────
# CLI config
# ──────────────────────────────────────────────────────────────

class DataProcessConfig(BaseModel):
    input_dir: str = "data/fastmri_knee_raw"
    output_dir: str = "data/mri-knee"
    acceleration: int = 4
    center_fraction: float = 0.08   # fraction of width always kept at centre
    test_fraction: float = 0.1
    seed: int = 42
    max_train_slices: Optional[int] = None
    max_test_slices: Optional[int] = None


cli = ArgParser()


# ──────────────────────────────────────────────────────────────
# MRI physics helpers
# ──────────────────────────────────────────────────────────────

def _centred_ifft2(kspace: np.ndarray) -> np.ndarray:
    """
    Centred 2-D inverse FFT.

    Assumes DC is at the centre of the input k-space (fastMRI convention).
    Returns the complex image with DC again at the centre.
    """
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1))
    return np.fft.fftshift(image, axes=(-2, -1))


def _centred_fft2(image: np.ndarray) -> np.ndarray:
    """Centred 2-D forward FFT — inverse of `_centred_ifft2`."""
    shifted = np.fft.ifftshift(image, axes=(-2, -1))
    k = np.fft.fft2(shifted, axes=(-2, -1))
    return np.fft.fftshift(k, axes=(-2, -1))


def rss_reconstruction(kspace: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the Root-Sum-of-Squares image from k-space.

    Input shape:
        (coils, H, W) complex   – multi-coil
        (H, W)        complex   – single-coil (treated as 1-coil RSS)

    Returns
    -------
    rss_normalised : (H, W) float32 in [0, 1]
    scale          : float, the pre-normalisation max used to scale rss
    """
    if kspace.ndim == 2:
        kspace = kspace[np.newaxis]              # (1, H, W)

    images = _centred_ifft2(kspace)              # complex (coils, H, W)
    rss = np.sqrt((np.abs(images) ** 2).sum(axis=0)).astype(np.float32)

    scale = float(rss.max())
    if scale > 0:
        rss = rss / scale

    return rss, scale


def build_cartesian_mask(width: int, acceleration: int, center_fraction: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    1-D Cartesian undersampling mask of length `width`.
    Always keeps the central `center_fraction * width` lines.
    Randomly samples the remaining lines to reach `width / acceleration` total.
    """
    num_keep   = max(1, int(round(width / acceleration)))
    num_center = max(1, int(round(width * center_fraction)))
    num_center = min(num_center, num_keep)
    num_random = max(0, num_keep - num_center)

    mask = np.zeros(width, dtype=np.float32)

    # Always-on centre block
    c_start = (width - num_center) // 2
    mask[c_start: c_start + num_center] = 1.0

    # Random peripheral lines
    peripheral = np.flatnonzero(mask == 0.0)
    if num_random > 0 and peripheral.size >= num_random:
        chosen = rng.choice(peripheral, size=num_random, replace=False)
        mask[chosen] = 1.0

    return mask


def kspace_to_input(virtual_kspace_masked: np.ndarray) -> np.ndarray:
    """
    Convert (H, W) complex masked k-space → model input array (2, H*W).
    Channel 0 is real, channel 1 is imaginary, both row-major flattened.

    NOTE: values are NOT renormalised — raw magnitudes are preserved so that
    the model's data-consistency block can blend predicted vs. observed
    k-space at a matching physical scale.
    """
    assert virtual_kspace_masked.ndim == 2, (
        f"Expected (H, W) input, got shape {virtual_kspace_masked.shape}"
    )
    real_part = virtual_kspace_masked.real.astype(np.float32).reshape(-1)
    imag_part = virtual_kspace_masked.imag.astype(np.float32).reshape(-1)
    return np.stack([real_part, imag_part], axis=0)


# ──────────────────────────────────────────────────────────────
# H5 file loading
# ──────────────────────────────────────────────────────────────

def load_h5_slices(filepath: str) -> List[np.ndarray]:
    """
    Load all axial slices from a fastMRI .h5 file.
    Returns a list of complex arrays, each of shape (coils, H, W) or (H, W).

    Handles both:
        * native complex-typed datasets (preferred fastMRI format)
        * structured-dtype legacy storage with {"r", "i"} fields
    """
    with h5py.File(filepath, "r") as f:
        if "kspace" not in f:
            raise KeyError(f"{filepath!r} does not contain a 'kspace' dataset")
        kspace = f["kspace"][:]    # (slices, [coils,] H, W)

    if kspace.dtype.names is not None and {"r", "i"}.issubset(set(kspace.dtype.names)):
        kspace = kspace["r"].astype(np.float32) + 1j * kspace["i"].astype(np.float32)

    kspace = kspace.astype(np.complex64)

    # Sanity check on rank: expect (S, H, W) or (S, C, H, W).
    if kspace.ndim not in (3, 4):
        raise ValueError(
            f"{filepath!r}: unexpected kspace rank {kspace.ndim}, "
            f"shape={kspace.shape}; expected 3 (single-coil) or 4 (multi-coil)."
        )

    return [kspace[sl] for sl in range(kspace.shape[0])]


# ──────────────────────────────────────────────────────────────
# Per-split conversion
# ──────────────────────────────────────────────────────────────

def _slice_rng(seed: int, slice_index: int) -> np.random.Generator:
    """Per-slice deterministic RNG, independent of file ordering."""
    return np.random.default_rng(np.random.SeedSequence([seed, slice_index]))


def convert_subset(
    set_name: str,
    h5_files: List[str],
    config: DataProcessConfig,
    max_slices: Optional[int],
    height: int,
    width: int,
    base_slice_index: int,
) -> int:
    """
    Process a list of .h5 files and write the .npy arrays for one split.
    Returns the number of slices written.
    """
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    inputs_list:  List[np.ndarray] = []
    labels_list:  List[np.ndarray] = []
    masks_list:   List[np.ndarray] = []
    scales_list:  List[float]      = []

    skipped = 0
    total   = 0
    is_multicoil = False
    global_index = base_slice_index

    for filepath in tqdm(h5_files, desc=f"Processing {set_name}"):
        try:
            slices = load_h5_slices(filepath)
        except Exception as exc:
            print(f"  [WARN] Failed to load {filepath}: {exc}")
            continue

        for sl_kspace in slices:
            if max_slices is not None and total >= max_slices:
                break

            # Spatial dims (last two axes)
            h, w = sl_kspace.shape[-2], sl_kspace.shape[-1]

            # Enforce consistent spatial size across all files
            if h != height or w != width:
                skipped += 1
                global_index += 1
                continue

            if sl_kspace.ndim == 3:
                is_multicoil = True

            # ── Ground-truth label (fully-sampled RSS image) ──────────────
            label_image, scale = rss_reconstruction(sl_kspace)   # (H, W) ∈ [0,1]

            # ── Random Cartesian mask, deterministic per global slice index
            slice_rng = _slice_rng(config.seed, global_index)
            mask = build_cartesian_mask(width, config.acceleration, config.center_fraction, slice_rng)

            # ── Virtual single-coil k-space matching the magnitude target ──
            # Using FFT(label) * scale ensures that data-consistency in the
            # training loop sees observations on the same scale as the
            # network prediction (which lives in [0, 1]).  We multiply by
            # `scale` so that the saved k-space recovers the *un-normalised*
            # magnitude image when the model multiplies its prediction by
            # `scale`.
            virtual_kspace = _centred_fft2(label_image.astype(np.complex64) * scale)
            virtual_kspace_masked = virtual_kspace * mask          # (H, W) complex

            # ── Convert to model input ────────────────────────────────────
            inp = kspace_to_input(virtual_kspace_masked)

            inputs_list.append(inp)
            labels_list.append(label_image.reshape(-1).astype(np.float32))   # (H*W,)
            masks_list.append(mask)                                            # (W,)
            scales_list.append(scale)

            total += 1
            global_index += 1

        if max_slices is not None and total >= max_slices:
            break

    if skipped > 0:
        print(f"  [INFO] {set_name}: skipped {skipped} slices with mismatched spatial size "
              f"(expected {height}×{width}).")

    if total == 0:
        raise RuntimeError(f"No slices were written for split '{set_name}'. "
                           f"Check --input-dir and that files have size {height}×{width}.")

    # Stack and save
    np.save(os.path.join(save_dir, "all__inputs.npy"),
            np.stack(inputs_list, axis=0).astype(np.float32))   # (N, 2, H*W)
    np.save(os.path.join(save_dir, "all__labels.npy"),
            np.stack(labels_list, axis=0).astype(np.float32))   # (N, H*W)
    np.save(os.path.join(save_dir, "all__masks.npy"),
            np.stack(masks_list,  axis=0).astype(np.float32))   # (N, W)
    np.save(os.path.join(save_dir, "all__scales.npy"),
            np.array(scales_list, dtype=np.float32))             # (N,)

    metadata = MRIDatasetMetadata(
        height=height,
        width=width,
        seq_len=height * width,
        acceleration=config.acceleration,
        center_fraction=config.center_fraction,
        is_multicoil=is_multicoil,
        total_slices=total,
        sets=["all"],
    )
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    print(f"  [OK] {set_name}: wrote {total} slices → {save_dir}")
    return total


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def _discover_h5_files(input_dir: str) -> List[str]:
    files = sorted(
        str(p) for p in Path(input_dir).rglob("*.h5")
    )
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {input_dir}")
    return files


def _infer_spatial_size(files: List[str]) -> Tuple[int, int]:
    """Read the first file to infer the canonical (H, W)."""
    slices = load_h5_slices(files[0])
    sl = slices[0]
    h, w = sl.shape[-2], sl.shape[-1]
    print(f"  [INFO] Inferred spatial size from first file: {h}×{w}")
    return h, w


def convert_dataset(config: DataProcessConfig) -> None:
    # Use a top-level RNG only for the train/test file split.  Per-slice
    # mask sampling uses its own RNG keyed on (seed, slice_index).
    split_rng = np.random.default_rng(config.seed)

    all_files = _discover_h5_files(config.input_dir)
    print(f"Found {len(all_files)} .h5 files in {config.input_dir}")

    height, width = _infer_spatial_size(all_files)

    # Train / test split (file-level, deterministic permutation).
    perm = split_rng.permutation(len(all_files))
    all_files = [all_files[i] for i in perm]
    n_test  = max(1, int(round(len(all_files) * config.test_fraction)))
    n_train = len(all_files) - n_test
    train_files = all_files[:n_train]
    test_files  = all_files[n_train:]

    print(f"Split: {n_train} train files, {n_test} test files")

    n_train_slices = convert_subset(
        "train", train_files, config,
        max_slices=config.max_train_slices,
        height=height, width=width,
        base_slice_index=0,
    )
    convert_subset(
        "test", test_files, config,
        max_slices=config.max_test_slices,
        height=height, width=width,
        # Offset so train and test slice rng streams never collide.
        base_slice_index=n_train_slices + 10**6,
    )


@cli.command(singleton=True)
def main(config: DataProcessConfig) -> None:
    convert_dataset(config)


if __name__ == "__main__":
    cli()

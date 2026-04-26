# TRM-MRI
 
**Tiny Recursive Models for accelerated MRI reconstruction.**
 
This repo adapts Samsung SAIL's [Tiny Recursive Model (TRM)](https://arxiv.org/abs/2510.04871) — the 7M-parameter architecture that punches well above its weight on ARC-AGI — to a very different problem: reconstructing MR images from undersampled k-space. The hope is that TRM's recursive reasoning loop, which lets a small network "think longer" by repeatedly refining a latent state, can replace the deep cascades of large unrolled networks (E2E-VarNet, MoDL, RecurrentVarNet) that currently dominate MRI reconstruction.
 
It's a research project. Things may change. Some choices are deliberately conservative; others are experiments. PRs welcome.
 
---
 
## Why this is interesting
 
Modern MR scanners trade off scan time against image quality by *undersampling* — collecting fewer frequency-domain (k-space) lines than Nyquist requires, and then asking software to fill in the gaps. State-of-the-art reconstruction networks are big: tens of millions of parameters, deep cascades, careful coil-aware architecture. They work well, but they're also expensive to train and unwieldy to deploy on edge hardware.
 
TRM goes the other direction. The whole model is a small CNN encoder, a *single* shared transformer block, a tiny CNN decoder, and two recursive latents (`z_H`, `z_L`) that get refined for `H_cycles × L_cycles` steps before producing an output. The depth comes from the recursion, not the parameter count. On ARC-AGI this gives roughly the same accuracy as a 27M-parameter HRM at a fraction of the compute. The bet here is that the same trick — *recurse, don't grow* — transfers to image reconstruction, where iterative refinement (data consistency, denoise, repeat) is already the standard mental model.
 
There's also a halting head (Q-head, ACT-style) that learns when to stop iterating per sample. Easy slices halt early; hard ones keep refining. That gives you a knob between latency and quality at inference time without retraining.
 
---
 
## Architecture at a glance
 
```
masked k-space (B, 2, H*W) ──┐
                             ├─→ MRIEncoder ──→ input_embeddings (B, H*W, D)
mask (B, W) ─────────────────┘                       │
                                                     ▼
                                  ┌────── recursive reasoning loop ────────┐
                                  │   for h in 1..H_cycles:                 │
                                  │     for l in 1..L_cycles:               │
                                  │       z_L ← TRMBlock(z_L, z_H + input)  │
                                  │     z_H ← TRMBlock(z_H, z_L)            │
                                  │   (T-1 cycles no-grad, last cycle grad) │
                                  └─────────────────────────────────────────┘
                                                     │
                                                     ▼
                                              MRIDecoder
                                                     │
                                                     ▼
                                         pred_image (B, H*W)
                                                     │
                                                     ▼
                                          Data Consistency
                                  (FFT, blend with observed lines, iFFT)
                                                     │
                                                     ▼
                                          final reconstruction
```
 
The transformer block is a vanilla post-norm RMSNorm + multi-head attention + SwiGLU MLP. RoPE for positional encoding. Everything runs in bfloat16 except the FFT/data-consistency block, which is forced to float32 for numerical stability.
 
Data consistency is the one place where MRI physics shows up explicitly. After the decoder produces a candidate image we take its centred FFT, blend its k-space with the observed (masked) k-space line-by-line — keeping what the scanner actually saw and only filling in unobserved lines from the network — and inverse-FFT back. This is the standard "DC layer" used in nearly every learned MRI reconstruction since DC-CNN (Schlemper et al., 2017).
 
---
 
## Quick start
 
### 1. Environment
 
You'll need Python 3.10, CUDA 12.1, and a recent PyTorch. Install the pinned dependencies:
 
```bash
pip install -r specific_requirements.txt
```
 
If you'd rather pick versions yourself, `requirements.txt` has the unpinned list.
 
A few of the heavier dependencies are worth flagging: `adam-atan2` (the Atan2-stabilised Adam variant TRM uses), `wandb` for logging, `hydra-core` for the config system, and `h5py` for reading raw fastMRI files.
 
### 2. Get fastMRI knee data
 
This repo trains on the [fastMRI knee dataset](https://fastmri.med.nyu.edu/). After signing up and downloading, you'll have a folder full of `.h5` files. Point the data builder at it:
 
```bash
python -m datasets.build_mri_dataset \
    --input-dir  data/fastmri_knee_raw \
    --output-dir data/mri-knee \
    --acceleration 4 \
    --center-fraction 0.08
```
 
This converts the raw `.h5` files into four NumPy arrays per split:
 
| File | Shape | What it is |
|---|---|---|
| `all__inputs.npy` | `(N, 2, H*W)` | masked single-coil-equivalent k-space, real & imag stacked |
| `all__labels.npy` | `(N, H*W)` | RSS magnitude image, normalised to [0, 1] |
| `all__masks.npy`  | `(N, W)` | 1-D Cartesian undersampling mask (0 = unsampled, 1 = kept) |
| `all__scales.npy` | `(N,)` | per-slice RSS max — needed for scale-aware data consistency |
 
Plus a `dataset.json` with the metadata (height, width, acceleration, etc.).
 
Two things worth knowing about the converter:
 
It uses centred FFTs throughout (`ifftshift → ifft2 → fftshift`) to match the fastMRI convention where DC sits at the centre of k-space. Get this wrong and you produce shifted images that look fine but aren't aligned with their labels.
 
For multi-coil data it does per-coil iFFT first, then RSS-combines, then writes a *virtual single-coil* k-space defined as `FFT(rss_image × scale)`. This keeps the input/label pair self-consistent for data consistency without forcing the model to handle coil sensitivity maps.
 
### 3. Train
 
Single GPU:
 
```bash
python pretrain.py --config-name cfg_mri_pretrain \
    data_paths='[data/mri-knee]' \
    epochs=50000 eval_interval=5000 \
    lr=1e-4 global_batch_size=4 ema=True
```
 
Four GPUs:
 
```bash
torchrun --nproc-per-node 4 pretrain.py --config-name cfg_mri_pretrain \
    data_paths='[data/mri-knee]' \
    epochs=50000 eval_interval=5000 \
    lr=1e-4 ema=True
```
 
Anything you put on the command line overrides the YAML — that's [Hydra](https://hydra.cc/) doing its thing. Useful overrides:
 
```bash
# Quick smoke test — 200 epochs, eval every 50
epochs=200 eval_interval=50
 
# Bigger model — more recursion depth
arch.H_cycles=4 arch.L_cycles=4 arch.hidden_size=384
 
# Different acceleration factor (re-run build_mri_dataset first!)
data_paths='[data/mri-knee-x8]'
```
 
Set `WANDB_MODE=disabled` if you don't want to log to W&B.
 
---
 
## Configuration
 
The config lives at `config/cfg_mri_pretrain.yaml`. The architecture itself is at `config/arch/trm_mri.yaml`. The defaults are tuned for fastMRI knee at 4× acceleration, single GPU, batch size 4. You'll want to tweak:
 
- `lr` — learning rate. 1e-4 with cosine schedule is a sane starting point.
- `lr_min_ratio` — how far the cosine schedule decays. Default 0.1, meaning LR ends at `0.1 × lr`. Set to 1.0 to disable decay.
- `global_batch_size` — *global* across all GPUs. Per-GPU batch is `global_batch_size / world_size`.
- `arch.H_cycles` / `arch.L_cycles` — outer/inner recursion depth. More = better reasoning but quadratic compute.
- `arch.halt_max_steps` — how many ACT steps the model can take. 1 disables ACT.
- `arch.no_ACT_continue` — when `True` (default), only the halt-logit decides when to stop. When `False`, the model also learns a continue-logit and uses Q-learning bootstrap targets, matching upstream TRM exactly.
- `ema=True` — enable exponential moving average of weights for evaluation. Costs almost nothing, usually helps.
---
 
## Repo layout
 
```
config/
  cfg_mri_pretrain.yaml          # main training config
  arch/trm_mri.yaml              # architecture (cycles, layers, hidden size, ...)
 
datasets/
  build_mri_dataset.py           # fastMRI .h5 → .npy converter
 
models/
  layers.py                      # CastedLinear, RoPE, SwiGLU, Attention, RMSNorm
  common.py                      # trunc_normal init
  ema.py                         # EMA helper (now also tracks persistent buffers)
  losses.py                      # ACTLossHead — used by ARC configs, kept for parity
  losses_mri.py                  # MRILossHead — MSE + Q-halt + Q-continue
  sparse_embedding.py            # ARC-only sparse embedding & SignSGD optimizer
  recursive_reasoning/
    trm.py                       # canonical Samsung TRM (ARC)
    trm_mri.py                   # TRM adapted for MRI (CNN encoder/decoder, DC layer)
    hrm.py                       # HRM baseline
    transformers_baseline.py     # plain transformer baseline
    trm_singlez.py, trm_hier6.py # TRM variants
 
utils/
  functions.py                   # load_model_class, get_model_source_path
 
pretrain.py                      # the actual training loop
```
 
---
 
## What's different from upstream Samsung TRM
 
The architecture is essentially unchanged — same recursive z_H/z_L latents, same shared TRMBlock, same ACT halting head with exploration. The MRI-specific changes are bolted on around it:
 
**MRIEncoder** — three input channels (real, imag, mask) through two depthwise-separable convs and a linear projection up to `hidden_size`. The mask is broadcast as a third channel so the network knows which k-space lines were observed.
 
**MRIDecoder** — single linear projection from `hidden_size` down to one real value per spatial position. No sigmoid; the data-consistency layer downstream can push values outside [0, 1] and clamping there would break the gradient through DC.
 
**Data consistency** — float32 centred-FFT block, scale-aware blend using `all__scales.npy`, applied after every decoder pass. This is the only MRI-physics piece in the whole model.
 
**Q-head** — mean-pools over spatial positions instead of reading position 0, since position 0 is now a pixel (not the dedicated summary token ARC has).
 
**Loss** — MSE on the post-DC image instead of cross-entropy on logits, PSNR as the reported metric, Q-halt target derived from per-batch median MSE (a sample is "halted-correctly" if its MSE is below the batch median).
 
If you want the unmodified TRM for an ARC run, `arch=trm` and the original cross-entropy loss head still work — the repo is bilingual.
 
---
 
## Results
 
*Coming soon.* Currently training on fastMRI knee single-coil at 4× acceleration. Once I have stable PSNR / SSIM numbers I'll fill in a comparison table against E2E-VarNet, U-Net baseline, and zero-filled reconstruction.
 
If you reproduce the training and have numbers worth sharing, open an issue or a PR.
 
---
 
## Acknowledgements
 
This work builds directly on:
 
- **TinyRecursiveModels** by Alexia Jolicoeur-Martineau et al. at Samsung SAIL Montreal — the recursive architecture, ACT halting, and almost all of the training scaffolding (`losses.py`, the `ACTV1` carry, the AdamATan2 + cosine schedule recipe). See [the paper](https://arxiv.org/abs/2510.04871) and the [original repo](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
- **HRM** (Hierarchical Reasoning Model) by Wang et al. — TRM's predecessor, also included as a baseline in `models/recursive_reasoning/hrm.py`.
- **fastMRI** by NYU and Facebook AI Research — the knee/brain dataset and evaluation conventions that make any of this reproducible.
The MRI-specific adaptation, the data-consistency integration, the dataset converter, and all the bugs (which I will own when you find them) are mine.
 
---
 
## License
 
Apache 2.0 — same as upstream TRM. See `LICENSE`.
 
---
 
## Citation
 
If this code helps your research:
 
```bibtex
@misc{trm-mri,
  author       = {Sathvik Loke},
  title        = {TRM-MRI: Tiny Recursive Models for accelerated MRI reconstruction},
  year         = {2026},
  howpublished = {\url{https://github.com/sathvikloke/TRM-MRI}},
}
```
 
And please cite the upstream TRM and HRM papers — none of this exists without them.
 

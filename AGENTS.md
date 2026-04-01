# AGENTS.md

## Purpose
This repository is now an **OSTrack-only single-object tracking workspace**.
Agents operating here should optimize for the paper workflow:

1. extract frames from video
2. label one target per frame
3. convert annotations into GOT10K-like data
4. fine-tune OSTrack
5. run checkpoints on local videos and compare results

Do not invent support for the removed legacy tracker pipeline.

## Repository Reality
- Root workflow is centered on `ostrack/`.
- Legacy files such as `demo.py`, `train.py`, and `transformer_tracker.py` are no longer part of the supported workflow.
- Main retained assets:
  - `env.yml`
  - `readme.md`
  - `AGENTS.md`
  - `test.mp4`
  - `ostrack/`
- Vendored upstream code lives at `ostrack/vendor/OSTrack/`.

## Environment Setup
Base environment:

```bash
conda env create -f env.yml
conda activate transformer_tracker
```

OSTrack extras:

```bash
pip install -r ostrack/requirements.txt
```

Required MAE checkpoint path:

```bash
ostrack/vendor/OSTrack/pretrained_models/mae_pretrain_vit_base.pth
```

## No Formal Lint/Test Stack
There is still no formal pytest/ruff/mypy project setup at the repo root.

Closest verification steps are:

```bash
python -m py_compile ostrack/tools/extract_frames.py ostrack/tools/label_sequence.py ostrack/tools/prepare_dataset.py ostrack/train_ostrack.py ostrack/run_video.py ostrack/run_video_batch.py
python ostrack/train_ostrack.py --dry-run
python ostrack/run_video.py --help
python ostrack/run_video_batch.py --help
```

Use these as smoke checks only.

## Primary Scripts and Real Commands

### 1. Extract frames
```bash
python ostrack/tools/extract_frames.py --video test.mp4 --output-root ostrack\data\labeled --sequence-name my_sequence
```

### 2. Label frames
```bash
python ostrack/tools/label_sequence.py --frames-dir ostrack\data\labeled\my_sequence\frames --output-root ostrack\data\labeled --sequence-name my_sequence
```

### 3. Prepare GOT10K-like data
```bash
python ostrack/tools/prepare_dataset.py --sequence-dir ostrack\data\labeled\my_sequence\frames --gt-file ostrack\data\labeled\my_sequence\groundtruth.txt --output-root ostrack\data\processed --sequence-name my_sequence --val-ratio 0.2 --overwrite
```

### 4. Dry-run training setup
```bash
python ostrack/train_ostrack.py --dry-run
```

### 5. Launch training
```bash
python ostrack/train_ostrack.py --mode single --nproc-per-node 1 --use-wandb 0
```

### 6. Run one checkpoint on video
```bash
python ostrack/run_video.py --video test.mp4 --config vitb_256_mae_ce_32x4_custom_ep50 --checkpoint "ostrack/outputs/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_custom_ep50/OSTrack_ep0005.pth.tar" --init-bbox 181 386 223 186
```

### 7. Compare multiple checkpoints
```bash
python ostrack/run_video_batch.py --video test.mp4 --config vitb_256_mae_ce_32x4_custom_ep50 --epochs 1 3 5 --init-bbox 181 386 223 186
```

## Important Behavioral Notes
- `label_sequence.py` is interactive and uses OpenCV mouse-drag ROI selection.
- `prepare_dataset.py` expects `groundtruth.txt` in `x,y,w,h` format.
- `prepare_dataset.py` requires at least 20 frames.
- The prepared dataset is GOT10K-like, not full official GOT10K.
- The current custom config is intentionally biased toward quick single-GPU experimentation:
  - validation effectively disabled during the 50-epoch run
  - checkpoint saving enabled every epoch
  - batch size reduced for stability
- Checkpoints are expected under:

```bash
ostrack/outputs/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_custom_ep50/
```

## Code Style and Editing Guidance
- Use 4 spaces for indentation.
- Keep edits local and minimal.
- Prefer preserving upstream OSTrack structure inside `ostrack/vendor/OSTrack/`.
- When patching vendored code, make the smallest compatibility fix possible and avoid broad refactors.
- Keep new local tooling under `ostrack/` rather than the repo root when feasible.
- Use explicit CLI flags and clear path defaults.

## Known Local Modifications to Vendored OSTrack
Agents should assume the vendored OSTrack code is **not pristine upstream**. It has local adjustments for this workspace, including:
- custom GOT10K-like dataset names
- Windows path normalization in generated local environment files
- optional imports/fallbacks for some dependencies
- compatibility fixes for newer PyTorch/Python behavior
- training configured to save checkpoints every epoch

Do not casually revert these changes.

## Verification Guidance
When changing this repository:
- If you change labeling tools, run their `--help` and a `py_compile` smoke check.
- If you change dataset prep, verify the generated `got10k/train` and `got10k/val` layout.
- If you change training code, run `python ostrack/train_ostrack.py --dry-run` first.
- If you change video inference, run `python ostrack/run_video.py --help` and, when a checkpoint exists, test one local video.

## Cursor / Copilot Rules
No repository-specific Cursor or Copilot rule files were found.

## Agent Operating Rule
Treat this repository as an **OSTrack fine-tuning and evaluation workspace**, not as a general tracker playground.

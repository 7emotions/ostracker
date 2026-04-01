# OSTrack Fine-Tuning Workspace

This directory adds a paper-oriented OSTrack workflow without replacing the original repo.

## Layout

- `vendor/OSTrack/`: upstream OSTrack code (cloned locally)
- `tools/prepare_dataset.py`: convert one annotated single-object sequence into a GOT10K-like train/val layout
- `train_ostrack.py`: wrapper that initializes OSTrack local paths and launches training
- `requirements.txt`: extra Python packages commonly needed by OSTrack on top of the root environment

## Recommended environment

Start from a Python 3.8 environment with PyTorch installed, then add OSTrack extras:

```bash
conda activate transformer_tracker
pip install -r ostrack/requirements.txt
```

If you prefer the official upstream environment, see:

```bash
ostrack/vendor/OSTrack/ostrack_cuda113_env.yaml
```

## Required pretrained checkpoint

OSTrack training expects the MAE ViT-Base pretrained checkpoint at:

```bash
ostrack/vendor/OSTrack/pretrained_models/mae_pretrain_vit_base.pth
```

Official source:

```bash
https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
```

Create the directory if needed before downloading the file.

## Prepare a simple single-object dataset

### 1. Extract frames from video

```bash
python ostrack/tools/extract_frames.py ^
  --video test.mp4 ^
  --output-root ostrack\data\labeled ^
  --sequence-name my_sequence
```

This creates:

```bash
ostrack/data/labeled/my_sequence/frames/
```

### 2. Label the extracted frames

```bash
python ostrack/tools/label_sequence.py ^
  --frames-dir ostrack\data\labeled\my_sequence\frames ^
  --output-root ostrack\data\labeled ^
  --sequence-name my_sequence
```

Controls:

- drag left mouse: draw bbox
- `SPACE`: confirm current frame and advance
- `R`: clear current bbox
- `B`: go back one frame
- `Q` / `ESC`: save progress and quit

This writes:

```bash
ostrack/data/labeled/my_sequence/groundtruth.txt
```

and keeps a copy of labeled frames under the same sequence directory.

### 3. Convert labels into OSTrack training data

The provided preparation script assumes:

- one image sequence directory
- one text file with one bbox per line in `x,y,w,h` format
- frame order matches the sorted image filenames
- at least 20 frames so OSTrack's sampler can form valid pairs

Example:

```bash
python ostrack/tools/prepare_dataset.py ^
  --sequence-dir ostrack\data\labeled\my_sequence\frames ^
  --gt-file ostrack\data\labeled\my_sequence\groundtruth.txt ^
  --output-root ostrack\data\processed ^
  --sequence-name my_sequence ^
  --val-ratio 0.2 ^
  --overwrite
```

This writes a GOT10K-like layout under:

```bash
ostrack/data/processed/got10k/train/
ostrack/data/processed/got10k/val/
```

Frames are converted to `00000001.jpg` style names so they match OSTrack's GOT10K loader.

The vendored OSTrack code has been patched to recognize:

- `CUSTOM_GOT10K_train`
- `CUSTOM_GOT10K_val`

through the config file:

```bash
ostrack/vendor/OSTrack/experiments/ostrack/vitb_256_mae_ce_32x4_custom_ep50.yaml
```

## Dry-run the training wrapper

Use this first to verify paths before starting training:

```bash
python ostrack/train_ostrack.py --dry-run
```

## Launch fine-tuning

Single GPU example:

```bash
python ostrack/train_ostrack.py ^
  --data-root ostrack\data\processed ^
  --save-dir ostrack\outputs ^
  --config vitb_256_mae_ce_32x4_custom_ep50 ^
  --mode single ^
  --nproc-per-node 1 ^
  --use-wandb 0
```

## Important notes

- The current dataset tool is intentionally minimal and aimed at a simple single-object paper workflow.
- It prepares one sequence into train/val splits; if you run it multiple times with different `--sequence-name` values, new sequence names are appended to each split's `list.txt`.
- `label_sequence.py` will automatically use the parent folder name when `--frames-dir` ends with `frames`, but passing `--sequence-name` explicitly is still the clearest option.
- `train_ostrack.py` will fail fast if the MAE checkpoint or processed GOT10K-like folders are missing.
- Existing root scripts (`demo.py`, `train.py`) are unchanged and can still be used independently.

# AGENTS.md

## Purpose
This repository is an **OSTrack-only single-object tracking workspace**.
Agents should optimize for the paper workflow this repo actually supports:
1. extract frames from a local video
2. label one target per frame
3. convert annotations into GOT10K-like train/val data
4. fine-tune OSTrack
5. run checkpoints on local videos and compare outputs

Do **not** invent support for the removed legacy tracker pipeline.

## Repository Scope
- The supported workflow is rooted in `ostrack/`.
- Vendored upstream OSTrack code lives in `ostrack/vendor/OSTrack/`.
- Root-level legacy scripts are not the recommended path for this workspace.
- The main top-level files are `README.md`, `env.yml`, `AGENTS.md`, `test.mp4`, and `ostrack/`.

## Environment Setup
Create and activate the intended conda environment first:
```bash
conda env create -f env.yml
conda activate transformer_tracker
```

`env.yml` pins the base environment to **Python 3.8** and installs PyTorch, torchvision, OpenCV, NumPy, matplotlib, and pip.

Then install OSTrack-specific packages:
```bash
python -m pip install -r ostrack/requirements.txt
```

Use `python -m pip`, not bare `pip`, when possible.

## Required Assets
Training expects the MAE pretrained checkpoint here:
```bash
ostrack/vendor/OSTrack/pretrained_models/mae_pretrain_vit_base.pth
```

The upstream-style environment reference is:
```bash
ostrack/vendor/OSTrack/ostrack_cuda113_env.yaml
```

## Primary Workflow Commands
### 1. Extract frames from video
```bash
python ostrack/tools/extract_frames.py --video test.mp4 --output-root ostrack\data\labeled --sequence-name my_sequence
```

### 2. Label one target per frame
```bash
python ostrack/tools/label_sequence.py --frames-dir ostrack\data\labeled\my_sequence\frames --output-root ostrack\data\labeled --sequence-name my_sequence
```

### 3. Convert labels into GOT10K-like data
```bash
python ostrack/tools/prepare_dataset.py --sequence-dir ostrack\data\labeled\my_sequence\frames --gt-file ostrack\data\labeled\my_sequence\groundtruth.txt --output-root ostrack\data\processed --sequence-name my_sequence --val-ratio 0.2 --overwrite
```

### 4. Dry-run the training wrapper
```bash
python ostrack/train_ostrack.py --dry-run
```

### 5. Launch fine-tuning
```bash
python ostrack/train_ostrack.py --mode single --nproc-per-node 1 --use-wandb 0
```

### 6. Run one checkpoint on a local video
```bash
python ostrack/run_video.py --video test.mp4 --config vitb_256_mae_ce_32x4_custom_ep50 --checkpoint "ostrack/outputs/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_custom_ep50/OSTrack_ep0005.pth.tar" --init-bbox 181 386 223 186
```

### 7. Compare multiple checkpoints
```bash
python ostrack/run_video_batch.py --video test.mp4 --config vitb_256_mae_ce_32x4_custom_ep50 --epochs 1 3 5 --init-bbox 181 386 223 186
```

## Build / Lint / Test Reality
There is **no formal repo-wide lint, test, or typecheck stack** such as pytest, ruff, mypy, tox, or nox at the repository root.

For this repo, treat the following as the supported verification commands:
```bash
python -m py_compile ostrack/tools/extract_frames.py ostrack/tools/label_sequence.py ostrack/tools/prepare_dataset.py ostrack/train_ostrack.py ostrack/run_video.py ostrack/run_video_batch.py
python ostrack/train_ostrack.py --dry-run
python ostrack/run_video.py --help
python ostrack/run_video_batch.py --help
```

## Single-File / Single-Target Verification
If you change one Python file, the closest thing to a single-test command is:
```bash
python -m py_compile path\to\changed_file.py
```

Use targeted smoke checks depending on the file you changed:
- `ostrack/tools/extract_frames.py` → `python ostrack/tools/extract_frames.py --help`
- `ostrack/tools/label_sequence.py` → `python ostrack/tools/label_sequence.py --help`
- `ostrack/tools/prepare_dataset.py` → `python ostrack/tools/prepare_dataset.py --help`
- `ostrack/train_ostrack.py` → `python ostrack/train_ostrack.py --dry-run`
- `ostrack/run_video.py` → `python ostrack/run_video.py --help`
- `ostrack/run_video_batch.py` → `python ostrack/run_video_batch.py --help`

There is no meaningful pytest-style “run a single test” command because the repo does not define a test suite.

## Data and Directory Expectations
- Labeled frame sequences live under `ostrack/data/labeled/`.
- Processed GOT10K-like data lives under `ostrack/data/processed/got10k/`.
- Outputs live under `ostrack/outputs/`.
- Checkpoints are expected under:
```bash
ostrack/outputs/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_custom_ep50/
```

## Important Behavioral Constraints
- `label_sequence.py` is interactive and depends on OpenCV ROI selection.
- `prepare_dataset.py` expects `groundtruth.txt` lines in `x,y,w,h` format.
- `prepare_dataset.py` requires at least **20 frames**.
- The generated data is **GOT10K-like**, not the full official GOT10K dataset.
- The current custom experiment config is `vitb_256_mae_ce_32x4_custom_ep50`.
- The vendored workspace has local modifications for this repo; do not casually revert them.

## Code Style Guidelines
### General Rules
- Use **4 spaces** for indentation.
- Prefer **small, local edits** over broad refactors.
- Preserve the structure of `ostrack/vendor/OSTrack/` whenever possible.
- Put new repo-specific tooling under `ostrack/` instead of the repo root when feasible.
- Prefer explicit CLI flags and explicit path defaults.

### Imports and Formatting
- Keep imports grouped as standard library first, then third-party, with a blank line between groups.
- Prefer direct imports already used in nearby files.
- Do not introduce unused imports.
- Match surrounding formatting instead of reformatting unrelated code.
- Keep long argparse declarations readable with one argument per block.
- Avoid compressing multi-step logic into dense one-liners.

### Types and Naming
- Follow the existing lightweight style: `Path` for filesystem arguments, builtin types for argparse values, and simple annotations where they help.
- Do not add type noise to small scripts.
- Do not suppress type issues with `Any`, ignore comments, or similar escapes.
- Use descriptive snake_case for functions, variables, and helpers.
- Keep CLI flag names explicit and reuse local names like `sequence_name`, `output_root`, `checkpoint_path`, and `repo_root`.

### Paths, Errors, and Execution
- Prefer `pathlib.Path` over string path manipulation.
- Resolve relative paths against the repo layout the same way existing wrappers do.
- Create directories with `mkdir(parents=True, exist_ok=True)` when needed.
- Validate important files and directories before launching heavy work.
- Validate early and raise **clear `ValueError` messages** for invalid inputs or missing files.
- Fail fast on missing prerequisites rather than silently continuing.
- Do not add empty `except` blocks.
- Do not swallow subprocess failures; existing wrappers use `check=True` for a reason.
- Keep wrapper scripts explicit about the commands they build and print.
- Do not silently change default config names, checkpoint locations, or data roots.

## Vendored OSTrack Guidance
Assume `ostrack/vendor/OSTrack/` is **not pristine upstream**. Local changes may include custom dataset names like `CUSTOM_GOT10K_train` and `CUSTOM_GOT10K_val`, Windows path normalization, compatibility fixes, and single-GPU-friendly defaults. When patching vendored code, make the **smallest compatibility fix possible**.

## Verification Guidance by Change Type
- Tooling / labeling changes → run `--help` plus `py_compile`.
- Dataset-prep changes → verify the generated `got10k/train` and `got10k/val` layout.
- Training wrapper changes → run `python ostrack/train_ostrack.py --dry-run`.
- Video inference changes → run `python ostrack/run_video.py --help`; if a checkpoint exists, test one local video.

## Cursor / Copilot Rules
No repository-specific Cursor rules or Copilot instruction files were found.
- No `.cursor/rules/` directory was found.
- No `.cursorrules` file was found.
- No `.github/copilot-instructions.md` file was found.

## Agent Operating Rule
Treat this repository as an **OSTrack fine-tuning and evaluation workspace**, not as a general tracker playground.

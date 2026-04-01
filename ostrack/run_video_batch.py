import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple OSTrack checkpoints on the same video")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--config",
        type=str,
        default="vitb_256_mae_ce_32x4_custom_ep50",
        help="OSTrack config name under experiments/ostrack",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="*",
        default=None,
        help="Epoch numbers to evaluate, e.g. --epochs 1 3 5 10",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit checkpoint paths; overrides --epochs when provided",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ostrack") / "outputs" / "video_results",
        help="Directory for result videos and bbox txt files",
    )
    parser.add_argument(
        "--init-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="Optional fixed initial bbox for all checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands that would be executed",
    )
    return parser.parse_args()


def resolve_checkpoints(repo_root: Path, config_name: str, epochs, checkpoints):
    if checkpoints:
        resolved = []
        for checkpoint in checkpoints:
            path = checkpoint if checkpoint.is_absolute() else repo_root / checkpoint
            if not path.is_file():
                raise ValueError(f"Checkpoint does not exist: {path}")
            resolved.append(path)
        return resolved

    if not epochs:
        raise ValueError("Provide either --epochs or --checkpoints")

    checkpoint_dir = repo_root / "ostrack" / "outputs" / "checkpoints" / "train" / "ostrack" / config_name
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    resolved = []
    for epoch in epochs:
        checkpoint_path = checkpoint_dir / f"OSTrack_ep{epoch:04d}.pth.tar"
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint for epoch {epoch} does not exist: {checkpoint_path}")
        resolved.append(checkpoint_path)
    return resolved


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if not args.video.is_file():
        raise ValueError(f"Video file does not exist: {args.video}")

    checkpoints = resolve_checkpoints(repo_root, args.config, args.epochs, args.checkpoints)
    run_video_path = repo_root / "ostrack" / "run_video.py"

    for checkpoint_path in checkpoints:
        cmd = [
            sys.executable,
            str(run_video_path),
            "--video",
            str(args.video),
            "--config",
            args.config,
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.init_bbox is not None:
            cmd.extend(["--init-bbox", *[str(v) for v in args.init_bbox]])

        print("Running:")
        print(" ".join(cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()

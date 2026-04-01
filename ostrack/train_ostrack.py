import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Wrapper for local OSTrack fine-tuning")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("ostrack") / "data" / "processed",
        help="Root containing got10k/train and got10k/val prepared by prepare_dataset.py",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("ostrack") / "outputs",
        help="Directory for OSTrack outputs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vitb_256_mae_ce_32x4_custom_ep50",
        help="OSTrack experiment config name under experiments/ostrack",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multiple"],
        default="single",
        help="Training mode passed to OSTrack",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of GPUs per node when mode=multiple",
    )
    parser.add_argument(
        "--use-wandb",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to enable wandb logging",
    )
    parser.add_argument(
        "--pretrained-file",
        type=Path,
        default=Path("ostrack") / "vendor" / "OSTrack" / "pretrained_models" / "mae_pretrain_vit_base.pth",
        help="MAE pretrained checkpoint expected by the selected OSTrack config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands and checks without launching training",
    )
    return parser.parse_args()


def ensure_path(path: Path, description: str):
    if not path.exists():
        raise ValueError(f"{description} does not exist: {path}")


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = repo_root / "ostrack" / "vendor" / "OSTrack"
    ensure_path(vendor_root, "OSTrack vendor directory")

    got10k_train = repo_root / args.data_root / "got10k" / "train"
    got10k_val = repo_root / args.data_root / "got10k" / "val"
    ensure_path(got10k_train, "Prepared GOT10K-like train directory")
    ensure_path(got10k_val, "Prepared GOT10K-like val directory")
    ensure_path(got10k_train / "list.txt", "Train list.txt")
    ensure_path(got10k_val / "list.txt", "Val list.txt")
    ensure_path(repo_root / args.pretrained_file, "MAE pretrained checkpoint")

    save_dir = repo_root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    local_path_cmd = [
        sys.executable,
        "tracking/create_default_local_file.py",
        "--workspace_dir",
        ".",
        "--data_dir",
        str((repo_root / args.data_root).resolve()),
        "--save_dir",
        str(save_dir.resolve()),
    ]
    train_cmd = [
        sys.executable,
        "tracking/train.py",
        "--script",
        "ostrack",
        "--config",
        args.config,
        "--save_dir",
        str(save_dir.resolve()),
        "--mode",
        args.mode,
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--use_wandb",
        str(args.use_wandb),
    ]

    print("Local-path init command:")
    print(" ".join(local_path_cmd))
    print("Training command:")
    print(" ".join(train_cmd))

    if args.dry_run:
        return

    subprocess.run(local_path_cmd, cwd=vendor_root, check=True)
    subprocess.run(train_cmd, cwd=vendor_root, check=True)


if __name__ == "__main__":
    main()

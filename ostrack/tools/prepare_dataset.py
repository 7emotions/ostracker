import argparse
import shutil
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a single-object sequence into a GOT10K-like layout for OSTrack"
    )
    parser.add_argument("--sequence-dir", type=Path, required=True, help="Directory containing source frames")
    parser.add_argument("--gt-file", type=Path, required=True, help="Text file with one bbox per line: x,y,w,h")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ostrack") / "data" / "processed",
        help="Root directory for processed OSTrack data",
    )
    parser.add_argument(
        "--sequence-name",
        type=str,
        default=None,
        help="Optional base name for the generated sequence folders",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio for splitting one sequence into train/val subsets",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing processed train/val sequence folders before regenerating",
    )
    return parser.parse_args()


def list_frames(sequence_dir: Path):
    frames = [p for p in sequence_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    frames.sort()
    return frames


def parse_bbox_line(line: str):
    normalized = line.replace("\t", ",").replace(" ", ",")
    parts = [part for part in normalized.split(",") if part]
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox line: {line.strip()}")
    return [float(part) for part in parts]


def load_bboxes(gt_file: Path):
    return [parse_bbox_line(line) for line in gt_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_sequence(split_root: Path, sequence_name: str, frames, bboxes):
    sequence_root = split_root / sequence_name
    sequence_root.mkdir(parents=True, exist_ok=True)

    for index, (frame_path, bbox) in enumerate(zip(frames, bboxes), start=1):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"Failed to read frame: {frame_path}")
        target_name = f"{index:08d}.jpg"
        if not cv2.imwrite(str(sequence_root / target_name), image):
            raise ValueError(f"Failed to write converted frame: {sequence_root / target_name}")

    (sequence_root / "groundtruth.txt").write_text(
        "\n".join(",".join(f"{value:.4f}" for value in bbox) for bbox in bboxes),
        encoding="utf-8",
    )
    (sequence_root / "absence.label").write_text("\n".join("0" for _ in bboxes), encoding="utf-8")
    (sequence_root / "cover.label").write_text("\n".join("8" for _ in bboxes), encoding="utf-8")
    (sequence_root / "meta_info.ini").write_text(
        "[Sequence]\n"
        f"name: {sequence_name}\n"
        "im_dir: .\n"
        f"frame_num: {len(bboxes)}\n"
        "target_num: 1\n"
        "object_class: single_object\n"
        "motion_class: unknown\n"
        "major_class: custom\n"
        "root_class: custom\n"
        "motion_adverb: unknown\n",
        encoding="utf-8",
    )


def write_list_file(split_root: Path, sequence_name: str):
    list_path = split_root / "list.txt"
    existing = []
    if list_path.exists():
        existing = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sequence_name not in existing:
        existing.append(sequence_name)
    list_path.write_text("\n".join(existing) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    if not args.sequence_dir.is_dir():
        raise ValueError(f"Sequence directory does not exist: {args.sequence_dir}")
    if not args.gt_file.is_file():
        raise ValueError(f"Ground-truth file does not exist: {args.gt_file}")
    if not 0.0 < args.val_ratio < 0.5:
        raise ValueError("val_ratio must be between 0 and 0.5")

    frames = list_frames(args.sequence_dir)
    bboxes = load_bboxes(args.gt_file)
    if len(frames) != len(bboxes):
        raise ValueError(
            f"Frame count ({len(frames)}) does not match bbox count ({len(bboxes)})"
        )
    if len(frames) < 20:
        raise ValueError("Need at least 20 frames to build OSTrack train/val splits")

    val_count = max(2, int(round(len(frames) * args.val_ratio)))
    train_count = len(frames) - val_count
    if train_count < 2:
        raise ValueError("Training split would be too small; reduce val_ratio or add more frames")

    base_name = args.sequence_name or args.sequence_dir.name
    got10k_root = args.output_root / "got10k"
    train_root = got10k_root / "train"
    val_root = got10k_root / "val"
    train_sequence_name = f"{base_name}_train"
    val_sequence_name = f"{base_name}_val"

    if args.overwrite:
        shutil.rmtree(train_root / train_sequence_name, ignore_errors=True)
        shutil.rmtree(val_root / val_sequence_name, ignore_errors=True)

    write_sequence(train_root, train_sequence_name, frames[:train_count], bboxes[:train_count])
    write_sequence(val_root, val_sequence_name, frames[train_count:], bboxes[train_count:])
    write_list_file(train_root, train_sequence_name)
    write_list_file(val_root, val_sequence_name)

    print(f"Prepared train split: {train_root / train_sequence_name}")
    print(f"Prepared val split: {val_root / val_sequence_name}")
    print("Dataset root for OSTrack local paths:", got10k_root)


if __name__ == "__main__":
    main()

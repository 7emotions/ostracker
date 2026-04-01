import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video for manual labeling")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ostrack") / "data" / "labeled",
        help="Root directory for extracted sequences",
    )
    parser.add_argument(
        "--sequence-name",
        type=str,
        default=None,
        help="Optional output sequence name; defaults to video stem",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to export (0-based)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to export",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Export every Nth frame",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing extracted JPG frames before exporting",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.video.is_file():
        raise ValueError(f"Video file does not exist: {args.video}")
    if args.start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    sequence_name = args.sequence_name or args.video.stem
    frames_dir = args.output_root / sequence_name / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for jpg_path in frames_dir.glob("*.jpg"):
            jpg_path.unlink()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {args.video}")

    frame_index = 0
    saved_count = 0
    target_count = args.max_frames

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index < args.start_frame:
            frame_index += 1
            continue

        relative_index = frame_index - args.start_frame
        if relative_index % args.stride != 0:
            frame_index += 1
            continue

        if target_count is not None and saved_count >= target_count:
            break

        output_path = frames_dir / f"{saved_count + 1:08d}.jpg"
        if not cv2.imwrite(str(output_path), frame):
            raise ValueError(f"Failed to save frame: {output_path}")

        saved_count += 1
        frame_index += 1

    cap.release()

    print(f"Saved {saved_count} frame(s) to {frames_dir}")


if __name__ == "__main__":
    main()

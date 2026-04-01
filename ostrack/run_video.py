import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Run a fine-tuned OSTrack checkpoint on a local video")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--config",
        type=str,
        default="vitb_256_mae_ce_32x4_custom_ep50",
        help="OSTrack config name under experiments/ostrack",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path; defaults to the latest checkpoint for the selected config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ostrack") / "outputs" / "video_results",
        help="Directory for result video and bbox txt",
    )
    parser.add_argument(
        "--init-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="Optional initial bbox; otherwise select ROI interactively",
    )
    return parser.parse_args()


def resolve_vendor_root(repo_root: Path) -> Path:
    return repo_root / "ostrack" / "vendor" / "OSTrack"


def resolve_checkpoint(repo_root: Path, config_name: str, checkpoint_arg: Optional[Path]) -> Path:
    if checkpoint_arg is not None:
        checkpoint_path = checkpoint_arg if checkpoint_arg.is_absolute() else repo_root / checkpoint_arg
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path

    checkpoint_dir = repo_root / "ostrack" / "outputs" / "checkpoints" / "train" / "ostrack" / config_name
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    candidates = sorted(checkpoint_dir.glob("OSTrack_ep*.pth.tar"))
    if not candidates:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = resolve_vendor_root(repo_root)
    if not args.video.is_file():
        raise ValueError(f"Video file does not exist: {args.video}")

    checkpoint_path = resolve_checkpoint(repo_root, args.config, args.checkpoint)

    vendor_root_str = str(vendor_root)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)

    from lib.test.parameter.ostrack import parameters
    from lib.test.tracker.ostrack import OSTrack

    params = parameters(args.config)
    params.checkpoint = str(checkpoint_path)
    params.debug = 0
    params.save_all_boxes = False

    tracker = OSTrack(params, dataset_name="video")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {args.video}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise ValueError("Unable to read first frame from video")

    if args.init_bbox is None:
        display = first_frame.copy()
        cv2.putText(display, "Select target ROI and press ENTER", (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 255, 0), 1)
        init_bbox = cv2.selectROI("OSTrack Video Inference", display, fromCenter=False)
        cv2.destroyWindow("OSTrack Video Inference")
    else:
        init_bbox = tuple(args.init_bbox)

    if len(init_bbox) != 4 or init_bbox[2] <= 0 or init_bbox[3] <= 0:
        cap.release()
        raise ValueError("Initial bbox must be valid: x y w h")

    tracker.initialize(first_frame, {"init_bbox": list(init_bbox)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_name = args.video.stem
    output_video_path = args.output_dir / f"{video_name}_{checkpoint_path.stem}.mp4"
    output_bbox_path = args.output_dir / f"{video_name}_{checkpoint_path.stem}.txt"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_h, frame_w = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    output_boxes = [list(map(int, init_bbox))]
    first_draw = first_frame.copy()
    x, y, w, h = output_boxes[0]
    cv2.rectangle(first_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(first_draw, "Frame 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    writer.write(first_draw)

    frame_id = 1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        out = tracker.track(frame)
        state = [int(s) for s in out["target_bbox"]]
        output_boxes.append(state)

        frame_draw = frame.copy()
        cv2.rectangle(frame_draw, (state[0], state[1]), (state[0] + state[2], state[1] + state[3]), (0, 255, 0), 2)
        cv2.putText(frame_draw, f"Frame {frame_id + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write(frame_draw)
        frame_id += 1

    cap.release()
    writer.release()

    with output_bbox_path.open("w", encoding="utf-8") as f:
        for bbox in output_boxes:
            f.write("\t".join(str(v) for v in bbox) + "\n")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved result video to: {output_video_path}")
    print(f"Saved bbox trajectory to: {output_bbox_path}")


if __name__ == "__main__":
    main()

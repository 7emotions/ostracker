import argparse
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class SequenceLabeler:
    def __init__(self, frames_dir: Path, output_root: Path, sequence_name: str, overwrite: bool = False):
        self.frames_dir = frames_dir
        self.output_root = output_root
        self.sequence_name = sequence_name
        self.sequence_root = output_root / sequence_name
        self.output_frames_dir = self.sequence_root / "frames"
        self.gt_path = self.sequence_root / "groundtruth.txt"
        self.window_name = "OSTrack Sequence Labeler"

        self.frames = sorted(
            [path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        )
        if not self.frames:
            raise ValueError(f"No image frames found in {frames_dir}")

        self.sequence_root.mkdir(parents=True, exist_ok=True)
        self.output_frames_dir.mkdir(parents=True, exist_ok=True)
        if overwrite and self.gt_path.exists():
            self.gt_path.unlink()

        self.current_index = 0
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.current_bbox = None
        self.annotations = self._load_existing_annotations()

    def _load_existing_annotations(self):
        if not self.gt_path.exists():
            return []

        annotations = []
        for line in self.gt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [float(part) for part in line.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Invalid annotation line: {line}")
            annotations.append(tuple(int(round(value)) for value in parts))
        return annotations

    def _save_annotations(self):
        self.gt_path.write_text(
            "\n".join(
                ",".join(str(value) for value in bbox)
                for bbox in self.annotations
            )
            + ("\n" if self.annotations else ""),
            encoding="utf-8",
        )

    def _copy_frame(self, frame_path: Path, index: int):
        output_path = self.output_frames_dir / f"{index + 1:08d}.jpg"
        if output_path.exists():
            return

        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"Failed to read frame: {frame_path}")
        if not cv2.imwrite(str(output_path), image):
            raise ValueError(f"Failed to write labeled frame copy: {output_path}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            if self.start_point and self.end_point:
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.current_bbox = (x1, y1, x2 - x1, y2 - y1)

    def _draw_selection_box(self, frame):
        if self.selecting and self.start_point and self.end_point:
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 0), 2)

    def _draw_existing_bbox(self, frame):
        bbox = self.current_bbox
        if bbox is None and self.current_index < len(self.annotations):
            bbox = self.annotations[self.current_index]
        if bbox is None:
            return

        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def _confirm_current_frame(self):
        if self.current_bbox is None:
            return False

        self._copy_frame(self.frames[self.current_index], self.current_index)
        if self.current_index < len(self.annotations):
            self.annotations[self.current_index] = self.current_bbox
        else:
            self.annotations.append(self.current_bbox)
        self._save_annotations()
        self.current_index += 1
        self.current_bbox = None
        self.start_point = None
        self.end_point = None
        return True

    def _go_back(self):
        if self.current_index == 0:
            return
        self.current_index -= 1
        self.current_bbox = self.annotations[self.current_index] if self.current_index < len(self.annotations) else None
        self.start_point = None
        self.end_point = None

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        if self.annotations:
            self.current_index = min(len(self.annotations), len(self.frames) - 1)

        while self.current_index < len(self.frames):
            frame = cv2.imread(str(self.frames[self.current_index]))
            if frame is None:
                raise ValueError(f"Failed to read frame: {self.frames[self.current_index]}")

            display = frame.copy()
            self._draw_existing_bbox(display)
            self._draw_selection_box(display)

            cv2.putText(display, f"Frame {self.current_index + 1}/{len(self.frames)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Drag: bbox | SPACE: confirm | R: reset | B: back | Q/ESC: save&quit", (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(10) & 0xFF

            if key == ord(" "):
                self._confirm_current_frame()
            elif key in (ord("r"), ord("R")):
                self.current_bbox = None
                self.start_point = None
                self.end_point = None
            elif key in (ord("b"), ord("B")):
                self._go_back()
            elif key in (ord("q"), 27):
                break

        cv2.destroyAllWindows()
        print(f"Saved annotations to {self.gt_path}")
        print(f"Copied labeled frames to {self.output_frames_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Label a single-object frame sequence and export groundtruth.txt")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing extracted frames")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ostrack") / "data" / "labeled",
        help="Root directory for labeled sequences",
    )
    parser.add_argument(
        "--sequence-name",
        type=str,
        default=None,
        help="Optional output sequence name; defaults to frames directory name",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing groundtruth.txt before labeling",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.frames_dir.is_dir():
        raise ValueError(f"Frames directory does not exist: {args.frames_dir}")

    if args.sequence_name:
        sequence_name = args.sequence_name
    elif args.frames_dir.name.lower() == "frames":
        sequence_name = args.frames_dir.parent.name
    else:
        sequence_name = args.frames_dir.name

    labeler = SequenceLabeler(args.frames_dir, args.output_root, sequence_name, overwrite=args.overwrite)
    labeler.run()


if __name__ == "__main__":
    main()

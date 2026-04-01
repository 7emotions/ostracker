import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(
    r"\[train:\s*(?P<epoch>\d+),\s*(?P<step>\d+)\s*/\s*(?P<steps_per_epoch>\d+)\]"
    r".*?Loss/total:\s*(?P<loss_total>[\d.]+)"
    r".*?Loss/giou:\s*(?P<loss_giou>[\d.]+)"
    r".*?Loss/l1:\s*(?P<loss_l1>[\d.]+)"
    r".*?Loss/location:\s*(?P<loss_location>[\d.]+)"
    r".*?IoU:\s*(?P<iou>[\d.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-ready charts from OSTrack outputs")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("ostrack") / "outputs" / "logs" / "ostrack-vitb_256_mae_ce_32x4_custom_ep50.log",
        help="Training log file to parse",
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=Path("ostrack") / "outputs" / "video_results" / "test_OSTrack_ep0005.pth.txt",
        help="Predicted bbox txt file from run_video.py",
    )
    parser.add_argument(
        "--groundtruth-file",
        type=Path,
        default=Path("ostrack") / "data" / "labeled" / "my_sequence" / "groundtruth.txt",
        help="Ground-truth bbox txt file",
    )
    parser.add_argument(
        "--train-meta-file",
        type=Path,
        default=Path("ostrack") / "data" / "processed" / "got10k" / "train" / "my_sequence_train" / "meta_info.ini",
        help="Train split meta_info.ini file",
    )
    parser.add_argument(
        "--val-meta-file",
        type=Path,
        default=Path("ostrack") / "data" / "processed" / "got10k" / "val" / "my_sequence_val" / "meta_info.ini",
        help="Val split meta_info.ini file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ostrack") / "outputs" / "paper_figures",
        help="Directory to write figures and summary tables",
    )
    return parser.parse_args()


def ensure_file(path: Path, description: str):
    if not path.is_file():
        raise ValueError(f"{description} does not exist: {path}")


def parse_training_log(log_file: Path):
    records = []
    for raw_line in log_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LOG_PATTERN.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        step = int(match.group("step"))
        steps_per_epoch = int(match.group("steps_per_epoch"))
        global_step = (epoch - 1) * steps_per_epoch + step
        records.append(
            {
                "epoch": epoch,
                "step": step,
                "steps_per_epoch": steps_per_epoch,
                "global_step": global_step,
                "loss_total": float(match.group("loss_total")),
                "loss_giou": float(match.group("loss_giou")),
                "loss_l1": float(match.group("loss_l1")),
                "loss_location": float(match.group("loss_location")),
                "iou": float(match.group("iou")),
            }
        )
    if not records:
        raise ValueError(f"No parsable training records found in {log_file}")
    return records


def parse_bbox_file(file_path: Path, delimiter: str):
    boxes = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = line.replace("\t", delimiter).replace(" ", delimiter)
        parts = [part for part in normalized.split(delimiter) if part]
        if len(parts) != 4:
            raise ValueError(f"Invalid bbox line in {file_path}: {line}")
        boxes.append([float(part) for part in parts])
    if not boxes:
        raise ValueError(f"No bbox rows found in {file_path}")
    return boxes


def parse_meta_frame_count(meta_file: Path):
    for raw_line in meta_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("frame_num:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"frame_num not found in {meta_file}")


def bbox_center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def bbox_area(box):
    return box[2] * box[3]


def bbox_iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = bbox_area(box_a) + bbox_area(box_b) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def center_error(box_a, box_b):
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return math.hypot(ax - bx, ay - by)


def make_training_figure(records, output_path: Path):
    steps = [record["global_step"] for record in records]
    epochs = sorted({record["epoch"] for record in records})
    epoch_boundaries = []
    for epoch in epochs[:-1]:
        epoch_records = [record for record in records if record["epoch"] == epoch]
        epoch_boundaries.append(epoch_records[-1]["global_step"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(steps, [record["loss_total"] for record in records], label="Loss/total", linewidth=2)
    axes[0].plot(steps, [record["loss_location"] for record in records], label="Loss/location", linewidth=1.8)
    axes[0].plot(steps, [record["loss_giou"] for record in records], label="Loss/giou", linewidth=1.6)
    axes[0].plot(steps, [record["loss_l1"] for record in records], label="Loss/l1", linewidth=1.6)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("OSTrack Training Curves")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, [record["iou"] for record in records], color="tab:green", label="Train IoU", linewidth=2)
    axes[1].set_xlabel("Global step")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Training IoU Progress")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for boundary in epoch_boundaries:
        axes[0].axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        axes[1].axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_tracking_figure(frames, gt_boxes, pred_boxes, output_path: Path):
    gt_center_x = [bbox_center(box)[0] for box in gt_boxes]
    gt_center_y = [bbox_center(box)[1] for box in gt_boxes]
    pred_center_x = [bbox_center(box)[0] for box in pred_boxes]
    pred_center_y = [bbox_center(box)[1] for box in pred_boxes]
    ious = [bbox_iou(gt_box, pred_box) for gt_box, pred_box in zip(gt_boxes, pred_boxes)]
    errors = [center_error(gt_box, pred_box) for gt_box, pred_box in zip(gt_boxes, pred_boxes)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(frames, gt_center_x, label="GT center x", linewidth=2)
    axes[0].plot(frames, pred_center_x, label="Pred center x", linewidth=2)
    axes[0].plot(frames, gt_center_y, label="GT center y", linewidth=1.8)
    axes[0].plot(frames, pred_center_y, label="Pred center y", linewidth=1.8)
    axes[0].set_ylabel("Center coord")
    axes[0].set_title("Target Center Trajectory")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2)

    axes[1].plot(frames, ious, color="tab:green", linewidth=2, label="Per-frame IoU")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Prediction Overlap with Ground Truth")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(frames, errors, color="tab:red", linewidth=2, label="Center error")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Pixels")
    axes[2].set_title("Center Error over Time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_bbox_size_figure(frames, gt_boxes, pred_boxes, output_path: Path):
    gt_width = [box[2] for box in gt_boxes]
    pred_width = [box[2] for box in pred_boxes]
    gt_height = [box[3] for box in gt_boxes]
    pred_height = [box[3] for box in pred_boxes]
    gt_area = [bbox_area(box) for box in gt_boxes]
    pred_area = [bbox_area(box) for box in pred_boxes]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(frames, gt_width, label="GT width", linewidth=2)
    axes[0].plot(frames, pred_width, label="Pred width", linewidth=2)
    axes[0].set_ylabel("Pixels")
    axes[0].set_title("Bounding Box Width")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(frames, gt_height, label="GT height", linewidth=2)
    axes[1].plot(frames, pred_height, label="Pred height", linewidth=2)
    axes[1].set_ylabel("Pixels")
    axes[1].set_title("Bounding Box Height")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(frames, gt_area, label="GT area", linewidth=2)
    axes[2].plot(frames, pred_area, label="Pred area", linewidth=2)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Pixels²")
    axes[2].set_title("Bounding Box Area")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_dataset_figure(train_count: int, val_count: int, output_path: Path):
    labels = ["Train", "Val"]
    counts = [train_count, val_count]
    total = train_count + val_count
    ratios = [count / total for count in counts]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    bars = axes[0].bar(labels, counts, color=["tab:blue", "tab:orange"])
    axes[0].set_ylabel("Frames")
    axes[0].set_title("Dataset Split by Frame Count")
    axes[0].grid(True, axis="y", alpha=0.3)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, count + 0.5, str(count), ha="center", va="bottom")

    axes[1].pie(ratios, labels=[f"{label} ({ratio:.1%})" for label, ratio in zip(labels, ratios)], autopct="%.1f%%")
    axes[1].set_title("Dataset Split Ratio")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(records, gt_boxes, pred_boxes, train_count: int, val_count: int, output_path: Path):
    per_frame_ious = [bbox_iou(gt_box, pred_box) for gt_box, pred_box in zip(gt_boxes, pred_boxes)]
    per_frame_errors = [center_error(gt_box, pred_box) for gt_box, pred_box in zip(gt_boxes, pred_boxes)]

    last_record = records[-1]
    best_iou_record = max(records, key=lambda record: record["iou"])
    rows = [
        ["training_epochs_seen", max(record["epoch"] for record in records)],
        ["training_log_points", len(records)],
        ["final_train_loss_total", f"{last_record['loss_total']:.6f}"],
        ["final_train_iou", f"{last_record['iou']:.6f}"],
        ["best_logged_train_iou", f"{best_iou_record['iou']:.6f}"],
        ["best_logged_train_iou_epoch", best_iou_record["epoch"]],
        ["mean_video_iou", f"{sum(per_frame_ious) / len(per_frame_ious):.6f}"],
        ["min_video_iou", f"{min(per_frame_ious):.6f}"],
        ["max_video_iou", f"{max(per_frame_ious):.6f}"],
        ["mean_center_error", f"{sum(per_frame_errors) / len(per_frame_errors):.6f}"],
        ["max_center_error", f"{max(per_frame_errors):.6f}"],
        ["frame_count", len(gt_boxes)],
        ["train_split_frames", train_count],
        ["val_split_frames", val_count],
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def main():
    args = parse_args()
    ensure_file(args.log_file, "Training log file")
    ensure_file(args.prediction_file, "Prediction bbox file")
    ensure_file(args.groundtruth_file, "Ground-truth bbox file")
    ensure_file(args.train_meta_file, "Train meta file")
    ensure_file(args.val_meta_file, "Val meta file")

    records = parse_training_log(args.log_file)
    gt_boxes = parse_bbox_file(args.groundtruth_file, ",")
    pred_boxes = parse_bbox_file(args.prediction_file, " ")
    if len(gt_boxes) != len(pred_boxes):
        raise ValueError(
            f"Ground-truth frame count ({len(gt_boxes)}) does not match prediction count ({len(pred_boxes)})"
        )

    train_count = parse_meta_frame_count(args.train_meta_file)
    val_count = parse_meta_frame_count(args.val_meta_file)
    frames = list(range(1, len(gt_boxes) + 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    make_training_figure(records, args.output_dir / "training_curves.png")
    make_tracking_figure(frames, gt_boxes, pred_boxes, args.output_dir / "tracking_quality.png")
    make_bbox_size_figure(frames, gt_boxes, pred_boxes, args.output_dir / "bbox_dynamics.png")
    make_dataset_figure(train_count, val_count, args.output_dir / "dataset_split.png")
    write_summary_csv(records, gt_boxes, pred_boxes, train_count, val_count, args.output_dir / "figure_summary.csv")

    print(f"Generated paper figures in: {args.output_dir}")


if __name__ == "__main__":
    main()

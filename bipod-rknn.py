#!/usr/bin/env python3
"""
Batch-sort images into folders based on YOLO/OpenVINO object detections.
Requires: pip install opencv-python numpy tqdm rknn-toolkit-lite2
"""

import os
import sys
import shutil
import argparse
import csv
import time
import cv2
import numpy as np
from datetime import datetime
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from rknnlite.api import RKNNLite

# --- CONFIGURATION ---
MODEL_PATH = "yolo11s.rknn"
DEFAULT_SOURCE = "~/Pictures/"
DEFAULT_OUTPUT = "~/Pictures_Sorted"
DEFAULT_CONF = 0.40
DEFAULT_MODE = "auto"
DEFAULT_TILE_THRESHOLD = 1000
DEFAULT_TILE_SIZE = 640
DEFAULT_TILE_OVERLAP = 100
DEFAULT_MAX_EDGE = 1920

PRIORITY = [
    "person", "car", "bus", "truck",
    "motorcycle", "bicycle", "train", "boat",
    "cow", "sheep", "horse", "bird", "dog", "cat"
]

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(prog="dsbatch-rknn")
    p.add_argument("--input", "-i", default=DEFAULT_SOURCE)
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    p.add_argument("--conf", "-c", type=float, default=DEFAULT_CONF)
    p.add_argument("--model", "-m", default=MODEL_PATH)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mode", choices=["resize", "tile", "auto"], default=DEFAULT_MODE)
    p.add_argument("--tile-threshold", type=int, default=DEFAULT_TILE_THRESHOLD)
    p.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p.add_argument("--tile-overlap", type=int, default=DEFAULT_TILE_OVERLAP)
    p.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE)
    return p.parse_args()


def pick_primary(detected):
    for cls in PRIORITY:
        if cls in detected:
            return cls
    return sorted(detected)[0] if detected else "empty_or_unknown"


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0] - nh) // 2
    bottom = new_shape[0] - nh - top
    left = (new_shape[1] - nw) // 2
    right = new_shape[1] - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def log_failure(log_path, stage, img_path, reason):
    p = Path(log_path)
    new = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if new:
            w.writerow(["timestamp", "stage", "path", "reason"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), stage, str(img_path), reason])


def is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def make_destination(img_path, source_dir, target_folder):
    rel_path = os.path.relpath(img_path, source_dir)
    base_name = os.path.basename(img_path)
    relative_parent = os.path.dirname(rel_path)
    if relative_parent:
        dir_prefix = relative_parent.replace(os.sep, "__")
        stem, extension = os.path.splitext(base_name)
        new_filename = f"{stem}__{dir_prefix}{extension}"
    else:
        new_filename = base_name
    return os.path.join(target_folder, relative_parent, new_filename)


def process_outputs_v5(outputs, conf_thresh=0.40):
    detected_classes = set()
    for output in outputs:
        if output.ndim != 4 or output.shape[1] != 255:
            continue
        _, _, H, W = output.shape
        o = output.reshape(1, 3, 85, H, W)
        o = o.transpose(0, 1, 3, 4, 2).reshape(-1, 85)
        obj = o[:, 4]
        mask = obj > conf_thresh
        if not np.any(mask):
            continue
        rows = o[mask]
        class_scores = rows[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        final_scores = rows[:, 4] * class_scores[np.arange(len(rows)), class_ids]
        keep = final_scores > conf_thresh
        for cid in class_ids[keep]:
            if 0 <= cid < len(CLASSES):
                detected_classes.add(CLASSES[cid])
    return list(detected_classes)


def process_outputs_v11_e2e(outputs, conf_thresh=0.40):
    detected_classes = set()
    o = np.squeeze(outputs[0]) 
    if o.ndim != 2 or o.shape[1] != 6:
        return []
    confidences = o[:, 4]
    class_ids = o[:, 5]
    keep = confidences > conf_thresh
    for cid in class_ids[keep]:
        cid = int(cid)
        if 0 <= cid < len(CLASSES):
            detected_classes.add(CLASSES[cid])
    return list(detected_classes)


def process_outputs_v11_raw(outputs, conf_thresh=0.40):
    detected_classes = set()
    for output in outputs:
        o = np.squeeze(output)
        if o.ndim != 2:
            continue
        if o.shape[0] == 84:
            o = o.T 
        elif o.shape[1] != 84:
            continue
        class_scores = o[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(o)), class_ids]
        keep = confidences > conf_thresh
        for cid in class_ids[keep]:
            if 0 <= cid < len(CLASSES):
                detected_classes.add(CLASSES[cid])
    return list(detected_classes)


def process_outputs_yolo11_zoo(outputs, conf_thresh=0.40):
    detected_classes = set()
    for i in range(0, len(outputs), 3):
        box = np.squeeze(outputs[i])
        cls = np.squeeze(outputs[i+1])
        obj = np.squeeze(outputs[i+2])
        conf = obj * cls
        class_ids, y_coords, x_coords = np.where(conf > conf_thresh)
        for cid in class_ids:
            if 0 <= cid < len(CLASSES):
                detected_classes.add(CLASSES[cid])
    return list(detected_classes)


def process_outputs(outputs, conf_thresh=0.40):
    if not outputs:
        return []
    first = outputs[0]
    if first.ndim == 4 and first.shape[1] == 255:
        return process_outputs_v5(outputs, conf_thresh)
    elif first.ndim == 3 and first.shape[2] == 6:
        return process_outputs_v11_e2e(outputs, conf_thresh)
    elif len(outputs) == 9 and first.shape[1] == 64:
        return process_outputs_yolo11_zoo(outputs, conf_thresh)
    elif 84 in first.shape:
        return process_outputs_v11_raw(outputs, conf_thresh)
    raise ValueError(f"Unrecognized model output shape: {first.shape}")


def infer_rknn(rknn_lite, img_bgr, conf_thresh, args):
    """Processes a single image array using RKNN, handling max-edge, resizing, and tiling."""
    long_edge = max(img_bgr.shape[:2])
    
    # 1. Max Edge Downscale (Safety net for massive images)
    if long_edge > args.max_edge:
        scale = args.max_edge / long_edge
        new_size = (max(1, int(img_bgr.shape[1] * scale)), max(1, int(img_bgr.shape[0] * scale)))
        img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)
        long_edge = args.max_edge

    # 2. Handle Resize Mode (Force downscale to tile-size max)
    if args.mode == "resize":
        if long_edge > args.tile_size:
            scale = args.tile_size / long_edge
            new_size = (max(1, int(img_bgr.shape[1] * scale)), max(1, int(img_bgr.shape[0] * scale)))
            img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)

    def run_single_infer(crop_bgr):
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img_lb = letterbox(img_rgb, new_shape=(args.tile_size, args.tile_size))
        img_input = np.expand_dims(img_lb, axis=0).astype(np.uint8)
        outputs = rknn_lite.inference(inputs=[img_input], data_format="nhwc")
        return process_outputs(outputs, conf_thresh)

    detected_classes = set()

    # 3. Decide on Tiling
    use_tiles = False
    if args.mode == "tile":
        use_tiles = True
    elif args.mode == "auto" and max(img_bgr.shape[:2]) >= args.tile_threshold:
        use_tiles = True

    if use_tiles:
        h, w = img_bgr.shape[:2]
        stride = args.tile_size - args.tile_overlap
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + args.tile_size, h)
                x2 = min(x + args.tile_size, w)
                
                crop = img_bgr[y:y2, x:x2]
                # Pad edges so the NPU always gets a perfect 640x640 square
                if crop.shape[0] < args.tile_size or crop.shape[1] < args.tile_size:
                    crop = cv2.copyMakeBorder(crop, 0, args.tile_size - crop.shape[0], 
                                              0, args.tile_size - crop.shape[1], 
                                              cv2.BORDER_CONSTANT, value=(114,114,114))
                
                classes_in_tile = run_single_infer(crop)
                detected_classes.update(classes_in_tile)
    else:
        detected_classes.update(run_single_infer(img_bgr))

    return detected_classes


def main():
    args = parse_args()

    source_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(args.input)))
    output_root = os.path.abspath(os.path.expandvars(os.path.expanduser(args.output)))

    if not os.path.isdir(source_dir):
        print(f"Error: source directory does not exist: {source_dir}")
        return 1
    if not os.path.isfile(args.model):
        print(f"Error: model not found: {args.model}")
        return 1
    if is_within(output_root, source_dir):
        print("Error: output directory cannot be inside input directory.", file=sys.stderr)
        return 2

    print(f"--> Loading RKNN Model: {os.path.basename(args.model)}")
    rknn_lite = RKNNLite()
    if rknn_lite.load_rknn(args.model) != 0:
        print("Error: Could not load the model.")
        return 1

    print("--> Initialising NPU Hardware Context...")
    if rknn_lite.init_runtime() != 0:
        print("Error: NPU Hardware initialization failed!")
        return 1

    if not args.dry_run:
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(os.path.join(output_root, "empty_or_unknown"), exist_ok=True)

    print(f"--> Indexing files in {source_dir}...")
    all_image_paths = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if not is_within(os.path.join(root, d), output_root)]
        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                all_image_paths.append(os.path.join(root, file))

    total_files = len(all_image_paths)
    print(f"--> Found {total_files} images to analyze and sort.")
    if total_files == 0:
        rknn_lite.release()
        return 0

    LOG_PATH = os.path.join(output_root, "failures.csv")
    SUMMARY_PATH = os.path.join(output_root, "run_summary.txt")
    class_counts = Counter()
    total_detections = 0
    last_update = time.time()
    pbar = tqdm(all_image_paths, desc="Processing Images", unit="img")

    for img_path in pbar:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            log_failure(LOG_PATH, "read", img_path, "cv2.imread returned None")
            continue

        try:
            detected_objects = infer_rknn(rknn_lite, img, args.conf, args)
        except Exception as e:
            print(f"\n[skip] {filename}: inference failed: {e}", file=sys.stderr)
            log_failure(LOG_PATH, "inference", img_path, str(e))
            continue

        if detected_objects:
            total_detections += 1
            primary_class = pick_primary(detected_objects)
            target_folder = os.path.join(output_root, primary_class)
            class_counts[primary_class] += 1
        else:
            target_folder = os.path.join(output_root, "empty_or_unknown")
            class_counts["empty_or_unknown"] += 1

        dest_path = make_destination(img_path, source_dir, target_folder)

        if not args.dry_run:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Handle duplicate filenames
            if os.path.exists(dest_path):
                stem, ext = os.path.splitext(dest_path)
                counter = 1
                while os.path.exists(f"{stem}_{counter}{ext}"):
                    counter += 1
                dest_path = f"{stem}_{counter}{ext}"

            try:
                shutil.copy2(img_path, dest_path)
            except Exception as e:
                print(f"[copy-fail] {filename}: {e}", file=sys.stderr)
                log_failure(LOG_PATH, "copy", img_path, str(e))

        # Update rolling progress bar
        if time.time() - last_update > 0.5:
            top_classes = ", ".join(f"{name}:{count}" for name, count in class_counts.most_common(3))
            pbar.set_postfix_str(top_classes)
            last_update = time.time()

    rknn_lite.release()

    print("\n-- Class summary --")
    for name, count in class_counts.most_common():
        print(f"  {name:20s} {count}")

    if not args.dry_run:
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write(f"Batch Run Summary: {datetime.now().isoformat()}\n")
            f.write(f"Input Directory: {source_dir}\n")
            f.write(f"Output Directory: {output_root}\n")
            f.write(f"Total Images Scanned: {total_files}\n")
            f.write(f"Images with Detections: {total_detections}\n")
            f.write(f"Images without Detections: {class_counts['empty_or_unknown']}\n\n")
            f.write("Detected Classes:\n")
            for name, count in class_counts.most_common():
                if name != "empty_or_unknown":
                    f.write(f"  - {name}: {count}\n")
        print(f"\nRun summary saved to: {SUMMARY_PATH}")

    print(f"\nAll done! Your files are waiting in: {output_root}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

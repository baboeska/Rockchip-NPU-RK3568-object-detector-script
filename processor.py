import os
import sys
import shutil
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from rknnlite.api import RKNNLite

# --- CONFIGURATION ---
# Defaults (you can override these using command-line flags)
MODEL_PATH = "yolo11s.rknn"

DEFAULT_SOURCE = "/home/mingus/Pictures/"
DEFAULT_OUTPUT = "/home/mingus/Camera1_Sorted"
DEFAULT_CONF   = 0.40

PRIORITY = ["person", "car", "bus", "truck", "motorcycle", "bicycle", "train", "boat"]

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


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0] - nh) // 2
    bottom = new_shape[0] - nh - top
    left = (new_shape[1] - nw) // 2
    right = new_shape[1] - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)


def parse_args():
    p = argparse.ArgumentParser(
        prog="dsbatch",
        description="Batch-sort images with YOLO on RKNPU.",
    )
    p.add_argument("--input", "-i", default=DEFAULT_SOURCE,
                   help=f"source directory to scan (default: {DEFAULT_SOURCE})")
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                   help=f"destination root for sorted folders (default: {DEFAULT_OUTPUT})")
    p.add_argument("--conf", "-c", type=float, default=DEFAULT_CONF,
                   help=f"confidence threshold (default: {DEFAULT_CONF})")
    p.add_argument("--model", "-m", default=MODEL_PATH,
                   help=f"path to the .rknn model file (default: {MODEL_PATH})")
    p.add_argument("--dry-run", action="store_true",
                   help="print routing decisions without copying files")
    return p.parse_args()


def pick_primary(detected):
    for cls in PRIORITY:
        if cls in detected:
            return cls
    return sorted(detected)[0]


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
    """Decode YOLO11 End-to-End outputs, shape (1, 300, 6)."""
    detected_classes = set()
    o = np.squeeze(outputs[0]) 
    if o.ndim != 2 or o.shape[1] != 6:
        return []
    
    # Columns: 0=x1, 1=y1, 2=x2, 3=y2, 4=confidence, 5=class_id
    confidences = o[:, 4]
    class_ids = o[:, 5]
    keep = confidences > conf_thresh
    for cid in class_ids[keep]:
        cid = int(cid)
        if 0 <= cid < len(CLASSES):
            detected_classes.add(CLASSES[cid])
    return list(detected_classes)


def process_outputs_v11_raw(outputs, conf_thresh=0.40):
    """Decode YOLO11 Raw outputs, shape (1, 84, 8400)."""
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
    """Decode RKNN Model Zoo YOLO11 outputs (9 tensors)."""
    detected_classes = set()

    # Outputs come in groups of 3 for each stride (80x80, 40x40, 20x20)
    for i in range(0, len(outputs), 3):
        box = np.squeeze(outputs[i])     # Shape: (64, H, W)
        cls = np.squeeze(outputs[i+1])   # Shape: (80, H, W)
        obj = np.squeeze(outputs[i+2])   # Shape: (1, H, W)

        # Multiply objectness by class scores to get final confidence
        # obj shape is (1, H, W) so we broadcast it against cls (80, H, W)
        conf = obj * cls
        
        # Find where confidence is above threshold
        # np.where returns (class_ids, y_coords, x_coords)
        class_ids, y_coords, x_coords = np.where(conf > conf_thresh)
        
        for cid in class_ids:
            if 0 <= cid < len(CLASSES):
                detected_classes.add(CLASSES[cid])

    return list(detected_classes)

def process_outputs(outputs, conf_thresh=0.40):
    """Auto-detects the model output format."""
    if not outputs:
        return []
    
    first = outputs[0]
    
    # YOLOv5 outputs (1, 255, H, W)
    if first.ndim == 4 and first.shape[1] == 255:
        return process_outputs_v5(outputs, conf_thresh)
        
    # YOLO26 E2E outputs (1, 300, 6)
    elif first.ndim == 3 and first.shape[2] == 6:
        return process_outputs_v11_e2e(outputs, conf_thresh)
        
    # YOLO11 Model Zoo outputs (9 tensors, first is 64 channels)
    elif len(outputs) == 9 and first.shape[1] == 64:
        return process_outputs_yolo11_zoo(outputs, conf_thresh)
        
    # YOLO11 Standard outputs (1, 84, 8400)
    elif 84 in first.shape:
        return process_outputs_v11_raw(outputs, conf_thresh)
        
    raise ValueError(f"Unrecognized model output shape: {first.shape}")


def main():
    args = parse_args()

    source_dir  = os.path.abspath(os.path.expanduser(args.input))
    output_root = os.path.abspath(os.path.expanduser(args.output))
    conf_thresh = args.conf

    if not os.path.isdir(source_dir):
        print(f"Error: source directory does not exist: {source_dir}")
        return 1
    if not os.path.isfile(args.model):
        print(f"Error: model not found: {args.model}")
        return 1

    print(f"--> Loading RKNN Model: {os.path.basename(args.model)}")
    rknn_lite = RKNNLite()
    if rknn_lite.load_rknn(args.model) != 0:
        print(f"Error: Could not load the model.")
        return 1

    print("--> Initialising NPU Hardware Context...")
    if rknn_lite.init_runtime() != 0:
        print("Error: NPU Hardware initialization failed!")
        return 1

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, "empty_or_unknown"), exist_ok=True)

    print(f"--> Indexing files in {source_dir}... (This might take a moment)")
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_image_paths = []

    for root, dirs, files in os.walk(source_dir):
        if os.path.abspath(root).startswith(output_root):
            continue
        for file in files:
            if file.lower().endswith(image_extensions):
                all_image_paths.append(os.path.join(root, file))

    total_files = len(all_image_paths)
    print(f"--> Found {total_files} images to analyze and sort.")

    if total_files == 0:
        rknn_lite.release()
        return 0

    for img_path in tqdm(all_image_paths, desc="Processing Images", unit="img"):
        filename = os.path.basename(img_path)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_letterboxed = letterbox(img_rgb, new_shape=(640, 640))
        img_input = np.expand_dims(img_letterboxed, axis=0).astype(np.uint8)

        try:
            outputs = rknn_lite.inference(inputs=[img_input], data_format="nhwc")
            detected_objects = process_outputs(outputs, conf_thresh=conf_thresh)
        except Exception as e:
            print(f"\n[skip] {filename}: {e}", file=sys.stderr)
            continue

        if detected_objects:
            primary_class = pick_primary(detected_objects)
            target_folder = os.path.join(output_root, primary_class)
        else:
            target_folder = os.path.join(output_root, "empty_or_unknown")

        # --- NEW PATH LOGIC ---
        rel_path = os.path.relpath(img_path, source_dir)
        base_name = os.path.basename(img_path)
        dir_prefix = os.path.dirname(rel_path).replace(os.sep, "__")
        
        # If the image is in a subfolder, append that folder structure to the filename
        if dir_prefix:
            new_filename = f"{os.path.splitext(base_name)[0]}__{dir_prefix}{os.path.splitext(base_name)[1]}"
        else:
            new_filename = base_name
            
        # Keep the directory structure in the output folder
        dest_path = os.path.join(target_folder, os.path.dirname(rel_path), new_filename)
        out_dir = os.path.dirname(dest_path)
        # ----------------------

        if args.dry_run:
            # Show exactly what the output path will look like
            print(f"{rel_path} -> {os.path.relpath(dest_path, output_root)}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        try:
            shutil.copy(img_path, dest_path)
        except Exception as e:
            print(f"[copy-fail] {filename}: {e}", file=sys.stderr)

    rknn_lite.release()
    print(f"\nAll done! Your files are waiting in: {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

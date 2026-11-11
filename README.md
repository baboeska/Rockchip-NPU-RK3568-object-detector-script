# BIPOD-RKNN

Batch image processor for Rockchip RK3568 (0.8 TOPS NPU)

Runs YOLO11 and YOLOv5 on the RK3568 NPU. Requires `rknn-toolkit-lite2` 2.5.0
and `librknnrt.so` 2.5.0 on the board.

Part of [AI BIPOD](https://github.com/baboeska/bipod)

## Performance
| Platform | Model | Throughput |
|---|---|---|
| RK3568 NPU (0.8 TOPS) | YOLO11s INT8 | ~6.5 img/s |

## Honest assessment

Best as a project rather than a daily driver — the Debian 13 image ships without
GL/GLES/V4L2 acceleration, so the desktop is sluggish. If you own an RK3568,
Armbian with an open kernel is more usable overall. Run the sorter headless
over SSH, or, try a Debian 11/12 image for limited GPU support.

Intended use: a second layer on MotionEye or other security camera/PVR
software, auto-sorting detections into folders.

## IMPORTANT: separate venv


## Requirements

- Python 3.12
- `rknn-toolkit-lite2` 2.5.0 (Rockchip vendor tool — not bundled, install from Rockchip)
- `librknnrt.so` 2.5.0 (runtime, must be on the board, or, 2.3.0, if using yolo_v5)
- opencv-python, numpy, tqdm
- setuptools<81 (newer versions break the toolkit's build)
- RKNPU visible in Kernel, Rockchip proprietary kernel

## Setup

### 1. System packages

bash

    sudo apt update && sudo apt install -y build-essential cmake libopencv-dev
    sudo usermod -aG video,render,disk $USER   # then log out and back in

2. Python environment
bash

        sudo apt install -y python3.12 python3.12-venv
        python3.12 -m venv ~/npu_env
        source ~/npu_env/bin/activate
        pip install opencv-python numpy tqdm "setuptools<81"
  
3. Install rknn-toolkit-lite2

   git clone https://github.com/rockchip-linux/rknn-toolkit2

   Then look in /rknn-toolkit2/tree/master/rknn_toolkit_lite2/packages for Python wheels.
   

The wheel is architecture-specific (aarch64) and must be installed on the board,
not copied from another machine.
bash

    pip install /path/to/rknn_toolkit_lite2-2.5.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
        sudo usermod -aG video,render,disk $USER   # then log out and back in

4. Run program

        python3 bipod-rknn.py -i=/your/images -o=/sorted/images

## Licensing

This project uses a **split-license** model:
- **Code** (Python scripts, etc.) → **AGPL-3.0** (see `LICENSE`)
- **RKNN model files** → **RKNN SDK License** (see `LICENSE-RKNN`)

The RKNN SDK itself is **not** bundled here; users must obtain it from Rockchip separately.

It is not perfect and has no warranty, nor is this version intended to be bulletproof.

If you alter this code for other hardware, please share what you make on github or somewhere where people who want it are likely to find it.

Deepseek R1 free tier and Google Search AI helped me make this, very amazing work.

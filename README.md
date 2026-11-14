# BIPOD-RKNN

Batch image processor for Rockchip RK3568 (0.8 TOPS NPU)

Runs YOLO11 and YOLOv5 on the RK3568 NPU. Requires `rknn-toolkit-lite2` 2.5.0
and `librknnrt.so` 2.5.0 on the board for YOLOv11.

Part of [AI BIPOD](https://github.com/baboeska/bipod)

## Performance

| RK3568 NPU (0.8 TOPS) | YOLO11s INT8 | ~6.5 img/s resize | ~1.5 img/s tile |

## Honest assessment

Best as a project rather than a daily driver — the Debian 13 image ships without
GL/GLES/V4L2 acceleration, so the desktop is sluggish. If you own an RK3568 with HDMI,
Armbian with an open kernel and no NPU support is more usable overall. Run the sorter headless
over SSH, or, try a Debian 11/12 image for limited GPU support.

Intended use: a second layer on MotionEye or other security camera/PVR
software, auto-sorting detections into folders.

You don't necessarily need a second VENV, this works with Python 3.9, but, a VENV is a good idea.


## Requirements

- Python 3.9-3.12
- `rknn-toolkit-lite2` 2.5.0 (Rockchip vendor tool — not bundled, install from Rockchip)
- `librknnrt.so` 2.5.0 (runtime, must be on the board, or, 2.3.0, if using yolo_v5)
- opencv-python, numpy, tqdm
- setuptools<81 (newer versions break the toolkit's build)
- RKNPU visible in Kernel, Rockchip proprietary kernel

## Setup


### 0. Prerequisites


1. The librknnrt.so Runtime Library (Manual Install)
You must download the librknnrt.so file from the airockchip/rknn-toolkit2 GitHub repository (look inside rknpu2/runtime/RK356X/Linux/librknn_api/aarch64/) and copy it to your board:

bash
sudo cp librknnrt.so /usr/lib/

sudo ldconfig

(Many pre-built Rockchip Debian images include this by default, but if you are using a custom or minimal image, you must do this manually.) 


2. The NPU Kernel Driver

The librknnrt.so is just the userspace runtime. The board's Linux kernel must also have the Rockchip NPU driver enabled and loaded. You can usually check this with:

bash

dmesg | grep rknpu

If you see NPU initialization messages, the driver is working.



3. Version Matching (Crucial)


Your Python wheel (rknn_toolkit_lite2) and the C runtime (librknnrt.so) must be the same major version. Since you are using the 2.5.0 wheel, you need the 2.5.0 librknnrt.so . A mismatch here is a very common cause of mysterious initialization failures.


🛠️ How to Verify It's Working


Before running the full BIPOD script, run this quick test on your board to confirm the runtime is correctly installed and visible:


bash

# Check if the system can find the runtime library

ldconfig -p | grep rknnrt


# Check the version of the library you installed

strings /usr/lib/librknnrt.so | grep librknnrt


The second command should output something like librknnrt version: 2.5.0 (...).


### 1. System packages

bash

    sudo apt update && sudo apt install -y build-essential cmake libopencv-dev git
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
Re wheel, ensure the cp***-cp*** matches python3 --version output for the numbers.
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

Deepseek R1 free tier and Google Search AI and Gemma4 and ChatGPT helped me make this, very amazing work.

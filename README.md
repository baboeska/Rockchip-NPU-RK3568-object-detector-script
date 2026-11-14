#BIPOD-RKNN
#Batch image processor for Rockchip RK3568 (0.8 TOPS NPU)

Runs YOLO11 and YOLOv5 on the RK3568 NPU. Requires rknn-toolkit-lite2 2.5.0 and librknnrt.so 2.5.0 on the board for YOLOv11.

Part of AI BIPOD (https://github.com/baboeska/bipod)

#PERFORMANCE

RK3568 NPU (0.8 TOPS) with YOLO11s INT8:

Resize mode: ~6.5 img/s

Tile mode: ~1.5 img/s

#HONEST ASSESSMENT

Best as a project rather than a daily driver. The Debian 13 image ships without GL/GLES/V4L2 acceleration, so the desktop is sluggish. If you own an RK3568 with HDMI, Armbian with an open kernel with no NPU support is more usable overall. Run the sorter headless over SSH, or try a Debian 11/12 image for limited GPU support.

Intended use: a second layer on MotionEye or other security camera/PVR software, auto-sorting detections into folders.

You don't necessarily need a second VENV; this works with Python 3.9, but a VENV is a good idea.

#REQUIREMENTS

Python 3.9 to 3.12

rknn-toolkit-lite2 2.5.0 (Rockchip vendor tool, not bundled, install from Rockchip)

librknnrt.so 2.5.0 (runtime, must be on the board, or 2.3.0 if using YOLOv5)

opencv-python, numpy, tqdm

setuptools < 81 (newer versions break the toolkit's build)

RKNPU visible in kernel (Rockchip proprietary kernel)

##SETUP

#Prerequisites

The librknnrt.so Runtime Library (Manual Install)

You must download the librknnrt.so file from the airockchip/rknn-toolkit2 GitHub repository (look inside rknpu2/runtime/RK356X/Linux/librknn_api/aarch64/) and copy it to your board:

sudo cp librknnrt.so /usr/lib/
sudo ldconfig

Many pre-built Rockchip Debian images include this by default, but if you are using a custom or minimal image, you must do this manually.

The NPU Kernel Driver

The librknnrt.so is just the userspace runtime. The board's Linux kernel must also have the Rockchip NPU driver enabled and loaded. You can usually check this with:

dmesg | grep rknpu

If you see NPU initialization messages, the driver is working.

Version Matching (Crucial)

Your Python wheel (rknn_toolkit_lite2) and the C runtime (librknnrt.so) must be the same major version. Since you are using the 2.5.0 wheel, you need the 2.5.0 librknnrt.so. A mismatch here is a very common cause of mysterious initialization failures.

How to verify it's working:

Before running the full BIPOD script, run this quick test on your board:

ldconfig -p | grep rknnrt

strings /usr/lib/librknnrt.so | grep librknnrt

The second command should output something like "librknnrt version: 2.5.0 (...)".

System packages

sudo apt update && sudo apt install -y build-essential cmake libopencv-dev git
sudo usermod -aG video,render,disk $USER

(Then log out and back in for the group change to take effect.)

Python environment

sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv ~/npu_env
source ~/npu_env/bin/activate
pip install opencv-python numpy tqdm "setuptools<81"

Install rknn-toolkit-lite2

git clone https://github.com/rockchip-linux/rknn-toolkit2

Then look in rknn-toolkit2/rknn_toolkit_lite2/packages for Python wheels. The wheel is architecture-specific (aarch64) and must be installed on the board, not copied from another machine.

Ensure the cp*-cp* in the wheel name matches your python3 --version output.

pip install /path/to/rknn_toolkit_lite2-2.5.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

Run

python3 bipod-rknn.py -i ~/Pictures -o ~/Pictures_Sorted

Available modes:

--mode auto (default): downscale if >1920px, then tile if >1000px

--mode tile: unconditional tiling

--mode resize: downscale to 640px max, single pass

#LICENSING

This project uses a split-license model:

Code (Python scripts, etc.) is AGPL-3.0 (see LICENSE)

RKNN model files are RKNN SDK License (see LICENSE-RKNN)

The RKNN SDK itself is not bundled here; users must obtain it from Rockchip separately.

It is not perfect and has no warranty, nor is this version intended to be bulletproof. Do good things and share and share alike.

If you alter this code for other hardware, please share what you make on GitHub or somewhere where people who want it are likely to find it.

DeepSeek R1 free tier and Google Search AI helped me make this, very amazing work.

This script runs YOLO11 and YOLOv5 on the Rockchip RK3568 NPU. It requires rknn-toolkit-lite2 version 2.5.0 and librknnrt.so 2.5.0 on the board.

Requirements:
Python 3.12
rknn-toolkit-lite2
opencv-python
numpy
tqdm
setuptools<81

Usage:
Build a python 3.12 virtual environment. Install rockchip toolkit 2.5.0 and Ultralytics.
Then run via "python processor.py -i="input directory" -o="output directory"

For any issues please raise.
This is a project that may require some thought to process.
At some point in the near future it may be converted to run on OpenVino for Intel cpu support, but is likely to stay a linux program, so, wsl, if you need this.

it's intended purpose is to allow fast batch processing of images as a second layer on motioneye or other security camera/pvr software, automatically sorting into detected object folders.

It is not perfect and has no warranty.

If you alter this code for other hardware, please share what you make on github or somewhere where people who want it are likely to find it.

Deepseek R1 free tier and Google Search AI helped me make this, very amazing work.

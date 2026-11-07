import numpy as np
import cv2
from rknnlite.api import RKNNLite

MODEL = "yolo11s.rknn"
IMG   = "bus.jpg"   # <-- change this

rknn = RKNNLite()
rknn.load_rknn(MODEL)
rknn.init_runtime()
print("runtime OK")

img = cv2.imread(IMG)
print("img shape:", img.shape)

# match the C demo: NHWC uint8, letterbox-ish resize (no pad here, just check tensor path)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (640, 640))
inp = np.expand_dims(resized, axis=0).astype(np.uint8)  # NHWC uint8
print("inp:", inp.shape, inp.dtype)

outs = rknn.inference(inputs=[inp], data_format="nhwc")
print("num outputs:", len(outs))
for i, o in enumerate(outs):
    print(f"out[{i}]: shape={o.shape} dtype={o.dtype} min={float(o.min()):.4f} max={float(o.max()):.4f} mean={float(o.mean()):.4f}")

rknn.release()

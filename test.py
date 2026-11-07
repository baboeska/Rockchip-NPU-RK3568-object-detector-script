import cv2
import numpy as np
from rknnlite.api import RKNNLite

MODEL = "yolo11s.rknn"
IMG = "bus.jpg" # MAKE SURE THIS IS THE CORRECT PATH TO YOUR IMAGE

rknn = RKNNLite()
rknn.load_rknn(MODEL)
rknn.init_runtime()

img = cv2.imread(IMG)
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
inp = np.expand_dims(img, 0).astype(np.uint8)

outs = rknn.inference(inputs=[inp], data_format="nhwc")
o = outs[0][0] # Shape (300, 6)

print("First 5 rows of output:")
for row in o[:5]:
    print(f"x1:{row[0]:.1f}, y1:{row[1]:.1f}, x2:{row[2]:.1f}, y2:{row[3]:.1f}, col4:{row[4]:.4f}, col5:{row[5]:.4f}")

# Let's find the maximum values in columns 4 and 5
print(f"\nMax in col4: {np.max(o[:, 4])}")
print(f"Max in col5: {np.max(o[:, 5])}")
rknn.release()

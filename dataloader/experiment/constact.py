import numpy as np
import cv2
farina = cv2.imread("im0.png")

Imax = np.max(farina)
Imin = np.min(farina)
MAX = 255
MIN = 0
farina_cs = (farina - Imin) / (Imax - Imin) * (MAX - MIN) + MIN
cv2.imshow("farina_cs", farina_cs.astype("uint8"))
cv2.waitKey()

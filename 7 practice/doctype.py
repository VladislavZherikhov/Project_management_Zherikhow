import cv2
import numpy as np

HOG = cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)


def features(image):
    gray = np.asarray(image.convert("L").resize((96, 96)))
    # Pool across the four right-angle orientations.
    vectors = [HOG.compute(np.ascontiguousarray(np.rot90(gray, k))).ravel()
               for k in range(4)]
    return np.mean(vectors, axis=0)

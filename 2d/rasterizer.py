import numpy as np
import cv2

class Rasterizer:
    def __init__(self, height, width):
        self.MAX_PX_VAL = 255
        self.height = height
        self.width = width
        self.canvas = np.ones((height, width)) * MAX_PX_VAL
        self.num_primitives = 0

    def line(x0, y0, x1, y1):
        pass


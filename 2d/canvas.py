import numpy as np
import cv2

class Canvas
    def __init__(self, height, width):
        self.MAX_PX_VAL = 255
        self.height = height
        self.width = width
        self.primitives = []
        self.canvas = np.ones((height, width)) * MAX_PX_VAL

    def add_line(x0, y0, x1, y1):
        return self._add_primitive_struct('line', [x0, y0, x1, y1])

    def _add_primitive_struct(self, prim_name, args):
        struct = (prim_name, args)
        self.primitives.append(struct)
        return struct

    def render_canvas(self):
        # TODO: calls all the drawing functions to draw structs
        # in self.primitives to the canvas
        pass

    def render_line(x0, y0, x1, y1):
        # TODO: bresenham
        pass


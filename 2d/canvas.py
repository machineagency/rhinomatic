import numpy as np
import imageio

class Canvas
    def __init__(self, height, width):
        self.BLACK = 255
        self.height = height
        self.width = width
        self.primitives = []
        self.canvas = np.ones((height, width)) * BLACK

    def add_line(x0, y0, x1, y1):
        return self._add_primitive_struct('line', [x0, y0, x1, y1])

    def _add_primitive_struct(self, prim_name, args):
        struct = (prim_name, args)
        self.primitives.append(struct)
        return struct

    def set_pixel(self, x, y):
        self.canvas[x, y] = self.BLACK

    def render_canvas(self):
        # TODO: calls all the drawing functions to draw structs
        # in self.primitives to the canvas
        pass

    def render_line(x0, y0, x1, y1):
        # TODO: bresenham
        if abs(y1 - y0) < abs(x1 - x0):
            if x0 > x1:
                self._render_line_low(x1, y1, x0, y0)
            else:
                self._render_line_low(x0, y0, x1, y1)
        else:
            if y0 > y1:
                self._render_line_high(x1, y1, x0, y0)
            else:
                self._render_line_high(x0, y0, x1, y1)

    def _render_line_low(x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        D = 2 * dy - dx
        y = y0

        for x in range(x0, x1):
            self.set_pixel(x, y)
            if D > 0:
                y += yi
                D -= 2 * dx
            D += 2 * dy

    def _render_line_high(x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        xi = 1
        if dy < 0:
            xi = -1
            dx = -dx
        D = 2 * dx - dy
        x = x0

        for y in range(y0, y1):
            self.set_pixel(x, y)
            if D > 0:
                x += xi
                D -= 2 * dy
            D += 2 * dx

    def save_canvas(self, filename='img'):
        imageio.imwrite(filename, self.canvas, 'png')
        return filename


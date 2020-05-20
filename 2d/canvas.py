import numpy as np
import imageio

class Canvas:
    def __init__(self, height, width):
        self.WHITE = 255
        self.BLACK = 0
        self.height = height
        self.width = width
        self.primitives = []
        self.canvas = np.ones((height, width), dtype='uint8') * self.WHITE

    def add_line(self, x0, y0, x1, y1):
        return self._add_primitive_struct('line', [x0, y0, x1, y1])

    def _add_primitive_struct(self, prim_name, args):
        struct = (prim_name, args)
        self.primitives.append(struct)
        return struct

    def write_spec(self, filename='spec'):
        spec_file = open(f'{filename}.txt', 'w+')
        for struct in self.primitives:
            command_name, command_args = struct[0], tuple(struct[1])
            spec_file.write(f'{command_name}{command_args}\n')
        spec_file.write('\n')
        spec_file.close()

    def set_pixel(self, x, y):
        self.canvas[x, y] = self.BLACK

    def clear_canvas(self):
        self.canvas = np.ones((self.height, self.width), dtype='uint8')\
                        * self.WHITE

    def render_canvas(self):
        self.clear_canvas()
        for struct in self.primitives:
            command_name = struct[0]
            command_args = struct[1]
            if command_name == 'line':
                self.render_line(*command_args)
        return self.canvas

    def render_line(self, x0, y0, x1, y1):
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

    def _render_line_low(self, x0, y0, x1, y1):
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

    def _render_line_high(self, x0, y0, x1, y1):
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
        FILETYPE = 'png'
        imageio.imwrite(f'{filename}.{FILETYPE}', self.canvas, FILETYPE)
        return filename

if __name__ == '__main__':
    c = Canvas(265, 256)
    c.add_line(0, 0, 80, 20)
    c.add_line(50, 20, 50, 150)
    c.add_line(20, 50, 150, 50)
    c.render_canvas()
    c.save_canvas()
    c.write_spec()


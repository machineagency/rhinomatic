import numpy as np
import imageio
from random import randrange

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

    def add_rectangle(self, x, y, h, w):
        return self._add_primitive_struct('rectangle', [x, y, h, w])

    def add_circle(self, x, y, r):
        return self._add_primitive_struct('circle', [x, y, r])

    def _add_primitive_struct(self, prim_name, args):
        struct = (prim_name, args)
        self.primitives.append(struct)
        return struct

    def clear_primitives(self):
        self.primitives = []

    def write_spec(self, filename='spec'):
        spec_file = open(f'{filename}.txt', 'w+')
        for struct in self.primitives:
            command_name, command_args = struct[0], tuple(struct[1])
            spec_file.write(f'{command_name}{command_args}\n')
        spec_file.write('\n')
        spec_file.close()

    def set_pixel(self, x, y):
        if x >= 0 and x < self.width and y >= 0 and y < self.height:
            self.canvas[y, x] = self.BLACK

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
            if command_name == 'rectangle':
                self.render_rectangle(*command_args)
            if command_name == 'circle':
                self.render_circle(*command_args)
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

    def render_rectangle(self, x, y, h, w):
        for y_idx in range(y, y + h):
            for x_idx in range(x, x + w):
                self.set_pixel(x_idx, y_idx)

    def render_circle(self, x, y, r):
        for y_idx in range(y - r, y + r + 1):
            for x_idx in range(x - r, x + r + 1):
                dist_vect = (x_idx - x, y_idx - y)
                if np.linalg.norm(dist_vect) <= r:
                    self.set_pixel(x_idx, y_idx)

    def save_canvas(self, filename='img'):
        FILETYPE = 'png'
        imageio.imwrite(f'{filename}.{FILETYPE}', self.canvas, FILETYPE)
        return filename

    def make_random_drawings(self, n):
        MAX_PRIMITIVES = 2
        for drawing_number in range(n):
            num_primitives = randrange(1, MAX_PRIMITIVES)
            self.clear_primitives()
            self.clear_canvas()
            for primitive_number in range(num_primitives):
                x = randrange(100)
                y = randrange(100)
                r = randrange(50)
                c.add_circle(x, y, r)
            self.render_canvas()
            self.save_canvas(f'data/drawings/{drawing_number}')
            self.write_spec(f'data/specs/{drawing_number}')

    def make_easy_drawings(self, n):
        # anchor_points = [(20, 10), (10, 20)]
        for drawing_number in range(n):
            if drawing_number % 1000 == 0:
                print(drawing_number)
            self.clear_primitives()
            self.clear_canvas()
            # for pt in anchor_points:
            for _ in range(3):
                pt = (np.random.randint(0, 32), np.random.randint(0, 32))
                h = np.random.randint(8, 16)
                w = np.random.randint(8, 16)
                r = np.random.randint(4, 8)

                # h = 8
                # w = 8
                # r = 4
                dice_roll = np.random.rand()
                if dice_roll < 0.33:
                    c.add_rectangle(*pt, h, w)
                elif dice_roll < 0.67:
                    c.add_circle(*pt, r)
                else:
                    # Don't do draw anything here
                    pass
            self.render_canvas()
            self.save_canvas(f'data/drawings/{drawing_number}')
            self.write_spec(f'data/specs/{drawing_number}')

if __name__ == '__main__':
    batch_size = 32
    c = Canvas(32, 32)
    c.make_easy_drawings(1000 * batch_size)


import numpy as np
import imageio
from random import randrange

class Environment:
    def __init__(self, height, width):
        self.MAX_PRIMITIVES = 3
        self.MAX_ACTIONS = 100
        self.WHITE = 255
        self.BLACK = 0
        self.height = height
        self.width = width
        self.primitives = []
        self.canvas = np.ones((height, width), dtype='uint8') * self.WHITE
        self.actions_done = 0

    def reset(self):
        self.clear_canvas()
        self.clear_primitives()
        self.actions_done = 0

    def do_action(self, action):
        if action[0] == 'STOP':
            return False
        if action[0] == 'MODIFY':
            self._modify_last_primitive_struct(action[1])
        else:
            self._add_primitive_struct(action[0], *action[1:])
        self.render_canvas()
        self.actions_done += 1
        return True

    def peek_action(self, action):
        """
        Action is a primitive struct of the form:
        (prim_name, [args_or_mods]),
        where PRIM_NAME can also be 'STOP' or 'MODIFY'.
        """
        orig_primitives = self.primitives[::]
        orig_canvas = self.canvas.copy()
        success = self.do_action(action)
        new_canvas = self.canvas.copy()
        self.primitives = orig_primitives
        self.canvas = orig_canvas
        return new_canvas

    def get_actions(self):
        """
        An action a 2-tuple is of the form:
        (name, params)
        where:
            - NAME is 'MOD' if we are modifying the current primitive,
              'STOP' to terminate the episode, and the name of a new primitve
              otherwise
            - PARAMS is a list of modifications to the currrent primitive's
              arguments e.g. [0, 0, -1] to decrease the radius of a circle,
              otherwise it is just the list of arguments for the new primitive.
        The actions available depend on the current set of primitives in
        self.primitives, as well as self.MAX_PRIMITIVES.
        """
        actions = [('STOP', [])]
        if self.actions_done >= self.MAX_ACTIONS - 1:
            return actions
        last_primitive = self._get_last_primitive_struct()
        if last_primitive:
            last_primitive_num_args = len(last_primitive[1])
            for i in range(last_primitive_num_args):
                mods_pos = [0] * last_primitive_num_args
                mods_neg = [0] * last_primitive_num_args
                mods_pos[i] = 1
                mods_neg[i] = -1
                actions.append(('MODIFY', mods_pos))
                actions.append(('MODIFY', mods_neg))
        if len(self.primitives) < self.MAX_PRIMITIVES:
            rect_args = [15, 15, 10, 10]
            circle_args = [15, 15, 6]
            actions.append(('rectangle', rect_args))
            actions.append(('circle', circle_args))
        return actions

    def intersection_over_union(self, spec):
        self.render_canvas()
        canvas_binarized = np.where(self.canvas == self.BLACK, 1, 0)
        spec_binarized = np.where(spec == self.BLACK, 1, 0)
        added = canvas_binarized + spec_binarized
        union = np.where(added > 0, 1, 0)
        intersection = np.where(added > 1, 1, 0)
        return np.count_nonzero(intersection) / np.count_nonzero(union)

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

    def _get_last_primitive_struct(self):
        if len(self.primitives) > 0:
            return self.primitives[len(self.primitives) - 1]

    def _modify_last_primitive_struct(self, arg_mods):
        last_primitive = self._get_last_primitive_struct()
        if last_primitive:
            last_args = last_primitive[1]
            if len(arg_mods) != len(last_args):
                raise ValueError()
            for i, mod in enumerate(arg_mods):
                last_args[i] += mod
            return last_primitive

    def _remove_last_primitive_struct(self):
        if len(self.primitives) > 0:
            self.primitives = self.primitives[:len(self.primitives) - 1]

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
        for drawing_number in range(n):
            num_primitives = randrange(1, self.MAX_PRIMITIVES)
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

    def make_easy_stacked_drawings(self, n):
        # TODO: maybe, have the first be spec, second x, third x'
        num_canvases = 3
        for drawing_number in range(n):
            canvases = []
            if drawing_number % 1000 == 0:
                print(drawing_number)
            for c_idx in range(num_canvases):
                self.clear_primitives()
                self.clear_canvas()
                for _ in range(self.MAX_PRIMITIVES):
                    pt = (np.random.randint(0, 32), np.random.randint(0, 32))
                    h = np.random.randint(8, 16)
                    w = np.random.randint(8, 16)
                    r = np.random.randint(4, 8)

                    dice_roll = np.random.rand()
                    if dice_roll < 0.33:
                        c.add_rectangle(*pt, h, w)
                    elif dice_roll < 0.67:
                        c.add_circle(*pt, r)
                    else:
                        # Don't do draw anything here
                        pass
                self.render_canvas()
                canvases.append(self.canvas.copy())
            final_canvas = np.stack(canvases, 2)
            self.canvas = final_canvas
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
    c.make_easy_stacked_drawings(1000 * batch_size)


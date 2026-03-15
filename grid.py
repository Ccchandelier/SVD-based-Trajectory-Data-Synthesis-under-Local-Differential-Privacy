import random

class Grid:
    def __init__(self, x0, y0, dx, dy, idx):
        self.min_x = x0
        self.min_y = y0
        self.max_x = x0 + dx
        self.max_y = y0 + dy
        self.index = idx

    def in_cell(self, p):
        return self.min_x <= p[0] <= self.max_x and self.min_y <= p[1] <= self.max_y

    def sample_point(self):
        x = self.min_x + random.random() * (self.max_x - self.min_x)
        y = self.min_y + random.random() * (self.max_y - self.min_y)
        return x, y

    def equal(self, other):
        return self.index == other.index

    def __eq__(self, other):
        return type(other) == Grid and self.index == other.index

    def __hash__(self):
        return hash(self.index)


class Transition:
    def __init__(self, g1, g2, flag=0):
        self.g1 = g1
        self.g2 = g2
        self.flag = flag

    def __eq__(self, other):
        return type(other) == Transition and self.g1 == other.g1 and self.g2 == other.g2 and self.flag == other.flag

    def __hash__(self):
        return hash(self.g1.index + self.g2.index + (self.flag,))


class GridMap:
    def __init__(self, n, min_x, min_y, max_x, max_y):
        min_x -= 1e-6
        min_y -= 1e-6
        max_x += 1e-6
        max_y += 1e-6
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.step_x = (max_x - min_x) / n
        self.step_y = (max_y - min_y) / n
        self.map = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(Grid(min_x + self.step_x * i, min_y + self.step_y * j, self.step_x, self.step_y, (i, j)))
            self.map.append(row)

    def get_adjacent(self, g):
        i, j = g.index
        xs = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1),
              (i, j - 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1)]
        ys = []
        for a, b in xs:
            if 0 <= a < len(self.map) and 0 <= b < len(self.map[0]):
                ys.append((a, b))
        return ys

    def is_adjacent_grids(self, g1, g2):
        return g2.index in self.get_adjacent(g1)

    def get_list_map(self):
        out = []
        for row in self.map:
            out.extend(row)
        return out

    def get_all_transition(self):
        out = []
        for g in self.get_list_map():
            out.append(Transition(g, g, 1))
            out.append(Transition(g, g, 0))
            for i, j in self.get_adjacent(g):
                out.append(Transition(g, self.map[i][j]))
            out.append(Transition(g, g, 2))
        return out

    def get_normal_transition(self):
        out = []
        for g in self.get_list_map():
            out.append(Transition(g, g, 0))
            for i, j in self.get_adjacent(g):
                out.append(Transition(g, self.map[i][j]))
        return out

    @property
    def size(self):
        return len(self.map) * len(self.map[0])

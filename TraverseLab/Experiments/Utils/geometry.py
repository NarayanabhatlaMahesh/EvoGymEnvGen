# utils/geometry.py

from collections import deque


def rects_overlap(a, b):
    return not (
        a["x"] + a["w"] <= b["x"] or
        b["x"] + b["w"] <= a["x"] or
        a["y"] + a["h"] <= b["y"] or
        b["y"] + b["h"] <= a["y"]
    )


def inside_world(obj, world_w, world_h):
    return (
        0 <= obj["x"] and
        0 <= obj["y"] and
        obj["x"] + obj["w"] <= world_w and
        obj["y"] + obj["h"] <= world_h
    )



def valid_environment(self, env):

    if not np.any(env == 2):
        return False

    visited = set()
    coords = list(zip(*np.where(env == 2)))

    queue = deque([coords[0]])

    while queue:
        y, x = queue.popleft()
        if (y, x) in visited:
            continue
        visited.add((y, x))

        for ny, nx in [(y+1,x),(y-1,x),(y,x+1),(y,x-1)]:
            if 0 <= ny < self.WORLD_HEIGHT and 0 <= nx < self.WORLD_WIDTH:
                if env[ny, nx] == 2 and (ny, nx) not in visited:
                    queue.append((ny, nx))

    return len(visited) == len(coords)

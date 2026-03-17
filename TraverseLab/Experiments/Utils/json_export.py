import random
random.seed(42)

def neighbors4(idx, world_width):
    x = idx % world_width
    y = idx // world_width
    return [
        y * world_width + x - 1,
        y * world_width + x + 1,
        (y - 1) * world_width + x,
        (y + 1) * world_width + x,
    ]


def rect_to_object(rect, world_width, is_ground):
    """
    Convert a rectangle into EvoGym object format:
    {
        indices: [...],
        types: [...],
        neighbors: { idx: [neighbors] }
    }
    """
    indices = []
    neighbors = {}

    # collect voxel indices
    for dy in range(rect["h"]):
        for dx in range(rect["w"]):
            x = rect["x"] + dx
            y = rect["y"] + dy
            idx = y * world_width + x
            indices.append(idx)

    index_set = set(indices)

    # build 4-connected neighbors
    for i in indices:
        neighbors[str(i)] = [
            n for n in neighbors4(i, world_width) if n in index_set
        ]

    # ground = type 1, others random material
    types = [
        1 if is_ground else random.randint(2, 5)
        for _ in indices
    ]

    return {
        "indices": indices,
        "types": types,
        "neighbors": neighbors,
    }


import math

def object_to_rect(obj, world_width):
    """
    Converts an EvoGym object format back into a rect dictionary and its metadata.
    """
    indices = obj["indices"]
    if not indices:
        return None

    # 1. Convert 1D indices back to 2D coordinates
    coords = []
    for idx in indices:
        x = idx % world_width
        y = idx // world_width
        coords.append((x, y))

    # 2. Find the bounding box (min/max x and y)
    min_x = min(c[0] for c in coords)
    max_x = max(c[0] for c in coords)
    min_y = min(c[1] for c in coords)
    max_y = max(c[1] for c in coords)

    # 3. Reconstruct the rect dictionary
    rect = {
        "x": min_x,
        "y": min_y,
        "w": (max_x - min_x) + 1,
        "h": (max_y - min_y) + 1
    }

    return rect

def grid_to_json(self, env):
    indices = []

    for y in range(self.WORLD_HEIGHT):
        for x in range(self.WORLD_WIDTH):
            if env[y, x] == 2:
                idx = y * self.WORLD_WIDTH + x
                indices.append(idx)

    return {
        "objects": {
            "terrain": {
                "indices": indices
            }
        }
    }

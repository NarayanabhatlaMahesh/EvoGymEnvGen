# utils/voxel.py

def neighbors4(idx, world_w):
    x = idx % world_w
    y = idx // world_w
    return [
        y * world_w + x - 1,
        y * world_w + x + 1,
        (y - 1) * world_w + x,
        (y + 1) * world_w + x,
    ]





def is_connected(indices, world_w):
    visited = set()
    stack = [next(iter(indices))]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for n in neighbors4(cur, world_w):
            if n in indices:
                stack.append(n)
    return len(visited) == len(indices)

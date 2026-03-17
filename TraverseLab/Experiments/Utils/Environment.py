import random


class Environment:

    def __init__(self, width, height, ground_height, curriculum):

        self.width = width
        self.height = height
        self.ground_height = ground_height

        self.obj_size = curriculum["size"]
        self.obj_count = curriculum["count"]

        self.objects = []
        self.full_json = {}

    # ------------------------------------------------
    # BUILD FINAL JSON
    # ------------------------------------------------

    def build_environment(self):

        env = {
            "grid_width": self.width,
            "grid_height": self.height,
            "objects": {}
        }

        env["objects"]["ground"] = self.generate_ground()

        for i, obj in enumerate(self.objects):
            env["objects"][f"terrain_{i}"] = obj

        self.full_json = env

        return env


    # ------------------------------------------------
    # GENERATE GROUND
    # ------------------------------------------------

    def generate_ground(self):

        indices = []
        neighbors = {}

        for x in range(self.width):

            idx = x
            indices.append(idx)

        for i in range(len(indices)):

            n = []

            if i > 0:
                n.append(indices[i - 1])

            if i < len(indices) - 1:
                n.append(indices[i + 1])

            neighbors[str(indices[i])] = n

        return {
            "indices": indices,
            "types": [1] * len(indices),
            "neighbors": neighbors
        }


    # ------------------------------------------------
    # GENERATE OBJECT
    # ------------------------------------------------

    def generate_object(self, size, slot_start, slot_end):

        obj = {
            "indices": [],
            "types": [],
            "neighbors": {}
        }

        base_y = self.ground_height

        start_x = random.randint(slot_start, slot_end - size)

        coords = []

        for i in range(size):
            for j in range(size):

                x = start_x + j
                y = base_y + i

                idx = y * self.width + x

                obj["indices"].append(idx)
                obj["types"].append(1)

                coords.append((idx, x, y))

        for idx, x, y in coords:

            nbs = []

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:

                nx = x + dx
                ny = y + dy

                nid = ny * self.width + nx

                if nid in obj["indices"]:
                    nbs.append(nid)

            obj["neighbors"][str(idx)] = nbs

        return obj

class ParentEnvironment(Environment):

    def __init__(self, width, height, ground_height, curriculum):

        super().__init__(width, height, ground_height, curriculum)

        slot = width // self.obj_count

        for i in range(self.obj_count):

            slot_start = i * slot
            slot_end = slot_start + slot - 1

            obj = self.generate_object(self.obj_size, slot_start, slot_end)

            self.objects.append(obj)

        self.build_environment()

class ChildEnvironment(Environment):

    def __init__(self, parent1, parent2):

        super().__init__(
            parent1.width,
            parent1.height,
            parent1.ground_height,
            {"size": parent1.obj_size, "count": parent1.obj_count}
        )

        split = random.randint(1, len(parent1.objects) - 1)

        self.objects = parent1.objects[:split] + parent2.objects[split:]

        self.build_environment()


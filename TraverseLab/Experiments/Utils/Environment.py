import random
import json

from ollama import chat


class Environment:

    def __init__(self, width, height, ground_height, curriculum):

        self.width = width
        self.height = height
        self.ground_height = ground_height

        self.obj_size = curriculum["size"]
        self.obj_count = curriculum["count"]

        self.objects = []
        self.full_json = {}

        # ---------------- LLM PROMPT ----------------
        self.LLM_MUTATION_PROMPT = """
You are mutating a grid-based EvoGym environment.

STRICT RULES:
- Output MUST be valid JSON
- Do NOT change grid_width or grid_height
- Ground object MUST NOT change
- Modify ONLY terrain objects
- Mutations must be VERY SMALL (1–2 voxel changes)
- Objects must remain connected
- No overlaps
- Indices must be in bounds
- neighbors must be 4-connected
- types length must match indices

Return JSON ONLY.
"""

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
            "types": [5] * len(indices),
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
                obj["types"].append(5)

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

    # ------------------------------------------------
    # CONNECTIVITY CHECK
    # ------------------------------------------------

    def is_connected(self, indices):

        if len(indices) == 0:
            return False

        visited = set()
        stack = [next(iter(indices))]

        while stack:

            cur = stack.pop()

            if cur in visited:
                continue

            visited.add(cur)

            x = cur % self.width
            y = cur // self.width

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:

                nx = x + dx
                ny = y + dy
                nid = ny * self.width + nx

                if nid in indices:
                    stack.append(nid)

        return len(visited) == len(indices)

    # ------------------------------------------------
    # LLM MUTATION (NEW)
    # ------------------------------------------------

    def llm_mutate(self):

        env_json = self.build_environment()

        prompt = (
            self.LLM_MUTATION_PROMPT
            + "\nINPUT:\n"
            + json.dumps(env_json)
            + "\nOUTPUT:\n"
        )

        for _ in range(3):  # retry

            try:
                response = chat(
                    model="gpt-oss:120b-cloud",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0}
                )

                raw = response["message"]["content"]
                start, end = raw.find("{"), raw.rfind("}")

                if start == -1 or end == -1:
                    continue

                parsed = json.loads(raw[start:end + 1])

                if "objects" not in parsed:
                    continue

                new_objects = []

                for key, obj in parsed["objects"].items():

                    if key == "ground":
                        continue

                    if "indices" not in obj or "neighbors" not in obj:
                        continue

                    # basic connectivity check
                    if not self.is_connected(set(obj["indices"])):
                        continue

                    new_objects.append(obj)

                if len(new_objects) == len(self.objects):
                    self.objects = new_objects
                    return self

            except Exception:
                continue

        return self  # fallback


# ==================================================
# PARENT
# ==================================================

class ParentEnvironment(Environment):

    def __init__(self, width, height, ground_height, curriculum):

        super().__init__(width, height, ground_height, curriculum)

        slot = width // self.obj_count

        for i in range(self.obj_count):

            slot_start = i * slot
            slot_end = slot_start + slot - 1

            obj = self.generate_object(self.obj_size, slot_start, slot_end)
            self.objects.append(obj)

        print('in parent env ', width, height, ground_height, curriculum)

        self.build_environment()


# ==================================================
# CHILD
# ==================================================

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

    # ------------------------------------------------
    # MUTATION ENTRY POINT (NEW)
    # ------------------------------------------------

    def mutate(self, mutation_type="random", mutation_prob=0.1):

        if mutation_type == "LLM":
            return self.llm_mutate()

        # ---------- RANDOM MUTATION ----------
        new_objects = []

        for i in range(len(self.objects)):

            slot = self.width // len(self.objects)
            slot_start = i * slot
            slot_end = slot_start + slot - 1

            new_obj = self.mutate_object(
                self.objects[i],
                slot_start,
                slot_end,
                self.obj_size,
                mutation_prob
            )

            new_objects.append(new_obj)

        self.objects = new_objects
        return self

    # ------------------------------------------------
    # VOXEL MUTATION (UNCHANGED)
    # ------------------------------------------------

    def mutate_object(self, obj, slot_start, slot_end, obj_size, mutation_prob):

        indices = set(obj["indices"])

        num_mutations = max(1, int(mutation_prob * len(indices)))

        for _ in range(num_mutations):

            # REMOVE
            if random.random() < 0.5 and len(indices) > 1:

                idx = random.choice(list(indices))
                indices.remove(idx)

                if not self.is_connected(indices):
                    indices.add(idx)

            # ADD
            else:

                idx = random.choice(list(indices))

                x = idx % self.width
                y = idx // self.width

                dx, dy = random.choice([(1,0),(-1,0),(0,1),(0,-1)])

                nx = x + dx
                ny = y + dy

                if (
                    0 <= nx < self.width and
                    self.ground_height <= ny < self.height and
                    slot_start <= nx <= slot_end
                ):
                    new_idx = ny * self.width + nx
                    indices.add(new_idx)

        new_obj = {
            "indices": list(indices),
            "types": [5] * len(indices),
            "neighbors": {}
        }

        for idx in indices:

            x = idx % self.width
            y = idx // self.width

            nbs = []

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:

                nx = x + dx
                ny = y + dy
                nid = ny * self.width + nx

                if nid in indices:
                    nbs.append(nid)

            new_obj["neighbors"][str(idx)] = nbs

        return new_obj
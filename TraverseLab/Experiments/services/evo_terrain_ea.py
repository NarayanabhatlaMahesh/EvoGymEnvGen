import numpy as np
import random
from ..Utils.evolution import run_evolution
from ..Utils.fitness import ppo_fitness
from ..Utils.robot_loader import load_robot_from_csv
from ..Utils.rendering import normalize_frame
from ..Utils.config import LLM_MUTATION_PROMPT
import json
from ollama import ChatResponse, chat
random.seed(42)


class EvoGymTerrainEA:

    def __init__(self, vals):

        self.WORLD_WIDTH = int(vals.get("WORLD_WIDTH", 40))
        self.WORLD_HEIGHT = int(vals.get("WORLD_HEIGHT", 6))
        self.GROUND_HEIGHT = int(vals.get("GROUND_HEIGHT", 1))

        self.POPULATION_SIZE = int(vals.get("POPULATION_SIZE", 6))
        self.ELITE_COUNT = int(vals.get("ELITE_COUNT", 3))
        self.GENERATIONS = int(vals.get("GENERATIONS", 3))
        self.MUTATION_PROB = float(vals.get("MUTATION_PROB", 0.5))

        self.ROBOT_CSV = vals.get(
            "ROBOT_CSV",
            r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\evogym_high_reward_distinct_envs.csv"
        )


        self.MAX_STEPS = int(vals.get("MAX_STEPS", 40))
        self.PPO_TRAIN_TIMESTEPS = int(vals.get("PPO_TRAIN_TIMESTEPS", 200))

        self.CURRICULUM = {
            "easy": {
                "objects_count": int(vals.get("CURRICULUM_1_OBJECTS_COUNT", 3)),
                "object_size": int(vals.get("CURRICULUM_1_OBJECT_SIZE", 1)),
            },
            "medium": {
                "objects_count": int(vals.get("CURRICULUM_2_OBJECTS_COUNT", 5)),
                "object_size": int(vals.get("CURRICULUM_2_OBJECT_SIZE", 2)),
            },
            "hard": {
                "objects_count": int(vals.get("CURRICULUM_3_OBJECTS_COUNT", 6)),
                "object_size": int(vals.get("CURRICULUM_3_OBJECT_SIZE", 3)),
            },
        }

        # ================= MODEL =================
        self.model_name = "ppo_ObstacleTraverser__480000_steps"

        self.MODEL_PATH = rf"C:\Users\numam\EALLMs\EnvGenPrmtOpt\saved_models\checkpoints\ObstacleTraverser-v0-7\{self.model_name}.zip"

        self.ROBOT_NAME = vals.get("ROBOT_NAME", "robot")


        # ================= RENDERING =================
        self.TARGET_W = 3260
        self.TARGET_H = 1150

        print("MODEL_PATH:", self.MODEL_PATH)
        print("Has initialise_population:", hasattr(self, "initialise_population"))
        print("Has normalize_frame:", hasattr(self, "normalize_frame"))



    # ======================================================
    # GENERATION
    # ======================================================

    def generate_valid_environment(self, cfg):

        envs = []

        while len(envs) < self.POPULATION_SIZE:

            env = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH), dtype=int)

            # -------- GROUND --------
            env[0:self.GROUND_HEIGHT, :] = 1

            s = cfg["object_size"]     # object size
            g = 1                      # required gap
            W = self.WORLD_WIDTH
            y0 = self.GROUND_HEIGHT

            # -------- MAX WINDOWS (FORMULA) --------
            max_objects = W // (s + g)

            # curriculum might ask more than physically possible
            n = min(cfg["objects_count"], max_objects)

            # choose window indices randomly
            window_ids = list(range(max_objects))
            random.shuffle(window_ids)
            window_ids = window_ids[:n]

            # -------- Make And PLACE OBJECTS --------
            for i in window_ids:

                r = random.randint(0, g)      # shift inside gap
                x0 = i*(s+g) + r

                if x0 + s < W:
                    env[y0:y0+s, x0:x0+s] = 2

            envs.append(env.copy())

        return envs



    # ======================================================
    # MUTATION (CONNECTIVITY SAFE)
    # ======================================================

    def mutate_environment(self, env, difficulty):

        new_env = env.copy()

        size = self.CURRICULUM[difficulty]["object_size"]

        # Expand terrain by 1 connected voxel
        terrain_positions = np.argwhere(new_env == 2)

        if len(terrain_positions) > 0:

            y, x = random.choice(terrain_positions)

            for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:

                ny, nx = y + dy, x + dx

                if (
                    0 <= ny < self.WORLD_HEIGHT and
                    0 <= nx < self.WORLD_WIDTH and
                    new_env[ny, nx] == 0
                ):
                    new_env[ny, nx] = 2
                    break

        return new_env
    
    def initialise_population(self, curriculum_params):
        return self.generate_valid_environment(curriculum_params)


    # ======================================================
    # VALIDATION
    # ======================================================

    def valid_environment(self, env):

        # Must have terrain
        if not np.any(env == 2):
            return False

        # Ground must remain intact
        if not np.all(env[0:self.GROUND_HEIGHT, :] == 1):
            return False

        # Ensure terrain is connected
        return self.is_connected(env)

    def is_connected(self, env):

        visited = set()
        terrain = np.argwhere(env == 2)

        if len(terrain) == 0:
            return False

        stack = [tuple(terrain[0])]

        while stack:
            y, x = stack.pop()
            if (y, x) in visited:
                continue
            visited.add((y, x))

            for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < self.WORLD_HEIGHT and
                    0 <= nx < self.WORLD_WIDTH and
                    env[ny, nx] == 2 and
                    (ny, nx) not in visited
                ):
                    stack.append((ny, nx))

        return len(visited) == len(terrain)

    # ======================================================
    # WRAPPERS
    # ======================================================

    def run(self, mutation_type="LLM"):
        return run_evolution(self, mutation_type="LLM")

    def ppo_fitness(self, env_json_path, cfg):
        return ppo_fitness(env_json_path, cfg)

    def load_robot_from_csv(self):
        return load_robot_from_csv(self.ROBOT_CSV, self.ROBOT_NAME, 7)
    
    def normalize_frame(self, frame, TARGET_H, TARGET_W):
        return normalize_frame(frame, self.TARGET_W, self.TARGET_H)
    
    def llm_mutation_prompt(self,obj, difficulty):

        grid_txt = "\n".join(
            " ".join(str(int(v)) for v in row)
            for row in obj
        )

        return f"""
        You are a LOCAL mutation operator inside an evolutionary algorithm.

        You are NOT generating a new object.
        You are modifying an existing one.

        The object below is a 2D voxel terrain component:
        2 = solid voxel
        0 = empty space

        Perform ONLY ONE small local change near an existing solid voxel (2):

        Allowed operations:
        - add one voxel
        - remove one voxel
        - extend surface by one voxel
        - carve one voxel
        - form a small step
        - form a small slope
        - form a ledge

        Constraints:
        - Result MUST remain one 4-connected component
        - NO floating voxels (every 2 must have support below or beside)
        - DO NOT expand bounding box
        - DO NOT delete entire object
        - Keep shape mostly similar

        Difficulty target:
        easy   → flat, wide, blocky
        medium → gentle slope or step
        hard   → narrow step, ledge or small gap

        Output ONLY the mutated grid using 0 and 2.
        No text.
        No explanation.
        No markdown.
        Same dimensions as input.

        INPUT
        {grid_txt}

        OUTPUT
    """
    def LLM_mutate_environment(self, env, difficulty):

        print("\n--- LLM OBJECT MUTATION START ---")

        new_env = env.copy()

        try:

            # ---------------- FIND TERRAIN COMPONENTS ----------------
            visited = set()
            components = []

            for y in range(self.WORLD_HEIGHT):
                for x in range(self.WORLD_WIDTH):

                    if new_env[y, x] == 2 and (y, x) not in visited:

                        stack = [(y, x)]
                        comp = []

                        while stack:
                            cy, cx = stack.pop()

                            if (cy, cx) in visited:
                                continue

                            visited.add((cy, cx))
                            comp.append((cy, cx))

                            for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                                ny, nx = cy+dy, cx+dx
                                if (
                                    0 <= ny < self.WORLD_HEIGHT and
                                    0 <= nx < self.WORLD_WIDTH and
                                    new_env[ny, nx] == 2 and
                                    (ny, nx) not in visited
                                ):
                                    stack.append((ny, nx))

                        components.append(comp)

            if len(components) == 0:
                return env

            # ---------------- PICK RANDOM OBJECT ----------------
            comp = random.choice(components)

            ys = [p[0] for p in comp]
            xs = [p[1] for p in comp]

            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)

            obj = new_env[y0:y1+1, x0:x1+1].copy()
            obj[obj != 2] = 0

            # ---------------- CALL LLM ----------------
            prompt = self.llm_mutation_prompt(obj, difficulty)

            response: ChatResponse = chat(
                model="gpt-oss:120b-cloud",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0}
            )

            raw = response["message"]["content"]

            lines = [
                l.strip()
                for l in raw.split("\n")
                if l.strip()
            ]

            grid = np.array([
                [int(x) for x in r.split()]
                for r in lines
            ])

            if grid.shape != obj.shape:
                print("[DEBUG] ❌ Shape mismatch → returning original env")
                return env

            if not np.any(grid == 2):
                print("[DEBUG] ❌ Empty object → returning original env")
                return env

            # ---------------- REINSERT ----------------
            patch = new_env[y0:y1+1, x0:x1+1]
            patch[:] = 0
            patch[grid == 2] = 2

            # keep ground intact
            new_env[0:self.GROUND_HEIGHT, :] = 1

            if not self.valid_environment(new_env):
                print("[DEBUG] ❌ Post-mutation invalid → returning original env")
                return env

            diff = np.sum(new_env != env)
            print(f"[DEBUG] ✅ Object mutated | {diff} cells changed")

            print("--- LLM OBJECT MUTATION END ---\n")
            return new_env

        except Exception as e:
            print(f"[DEBUG] ❌ Exception: {e} → returning original env")
            return env
import json
import random
import copy
import os
import re
import tempfile
from typing import Dict, Any

from anyio import Path
from anyio import Path
import cv2
import imageio
import pandas as pd
import ast
import numpy as np

from evogym.world import EvoWorld
from django.utils import timezone

from .models import VisualiseEnvs, EnvImages, TimestampEnvGenerated
from .JsonWorldEnv import JsonWorldEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ollama import chat
from ollama import ChatResponse

random.seed(42)

class EvoGymTerrainEA:
    def __init__(self, vals):
        self.WORLD_WIDTH = int(vals["WORLD_WIDTH"])
        self.WORLD_HEIGHT = int(vals["WORLD_HEIGHT"])
        self.GROUND_HEIGHT = int(vals["GROUND_HEIGHT"])

        self.POPULATION_SIZE = int(vals["POPULATION_SIZE"])
        self.ELITE_COUNT = int(vals["ELITE_COUNT"])
        self.GENERATIONS = int(vals["GENERATIONS"])
        self.MUTATION_PROB = float(vals["MUTATION_PROB"])

        self.MAX_STEPS = int(vals["MAX_STEPS"])

        self.N_ENVS = int(vals["N_ENVS"])
        self.N_STEPS = int(vals["N_STEPS"])
        self.PPO_TRAIN_TIMESTEPS = int(vals["PPO_TRAIN_TIMESTEPS"])
        self.BATCH_SIZE = int(vals["BATCH_SIZE"])

        self.ROBOT_NAME = vals["ROBOT_NAME"]
        self.ROBOT_CSV = vals["ROBOT_CSV"]
        self.DEVICE = vals["DEVICE"]
        self.paths=[]

        self.LLM_MUTATION_PROMPT = """
        You are mutating a grid-based EvoGym environment.

        CRITICAL RULES (FAIL IF ANY ARE VIOLATED):

        1. Output MUST be valid JSON.
        2. Output MUST match the EXACT schema.
        3. Do NOT change grid_width or grid_height.
        4. Object "new_object_1" is the ground and MUST NOT change.
        5. You may ONLY mutate objects new_object_2, new_object_3, etc.
        6. All mutated objects MUST touch the ground.
        7. Objects must be SMALL.
        8. Mutations must be SMALL.
        9. Object connectivity MUST be preserved.
        10. No overlapping indices.
        11. Indices must stay in bounds.
        12. neighbors MUST be 4-connected.
        13. types length MUST match indices.

        Return JSON ONLY.
        """

        self.WORLD_PATH = r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\RunExp\Generation\LLM_Envs\gen10_env19.json"
        # ROBOT_CSV  = r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\data\evogym_env_body_connections.csv"
        self.ROBOT_CSV  = r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\evogym_high_reward_distinct_envs.csv"
        self.model_name='ppo_ObstacleTraverser__480000_steps'
        self.MODEL_PATH = rf"C:\Users\numam\EALLMs\EnvGenPrmtOpt\saved_models\checkpoints\ObstacleTraverser-v0-7\{self.model_name}"
        self.ROBOT_NAME = "robot"
        # MAX_STEPS = 1000

        self.USE_PPO = True   # ← switch here

        self.TARGET_W, self.TARGET_H = 3260, 1150   # resize AFTER full capture
        self.MAX_STEPS_FOR_GIF = 100
        self.ROBOT_NAME = "robot"

    def normalize_frame(self, img, H, W):
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        return img
    # ---------------- GEOMETRY ----------------
    def rects_overlap(self, a, b):
        return not (
            a["x"] + a["w"] <= b["x"] or
            b["x"] + b["w"] <= a["x"] or
            a["y"] + a["h"] <= b["y"] or
            b["y"] + b["h"] <= a["y"]
        )

    def inside_world(self, obj):
        return (
            0 <= obj["x"] and
            0 <= obj["y"] and
            obj["x"] + obj["w"] <= self.WORLD_WIDTH and
            obj["y"] + obj["h"] <= self.WORLD_HEIGHT
        )

    def valid_environment(self, env):
        objs = env["objects"]
        for i, o in enumerate(objs):
            if not self.inside_world(o):
                return False
            for j in range(i):
                if self.rects_overlap(o, objs[j]):
                    return False
        return True

    # ---------------- INIT ENV ----------------
    def ground_object(self):
        return {"x": 0, "y": 0, "w": self.WORLD_WIDTH, "h": self.GROUND_HEIGHT}

    def terrain_bump(self):
        w = random.randint(1, 3)
        h = random.randint(1, 2)
        x = random.randint(0, self.WORLD_WIDTH - w)
        return {"x": x, "y": self.GROUND_HEIGHT, "w": w, "h": h}

    def generate_valid_environment(self, n):
        while True:
            objs = [self.ground_object()]
            while len(objs) < n + 1:
                cand = self.terrain_bump()
                if any(self.rects_overlap(cand, o) for o in objs):
                    continue
                objs.append(cand)
            env = {"objects": objs}
            if self.valid_environment(env):
                return env

    def initialise_population(self):
        return [self.generate_valid_environment(6) for _ in range(self.POPULATION_SIZE)]

    # ---------------- VOXEL HELPERS ----------------
    def neighbors4(self, idx):
        x = idx % self.WORLD_WIDTH
        y = idx // self.WORLD_WIDTH
        return [
            y * self.WORLD_WIDTH + x - 1,
            y * self.WORLD_WIDTH + x + 1,
            (y - 1) * self.WORLD_WIDTH + x,
            (y + 1) * self.WORLD_WIDTH + x,
        ]

    def rect_to_indices(self, rect):
        s = set()
        for dy in range(rect["h"]):
            for dx in range(rect["w"]):
                x = rect["x"] + dx
                y = rect["y"] + dy
                s.add(y * self.WORLD_WIDTH + x)
        return s

    def indices_to_rect(self, indices):
        xs = [i % self.WORLD_WIDTH for i in indices]
        ys = [i // self.WORLD_WIDTH for i in indices]
        return {
            "x": min(xs),
            "y": min(ys),
            "w": max(xs) - min(xs) + 1,
            "h": max(ys) - min(ys) + 1
        }

    def is_connected(self, indices):
        visited = set()
        stack = [next(iter(indices))]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for n in self.neighbors4(cur):
                if n in indices:
                    stack.append(n)
        return len(visited) == len(indices)

    # ---------------- RANDOM MUTATION ----------------
    def mutate_environment(self, env):
        if random.random() > self.MUTATION_PROB:
            return env

        new = copy.deepcopy(env)

        for i in range(1, len(new["objects"])):
            rect = new["objects"][i]
            voxels = self.rect_to_indices(rect)

            boundary = []
            for v in voxels:
                for n in self.neighbors4(v):
                    x, y = n % self.WORLD_WIDTH, n // self.WORLD_WIDTH
                    if (
                        n not in voxels and
                        0 <= x < self.WORLD_WIDTH and
                        self.GROUND_HEIGHT <= y < self.WORLD_HEIGHT
                    ):
                        boundary.append(n)

            if random.random() < 0.5 and boundary:
                voxels.add(random.choice(boundary))
            elif len(voxels) > 1:
                rem = random.choice(list(voxels))
                voxels.remove(rem)
                if not self.is_connected(voxels):
                    voxels.add(rem)

            new["objects"][i] = self.indices_to_rect(voxels)

        return new if self.valid_environment(new) else env

    # ---------------- LLM MUTATION ----------------
    def LLM_mutate_environment(self, env):
        prompt = (
            self.LLM_MUTATION_PROMPT
            + "\nINPUT ENVIRONMENT:\n"
            + json.dumps(self.to_object_json(env))
            + "\nOUTPUT:\n"
        )

        response: ChatResponse = chat(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0
            }
        )

        raw = response["message"]["content"]
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return env

        try:
            parsed = json.loads(raw[start:end + 1])
            objs = []
            for obj in parsed["objects"].values():
                xs = [i % self.WORLD_WIDTH for i in obj["indices"]]
                ys = [i // self.WORLD_WIDTH for i in obj["indices"]]
                objs.append({
                    "x": min(xs),
                    "y": min(ys),
                    "w": max(xs) - min(xs) + 1,
                    "h": max(ys) - min(ys) + 1
                })
            env_out = {"objects": objs}
            return env_out if self.valid_environment(env_out) else env
        except Exception:
            return env

    # ---------------- JSON EXPORT ----------------
    def rect_to_object(self, rect, is_ground):
        indices = []
        neighbors = {}

        for dy in range(rect["h"]):
            for dx in range(rect["w"]):
                x = rect["x"] + dx
                y = rect["y"] + dy
                idx = y * self.WORLD_WIDTH + x
                indices.append(idx)

        s = set(indices)
        for i in indices:
            neighbors[str(i)] = [n for n in self.neighbors4(i) if n in s]

        types = [1 if is_ground else random.randint(1, 2) for _ in indices]
        return {"indices": indices, "types": types, "neighbors": neighbors}

    def to_object_json(self, env):
        return {
            "grid_width": self.WORLD_WIDTH,
            "grid_height": self.WORLD_HEIGHT,
            "objects": {
                f"new_object_{i+1}": self.rect_to_object(rect, i == 0)
                for i, rect in enumerate(env["objects"])
            }
        }
    
    # ---------------- ROBOT IMPORT ----------------
    def parse_body(self,body_str):
        # Find all array([...]) blocks
        arrays = re.findall(r'array\(\[([^\]]+)\]\)', body_str)

        # Convert each block to numpy array
        return np.array([
            np.fromstring(a, sep=',', dtype=int)
            for a in arrays
        ])

    def parse_connections(self,conn_str):
        arrays = re.findall(r'array\(\[([^\]]+)\]\)', conn_str)

        return np.array([
            np.fromstring(a, sep=',', dtype=int)
            for a in arrays
        ])
    
    def load_robot_from_csv(self):
        df = pd.read_csv(self.ROBOT_CSV)
        df = df.where(df['env_name'] == 'ObstacleTraverser-v0').dropna().sort_values(by='reward').reset_index(drop=True)
        row = df.iloc[7]
        body = self.parse_body(row["body"])
        connections = self.parse_connections(row["connections"])

        return body, connections

    # ---------------- SAVE GENERATION ----------------
    def save_generation(self, population, gen, root, visualise_entry):
        gen_dir = os.path.join(root, f"gen_{gen}")
        os.makedirs(gen_dir, exist_ok=True)
        body, connections = self.load_robot_from_csv()

        

        for i, env in enumerate(population):
            path = os.path.join(gen_dir, f"env_{i}.json")
            self.paths.append(path)
            with open(path, "w") as f:
                json.dump(self.to_object_json(env), f, indent=2)

        body, con = self.load_robot_from_csv()
        i=0
        for file in self.paths:
            # break
            world = None
            world = EvoWorld.from_json(file)
            run = True
            x_v = 4
            x_y = 10
            while(run):
                try:
                    world.add_from_array(
                        name=self.ROBOT_NAME,
                        structure=body,
                        connections=con,
                        x=x_v,
                        y=x_y,
                    )
                    
                    run = False
                except:
                    x_v+=1
                    x_y+=1
                    run = True

            env = DummyVecEnv([
                lambda: JsonWorldEnv(
                    world=world,
                    robot_name=self.ROBOT_NAME,
                    render_mode='img',
                    total_timesteps=1 * 100,
                )
            ])

            obs = env.reset()

            img0 = env.env_method('render')[0]
            img0 = self.normalize_frame(img0, self.TARGET_H, self.TARGET_W)
            img_gen_dir = os.path.join(root, f"gen_{gen}", "images")
            os.makedirs(img_gen_dir, exist_ok=True)
            imgpath = rf"{img_gen_dir}\{gen}_{i}.png"
            imageio.imwrite(imgpath, img0)
            print(imgpath)
            # Save image path to the database
            env_image = EnvImages(
                visualise_env=visualise_entry,
                image_path=imgpath
            )
            env_image.save()

            i+=1
        self.paths=[]

        return path

    # ---------------- FITNESS ----------------
    def ppo_fitness(self, env_desc):
        world_json = self.to_object_json(env_desc)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(world_json, f)
            path = f.name

        try:
            world = EvoWorld.from_json(path)

            df = pd.read_csv(self.ROBOT_CSV)
            row = df.iloc[1]
            body = self.parse_body(row["body"])
            connections = self.parse_connections(row["connections"])

            world.add_from_array(
                name=self.ROBOT_NAME,
                structure=body,
                connections=connections,
                x=1,
                y=10,
            )

            env = DummyVecEnv(
                [lambda: JsonWorldEnv(world=world, 
                                      robot_name=self.ROBOT_NAME, 
                                      total_timesteps=self.PPO_TRAIN_TIMESTEPS
                                      ) for _ in range(self.N_ENVS)]
            )

            model = PPO(
                "MlpPolicy",
                env,
                n_steps=self.N_STEPS,
                batch_size=self.BATCH_SIZE,
                learning_rate=1e-3,
                device=self.DEVICE,
                verbose=0,
            )

            model.learn(total_timesteps=self.PPO_TRAIN_TIMESTEPS)

            obs = env.reset()
            total_reward = 0.0
            for _ in range(self.MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, _ = env.step(action)
                total_reward += reward.mean()
                if dones.any():
                    break

            env.close()
            return total_reward
        finally:
            os.unlink(path)

    # ---------------- EVOLUTION ----------------
    def run(self, mutation_type="random"):
        population = self.initialise_population()
        timestmp = timezone.now()           # ✅ real datetime
        val = timestmp.strftime("%Y%m%d_%H%M%S")
        timestampenvgen = TimestampEnvGenerated(timestamp=timestmp)
        timestampenvgen.save()
        genroot = os.path.join("generated_envs", val)
        for gen in range(1, self.GENERATIONS + 1):
            print(f"\n🧬 Generation {gen}")

            # SAVE CURRENT POPULATION
            gen_dir = os.path.join(genroot, f"gen_{gen}")
            visualise_entry = VisualiseEnvs(
                path=gen_dir,
                generation=gen,
                timestampenvgenerated=timestampenvgen
            )
            visualise_entry.save()
            path = self.save_generation(population, gen, genroot, visualise_entry)

            

            scored = [(env, self.ppo_fitness(env)) for env in population]
            scored.sort(key=lambda x: x[1], reverse=True)

            elites = [env for env, _ in scored[:self.ELITE_COUNT]]
            next_pop = elites[:]

            while len(next_pop) < self.POPULATION_SIZE:
                parent = random.choice(elites)
                if mutation_type == "LLM":
                    next_pop.append(self.LLM_mutate_environment(parent))
                else:
                    next_pop.append(self.mutate_environment(parent))

            population = next_pop
            print("✅ Done")
        return self.paths

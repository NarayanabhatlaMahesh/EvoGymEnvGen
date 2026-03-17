# evolution.py

import os
import json
import random
import imageio
import numpy as np

from django.utils import timezone

from ..models import (
    VisualiseEnvs,
    EnvImages,
    TimestampEnvGenerated,
    Curriculum,
    GeneratedEnv
)

from evogym.world import EvoWorld
from ..JsonWorldEnv import JsonWorldEnv
from stable_baselines3.common.vec_env import DummyVecEnv

random.seed(42)


# ============================================================
# RUN EVOLUTION
# ============================================================

def run_evolution(self, mutation_type="LLM"):

    timestmp = timezone.now()
    timestamp_str = timestmp.strftime("%Y%m%d_%H%M%S")

    experiment = TimestampEnvGenerated.objects.create(
        timestamp=timestmp
    )

    genroot = os.path.join("generated_envs", timestamp_str)

    for difficulty in ["easy", "medium", "hard"]:

        print(f"\n🚀 Starting difficulty: {difficulty}")

        curriculum_obj = Curriculum.objects.create(
            experiment=experiment,
            difficulty=difficulty,
            object_size=self.CURRICULUM[difficulty]['object_size'],
            object_count=self.CURRICULUM[difficulty]['objects_count'],
            description=f"{difficulty} CURRICULUM"
        )

        population = self.initialise_population(
            self.CURRICULUM[difficulty]
        )

        for gen in range(1, self.GENERATIONS + 1):

            print(f"\n🧬 Generation {gen}")

            gen_dir = os.path.join(genroot, difficulty, f"gen_{gen}")
            os.makedirs(gen_dir, exist_ok=True)

            visualise_entry = VisualiseEnvs.objects.create(
                path=gen_dir,
                generation=gen,
                curriculum=curriculum_obj
            )

            # ---- SAVE JSON + IMAGES ----
            json_paths = save_generation(
                self=self,
                population=population,
                gen=gen,
                root=gen_dir,
                visualise_entry=visualise_entry
            )

            # ---- FITNESS ----
            scored = [
                (population[i], self.ppo_fitness(json_paths[i], self))
                for i in range(len(population))
            ]

            


            scored.sort(key=lambda x: x[1], reverse=True)
            best_ratio = scored[0][1]
            avg_ratio  = sum([s for _, s in scored]) / len(scored)


            # from ..models import GeneratedEnv

            GeneratedEnv.objects.create(
                visualise_Envs = visualise_entry,
                avgPPO = avg_ratio,
                maxPPO = best_ratio,
                avgIOU = 0,
                avgNCD = 0,
                maxIOU = 0,
                maxNCD = 0
            )


            elites = [env for env, _ in scored[:self.ELITE_COUNT]]

            # ---- CROSSOVER ----
            next_pop = cross_over(elites, self)

            # ---- MUTATION ----
            new_pop = []
            
            while len(new_pop) < self.POPULATION_SIZE:
                parent = random.choice(next_pop)

                if mutation_type == "LLM":
                    child = self.LLM_mutate_environment(parent, difficulty)
                else:
                    child = self.mutate_environment(parent, difficulty)

                new_pop.append(child)

            population = new_pop

            print("✅ Generation Complete")

    print("\n🎉 Evolution Finished")
    return True


# ============================================================
# CROSSOVER (GRID VERSION)
# ============================================================



def cross_over(elites, self):

    envs = []

    while len(envs) < self.POPULATION_SIZE:

        p1, p2 = random.sample(elites, 2)

        child = p1.copy()

        split = random.randint(1, self.WORLD_WIDTH - 2)
        child[:, split:] = p2[:, split:]

        # keep ground intact
        child[0:self.GROUND_HEIGHT, :] = 1

        # ONLY accept if connected
        if np.any(child == 2):
            envs.append(child)

    return envs



# ============================================================
# GRID → EVOGYM JSON FORMAT
# ============================================================


def grid_to_json(env, cfg):

    objects = {}

    H, W = cfg.WORLD_HEIGHT, cfg.WORLD_WIDTH
    visited = set()
    comp_id = 0

    # -------- GROUND --------
    ground_idx = []
    ground_nb  = {}

    for y in range(H):
        for x in range(W):
            if env[y,x] == 1:
                idx = y*W + x
                ground_idx.append(idx)

    for idx in ground_idx:
        ground_nb[str(idx)] = []
        x = idx % W
        y = idx // W

        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx,y+dy
            if 0<=nx<W and 0<=ny<H:
                n = ny*W+nx
                if n in ground_idx:
                    ground_nb[str(idx)].append(n)

    objects["ground"] = {
        "indices": ground_idx,
        "types":[5]*len(ground_idx),
        "neighbors":ground_nb
    }

    # -------- TERRAIN SPLIT --------
    for y in range(H):
        for x in range(W):

            if env[y,x]==2 and (y,x) not in visited:

                stack=[(y,x)]
                comp=[]

                while stack:
                    cy,cx=stack.pop()
                    if (cy,cx) in visited:
                        continue

                    visited.add((cy,cx))
                    comp.append((cy,cx))

                    for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                        ny,nx=cy+dy,cx+dx
                        if 0<=ny<H and 0<=nx<W:
                            if env[ny,nx]==2 and (ny,nx) not in visited:
                                stack.append((ny,nx))

                idx=[]
                nb={}

                for cy,cx in comp:
                    idx.append(cy*W+cx)

                for i in idx:
                    nb[str(i)]=[]
                    x=i%W
                    y=i//W

                    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nx,ny=x+dx,y+dy
                        if 0<=nx<W and 0<=ny<H:
                            n=ny*W+nx
                            if n in idx:
                                nb[str(i)].append(n)

                objects[f"terrain_{comp_id}"]={
                    "indices":idx,
                    "types":[1]*len(idx),
                    "neighbors":nb
                }

                comp_id+=1

    return {
        "grid_width":W,
        "grid_height":H,
        "objects":objects
    }



# ============================================================
# SAVE GENERATION
# ============================================================

def save_generation(self, population, gen, root, visualise_entry):

    os.makedirs(root, exist_ok=True)

    body, connections = self.load_robot_from_csv()

    json_paths = []

    # ---------------- SAVE JSON ----------------
    for i, env in enumerate(population):

        # Debug info
        print(f"\n[SAVE DEBUG] ENV {i}")
        print("Ground voxels :", np.sum(env == 1))
        print("Terrain voxels:", np.sum(env == 2))
        print("Unique values :", np.unique(env))
        print("-----------------------------------")

        json_path = os.path.join(root, f"env_{i}.json")

        json_data = grid_to_json(env, self)

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        json_paths.append(json_path)

    # ---------------- SAVE IMAGES ----------------
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)

    for i, file_path in enumerate(json_paths):

        world = EvoWorld.from_json(file_path)

        # Safe robot placement
        x_v, x_y = 1, self.GROUND_HEIGHT + 1

        while True:
            try:
                world.add_from_array(
                    name=self.ROBOT_NAME,
                    structure=body,
                    connections=connections,
                    x=x_v,
                    y=x_y,
                )
                break
            except Exception:
                x_v += 1
                if x_v > self.WORLD_WIDTH - 3:
                    break

        env = DummyVecEnv([
            lambda: JsonWorldEnv(
                world=world,
                robot_name=self.ROBOT_NAME,
                render_mode="img",
                total_timesteps=100,
            )
        ])

        env.reset()

        img = env.env_method("render")[0]
        img = self.normalize_frame(img, self.TARGET_H, self.TARGET_W)

        img_path = os.path.join(img_dir, f"{gen}_{i}.png")

        imageio.imwrite(img_path, img)

        EnvImages.objects.create(
            visualise_env=visualise_entry,
            image_path=img_path
        )

    print(f"📸 Saved {len(json_paths)} environments for generation {gen}")

    return json_paths

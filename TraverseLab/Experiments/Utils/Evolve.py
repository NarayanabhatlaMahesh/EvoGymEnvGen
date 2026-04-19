import random
import json
import os
import imageio
import numpy as np
import pandas as pd
from pathlib import Path

from .robot_loader import parse_array_blocks
from .Environment import ParentEnvironment, ChildEnvironment
from .JsonWorldEnv import JsonWorldEnv
from .walkerenv import JsonWorldEnv as WalkerEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from Experiments.models import (
    Curriculum,
    VisualiseEnvs,
    EnvImages,
    GeneratedEnv
)

from evogym import EvoWorld
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class Evolve:

    # ------------------------------------------------
    # GET PPO PATHS
    # ------------------------------------------------
    def get_ppo_paths(self, env_map, base_dir):
        ppo_paths = {}

        for env_name in env_map.keys():
            env_dir = base_dir / env_name

            if not env_dir.exists():
                continue

            robot_models = []

            for robot_dir in sorted(env_dir.glob("robot_*")):
                model_path = robot_dir / "final_model.zip"

                if model_path.exists():
                    robot_models.append(str(model_path))

            if robot_models:
                ppo_paths[env_name] = robot_models

        return ppo_paths

    # ------------------------------------------------
    # BUILD MODEL-ROBOT MAP
    # ------------------------------------------------
    def build_model_robot_map(self, ppo_paths):

        df = pd.read_csv(self.cfg.ROBOT_CSV)
        df = df[df["env_name"].isin(self.ENV_MAP.keys())]

        model_map = {}

        for env_name, model_list in ppo_paths.items():

            env_df = df[df["env_name"] == env_name]

            if env_df.empty:
                continue

            env_df = env_df.sort_values(by="reward", ascending=False).reset_index(drop=True)

            for i, model_path in enumerate(model_list):

                if i >= len(env_df):
                    break

                body = parse_array_blocks(env_df.iloc[i]["body"])
                connections = parse_array_blocks(env_df.iloc[i]["connections"])

                model_map[model_path] = {
                    "model": PPO.load(model_path, device="cpu"),
                    "body": body,
                    "connections": connections,
                    "env_class": self.ENV_MAP[env_name]
                }

        return model_map

    # ------------------------------------------------
    # INIT
    # ------------------------------------------------
    def __init__(
        self,
        cfg,
        curriculum,
        difficulty,
        generations,
        population_size,
        mutation_prob,
        elite_count,
        max_steps,
        world_width,
        world_height,
        ground_height,
        timestamp_obj,
        timestamp_str,
        save_path="generated_envs"
    ):

        self.cfg = cfg
        self.curriculum = curriculum
        self.difficulty = difficulty

        self.experiment = timestamp_obj
        self.timestamp_str = timestamp_str

        self.GENERATIONS = generations
        self.POPULATION_SIZE = population_size
        self.MUTATION_PROB = mutation_prob
        self.ELITE_COUNT = elite_count
        self.MAX_STEPS = max_steps

        self.WORLD_WIDTH = world_width
        self.WORLD_HEIGHT = world_height
        self.GROUND_HEIGHT = ground_height

        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.ENV_MAP = {
            "ObstacleTraverser-v0": JsonWorldEnv,
            "Traverser-v0": JsonWorldEnv,
            "Climber-v0": JsonWorldEnv,
            "Walker-v0": WalkerEnv
        }

        BASE_DIR = Path("C:/Users/numam/EALLMs/EnvGenPrmtOpt/saved_modelsNEWW")

        print("📦 Loading PPO models + robots...")

        ppo_paths = self.get_ppo_paths(self.ENV_MAP, BASE_DIR)
        self.model_robot_map = self.build_model_robot_map(ppo_paths)

        print(f"✅ Loaded {len(self.model_robot_map)} model-robot pairs")

    # ------------------------------------------------
    # SAFE ROBOT PLACEMENT
    # ------------------------------------------------
    def safe_add_robot(self, world, body, connections):

        for dy in range(1, self.WORLD_HEIGHT):
            try:
                world.add_from_array(
                    name=self.cfg.ROBOT_NAME,
                    structure=body,
                    connections=connections,
                    x=1,
                    y=self.GROUND_HEIGHT + dy
                )
                return True
            except:
                continue

        return False

    # ------------------------------------------------
    # RUN EVOLUTION
    # ------------------------------------------------
    def run_evolution(self):

        print(f"\n🚀 Starting difficulty: {self.difficulty}")

        genroot = os.path.join(self.save_path, self.timestamp_str)

        curriculum_obj = Curriculum.objects.create(
            experiment=self.experiment,
            difficulty=self.difficulty,
            object_size=self.curriculum["size"],
            object_count=self.curriculum["count"],
            description=f"{self.difficulty} CURRICULUM"
        )

        population = self.initialize_population()

        for gen in range(1, self.GENERATIONS + 1):

            print(f"\n🧬 Generation {gen}")

            gen_dir = os.path.join(genroot, self.difficulty, f"gen_{gen}")
            os.makedirs(gen_dir, exist_ok=True)

            visualise_entry = VisualiseEnvs.objects.create(
                path=gen_dir,
                generation=gen,
                curriculum=curriculum_obj
            )

            json_paths = self.save_generation(
                population,
                gen,
                gen_dir,
                visualise_entry
            )

            scores = [self.run_evogym(p) for p in json_paths]

            best = max(scores)
            avg = sum(scores) / len(scores)

            GeneratedEnv.objects.create(
                visualise_Envs=visualise_entry,
                avgPPO=avg,
                maxPPO=best,
                avgIOU=0,
                maxIOU=0,
                avgNCD=0,
                maxNCD=0
            )

            scored = list(zip(population, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            elites = [env for env, _ in scored[:self.ELITE_COUNT]]
            next_pop = self.crossover(elites)

            new_pop = []
            while len(new_pop) < self.POPULATION_SIZE:
                parent = random.choice(next_pop)
                child = self.mutate_environment(parent)
                new_pop.append(child)

            population = new_pop

            print("✅ Generation Complete")

        print(f"🎯 Finished {self.difficulty}")

    # ------------------------------------------------
    def initialize_population(self):
        return [
            ParentEnvironment(
                width=self.WORLD_WIDTH,
                height=self.WORLD_HEIGHT,
                ground_height=self.GROUND_HEIGHT,
                curriculum=self.curriculum
            )
            for _ in range(self.POPULATION_SIZE)
        ]

    # ------------------------------------------------
    # SAVE (NO ROBOT)
    # ------------------------------------------------
    def save_generation(self, population, gen, root, visualise_entry):

        json_paths = []

        for i, env in enumerate(population):
            path = os.path.join(root, f"env_{i}.json")

            with open(path, "w") as f:
                json.dump(env.build_environment(), f, indent=2)

            json_paths.append(path)

        img_dir = os.path.join(root, "images")
        os.makedirs(img_dir, exist_ok=True)

        for i, path in enumerate(json_paths):

            world = EvoWorld.from_json(path)

            env = DummyVecEnv([
                lambda: JsonWorldEnv(
                    world=world,
                    robot_name=self.cfg.ROBOT_NAME,
                    render_mode="img",
                    total_timesteps=100
                )
            ])

            env.reset()
            img = env.env_method("render")[0]

            img_path = os.path.join(img_dir, f"{gen}_{i}.png")
            imageio.imwrite(img_path, img)

            EnvImages.objects.create(
                visualise_env=visualise_entry,
                image_path=img_path
            )

            env.close()

        return json_paths

    # ------------------------------------------------
    # FITNESS (PARALLEL OVER ALL ROBOTS)
    # ------------------------------------------------
    
    def run_evogym(self, json_path):

        jobs = list(self.model_robot_map.values())
        n = len(jobs)

        if n == 0:
            return 0.0

        # -------- create parallel envs --------
        def make_env(job):
            def _init():
                world = EvoWorld.from_json(json_path)

                if not self.safe_add_robot(world, job["body"], job["connections"]):
                    raise RuntimeError("Robot placement failed")

                return job["env_class"](
                    world=world,
                    robot_name=self.cfg.ROBOT_NAME,
                    total_timesteps=self.cfg.PPO_TRAIN_TIMESTEPS
                )
            return _init

        try:
            envs = SubprocVecEnv([make_env(job) for job in jobs])
        except:
            return 0.0

        obs = envs.reset()

        x0 = np.zeros(n)
        max_x = np.zeros(n)

        # initial positions
        for i in range(n):
            try:
                pos = envs.env_method("get_pos_com_obs", self.cfg.ROBOT_NAME, indices=i)[0][0]
            except:
                pos = 0.0
            x0[i] = pos
            max_x[i] = pos

        # -------- rollout --------
        for _ in range(self.cfg.MAX_STEPS):

            actions = []
            for i, job in enumerate(jobs):
                try:
                    action, _ = job["model"].predict(obs[i])
                except:
                    action = envs.action_space.sample()
                actions.append(action)

            obs, _, dones, _ = envs.step(actions)

            for i in range(n):
                try:
                    cur = envs.env_method("get_pos_com_obs", self.cfg.ROBOT_NAME, indices=i)[0][0]
                    max_x[i] = max(max_x[i], cur)
                except:
                    continue

            if dones.any():
                break

        envs.close()

        scores = (max_x - x0) / self.WORLD_WIDTH
        scores = np.array(scores)

        if len(scores) == 0:
            return 0.0

        # normalization (same as before)
        scores = scores / (np.abs(scores).max() + 1e-8)

        return float(np.std(scores))




    # ------------------------------------------------
    def crossover(self, parents):

        children = []

        while len(children) < self.POPULATION_SIZE - self.ELITE_COUNT:
            p1, p2 = random.sample(parents, 2)
            children.append(ChildEnvironment(p1, p2))

        return children

    # ------------------------------------------------
    def mutate_environment(self, env):

        for i in range(len(env.objects)):

            if random.random() < self.MUTATION_PROB:

                slot = self.WORLD_WIDTH // env.obj_count
                slot_start = i * slot
                slot_end = slot_start + slot - 1

                env.objects[i] = env.mutate_object(
                    env.objects[i],
                    slot_start,
                    slot_end,
                    env.obj_size,
                    self.MUTATION_PROB
                )

        return env
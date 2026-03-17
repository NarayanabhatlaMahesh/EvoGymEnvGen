import random
import json
import os
import numpy as np

from .robot_loader import load_robot_from_csv
from .Environment import ParentEnvironment, ChildEnvironment

# EvoGym
from evogym import EvoWorld
from .JsonWorldEnv import JsonWorldEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class Evolve:

    # ------------------------------------------------
    # INIT
    # ------------------------------------------------

    def __init__(
        self,
        cfg,
        curriculum,
        generations,
        population_size,
        mutation_prob,
        elite_count,
        max_steps,
        world_width,
        world_height,
        ground_height,
        save_path="generated_envs"
    ):

        self.cfg = cfg
        self.curriculum = curriculum

        self.generations = generations
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.elite_count = elite_count
        self.max_steps = max_steps

        self.width = world_width
        self.height = world_height
        self.ground_height = ground_height

        self.save_path = save_path

        os.makedirs(save_path, exist_ok=True)

        # Load PPO model once
        print("Loading PPO model...")
        self.model = PPO.load(cfg.MODEL_PATH, device="cpu")


    # ------------------------------------------------
    # RUN GA
    # ------------------------------------------------

    def run_ga(self):

        print("\nInitializing population")

        population = self.initialize_population()

        for g in range(self.generations):

            print(f"\nGeneration {g}")

            fitness_scores = self.run_fitness(population, g)

            ranked = list(zip(population, fitness_scores))
            ranked.sort(key=lambda x: x[1], reverse=True)

            population = [x[0] for x in ranked]

            print("Best fitness:", ranked[0][1])

            parents = population[:self.elite_count]

            children = self.generate_crossover(parents)

            self.mutate(children)

            population = parents + children

        return population


    # ------------------------------------------------
    # INITIAL POPULATION
    # ------------------------------------------------

    def initialize_population(self):

        population = []

        for i in range(self.population_size):

            env = ParentEnvironment(
                width=self.width,
                height=self.height,
                ground_height=self.ground_height,
                curriculum=self.curriculum
            )

            population.append(env)

        return population


    # ------------------------------------------------
    # FITNESS
    # ------------------------------------------------

    def run_fitness(self, envs, generation):

        fitness_scores = []

        for i, env in enumerate(envs):

            env_json = env.build_environment()

            file_path = os.path.join(
                self.save_path,
                f"env_g{generation}_i{i}.json"
            )

            with open(file_path, "w") as f:
                json.dump(env_json, f, indent=2)

            score = self.run_evogym(file_path)

            fitness_scores.append(score)

        return fitness_scores


    # ------------------------------------------------
    # RUN EVOGYM
    # ------------------------------------------------

    def run_evogym(self, json_path):

        try:

            # --------------------------------
            # LOAD WORLD
            # --------------------------------

            world = EvoWorld.from_json(json_path)

            body, con = load_robot_from_csv(
                self.cfg.ROBOT_CSV,
                self.cfg.ROBOT_NAME,
                7
            )

            world.add_from_array(
                name=self.cfg.ROBOT_NAME,
                structure=body,
                connections=con,
                x=1,
                y=10
            )

            # --------------------------------
            # ENV INITIALIZATION
            # --------------------------------

            env = DummyVecEnv([
                lambda: JsonWorldEnv(
                    world=world,
                    render_mode="img",
                    robot_name=self.cfg.ROBOT_NAME,
                    total_timesteps=self.cfg.PPO_TRAIN_TIMESTEPS
                )
            ])

            obs = env.reset()

            # --------------------------------
            # INITIAL ROBOT POSITION
            # --------------------------------

            x0 = env.env_method(
                "get_pos_com_obs",
                self.cfg.ROBOT_NAME
            )[0][0]

            # --------------------------------
            # RUN POLICY
            # --------------------------------

            for _ in range(self.cfg.MAX_STEPS):

                action, _ = self.model.predict(obs)

                obs, _, done, _ = env.step(action)

                if done.any():
                    break

            # --------------------------------
            # FINAL POSITION
            # --------------------------------

            xf = env.env_method(
                "get_pos_com_obs",
                self.cfg.ROBOT_NAME
            )[0][0]

            distance = xf - x0

            ratio = distance / self.cfg.WORLD_WIDTH

            return float(ratio)

        except Exception as e:

            print("EvoGym crash:", e)

            return 0.0


    # ------------------------------------------------
    # CROSSOVER
    # ------------------------------------------------

    def generate_crossover(self, parents):

        children = []

        while len(children) < self.population_size - self.elite_count:

            p1, p2 = random.sample(parents, 2)

            child = ChildEnvironment(p1, p2)

            children.append(child)

        return children


    # ------------------------------------------------
    # MUTATION
    # ------------------------------------------------

    def mutate(self, population):

        for env in population:

            if random.random() < self.mutation_prob:

                self.mutate_environment(env)


    # ------------------------------------------------
    # MUTATE ENV
    # ------------------------------------------------

    def mutate_environment(self, env):

        if len(env.objects) == 0:
            return

        obj_id = random.randint(0, len(env.objects) - 1)

        slot = self.width // env.obj_count

        slot_start = obj_id * slot
        slot_end = slot_start + slot - 1

        new_obj = env.generate_object(
            env.obj_size,
            slot_start,
            slot_end
        )

        env.objects[obj_id] = new_obj
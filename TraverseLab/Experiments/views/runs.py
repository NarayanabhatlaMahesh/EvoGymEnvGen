from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse

from Experiments.models import Run
from Metrics.models import RunMetricsSummary
from Artifacts.models import Artifact

from celery import shared_task

from ..Utils.Evolve import Evolve

import traceback


# ------------------------------------------------
# CONFIG CLASS
# ------------------------------------------------

class Config:

    def __init__(self):

        # -------------------------------
        # ROBOT + MODEL
        # -------------------------------

        self.ROBOT_CSV = r"C:\Users\numam\EALLMs\EnvGenPrmtOpt\evogym_high_reward_distinct_envs.csv"

        self.model_name = "ppo_ObstacleTraverser__480000_steps"

        self.MODEL_PATH = rf"C:\Users\numam\EALLMs\EnvGenPrmtOpt\saved_models\checkpoints\ObstacleTraverser-v0-7\{self.model_name}"

        self.ROBOT_NAME = "robot"

        # -------------------------------
        # PPO / SIMULATION
        # -------------------------------

        self.USE_PPO = True

        self.MAX_STEPS = 1000
        self.PPO_TRAIN_TIMESTEPS = 200

        # -------------------------------
        # RENDER
        # -------------------------------

        self.TARGET_W = 3260
        self.TARGET_H = 1150
        self.MAX_STEPS_FOR_GIF = 100

        # -------------------------------
        # WORLD (updated later)
        # -------------------------------

        self.WORLD_WIDTH = 40


# ------------------------------------------------
# REDIRECT
# ------------------------------------------------

def redirect_main(request):
    return redirect("/experiment")


# ------------------------------------------------
# RUN LIST
# ------------------------------------------------

def run_list(request):

    runs = (
        Run.objects
        .select_related("experiment")
        .order_by("-start_time")
    )

    return render(
        request,
        "experiments/run_list.html",
        {
            "runs": runs,
        }
    )


# ------------------------------------------------
# RUN DETAIL
# ------------------------------------------------

def run_detail(request, run_id):

    run = get_object_or_404(Run, id=run_id)

    metrics = RunMetricsSummary.objects.filter(run=run).first()
    artifacts = Artifact.objects.filter(run=run)

    rollout = artifacts.filter(artifact_type="gif").first()
    thumbnail = artifacts.filter(artifact_type="image").first()

    return render(
        request,
        "experiments/run_detail.html",
        {
            "run": run,
            "metrics": metrics,
            "rollout": rollout,
            "thumbnail": thumbnail,
            "artifacts": artifacts,
        }
    )


# ------------------------------------------------
# START GA ENV GENERATION
# ------------------------------------------------

def run_GA_env(request):

    if request.method == "POST":

        params = request.POST.dict()
        params.pop("csrfmiddlewaretoken", None)

        print("\nReceived Parameters")
        print("=" * 40)

        for k, v in params.items():
            print(f"{k:<35} : {v}")

        print("=" * 40)

        generate_environments.delay(params)

    return render(request, "experiments/GenEnvWorld.html")


# ------------------------------------------------
# CELERY TASK
# ------------------------------------------------

@shared_task(bind=True)
def generate_environments(self, params):

    try:

        print("\nStarting Terrain Evolution")
        print("-" * 40)

        # ------------------------------------------------
        # WORLD PARAMETERS
        # ------------------------------------------------

        WORLD_WIDTH = int(params.get("WORLD_WIDTH", 40))
        WORLD_HEIGHT = int(params.get("WORLD_HEIGHT", 6))
        GROUND_HEIGHT = int(params.get("GROUND_HEIGHT", 1))

        # ------------------------------------------------
        # GA PARAMETERS
        # ------------------------------------------------

        POPULATION_SIZE = int(params.get("POPULATION_SIZE", 6))
        GENERATIONS = int(params.get("GENERATIONS", 3))
        MUTATION_PROB = float(params.get("MUTATION_PROB", 0.3))

        ELITE_COUNT = POPULATION_SIZE // 2

        print(f"\nPopulation Size : {POPULATION_SIZE}")
        print(f"Generations     : {GENERATIONS}")
        print(f"Elite Count     : {ELITE_COUNT}")
        print(f"Mutation Prob   : {MUTATION_PROB}")

        # ------------------------------------------------
        # CONFIG
        # ------------------------------------------------

        cfg = Config()
        cfg.WORLD_WIDTH = WORLD_WIDTH   # override with UI value

        # ------------------------------------------------
        # CURRICULUM
        # ------------------------------------------------

        curriculum = {
            "easy": {
                "count": int(params.get("CURRICULUM_1_OBJECTS_COUNT", 3)),
                "size": int(params.get("CURRICULUM_1_OBJECT_SIZE", 1))
            },
            "medium": {
                "count": int(params.get("CURRICULUM_2_OBJECTS_COUNT", 5)),
                "size": int(params.get("CURRICULUM_2_OBJECT_SIZE", 2))
            },
            "hard": {
                "count": int(params.get("CURRICULUM_3_OBJECTS_COUNT", 6)),
                "size": int(params.get("CURRICULUM_3_OBJECT_SIZE", 3))
            }
        }

        # ------------------------------------------------
        # RUN CURRICULUM
        # ------------------------------------------------

        for difficulty, cur in curriculum.items():

            print("\n--------------------------------")
            print(f"Running difficulty: {difficulty}")
            print(f"Objects Count : {cur['count']}")
            print(f"Object Size   : {cur['size']}")
            print("--------------------------------")

            evo = Evolve(
                cfg=cfg,
                curriculum=cur,
                generations=GENERATIONS,
                population_size=POPULATION_SIZE,
                mutation_prob=MUTATION_PROB,
                elite_count=ELITE_COUNT,
                max_steps=cfg.MAX_STEPS,
                world_width=WORLD_WIDTH,
                world_height=WORLD_HEIGHT,
                ground_height=GROUND_HEIGHT
            )

            evo.run_ga()

        print("\nEvolution Finished")

        return True

    except Exception as e:

        print("\n" + "!" * 30 + " TRACEBACK " + "!" * 30)
        traceback.print_exc()
        print("!" * 71 + "\n")

        print(f"CRASH SUMMARY: {e}")

        return False
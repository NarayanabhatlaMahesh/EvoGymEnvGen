# utils/mutation.py

import copy, random, json
from ollama import chat
from .config import LLM_MUTATION_PROMPT
from ollama import chat
from ollama import ChatResponse
random.seed(42)


# def mutate_environment(env, cfg):
#     if random.random() > cfg.MUTATION_PROB:
#         return env

#     new = copy.deepcopy(env)

#     for i in range(1, len(new["objects"])):
#         rect = new["objects"][i]    
#         voxels = rect_to_indices(rect, cfg.WORLD_WIDTH)

#         boundary = []
#         for v in voxels:
#             for n in neighbors4(v, cfg.WORLD_WIDTH):
#                 x, y = n % cfg.WORLD_WIDTH, n // cfg.WORLD_WIDTH
#                 if (
#                     n not in voxels and
#                     0 <= x < cfg.WORLD_WIDTH and
#                     cfg.GROUND_HEIGHT <= y < cfg.WORLD_HEIGHT
#                 ):
#                     boundary.append(n)

#         if random.random() < 0.5 and boundary:
#             voxels.add(random.choice(boundary))
#         elif len(voxels) > 1:
#             rem = random.choice(list(voxels))
#             voxels.remove(rem)
#             if not is_connected(voxels, cfg.WORLD_WIDTH):
#                 voxels.add(rem)

#         new["objects"][i] = indices_to_rect(voxels, cfg.WORLD_WIDTH)

#     return new if valid_environment(new, cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT) else env



def mutate_environment(self, env, difficulty):

    new_env = env.copy()

    target_size = self.CURRICULUM[difficulty]['object_size'] ** 2

    # Count terrain voxels
    terrain_positions = [
        (y, x)
        for y in range(self.WORLD_HEIGHT)
        for x in range(self.WORLD_WIDTH)
        if new_env[y, x] == 2
    ]

    current_size = len(terrain_positions)

    # -----------------------------
    # GROW if too small
    # -----------------------------
    if current_size < target_size:

        boundary = []

        for y, x in terrain_positions:
            for ny, nx in [(y+1,x),(y-1,x),(y,x+1),(y,x-1)]:
                if (
                    self.GROUND_HEIGHT <= ny < self.WORLD_HEIGHT and
                    0 <= nx < self.WORLD_WIDTH and
                    new_env[ny, nx] == 0
                ):
                    boundary.append((ny, nx))

        if boundary:
            y, x = random.choice(boundary)
            new_env[y, x] = 2

    # -----------------------------
    # SHRINK if too large
    # -----------------------------
    elif current_size > target_size:

        y, x = random.choice(terrain_positions)
        new_env[y, x] = 0

    # Restore ground always
    new_env[0:self.GROUND_HEIGHT, :] = 1

    return new_env




# ---------------- LLM MUTATION ----------------
def LLM_mutate_environment(self, env):
    prompt = (
        LLM_MUTATION_PROMPT
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

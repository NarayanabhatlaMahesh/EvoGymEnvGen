from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from evogym.world import EvoWorld
from .robot_loader import load_robot_from_csv
from ..JsonWorldEnv import JsonWorldEnv


def ppo_fitness(json_path, cfg):

    # Load already-saved environment
    world = EvoWorld.from_json(json_path)

    body, con = load_robot_from_csv(
        cfg.ROBOT_CSV,
        cfg.ROBOT_NAME,
        7
    )

    world.add_from_array(
        name=cfg.ROBOT_NAME,
        structure=body,
        connections=con,
        x=1,
        y=10
    )

    env = DummyVecEnv([
        lambda: JsonWorldEnv(
            world=world,
            render_mode='img',
            robot_name=cfg.ROBOT_NAME,
            total_timesteps=cfg.PPO_TRAIN_TIMESTEPS,
        )
    ])

    model = PPO.load(cfg.MODEL_PATH, device="cpu")
    
    
    obs = env.reset()

    # initial x
    x0 = env.env_method(
        "get_pos_com_obs",
        cfg.ROBOT_NAME
    )[0][0]

    for _ in range(cfg.MAX_STEPS):

        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

        if done.any():
            break

    # final x
    xf = env.env_method(
        "get_pos_com_obs",
        cfg.ROBOT_NAME
    )[0][0]

    distance = xf - x0
    ratio = distance / cfg.WORLD_WIDTH

    return float(ratio)

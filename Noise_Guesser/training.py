from enviroment import CustomEnv
from stable_baselines3 import PPO
import os

def training(entrenamiento):
    models_dir = "Noise/models/PPO"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    env = CustomEnv()
    env.reset()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 10000
    for i in range(1, entrenamiento):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='PPO')
        model.save(f'{models_dir}/{TIMESTEPS*i}')

training(10000)


from world import Enviroment_double
from stable_baselines3 import PPO
import os


models_dir = "Double_Cable/models/PPO"
logdir = "Double_Cable/logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = Enviroment_double()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000
for i in range(1, 1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='PPO')
    model.save(f'{models_dir}/{TIMESTEPS*i}')


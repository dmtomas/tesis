from enviroment import CustomEnv
from stable_baselines3 import PPO, DQN
import os

def training(intervalos, tiempo, N, entrenamiento, entrenar):
    models_dir = "models/DQN"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    env = CustomEnv(N, intervalos, tiempo, entrenar) # El número acá es la cantidad de parámetros + cantidad de puntos + 1
    env.reset()

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 10000
    for i in range(1, entrenamiento):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='DQN')
        model.save(f'{models_dir}/{TIMESTEPS*i}')


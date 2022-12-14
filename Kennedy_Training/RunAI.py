from enviroment import CustomEnv
from stable_baselines3 import PPO
import keyboard

def run():
    models_dir = 'models/PPO'
    model_path = f"{models_dir}/3490000.zip"

    env = CustomEnv()
    model = PPO.load(model_path, env)
    for i in range(0, 1): 
        obs = env.reset()
        done  = False
        while not done:
            obs, reward, done, info = env.step(model.predict(obs)[0])
            if keyboard.is_pressed("p"):
                break

run()
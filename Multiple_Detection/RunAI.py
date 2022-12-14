from enviroment import CustomEnv
from stable_baselines3 import PPO
import keyboard

def run():
    models_dir = 'Multiple_Detection/models/PPO'
    model_path = f"{models_dir}/220000.zip"

    env = CustomEnv([-3, 0, 3])
    model = PPO.load(model_path, env)
    for i in range(0, 1): 
        obs = env.reset()
        done  = False
        while not done:
            print(model.predict(obs)[0])
            obs, reward, done, info = env.step(model.predict(obs)[0])
            if keyboard.is_pressed("p"):
                break

run()
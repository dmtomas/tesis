from stable_baselines3.common.env_checker import check_env
from enviroment import CustomEnv
import numpy as np
env = CustomEnv([-3, 0, 3])
episodes = 1

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:
		random_action = env.action_space.sample()
		obs, reward, done, info = env.step([1.5, 3])
		if done:
			break
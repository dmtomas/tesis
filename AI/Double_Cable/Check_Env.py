from stable_baselines3.common.env_checker import check_env
from world import Enviroment_double

env = Enviroment_double()
episodes = 50

for episode in range(episodes):
	done = False
	obs = env.reset()
	while not done:
		random_action = env.action_space.sample()
		obs, reward, done, info = env.step(random_action)
		print(obs)

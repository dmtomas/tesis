from enviroment import CustomEnv
from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt
import keyboard

def run(intervalos, tiempo, N, datos_y, entrenar):
    models_dir = 'models/DQN'
    model_path = f"{models_dir}/Bueno_fondo.zip"

    env = CustomEnv(N, intervalos, tiempo, entrenar) # El número acá es la cantidad de parámetros + cantidad de puntos + 1
    model = DQN.load(model_path, env)

    obs = env.reset()
    if datos_y != [0 for i in range(len(tiempo))]:
        env.resultado = datos_y
    done  = False
    while not done:
        obs, reward, done, info = env.step(model.predict(obs)[0])
        env.render()
        if keyboard.is_pressed("p"):
            break
    print([obs[i] for i in range(1, len(intervalos) + 1)])
    plt.show()
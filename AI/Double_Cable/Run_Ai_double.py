from world import Enviroment_double
from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt
import keyboard
import numpy as np
import pandas as pd
import seaborn as sns


def run(states, vueltas, model):
    obs = env.reset()
    done  = False
    env.states = states
    avr_rew = 0
    while not done:
        obs, reward, done, info = env.step(model.predict(obs)[0])
        action = model.predict(obs)[0]
        avr_rew += reward / (vueltas + 5)
    print(action)
    for i in range(vueltas):
        obs, reward, done, info = env.step(action)
        avr_rew += reward / (vueltas + 5)
        if keyboard.is_pressed("p"):
            break
    return avr_rew


if  __name__ == '__main__':
    N = 50
    sns.set_theme(style="whitegrid")
    abrir2 = np.array(pd.read_csv("Double_Cable/Prueba_Rendimiento_grande_doble.csv"))
    optimo2 = []

    for i in range(0, len(abrir2)):
        optimo2.append(float(abrir2[i]))
    states2 = np.linspace(0, 0.5, len(optimo2))
    plt.plot(states2, optimo2, color="#D81B60", label="Optimización numérica")

    states = np.linspace(0, 0.5, 50)
    Ia = []
    models_dir = 'Double_Cable/models/PPO'
    model_path = f"{models_dir}/597000.zip"
    env = Enviroment_double()
    model = PPO.load(model_path, env)
    for i in range(len(states)):
        Ia.append(run([-states[i], states[i]], N, model))
    print(Ia)
    plt.scatter(states, Ia, color="#1E88E5", label="IA", s=10)
    plt.fill_between(states, Ia - np.sqrt(np.array(Ia))/np.sqrt((N + 5) * 40), Ia + np.sqrt(np.array(Ia))/np.sqrt((N + 5) * 40), alpha=0.2, color="#1E88E5")
    plt.xlim(0, 0.5)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Probabilidad de éxito")
    plt.legend()
    plt.show()
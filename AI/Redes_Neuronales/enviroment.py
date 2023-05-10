import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

def Ajustar(b, t):
    return b[1] * np.e**(-t * b[0]) * np.cos(b[2] * np.sqrt(t) + b[3])

def Reward(a, b, action, inicial):
    ans = (b - a)**2 / b
    if action == 0 and b < a:
        ans *= -1
    elif action == 1 and b > a:
        ans *= -1
    elif action == 2:
        ans = 0
    return ans * 10


class CustomEnv(gym.Env):
    def __init__(self, N_CHANNELS, intervalos, tiempo, entrenamiento):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3) # N of discrete actions
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.infty, high=np.infty,
                                            shape=(N_CHANNELS,), dtype=np.float32)
        self.intervalos = intervalos
        self.tiempo = tiempo
        self.entrenamiento = entrenamiento

    def reset(self):
        self.buscado = [0 for i in range(len(self.intervalos))]
        self.done = False
        for i in range(0, len(self.buscado)):
            self.buscado[i] = np.random.uniform(self.intervalos[i][0], self.intervalos[i][1])
        if self.entrenamiento == False:
            print(self.buscado)
        self.delta = 0.01

        self.actual = [(self.intervalos[i][1] - self.intervalos[i][0]) / 2 + self.intervalos[i][0] for i in range(len(self.intervalos))]
        self.inicial = [(self.actual[i] - self.buscado[i])** 2 for i in range(len(self.intervalos))]
        self.steps = 0
        for i in range(0, len(self.actual)):
            self.steps += self.actual[i] * 2 / self.delta
        self.variable = 0
        self.resultado = np.random.normal(Ajustar(self.buscado, self.tiempo), np.abs(Ajustar(self.buscado, self.tiempo)*0.1))
        self.observation = np.array([self.variable] + list(self.actual) + list(self.resultado))
        return self.observation # Esta es la observación.


    def step(self, action):
        # Se intenta actualizar la siguiente variable.
        if self.variable < len(self.intervalos) - 1:
            self.variable += 1
        else:
            self.variable = 0
        self.steps -= 1
        if self.steps <= 0:
            self.done = True
        # Acción = 0 -> aumentá la variable.
        # Acción = 1 -> reducí la variable.
        # Acción = 2 -> no hagas nada con la variable.
        if action == 0:
            self.actual[self.variable] += self.delta
        elif action == 1:
            self.actual[self.variable] -= self.delta
        
        self.reward = Reward(self.actual[self.variable], self.buscado[self.variable], action, self.inicial[self.variable])
        self.observation = np.array([self.variable] + list(self.actual) + list(self.resultado))
        info = {}
        return self.observation, self.reward, self.done, info

    def render(self):
        self.vals = Ajustar(self.actual, self.tiempo)
        plt.clf()
        plt.plot(self.tiempo, self.resultado, label="Buscado")
        plt.plot(self.tiempo, self.vals, label="Resultado Actual")
        plt.plot(self.tiempo, self.resultado - self.vals, label="Error", linestyle="--", color="green")
        plt.legend()
        plt.pause(0.001) # Son 60fps.
        return 0

if __name__=="__main__":
    intervalos = [[1, 2], [1, 2], [10, 20], [1, 2]]
    tiempo = np.linspace(0, 5, 200) # Datos en el eje x
    enviroment = CustomEnv(205, intervalos, tiempo, False) # El número acá es la cantidad de parámetros + cantidad de puntos + 1
    enviroment.reset()
    for i in range(0, 10):
        action = int(input())
        enviroment.step(action)
        enviroment.render()
    plt.show()
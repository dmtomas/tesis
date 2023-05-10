from NeuralTraining import training
from run_AI import run
import numpy as np

if __name__ == "__main__":
    intervalos = [[1, 2], [1, 2], [10, 20], [1, 2]]     # Intervalos en los que se quiere entrenar a la IA.
    tiempo = np.linspace(0, 5, 200)                     # Datos en el eje x.
    datos_y = [0 for i in range(len(tiempo))]           # Datos en el eje y.
    entrenamiento = 101                                 # Cuantos bloques de 10000 se quieren entrenar.
    N = len(intervalos) + len(tiempo) + 1

    while True:
        action = int(input("1 para entrenar la IA, 2 para correr el modelo, 3 para salir: "))
        if action == 1:
            training(intervalos, tiempo, N, entrenamiento, True)
        elif action == 2:
            run(intervalos, tiempo, N, datos_y, False)
        else:
            break

# Para cambiar cual modelo se quiere selecciónar, cambiar la variable model_path en run_AI.py
# La función a ajustar está en enviroment.py
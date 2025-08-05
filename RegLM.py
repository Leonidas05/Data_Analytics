import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Funciones necesarias para la RLM
def calcular_matriz_y(Y):
    return np.array(Y).reshape(-1, 1)

def calcular_matriz_x(X):
    return X

def agregar_columna_unos(X):
    unos = np.ones((X.shape[0], 1))
    return np.concatenate((unos, X), axis=1)

def calcular_transpuesta(X):
    return X.T

def calcular_producto_punto(X_transpuesta, X):
    return np.dot(X_transpuesta, X)

def calcular_inversa(matriz):
    return np.linalg.inv(matriz)

def calcular_transpuesta_por_y(X_transpuesta, Y):
    return np.dot(X_transpuesta, Y)

def calcular_coeficientes(producto_punto_inversa, transpuesta_por_y):
    return np.dot(producto_punto_inversa, transpuesta_por_y)

def calcular_r2(Y_real, Y_pred):
    mean_y = np.mean(Y_real)
    ss_total = np.sum((Y_real - mean_y) ** 2)
    ss_residual = np.sum((Y_real - Y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

def calcular_rmse(Y_real, Y_pred):
    n = len(Y_real)
    mse = np.sum((Y_real - Y_pred) ** 2) / n
    rmse = np.sqrt(mse)
    return rmse

#Función principal de la RLM
def regresion_lineal_multiple(X, Y):
    matriz_y = calcular_matriz_y(Y)
    matriz_x = calcular_matriz_x(X)
    matriz_x_unos = agregar_columna_unos(matriz_x)
    matriz_x_transpuesta = calcular_transpuesta(matriz_x_unos)
    producto_punto = calcular_producto_punto(matriz_x_transpuesta, matriz_x_unos)
    inversa = calcular_inversa(producto_punto)
    transpuesta_por_y = calcular_transpuesta_por_y(matriz_x_transpuesta, matriz_y)
    coeficientes = calcular_coeficientes(inversa, transpuesta_por_y)
    Y_pred = np.dot(matriz_x_unos, coeficientes)
    r2 = calcular_r2(Y, Y_pred)
    rmse = calcular_rmse(Y, Y_pred)
    return coeficientes, Y_pred, r2, rmse

#Graficar en 3D
def graficar_regresion_multiple_3d(ax, X, Y_real, Y_pred, titulo):
    ax.scatter(X[:, 1], X[:, 2], Y_real, color='blue', label='Datos reales')
    ax.scatter(X[:, 1], X[:, 2], Y_pred, color='red', marker='x', label='Predicción')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title(titulo)
    ax.legend()

#Leer el dataset
df = pd.read_csv('SalaryMulti.csv')
X = df.iloc[:, :-1].values  # Variables independientes
Y = df.iloc[:, -1].values   # Variable dependiente

coeficientes, Y_pred, r2, rmse = regresion_lineal_multiple(X, Y)

#Imprimir los resultados
print("Valores estimados:")
print(Y_pred)
print("Valores de Beta obtenidos:")
print(coeficientes)
print("Coeficiente de determinación obtenido:")
print(r2)
print("Raíz del Error Cuadrático Medio obtenido:")
print(rmse)

# Graficar en 3D
fig_3d, axs_3d = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
graficar_regresion_multiple_3d(axs_3d[0, 0], np.concatenate((np.ones((X.shape[0], 1)), X[:, 1:3]), axis=1), Y, Y_pred, 'Y vs X1 y X2')
graficar_regresion_multiple_3d(axs_3d[0, 1], np.concatenate((np.ones((X.shape[0], 1)), X[:, [0, 2]]), axis=1), Y, Y_pred, 'Y vs X1 y X3')
graficar_regresion_multiple_3d(axs_3d[0, 2], np.concatenate((np.ones((X.shape[0], 1)), X[:, [0, 3]]), axis=1), Y, Y_pred, 'Y vs X1 y X4')
graficar_regresion_multiple_3d(axs_3d[1, 0], np.concatenate((np.ones((X.shape[0], 1)), X[:, [1, 2]]), axis=1), Y, Y_pred, 'Y vs X2 y X3')
graficar_regresion_multiple_3d(axs_3d[1, 1], np.concatenate((np.ones((X.shape[0], 1)), X[:, [1, 3]]), axis=1), Y, Y_pred, 'Y vs X2 y X4')
graficar_regresion_multiple_3d(axs_3d[1, 2], np.concatenate((np.ones((X.shape[0], 1)), X[:, [2, 3]]), axis=1), Y, Y_pred, 'Y vs X3 y X4')
plt.tight_layout()
plt.show()

# Predicciones de Y
print("Calcularemos el salario dependiente de los años de experiencia, experiencia como líder de equipo, experiencia como gerente de proyecto y número de certificados.")
while True:
    valores_x = input("Ingrese los valores de X1 a X4 para predecir el valor de Y (o 'q' para salir), separados por comas: ")
    if valores_x.lower() == 'q':
        print("Saliendo del programa.")
        break
    try:
        valores_x = [float(x.strip()) for x in valores_x.split(',')]
        print("Valores ingresados:", valores_x)
        if len(valores_x) != 4:
            raise ValueError
        valores_x_con_unos = np.concatenate(([1], valores_x))
        valor_y = np.dot(valores_x_con_unos, coeficientes)
        print(f"El valor predicho de Y para X1={valores_x[0]}, X2={valores_x[1]}, X3={valores_x[2]}, X4={valores_x[3]} es {valor_y[0]:.4f}")
    except ValueError:
        print("Por favor, ingrese cuatro valores numéricos separados por comas o 'q' para salir.")
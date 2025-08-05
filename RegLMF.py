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

def calcular_r2(y_real, y_pred):
    SST = ((y_real - np.mean(y_real)) ** 2).sum()
    SSR = ((y_pred - np.mean(y_real)) ** 2).sum()
    R2 = SSR / SST
    return R2

def calcular_rmse(y_real, y_pred):
    n = len(y_real)
    suma_cuadrados = sum((y_real[i] - y_pred[i]) ** 2 for i in range(n))
    rmse = (suma_cuadrados / n) ** 0.5
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

# Crear un grid para los hiperplanos
x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 50)
x2 = np.linspace(min(X[:, 1]), max(X[:, 1]), 50)
X1, X2 = np.meshgrid(x1, x2)

# Graficar los hiperplanos
fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
combinaciones = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3), (2, 3)
]
for i, (xi, xj) in enumerate(combinaciones):
    ax = axs[i // 3, i % 3]
    Z = coeficientes[0] + coeficientes[xi + 1] * X1 + coeficientes[xj + 1] * X2
    ax.scatter(X[:, xi], X[:, xj], Y, color='blue', label='Datos reales')
    ax.plot_surface(X1, X2, Z, alpha=0.5, color='red', label='Hiperplano de predicciones')
    ax.set_xlabel(f'X{xi + 1}')
    ax.set_ylabel(f'X{xj + 1}')
    ax.set_zlabel('Y')
    ax.set_title(f'Hiperplano de predicciones para X{xi + 1} y X{xj + 1}')
    ax.legend()

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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def imprimir_matrices(nombres_matrices, *matrices):
    for nombre, matriz in zip(nombres_matrices, matrices):
        print(f"{nombre}:")
        print(matriz)
        print()

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

def regresion_lineal_multiple(X, Y):
    matriz_y = calcular_matriz_y(Y)
    imprimir_matrices(["Matriz Y"], matriz_y)

    matriz_x = calcular_matriz_x(X)

    matriz_x_unos = agregar_columna_unos(matriz_x)
    imprimir_matrices(["Matriz X con unos"], matriz_x_unos)

    matriz_x_transpuesta = calcular_transpuesta(matriz_x_unos)
    imprimir_matrices(["Matriz X Transpuesta"], matriz_x_transpuesta)

    producto_punto = calcular_producto_punto(matriz_x_transpuesta, matriz_x_unos)
    imprimir_matrices(["Producto Punto (T*X)"], producto_punto)

    inversa = calcular_inversa(producto_punto)
    imprimir_matrices(["Inversa de T*X"], inversa)

    transpuesta_por_y = calcular_transpuesta_por_y(matriz_x_transpuesta, matriz_y)
    imprimir_matrices(["Transpuesta por Y"], transpuesta_por_y)

    coeficientes = calcular_coeficientes(inversa, transpuesta_por_y)
    return coeficientes

# Función para graficar regresión lineal múltiple en 3D
def graficar_regresion_multiple_3d(ax, X, Y_real, Y_pred, titulo):
    ax.scatter(X[:, 1], X[:, 2], Y_real, color='blue', label='Datos reales')
    ax.scatter(X[:, 1], X[:, 2], Y_pred, color='red', marker='x', label='Predicción')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title(titulo)
    ax.legend()

# Función para graficar regresión lineal múltiple en 2D
def graficar_regresion_multiple_2d(ax, X, Y_real, Y_pred, titulo):
    ax.scatter(X[:, 1], Y_real, color='blue', label='Datos reales')
    ax.scatter(X[:, 1], Y_pred, color='red', marker='x', label='Predicción')
    ax.set_xlabel('X1')
    ax.set_ylabel('Y')
    ax.set_title(titulo)
    ax.legend()

# Lectura del documento y asignación de los valores X y Y
df = pd.read_excel('RegLM.xlsx')
Y = df.iloc[:, 0].values   # Primera columna como variable dependiente
X = df.iloc[:, 1:].values  # Seleccionar las tres columnas restantes como variables independientes

# Calcular regresión lineal múltiple
coeficientes = regresion_lineal_multiple(X, Y)
print(f"Los coeficientes de las variables independientes son: {coeficientes}")

# Realizar predicción
valores_x = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Añadir columna de unos
Y_pred = np.dot(valores_x, coeficientes)

# Graficar en 3D
fig_3d, axs_3d = plt.subplots(1, 3, figsize=(15, 8), subplot_kw={'projection': '3d'})
graficar_regresion_multiple_3d(axs_3d[0], valores_x[:, [0, 1, 2]], Y, Y_pred, 'Y vs X1 y X2')
graficar_regresion_multiple_3d(axs_3d[1], valores_x[:, [0, 1, 3]], Y, Y_pred, 'Y vs X1 y X3')
graficar_regresion_multiple_3d(axs_3d[2], valores_x[:, [0, 2, 3]], Y, Y_pred, 'Y vs X2 y X3')
plt.tight_layout()
plt.show()

# Graficar en 2D
fig_2d, axs_2d = plt.subplots(1, 3, figsize=(15, 5))
graficar_regresion_multiple_2d(axs_2d[0], valores_x[:, [0, 1]], Y, Y_pred, 'Y vs X1')
graficar_regresion_multiple_2d(axs_2d[1], valores_x[:, [0, 2]], Y, Y_pred, 'Y vs X2')
graficar_regresion_multiple_2d(axs_2d[2], valores_x[:, [0, 3]], Y, Y_pred, 'Y vs X3')
plt.tight_layout()
plt.show()

# Predicciones de Y
while True:
    valores_x = input("Ingrese valores de X1, X2 y X3 para predecir el valor de Y (o 'q' para salir), separados por comas: ")
    if valores_x.lower() == 'q':
        print("Saliendo del programa.")
        break
    try:
        valores_x = [float(x.strip()) for x in valores_x.split(',')]
        print("Valores ingresados:", valores_x)
        if len(valores_x) != 3:
            raise ValueError
        valor_y = np.dot(np.append([1], valores_x), coeficientes)
        print(f"El valor predicho de Y para X1={valores_x[0]}, X2={valores_x[1]}, X3={valores_x[2]} es {valor_y[0]:.4f}")
    except ValueError:
        print("Por favor, ingrese tres valores numéricos separados por comas o 'q' para salir.")
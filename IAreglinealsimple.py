import pandas as pd
import matplotlib.pyplot as plt

def calcular_media(valores):
    return sum(valores) / len(valores)

def calcular_covarianza(x, y, media_x, media_y):
    cov = 0
    n = len(x)
    for i in range(n):
        cov += (x[i] - media_x) * (y[i] - media_y)
    return cov / n

def calcular_varianza(valores, media):
    var = 0
    n = len(valores)
    for i in range(n):
        var += (valores[i] - media) ** 2
    return var / n

def calcular_coeficientes(x, y):
    media_x = calcular_media(x)
    media_y = calcular_media(y)
    covarianza = calcular_covarianza(x, y, media_x, media_y)
    varianza_x = calcular_varianza(x, media_x)
    beta1 = covarianza / varianza_x
    beta0 = media_y - beta1 * media_x
    return beta0, beta1

def regresion_lineal_simple(x, y, valor_x):
    beta0, beta1 = calcular_coeficientes(x, y)
    return beta0 + beta1 * valor_x, beta0, beta1

#lectura del documento y asignacion de los valores X y Y
df = pd.read_excel('RegLS.xlsx')

X = df.iloc[:, 0].values  
Y = df.iloc[:, 1].values  

Y_pred, beta0, beta1 = regresion_lineal_simple(X, Y, X)
print(f"El valor de beta0 es de = {beta0} y el de beta1 es de = {beta1}")

#Graficar todo
plt.scatter(X, Y, label='Datos') #puntos
plt.plot(X, Y_pred, color='red', label='Regresión lineal') #línea
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresión Lineal Simple')
plt.legend()
plt.grid(True)
plt.show()

# Predicciones de Y
while True:
    valor_x = input("Ingrese un valor de X para predecir el valor de Y (o 'q' para salir): ")
    if valor_x.lower() == 'q':
        print("Saliendo del programa.")
        break
    try:
        valor_x = float(valor_x)
        valor_y = regresion_lineal_simple(X, Y, valor_x)
        print(f"El valor predicho de Y para X={valor_x} es {valor_y[0]:.2f}")
    except ValueError:
        print("Por favor, ingrese un valor numérico válido o 'q' para salir.")
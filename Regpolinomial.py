import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial y generación de datos
torch.manual_seed(0)  # Establecer una semilla para reproducibilidad en PyTorch
np.random.seed(0)  # Establecer una semilla para reproducibilidad en NumPy

def generate_poly_data(x):
    """
    Función para generar datos de ejemplo siguiendo una función polinomial.
    """
    return 2 * (x ** 2) - 3 * x + 5

X_np = np.random.rand(100, 1) * 10  # Generar 100 valores aleatorios de X en el rango [0, 10]
y_np = generate_poly_data(X_np) + np.random.randn(100, 1) * 2  # Generar valores de y con ruido gaussiano
X = torch.tensor(X_np, dtype=torch.float32)  # Convertir X a un tensor de PyTorch
y = torch.tensor(y_np, dtype=torch.float32)  # Convertir y a un tensor de PyTorch

# Normalización de los datos
X_mean = X.mean()  # Calcular la media de X
X_std = X.std()  # Calcular la desviación estándar de X
X = (X - X_mean) / X_std  # Normalizar X restando la media y dividiendo por la desviación estándar
y_mean = y.mean()  # Calcular la media de y
y_std = y.std()  # Calcular la desviación estándar de y
y = (y - y_mean) / y_std  # Normalizar y restando la media y dividiendo por la desviación estándar

X_train, X_test = X[:80], X[80:]  # Dividir X en conjuntos de entrenamiento (80%) y prueba (20%)
y_train, y_test = y[:80], y[80:]  # Dividir y en conjuntos de entrenamiento (80%) y prueba (20%)

# Definición del modelo polinomial
class PolynomialRegression(nn.Module):
    """
    Clase que define el modelo de regresión polinomial.
    """
    def __init__(self):
        super().__init__()
        self.poly_features = nn.Linear(3, 1)  # Capa lineal que acepta características polinomiales de grado 3

    def forward(self, x):
        """
        Función de forward pass del modelo.
        """
        return self.poly_features(x)

# Preparar datos para regresión polinomial
X_train_poly = torch.cat([X_train**i for i in range(1, 4)], 1)  # Generar características polinomiales para el conjunto de entrenamiento
X_test_poly = torch.cat([X_test**i for i in range(1, 4)], 1)  # Generar características polinomiales para el conjunto de prueba

# Instanciar modelo polinomial
poly_model = PolynomialRegression()

# Optimizador y criterio
optimizer_poly = optim.SGD(poly_model.parameters(), lr=0.01)  # Optimizador SGD con tasa de aprendizaje de 0.01
criterion = nn.MSELoss()  # Función de pérdida de error cuadrático medio

# Entrenamiento del modelo polinomial
num_epochs = 1000  # Número de épocas de entrenamiento
for epoch in range(num_epochs):
    optimizer_poly.zero_grad()  # Reiniciar los gradientes del optimizador
    predictions_poly = poly_model(X_train_poly)  # Realizar predicciones en el conjunto de entrenamiento
    loss_poly = criterion(predictions_poly, y_train)  # Calcular la pérdida
    loss_poly.backward()  # Realizar el backpropagation
    optimizer_poly.step()  # Actualizar los pesos del modelo

    # Imprimir el error (loss) cada 100 iteraciones
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_poly.item():.4f}")

# Predicciones para visualización
X_range = torch.linspace(X.min(), X.max(), 400).reshape(-1, 1)  # Generar un rango de valores de X para visualización
X_range_poly = torch.cat([X_range**i for i in range(1, 4)], 1)  # Generar características polinomiales para el rango de X
poly_model.eval()  # Poner el modelo en modo de evaluación
with torch.no_grad():  # Deshabilitar el cálculo de gradientes
    y_range_pred_poly = poly_model(X_range_poly)  # Realizar predicciones en el rango de X

# Desnormalización de las predicciones
y_range_pred_poly = y_range_pred_poly * y_std + y_mean  # Desnormalizar las predicciones

# Visualización de la regresión polinomial
plt.figure(figsize=(12, 8))
plt.scatter(X_train * X_std + X_mean, y_train * y_std + y_mean, color='blue', label='Datos de entrenamiento', alpha=0.5)
plt.scatter(X_test * X_std + X_mean, y_test * y_std + y_mean, color='green', label='Datos de prueba', alpha=0.5)
plt.plot(X_range * X_std + X_mean, y_range_pred_poly, color='red', linewidth=2, label='Regresión Polinomial')
plt.title('Regresión Polinomial en PyTorch')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
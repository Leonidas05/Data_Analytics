# Importar las librerías necesarias
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Ignorar advertencias
import warnings
warnings.filterwarnings('ignore')

# Definir el modelo de regresión lineal usando PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Cargar el conjunto de datos
data = pd.read_csv('Walmart_sales.csv')

# Preprocesamiento de los datos
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
X = data[['Temperature', 'Fuel_Price']]
y = data['Weekly_Sales']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.values.reshape(-1, 1)

# Convertir los datos a tensores de PyTorch
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train.astype(np.float32))
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y_train_tensor = torch.tensor(y_train.astype(np.float32))
y_test_tensor = torch.tensor(y_test.astype(np.float32))

# Instanciar el modelo
input_size = X_train_tensor.shape[1]
output_size = 1
model = LinearRegressionModel(input_size, output_size)

# Definir el criterio de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Entrenar el modelo
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluar el modelo
model.eval()  # Poner el modelo en modo de evaluación
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    loss = criterion(y_pred_tensor, y_test_tensor)
print(f'Loss on test data: {loss.item():.4f}')

# Interpretación de una predicción específica con PyTorch
sample_data = pd.DataFrame({
    'Temperature': [0.5],
    'Fuel_Price': [0.5]
})
sample_scaled = scaler.transform(sample_data)
sample_tensor = torch.tensor(sample_scaled.astype(np.float32))

with torch.no_grad():
    sample_pred = model(sample_tensor)
print(f"Bajo las condiciones de una temperatura estandarizada de 0.5 y un precio de combustible estandarizado de 0.5,")
print(f"el modelo predice ventas semanales de aproximadamente ${sample_pred.item():.2f}.")

# Análisis adicional para interpretar qué significan los valores estandarizados de 0.5
mean_temperature = data['Temperature'].mean()
std_temperature = data['Temperature'].std()
mean_fuel_price = data['Fuel_Price'].mean()
std_fuel_price = data['Fuel_Price'].std()

original_temp = 0.5 * std_temperature + mean_temperature
original_fuel_price = 0.5 * std_fuel_price + mean_fuel_price

print(f"Un valor estandarizado de 0.5 para la temperatura corresponde a una temperatura original de aproximadamente {original_temp:.2f} grados.")
print(f"Un valor estandarizado de 0.5 para el precio del combustible corresponde a un precio del combustible original de aproximadamente {original_fuel_price:.2f} unidades.")

# Visualización de la relación entre características y las ventas semanales
# Convertir tensores a numpy para visualización
X_test_numpy = X_test_tensor.numpy()
y_test_numpy = y_test_tensor.numpy().flatten()  # Aplanar el tensor de objetivos para visualización
y_pred_numpy = y_pred_tensor.numpy().flatten()  # Aplanar el tensor de predicciones

plt.figure(figsize=(14, 6))

# Temperatura vs. Ventas Semanales
plt.subplot(1, 2, 1)
plt.scatter(X_test_numpy[:, 0], y_test_numpy, color='blue', alpha=0.5, label='Real')
plt.scatter(X_test_numpy[:, 0], y_pred_numpy, color='red', alpha=0.5, label='Predicho')
plt.title('Temperatura vs. Ventas Semanales')
plt.xlabel('Temperatura (Estandarizada)')
plt.ylabel('Ventas Semanales')
plt.legend()

# Precio del Combustible vs. Ventas Semanales
plt.subplot(1, 2, 2)
plt.scatter(X_test_numpy[:, 1], y_test_numpy, color='blue', alpha=0.5, label='Real')
plt.scatter(X_test_numpy[:, 1], y_pred_numpy, color='red', alpha=0.5, label='Predicho')
plt.title('Precio del Combustible vs. Ventas Semanales')
plt.xlabel('Precio del Combustible (Estandarizado)')
plt.ylabel('Ventas Semanales')
plt.legend()

plt.tight_layout()
plt.show()
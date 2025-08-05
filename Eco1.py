import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# Cargar los datos de ventas_productos.csv
data = pd.read_csv('Eco1.csv')

# Preprocesamiento simple (codificación de variables categóricas)
data = pd.get_dummies(data, drop_first=True)

# Separar características y etiqueta
X = data.drop('Cantidad_Vendida', axis=1)
y = data['Cantidad_Vendida']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio (RMSE)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse}")

# Obtener el coeficiente de determinación R^2
r_squared = model.score(X_test, y_test)
print(f"R^2: {r_squared}")

# Obtener los coeficientes de la regresión
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coeficiente'])
print("Coeficientes de la regresión:")
print(coefficients)

# Obtener el intercepto (b0)
intercept = model.intercept_
print(f"Intercepto: {intercept}")

# Gráfico de dispersión (Scatter Plot) - Valores reales vs. Predicciones
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs. Predicciones')
plt.grid(True)
plt.show()

# Gráfico de residuos (Residual Plot)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Distribución de los Residuos')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Gráfico de los coeficientes de la regresión
coefficients.plot(kind='bar', figsize=(10, 6), color='green')
plt.title('Coeficientes de la Regresión Lineal')
plt.xlabel('Variables')
plt.ylabel('Valor del Coeficiente')
plt.grid(True)
plt.show()

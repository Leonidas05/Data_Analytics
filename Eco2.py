import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV
data = pd.read_csv('Eco2.csv')

# Preprocesamiento de datos
data = pd.get_dummies(data, columns=['Genero', 'Ubicacion', 'Nivel_Ingresos'], drop_first=True)

# Definir variables independientes (X) y dependiente (y)
X = data.drop('Compra_Significativa', axis=1)
y = data['Compra_Significativa'].apply(lambda x: 1 if x == 'Sí' else 0)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Importancia de las Características
importances = model.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importances')
plt.show()

# Distribución de los Datos
plt.figure(figsize=(12, 6))
sns.histplot(y, kde=True, discrete=True, color='skyblue')
plt.title('Distribution of Target Variable (Compra_Significativa)')
plt.xlabel('Compra_Significativa')
plt.ylabel('Frequency')
plt.show()

# Visualización del Árbol de Decisión
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Sí'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

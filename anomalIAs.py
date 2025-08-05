import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Creación de datos ficticios
num_samples = 1000
num_features = 10

data = np.random.randn(num_samples, num_features)
columns = [f'Feature_{i+1}' for i in range(num_features)]

# Crear DataFrame de pandas
df = pd.DataFrame(data, columns=columns)

# Agregar algunas anomalías ficticias
# Se eligen aleatoriamente algunas muestras y se modifica una característica aleatoria de cada una para introducir anomalías ficticias. 
# Esto simula un escenario realista donde solo una pequeña parte de los datos es anormal.
anomaly_indices = np.random.choice(num_samples, size=int(num_samples*0.05), replace=False)
for idx in anomaly_indices:
    # Introducir un valor anormal en una característica aleatoria
    feature_idx = np.random.randint(num_features)
    df.iloc[idx, feature_idx] += np.random.uniform(5, 10)

# Preprocesamiento básico
#Valores faltantes se rellenan hacia adelante. StandardScaler para estandarizar que tengan una media de 0 y una desviación estándar de 1. 
df.fillna(method='ffill', inplace=True)  # Manejo de valores faltantes
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.values)  # Normalización

# División entre datos de entrenamiento y prueba
#Se dividen los datos en conjuntos: 80% de los datos se utiliza para entrenamiento y el 20% para pruebas con una semilla que sera fija para controlar la consistencia de los datos.
X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

# Definición del autoencoder
# Clase base para todos los modelos en PyTorch. Permite definir y entrenar modelos de redes neuronales de PyTorch.
class Autoencoder(nn.Module):
    # Constructor de la clase. Recibe como argumento un input, que representa la dimensionalidad de los datos de entrada.
    def __init__(self, input_dim):
        # Inicializar la clase base antes de agregar nuestras propias inicializaciones.
        super(Autoencoder, self).__init__()

        #  Definimos las capas del codificador y del decodificador como secuencias de capas lineales
        # y funciones de activación ReLU para el codificador y el decodificador respectivamente
        # Los números 64, 32 y 16 representan el número de unidades (neuronas) en las capas ocultas del codificador y el decodificador. 
        # Estas son decisiones de diseño que determinan la dimensionalidad de la representación latente en el espacio de características de los datos.
        # La elección de estos números puede variar según la complejidad de los datos y los objetivos específicos del problema.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            #La función de activación sigmoide asegura que las salidas estén en el rango [0, 1].
            nn.Sigmoid())

    # Este método define cómo se propagan los datos a través del modelo.
    def forward(self, x):
        # Los datos de entrada se pasan a través del codificador para obtener la representación comprimida.
        x = self.encoder(x)
        # La representación comprimida se pasa a través del decodificador para obtener la reconstrucción de los datos originales.
        x = self.decoder(x)
        # Devolvemos la salida del decodificador, que es la reconstrucción de los datos originales.
        return x

# Entrenamiento del autoencoder

# Se calcula la dimensionalidad de los datos de entrada tomando la forma del segundo eje de la matriz, que representa el número de características.
input_dim = X_train.shape[1]
# Se instancia el autoencoder utilizando la clase que definimos anteriormente. 
autoencoder = Autoencoder(input_dim)
# Se define la función de pérdida como el error cuadrático medio proporcionada por PyTorch. 
# Calcula la diferencia entre las reconstrucciones del autoencoder y los datos originales.
criterion = nn.MSELoss()
# Se define el optimizador Adam, que es un algoritmo de optimización popular para entrenar redes neuronales. 
# Se pasa los parámetros del autoencoder al optimizador para que pueda ajustar los pesos durante el entrenamiento. Además, se establece una tasa de aprendizaje de 0.001
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
# Se establece el número de épocas de entrenamiento. Se refiere a una pasada completa a través de todos los datos de entrenamiento.
epochs = 20
for epoch in range(epochs):
    #  Se itera sobre cada muestra en el conjunto de datos de entrenamiento
    for data in X_train:
        # Se convierte la muestra actual en un tensor de PyTorch con tipo de datos float32.
        data = torch.tensor(data, dtype=torch.float32)
        # Se borran los gradientes acumulados en los parámetros del modelo antes de calcular los nuevos gradientes en esta iteración.
        optimizer.zero_grad()
        # Se pasa la muestra al autoencoder para obtener la reconstrucción.
        recon = autoencoder(data)
        # Se calcula la pérdida comparando la reconstrucción con los datos originales utilizando la función de pérdida definida anteriormente.
        loss = criterion(recon, data)
        # Calcular los gradientes de la pérdida con respecto a los parámetros del modelo.
        loss.backward()
        # Se actualizan los parámetros del modelo utilizando el algoritmo de optimización para minimizar la pérdida.
        optimizer.step()
    # Se imprime el progreso del entrenamiento mostrando el número de época actual y la pérdida asociada a esa época.    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Umbral basado en el error de reconstrucción
# Se reconstruyen los datos de prueba utilizando el autoencoder y se calcula el error de reconstrucción para cada muestra. 
# Este error representa la discrepancia entre la entrada original y su reconstrucción por parte del autoencoder.
with torch.no_grad():
    reconstructions = [autoencoder(torch.tensor(x, dtype=torch.float32)).numpy() for x in X_test]
    reconstruction_errors = [np.mean(np.square(x - recon)) for x, recon in zip(X_test, reconstructions)]

# Se calcula un umbral basado en la media y la desviación estándar de los errores de reconstrucción. 
# Las muestras cuyo error de reconstrucción supera este umbral se consideran anomalías.
threshold = np.mean(reconstruction_errors) + np.std(reconstruction_errors)

# Visualización de resultados
# Se visualiza la distribución de los errores de reconstrucción mediante un histograma. 
# Se traza una línea punteada en el umbral calculado para identificar visualmente las anomalías. 
# Esto ayuda a comprender cómo se distribuyen los errores y cómo se establece el umbral de detección de anomalías.
plt.hist(reconstruction_errors, bins=50)
plt.axvline(x=threshold, color='r', linestyle='--', label='Umbral')
plt.xlabel('Error de Reconstrucción')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores de Reconstrucción')
plt.legend()
plt.show()

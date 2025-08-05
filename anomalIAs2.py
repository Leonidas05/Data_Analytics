import pandas as pd
import torch
import torch.nn as nn
import random

# Paso 1: Cargar el modelo entrenado
# Aquí cargarías tu modelo de IA previamente entrenado para la detección de anomalías
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Definir arquitectura del autoencoder
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
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Paso 2: Función para leer datos en tiempo real desde un archivo CSV
def leer_datos_en_tiempo_real(ruta_archivo):
    try:
        data = pd.read_csv(ruta_archivo)
        return data
    except FileNotFoundError:
        print(f"Error: Archivo '{ruta_archivo}' no encontrado.")
        return None

# Paso 3: Función para detectar anomalías con el modelo de IA
def detectar_anomalias(datos, modelo):
    # Convertir los datos a tensores de PyTorch
    datos_tensor = torch.tensor(datos.values, dtype=torch.float32)
    
    # Pasar los datos a través del modelo de IA
    reconstrucciones = modelo(datos_tensor)
    
    # Calcular el error de reconstrucción entre los datos originales y reconstruidos
    errores_reconstruccion = torch.mean((datos_tensor - reconstrucciones)**2, dim=1)
    
    # Establecer un umbral de error (puedes ajustar este valor según tu modelo y datos)
    umbral = 0.1
    
    # Determinar si cada muestra supera el umbral de error
    anomalias_detectadas = errores_reconstruccion > umbral
    
    return anomalias_detectadas

# Paso 4: Función para enviar una advertencia en caso de detectar una anomalía
def enviar_advertencia():
    # Aquí implementarías el código para enviar una advertencia
    # Puedes utilizar notificaciones en pantalla, enviar un correo electrónico, enviar un mensaje de texto, etc.
    print("¡Se ha detectado una anomalía!")

# Paso 5: Cargar el modelo de IA previamente entrenado
# Reemplaza 'modelo_entrenado.pt' con la ruta de tu modelo de IA entrenado
ruta_modelo = 'modelo_entrenado.pt'
modelo = torch.load(ruta_modelo)

# Configuración del archivo CSV para leer datos en tiempo real
ruta_archivo_csv = 'RT_IOT2022.csv'

# Ciclo principal para monitorear la actividad en tiempo real
while True:
    # Leer datos en tiempo real desde el archivo CSV
    datos_en_tiempo_real = leer_datos_en_tiempo_real(ruta_archivo_csv)
    
    # Verificar si los datos se han leído correctamente
    if datos_en_tiempo_real is not None:
        # Detectar anomalías utilizando el modelo de IA
        anomalias_detectadas = detectar_anomalias(datos_en_tiempo_real, modelo)
        
        # Verificar si se detectaron anomalías
        if anomalias_detectadas.any():
            # Si se detecta una anomalía, enviar una advertencia
            enviar_advertencia()
        else:
            print("No se han detectado anomalías en los datos en tiempo real.")
    else:
        # Si no se pueden leer los datos, no se puede realizar la detección de anomalías
        print("No se puede realizar la detección de anomalías debido a errores de lectura de datos.")

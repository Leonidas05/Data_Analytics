import torch
import torch.nn as nn
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import scipy.io.arff as arff 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar y preprocesar los datos
archivo_entrenamiento = 'ECG5000_TRAIN.arff'
archivo_prueba = 'ECG5000_TEST.arff'

datos_entrenamiento, meta_entrenamiento = arff.loadarff(archivo_entrenamiento)
datos_prueba, meta_prueba = arff.loadarff(archivo_prueba)
entrenamiento = pd.DataFrame(datos_entrenamiento, columns=meta_entrenamiento.names())
prueba = pd.DataFrame(datos_prueba, columns=meta_prueba.names())
df = pd.concat([entrenamiento, prueba])

nuevas_columnas = list(df.columns)
nuevas_columnas[-1] = 'objetivo'
df.columns = nuevas_columnas
df_normal = df[df.objetivo == b'1']
df_anormal = df[df.objetivo != b'1']

print(entrenamiento.shape)
print(prueba.shape)
print(df.shape)

print(df.head())

fig = plt.figure(figsize=(12, 6))

plt.plot(df_normal.iloc[0, :-1])
plt.plot(df_normal.iloc[10, :-1])
plt.plot(df_normal.iloc[200, :-1])

plt.show()

fig = plt.figure(figsize=(12, 6))

plt.plot(df_anormal.iloc[0, :-1])
plt.plot(df_anormal.iloc[10, :-1])
plt.plot(df_anormal.iloc[200, :-1])

plt.show()

print(df_normal.shape)
print(df_anormal.shape)

# Información sobre los conjuntos de datos
print(f"Conjunto de datos normal: {df_normal.shape[0]}")
print(f"Conjunto de datos anormal: {df_anormal.shape[0]}")

tamaño_conjunto_datos = len(df_normal)
división = int(np.floor(0.15 * tamaño_conjunto_datos))
indices_entrenamiento, indices_validación = train_test_split(np.arange(tamaño_conjunto_datos), test_size=división, random_state=42)
indices_entrenamiento, indices_prueba = train_test_split(indices_entrenamiento, test_size=división, random_state=42)

print(f"Tamaño del conjunto de datos: {tamaño_conjunto_datos}")
print(f"División: {división}")
print(f"Indices de entrenamiento: {len(indices_entrenamiento)}")
print(f"Indices de validación: {len(indices_validación)}")
print(f"Indices de prueba: {len(indices_prueba)}")

# Definir la clase del conjunto de datos (Dataset)
class ECG5000(Dataset):
    def __init__(self, modo):
        assert modo in ['normal', 'anormal']
        df = pd.concat([entrenamiento, prueba])
        nuevas_columnas = list(df.columns)
        nuevas_columnas[-1] = 'objetivo'
        df.columns = nuevas_columnas

        if modo == 'normal':
            df = df[df.objetivo == b'1'].drop(labels='objetivo', axis=1)
        else:
            df = df[df.objetivo != b'1'].drop(labels='objetivo', axis=1)
        self.X = df.astype(np.float32).to_numpy()

    def obtener_tensor_torch(self):
        return torch.from_numpy(self.X)

    def __getitem__(self, índice):
        return torch.from_numpy(self.X[índice]).reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

# Definir la arquitectura del modelo
class Codificador(nn.Module):
    def __init__(self, longitud_secuencia, n_características, dim_embedding=64):
        super(Codificador, self).__init__()
        self.longitud_secuencia, self.n_características = longitud_secuencia, n_características
        self.dim_embedding, self.dim_oculta = dim_embedding, 2 * dim_embedding
        self.rnn1 = nn.LSTM(
            input_size=n_características,
            hidden_size=self.dim_oculta,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.dim_oculta,
            hidden_size=dim_embedding,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        tamaño_lote = x.shape[0]
        x = x.reshape((tamaño_lote, self.longitud_secuencia, self.n_características))
        x, (_, _) = self.rnn1(x)
        x, (oculta_n, _) = self.rnn2(x)
        return oculta_n.reshape((tamaño_lote, self.dim_embedding))

class Decodificador(nn.Module):
    def __init__(self, longitud_secuencia, dim_entrada=64, n_características=1):
        super(Decodificador, self).__init__()
        self.longitud_secuencia, self.dim_entrada = longitud_secuencia, dim_entrada
        self.dim_oculta, self.n_características = 2 * dim_entrada, n_características
        self.rnn1 = nn.LSTM(
            input_size=dim_entrada,
            hidden_size=dim_entrada,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=dim_entrada,
            hidden_size=self.dim_oculta,
            num_layers=1,
            batch_first=True
        )
        self.capa_salida = nn.Linear(self.dim_oculta, n_características)

    def forward(self, x):
        tamaño_lote = x.shape[0]
        x = x.repeat(self.longitud_secuencia, self.n_características)
        x = x.reshape((tamaño_lote, self.longitud_secuencia, self.dim_entrada))
        x, (oculta_n, celda_n) = self.rnn1(x)
        x, (oculta_n, celda_n) = self.rnn2(x)
        x = x.reshape((tamaño_lote, self.longitud_secuencia, self.dim_oculta))
        return self.capa_salida(x)

class AutoencoderRecurrente(nn.Module):
    def __init__(self, longitud_secuencia, n_características, dim_embedding=64, dispositivo='cuda', tamaño_lote=32):
        super(AutoencoderRecurrente, self).__init__()
        self.codificador = Codificador(longitud_secuencia, n_características, dim_embedding).to(dispositivo)
        self.decodificador = Decodificador(longitud_secuencia, dim_embedding, n_características).to(dispositivo)

    def forward(self, x):
        x = self.codificador(x)
        x = self.decodificador(x)
        return x

# Entrenamiento del modelo
conjunto_normal = ECG5000(modo='normal')
conjunto_anormal = ECG5000(modo='anormal')

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

longitud_secuencia, n_características = 140, 1
tamaño_lote = 512

modelo = AutoencoderRecurrente(longitud_secuencia, n_características=n_características, dim_embedding=128, dispositivo=dispositivo, tamaño_lote=tamaño_lote)

cargador_entrenamiento = torch.utils.data.DataLoader(conjunto_normal, batch_size=tamaño_lote, sampler=SubsetRandomSampler(indices_entrenamiento))
cargador_validación = torch.utils.data.DataLoader(conjunto_normal, batch_size=tamaño_lote, sampler=SubsetRandomSampler(indices_validación))
cargador_prueba = torch.utils.data.DataLoader(conjunto_normal, batch_size=tamaño_lote, sampler=SubsetRandomSampler(indices_prueba))
cargador_anomalía = torch.utils.data.DataLoader(conjunto_anormal, batch_size=tamaño_lote)

n_epocas = 20
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
criterio = nn.MSELoss(reduction='mean').to(dispositivo)
historia = dict(entrenamiento=[], validación=[])
mejores_pesos_modelo = copy.deepcopy(modelo.state_dict())
mejor_pérdida = 10000.0

for época in tqdm(range(1, n_epocas + 1)):
    modelo = modelo.train()

    pérdidas_entrenamiento = []
    pérdidas_validación = []
    pérdidas_prueba = []
    pérdidas_anomalía = []

    for i, secuencia_verdadera in enumerate(cargador_entrenamiento):
        optimizador.zero_grad()
        secuencia_verdadera = secuencia_verdadera.to(dispositivo)
        secuencia_predicha = modelo(secuencia_verdadera)
        pérdida = criterio(secuencia_predicha, secuencia_verdadera)
        pérdida.backward()
        optimizador.step()
        pérdidas_entrenamiento.append(pérdida.item())

    modelo = modelo.eval()
    with torch.no_grad():

        for i, secuencia_verdadera in enumerate(cargador_validación):
            secuencia_verdadera = secuencia_verdadera.to(dispositivo)
            secuencia_predicha = modelo(secuencia_verdadera)
            pérdida = criterio(secuencia_predicha, secuencia_verdadera)
            pérdidas_validación.append(pérdida.item())

        for i, secuencia_verdadera in enumerate(cargador_prueba):
            secuencia_verdadera = secuencia_verdadera.to(dispositivo)
            secuencia_predicha = modelo(secuencia_verdadera)
            pérdida = criterio(secuencia_predicha, secuencia_verdadera)
            pérdidas_prueba.append(pérdida.item())

        for i, secuencia_verdadera in enumerate(cargador_anomalía):
            secuencia_verdadera = secuencia_verdadera.to(dispositivo)
            secuencia_predicha = modelo(secuencia_verdadera)
            pérdida = criterio(secuencia_predicha, secuencia_verdadera)
            pérdidas_anomalía.append(pérdida.item())

    pérdida_entrenamiento = np.mean(pérdidas_entrenamiento)
    pérdida_validación = np.mean(pérdidas_validación)
    pérdida_prueba = np.mean(pérdidas_prueba)
    pérdida_anomalía = np.mean(pérdidas_anomalía)
    historia['entrenamiento'].append(pérdida_entrenamiento)
    if época % 10 == 0:
        print(f'Época {época}: pérdida de entrenamiento {pérdida_entrenamiento} {" "*6} pérdida de validación {pérdida_validación} {" "*6} pérdida de prueba {pérdida_prueba} {" "*6} pérdida de anomalía {pérdida_anomalía}')

modelo.load_state_dict(mejores_pesos_modelo)

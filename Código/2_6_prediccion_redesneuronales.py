import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

"""
Long Short-Term Memory (LSTM) o Memoria a Largo-Corto Plazo es un tipo de red neuronal recurrente (RNN).
Está diseñada para aprender dependencias a largo plazo en secuencias de datos.

Se usa para trabajar con datos que vienen en sucuencias. Puede recodar información durante un largo periodo de tiempo.

LSTM es una técnica de IA que se utiliza para analizar y hacer predicciones basadas en datos
que cambian con el tiempo, como los precios de las viviendas.

Este modelo es especialmente bueno para recordar información importante durante largos periodos de tiempo.
"""

"""
Este modelo se usa para predecir el precio de las viviendas en el futuro basado en datos históricos
como el precio anterior, la tasa de paro (desempleo) y una categoría descriptiva.
"""

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

# PREPARACIÓN DE LOS DATOS.

    ## Convierto la columna de fecha a datetime.

data['fecha'] = pd.to_datetime(data['fecha'], format='%m-%Y')

    ## Ordeno los datos por fecha.

data = data.sort_values('fecha')

    ## Seleccionp las columnas relevantes.

features = ['precio_m2', 'tasa_paro', 'categoria']
data_selected = data[features]

    ## Convierto la columna 'categoria' en variables dummy.

data_selected = pd.get_dummies(data_selected, columns=['categoria'], drop_first=True)

    ## Normalizo los datos (excluyendo las columnas dummy generadas).

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_selected)

    ## Convierto de nuevo a DataFrame.

data_scaled = pd.DataFrame(data_scaled, columns=data_selected.columns)

# CREAR SECUENCIAS DE DATOS.

    ## Creo secuencias de entrada y salida.

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data.iloc[i-seq_length:i].values)
        y.append(data.iloc[i, 0])
    return np.array(x), np.array(y)

SEQ_LENGTH = 120  # Usaremos 120 meses de historial para predecir
x, y = create_sequences(data_scaled, SEQ_LENGTH)

# DIVISIÓN DE LOS DATOS.

    ## Divido los datos en conjuntos de entrenamiento y prueba.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# DEFINO EL MODELO LSTM UTILIZANDO KERAS.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQ_LENGTH, x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

"""
Aquí he creado un modelo con dos capas LSTM con 50 unidades cada una, y capas de dropout para evitar el sobreajuste.
La última capa es una capa densa que produce la predicción final.
"""

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ENTRENAMIENTO DEL MODELO.

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

"""
Entrena el modelo con los datos de entrenamiento durante 50 ciclos (épocas)
para que pueda aprender los patrones en los datos.
"""

# EVALUACIÓN DEL MODELO.

loss = model.evaluate(x_test, y_test)
print(f'Loss: {loss}')

"""
Prueba el modelo con los datos de prueba para ver cuán bien predice los precios.
Imprime el valor de pérdida, que indica la precisión del modelo: un valor más bajo significa mejor precisión.
"""

# PREDICCIONES DEL MODELO.

y_pred = model.predict(x_test)

"""
Usa el modelo para predecir los precios en el conjunto de prueba.
"""

# DESNORMALIZACIÓN DE LAS PREDICCIONES.

y_test_scaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], data_selected.shape[1]-1))), axis=1))[:, 0]
y_pred_scaled = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((y_pred.shape[0], data_selected.shape[1]-1))), axis=1))[:, 0]

"""
Convierte las predicciones y los valores reales de vuelta a su escala original para que sean comprensibles.
"""

# VISUALIZACIÓN DE RESULTADOS.

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test_scaled, color='blue', label='Precios reales')
plt.plot(y_pred_scaled, color='red', label='Predicciones')
plt.title('Precios reales vs Precios predichos')
plt.xlabel('Tiempo')
plt.ylabel('Precio')
plt.legend()
plt.show()

"""
Loss: 0.021775271743535995
"""
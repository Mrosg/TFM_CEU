import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cargar el archivo
ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

# Convertir la columna de fecha a datetime
data['fecha'] = pd.to_datetime(data['fecha'], format='%m-%Y')

# Ordenar los datos por fecha
data = data.sort_values('fecha')

# Seleccionar las columnas relevantes incluyendo la tasa de emancipacion
features = ['precio_m2', 'tasa_paro', 'categoria', 'tasa_emancipacion']
data_selected = data[features]

# Convertir la columna 'categoria' en variables dummy
data_selected = pd.get_dummies(data_selected, columns=['categoria'], drop_first=True)

# Normalizar los datos (excluyendo las columnas dummy generadas)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_selected)

# Convertir de nuevo a DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=data_selected.columns)

# Crear secuencias de datos
def create_sequences(data, seq_length, target_col):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data.iloc[i-seq_length:i].values)
        y.append(data.iloc[i, data.columns.get_loc(target_col)])  # Predicción de la variable objetivo
    return np.array(x), np.array(y)

SEQ_LENGTH = 400  # Usaremos 400 registros de historial para predecir
x, y = create_sequences(data_scaled, SEQ_LENGTH, 'tasa_emancipacion')

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Definir el modelo LSTM utilizando Keras
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQ_LENGTH, x_train.shape[2])))
model.add(Dropout(0.5))
model.add(LSTM(units=50))
model.add(Dropout(0.5))
model.add(Dense(1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluación del modelo
loss = model.evaluate(x_test, y_test)
print(f'Loss: {loss}')

# Predicciones del modelo
y_pred = model.predict(x_test)

# Desnormalización de las predicciones
scaler_tasa = MinMaxScaler()
scaler_tasa.fit(data[['tasa_emancipacion']])

y_test_scaled = scaler_tasa.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_scaled = scaler_tasa.inverse_transform(y_pred).flatten()

# Calcular RMSE y MAE
rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))
mae = mean_absolute_error(y_test_scaled, y_pred_scaled)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.plot(y_test_scaled, color='blue', label='Tasa de Emancipación Real')
plt.plot(y_pred_scaled, color='red', label='Predicciones')
plt.title('Tasa de Emancipación Real vs Predicciones')
plt.xlabel('Tiempo')
plt.ylabel('Tasa de Emancipación')
plt.legend()
plt.show()

# Función de predicción con nuevos datos
def predecir_tasa_emancipacion(nuevo_precio_m2, nueva_tasa_paro, nueva_categoria):
    nuevos_datos = {
        'precio_m2': [nuevo_precio_m2],
        'tasa_paro': [nueva_tasa_paro],
        'tasa_emancipacion': [0]  # Placeholder for normalization
    }
    
    nuevas_categorias = pd.DataFrame({'categoria': [nueva_categoria]})
    nuevas_categorias = pd.get_dummies(nuevas_categorias, columns=['categoria'])

    # Asegurar que las nuevas categorías tienen las mismas columnas que las de entrenamiento
    categorias_entrenamiento = [col for col in data_selected.columns if 'categoria_' in col]
    for col in categorias_entrenamiento:
        if col not in nuevas_categorias.columns:
            nuevas_categorias[col] = 0
    nuevas_categorias = nuevas_categorias[categorias_entrenamiento]

    nuevos_datos = pd.concat([pd.DataFrame(nuevos_datos), nuevas_categorias], axis=1)

    # Normalizar los nuevos datos utilizando el mismo scaler
    nuevos_datos_scaled = scaler.transform(nuevos_datos)

    # Obtener las últimas (SEQ_LENGTH-1) filas de los datos de entrenamiento para formar la secuencia
    ultima_secuencia = data_scaled.iloc[-(SEQ_LENGTH-1):].values
    nueva_secuencia = np.vstack([ultima_secuencia, nuevos_datos_scaled])

    nueva_secuencia = nueva_secuencia.reshape((1, SEQ_LENGTH, nueva_secuencia.shape[1]))
    nueva_prediccion = model.predict(nueva_secuencia)

    nueva_prediccion_desnormalizada = scaler_tasa.inverse_transform(nueva_prediccion).flatten()

    return nueva_prediccion_desnormalizada[0]

# Ejemplo de uso de la función de predicción
nuevo_precio_m2 = 14.7
nueva_tasa_paro = 4.27
nueva_categoria = 'categoria_Medio-Alto'

prediccion = predecir_tasa_emancipacion(nuevo_precio_m2, nueva_tasa_paro, nueva_categoria)
print(f'Predicción de la tasa de emancipación para precio_m2={nuevo_precio_m2}, tasa_paro={nueva_tasa_paro} y categoria={nueva_categoria}: {prediccion:.4f}')
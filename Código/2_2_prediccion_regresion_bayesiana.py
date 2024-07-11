import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold
from sklearn.linear_model import BayesianRidge
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

# MODELO DE REGRESIÓN BAYESIANA SIN CATEGORÍA.

    ## Cargo mi dataset.

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

    ## Selección de características y variable objetivo

variables_independientes = ["precio_m2", "tasa_paro"]
variable_dependiente = "tasa_emancipacion"

x = data[variables_independientes].values
y = data[variable_dependiente].values

    ## Escalo mis datos.

escalar = StandardScaler()
x_scaled = escalar.fit_transform(x)

    ## Divido los datos en entrenamiento y prueba (70% entrenamiento y 30% prueba).

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, random_state = 42)

    ## Defino el modelo de regresión bayesiana.

model = BayesianRidge()

    ## Hago una validación cruzada con 5.000 simulaciones.

n_splits = 10  # Número de pliegues
n_repeats = 500  # Número de repeticiones (10 pliegues x 500 repeticiones = 5.000 simulaciones)
rkf = RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = 42)
cv_scores = cross_val_score(model, x_scaled, y, cv = rkf, scoring = "neg_mean_squared_error")
cv_mse = -cv_scores.mean()
cv_std = cv_scores.std()
print(f"ECM de la validación cruzada: {cv_mse:.4f} ± {cv_std:.4f}")

    ## Entreno el modelo con el conjunto de entrenamiento.

model.fit(x_train, y_train)

    ## Hago las predicciones sobre el conjunto de prueba.

y_pred = model.predict(x_test)

    ## Calculo el MSE (Error Cuadrático Medio).

mse = mean_squared_error(y_test, y_pred)
print(f"ECM: {mse:.4f}")

    ## Hago un dataframe para comparar las predicciones con los valores reales.

resultados = pd.DataFrame({
    "Real": y_test,
    "Predicción": y_pred
})

    ## Extraigo los coeficientes y el intercept del modelo.

intercept = model.intercept_
coef = model.coef_

    ## Obtengo la matriz de covarianza de los coeficientes.

cov_matrix = model.sigma_

    ## Calculo las desviaciones estándar de los coeficientes.

intercept_std = np.sqrt(model.alpha_)  # Desviación estándar del intercept.
coef_std = np.sqrt(np.diag(cov_matrix))  # Desviaciones estándar de los coeficientes.

    ## Hago un dataframe para los intervalos de confianza del modelo.

intervalos_confianza1 = pd.DataFrame({
    "Características": ["Intercept"] + variables_independientes,
    "Coeficiente": [intercept] + list(coef),
    "Límite Inferior": [intercept - 1.96 * intercept_std] + list(coef - 1.96 * coef_std),
    "Límite Superior": [intercept + 1.96 * intercept_std] + list(coef + 1.96 * coef_std)
})

    ## Vemos el resumen del modelo con intervalos de confianza.

print("Resumen del modelo de regresión bayesiana:")
print(intervalos_confianza1)

    ## Visualizo las distribuciones de probabilidad de los coeficientes.

x_vals = np.linspace(-30, 30, 400)
colors = ["blue", "green", "red"]

plt.figure(figsize = (12, 6))

for i, feature in enumerate(["Intercept"] + variables_independientes):
    mean = intervalos_confianza1["Coeficiente"].iloc[i]
    std = intercept_std if i == 0 else coef_std[i - 1]
    plt.plot(x_vals, stats.norm.pdf(x_vals, mean, std), label = feature, color = colors[i % len(colors)])
    
plt.legend()
plt.title("Distribuciones de probabilidad de los coeficientes del modelo de regresión bayesiana")
plt.xlabel("Valor del coeficiente")
plt.ylabel("Densidad de probabilidad")
plt.show()

    ## Visualizo las predicciones vs los valores reales.

plt.figure(figsize = (12, 6))
plt.plot(resultados["Real"].values, label = "Valores reales", color = "blue")
plt.plot(resultados["Predicción"].values, label = "Predicciones", color = "red", linestyle = "dashed")
plt.legend()
plt.title("Regresión bayesiana: comparación de valores reales y predicciones")
plt.xlabel("Índice")
plt.ylabel("Tasa de emancipación")
plt.show()

    ## Hago un histograma de los errores cuadráticos medios de la validación cruzada.

plt.figure(figsize = (12, 6))
plt.hist(-cv_scores, bins = 30, edgecolor = "black", alpha = 0.7, color = "turquoise")
plt.title("Distribución de los Errores Cuadráticos Medios en la validación cruzada")
plt.xlabel("Error Cuadrático Medio")
plt.ylabel("Frecuencia")
plt.show()

    ## Desnormalizo los coeficientes para interpretarlos en la escala original.

std_precio_m2 = data["precio_m2"].std()
std_tasa_paro = data["tasa_paro"].std()
std_tasa_emancipacion = data["tasa_emancipacion"].std()

coef_precio_m2_original = coef[0] * (std_precio_m2 / std_tasa_emancipacion)
coef_tasa_paro_original = coef[1] * (std_tasa_paro / std_tasa_emancipacion)

    ## Vemos los coeficientes desnormalizados.

print(f"Coeficiente desnormalizado de precio_m2: {coef_precio_m2_original:.4f}")
print(f"Coeficiente desnormalizado de tasa de paro: {coef_tasa_paro_original:.4f}")

    ## Interpretación del modelo.

print("\nInterpretación del modelo de regresión bayesiana:")
print(f"Intercepto: {intercept:.2f}")
print(f"Coeficiente de precio_m2: {coef_precio_m2_original:.4f} (Por cada incremento de 1 euro en el precio por metro cuadrado, la tasa de emancipación aumenta en aproximadamente 0.0249 puntos porcentuales.)")
print(f"Coeficiente de tasa_paro: {coef_tasa_paro_original:.4f} (Por cada incremento de 1 punto porcentual en la tasa de paro, la tasa de emancipación disminuye en aproximadamente -1.8964 puntos porcentuales.)")

    ## Simulaciones para hacer predicciones con nuevos datos.

def predecir_tasa_emancipacion(nuevo_precio_m2_1, nueva_tasa_paro_1):
        ### Crear el array con los nuevos datos.
    nuevo_dato = np.array([[nuevo_precio_m2_1, nueva_tasa_paro_1]])
        ### Normalizar el nuevo array utilizando el mismo StandardScaler.
    nuevo_dato_normalizado = escalar.transform(nuevo_dato)
        ### Hacer la predicción.
    nueva_prediccion = model.predict(nuevo_dato_normalizado)
    return nueva_prediccion[0]

        ### Ejemplo de uso de la función de predicción
nuevo_precio_m2_1 = 14.7 
nueva_tasa_paro_1 = 4.27 

prediccion_1 = predecir_tasa_emancipacion(nuevo_precio_m2_1, nueva_tasa_paro_1)
print(f"Predicción de la tasa de emancipación para precio_m2 = {nuevo_precio_m2_1} y tasa_paro = {nueva_tasa_paro_1} con una regresión bayesiana: {prediccion_1:.4f}")

"""
MODELO DE REGRESIÓN BAYESIANA. MODELO QUE TIENE EN CUENTA LA INCERTIDUMBRE.

- Intercepto: representa la tasa de emancipación promedio cuando las características (precio_m2 y tasa_paro)
        están en sus valores medios. En este caso, es aproximadamente 50.49%.
- Precio por m2: el coeficiente desnormalizado es 0.0249.
        Esto significa que por cada incremento de 1 euro en el precio por metro cuadrado,
        la tasa de emancipación aumenta en aproximadamente 0.0249 puntos porcentuales.
        Indica una relación positiva entre el precio de la vivienda y la tasa de emancipación, aunque el efecto es pequeño.
- Tasa de paro: el coeficiente es -1.8964. Esto significa que por cada incremento de 1 punto porcentual
        en la tasa de paro, la tasa de emancipación disminuye en aproximadamente 1.8964 puntos porcentuales.
        Indica una relación negativa entre el desempleo y la tasa de emancipación,
        sugiriendo que un mayor desempleo reduce la tasa de emancipación.

ERROR CUADRÁTICO MEDIO (ECM o MSE).

- El Error Cuadrático Medio es una medida para ver la precisión de las predicciones del modelo. Cuanto más bajo, mayor precisión.
- El ECM de este modelo es 3.8829 en la validazión cruzada y de 4.1579 en el conjunto de prueba:
las predicciones se acercan bastante a los valores reales.
- El histograma de ECM muestra estos valores y su frecuencia en las 20.000 simulaciones hechas con validación cruzada.

INTERVALOS DE CONFIANZA DEL MODELO.

- Intercepto: 50.485218
	•	Coeficiente: 50.485218
	•	Representa la tasa de emancipación promedio cuando las características precio_m2 y tasa_paro están en sus valores medios.
	•	Intervalo de Confianza: [49.472278, 51.498157]. Estamos un 95% seguros de que el valor verdadero del intercepto
        está dentro de este rango.

- precio_m2: 0.173302
	•	Coeficiente: 0.173302
	•	Por cada incremento de 1 euro en el precio por metro cuadrado, la tasa de emancipación aumenta
        en aproximadamente 0.173 puntos porcentuales.
	•	Intervalo de Confianza: [0.054661, 0.291943]. Estamos un 95% seguros de que el efecto verdadero
        del precio por metro cuadrado en la tasa de emancipación está dentro de este rango.
	•	El coeficiente positivo indica una relación positiva entre el precio de la vivienda y la tasa de emancipación.
    
- tasa_paro: -20.154752
	•	Coeficiente: -20.154752
	•	Por cada incremento de 1 punto porcentual en la tasa de paro, la tasa de emancipación disminuye en aproximadamente 20.15 puntos porcentuales.
	•	Intervalo de Confianza: [-20.273484, -20.036019]. Estamos un 95% seguros de que el efecto verdadero
        de la tasa de paro en la tasa de emancipación está dentro de este rango.
	•	El coeficiente negativo indica una fuerte relación negativa entre el desempleo y la tasa de emancipación.
"""

# MODELO DE REGRESIÓN BAYESIANA CON CATEGORÍA.

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

# Convertir la variable "categoria" en variables dummy (one-hot encoding)
data = pd.get_dummies(data, columns=["categoria"])

# Seleccionar las características y la variable objetivo
variables_independientes = ["precio_m2", "tasa_paro"] + [col for col in data.columns if "categoria_" in col]
variable_dependiente = "tasa_emancipacion"

x = data[variables_independientes].values
y = data[variable_dependiente].values

# Normalizar los datos
escalar = StandardScaler()
x_scaled = escalar.fit_transform(x)

# Dividir los datos en entrenamiento y prueba (70% entrenamiento y 30% prueba)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

# Definir el modelo de regresión bayesiana
model = BayesianRidge()

# Validación cruzada con 5.000 simulaciones
n_splits = 10  # Número de pliegues
n_repeats = 500  # Número de repeticiones (10 pliegues x 500 repeticiones = 5.000 simulaciones)
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
cv_scores = cross_val_score(model, x_scaled, y, cv=rkf, scoring="neg_mean_squared_error")
cv_mse = -cv_scores.mean()
cv_std = cv_scores.std()
print(f"Cross-Validated MSE: {cv_mse:.4f} ± {cv_std:.4f}")

# Entrenar el modelo con el conjunto de entrenamiento
model.fit(x_train, y_train)

# Hacer las predicciones sobre el conjunto de prueba
y_pred = model.predict(x_test)

# Calcular el MSE (Error Cuadrático Medio)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

# Hacer un dataframe para comparar las predicciones con los valores reales
resultados = pd.DataFrame({
    "Real": y_test,
    "Predicción": y_pred
})

# Extraer los coeficientes y el intercepto del modelo
intercept = model.intercept_
coef = model.coef_

# Obtener la matriz de covarianza de los coeficientes
cov_matrix = model.sigma_

# Calcular las desviaciones estándar de los coeficientes
intercept_std = np.sqrt(model.alpha_)  # Desviación estándar del intercepto
coef_std = np.sqrt(np.diag(cov_matrix))  # Desviaciones estándar de los coeficientes

# Hacer un dataframe para los intervalos de confianza del modelo
intervalos_confianza = pd.DataFrame({
    "Características": ["Intercept"] + variables_independientes,
    "Coeficiente": [intercept] + list(coef),
    "Límite Inferior": [intercept - 1.96 * intercept_std] + list(coef - 1.96 * coef_std),
    "Límite Superior": [intercept + 1.96 * intercept_std] + list(coef + 1.96 * coef_std)
})

# Vemos el resumen del modelo con intervalos de confianza
print("Resumen del modelo de regresión bayesiana:")
print(intervalos_confianza)

# Visualizar las distribuciones de probabilidad de los coeficientes
x_vals = np.linspace(-30, 30, 400)
colors = ["blue", "green", "red", "purple", "orange", "cyan", "magenta"]

plt.figure(figsize=(12, 6))
for i, feature in enumerate(["Intercept"] + variables_independientes):
    mean = intervalos_confianza["Coeficiente"].iloc[i]
    std = intercept_std if i == 0 else coef_std[i - 1]
    plt.plot(x_vals, stats.norm.pdf(x_vals, mean, std), label=feature, color=colors[i % len(colors)])
plt.legend()
plt.title("Distribuciones de probabilidad de los coeficientes del modelo de regresión bayesiana")
plt.xlabel("Valor del Coeficiente")
plt.ylabel("Densidad de Probabilidad")
plt.show()

# Visualizar las predicciones vs los valores reales
plt.figure(figsize=(12, 6))
plt.plot(resultados["Real"].values, label="Valores reales", color="blue")
plt.plot(resultados["Predicción"].values, label="Predicciones", color="red", linestyle="dashed")
plt.legend()
plt.title("Regresión bayesiana: comparación de valores reales y predicciones")
plt.xlabel("Índice")
plt.ylabel("Tasa de emancipación")
plt.show()

# Histograma de los errores cuadráticos medios de la validación cruzada
plt.figure(figsize=(12, 6))
plt.hist(-cv_scores, bins=30, edgecolor="black", alpha=0.7, color="turquoise")
plt.title("Distribución de los Errores Cuadráticos Medios en la validación cruzada")
plt.xlabel("Error Cuadrático Medio")
plt.ylabel("Frecuencia")
plt.show()

# Desnormalizar los coeficientes para interpretarlos en la escala original
std_precio_m2 = data["precio_m2"].std()
std_tasa_paro = data["tasa_paro"].std()
std_tasa_emancipacion = data["tasa_emancipacion"].std()

coef_precio_m2_original = coef[0] * (std_precio_m2 / std_tasa_emancipacion)
coef_tasa_paro_original = coef[1] * (std_tasa_paro / std_tasa_emancipacion)

# Vemos los coeficientes desnormalizados
print(f"Coeficiente desnormalizado de precio_m2: {coef_precio_m2_original:.4f}")
print(f"Coeficiente desnormalizado de tasa de paro: {coef_tasa_paro_original:.4f}")

# Interpretación del modelo
print("\nInterpretación del modelo de regresión bayesiana:")
print(f"Intercepto: {intercept:.2f}")
print(f"Coeficiente de precio_m2: {coef_precio_m2_original:.4f} (Por cada incremento de 1 euro en el precio por metro cuadrado, la tasa de emancipación aumenta en aproximadamente {coef_precio_m2_original:.4f} puntos porcentuales.)")
print(f"Coeficiente de tasa_paro: {coef_tasa_paro_original:.4f} (Por cada incremento de 1 punto porcentual en la tasa de paro, la tasa de emancipación disminuye en aproximadamente {coef_tasa_paro_original:.4f} puntos porcentuales.)")

# Simulaciones para hacer predicciones con nuevos datos
def predecir_tasa_emancipacion(nuevo_precio_m2_2, nueva_tasa_paro_2, nueva_categoria_2):
    # Crear el array con los nuevos datos
    nueva_categoria_dummies = [1 if f"categoria_{nueva_categoria_2}" in col else 0 for col in variables_independientes[2:]]
    nuevo_dato = np.array([[nuevo_precio_m2_2, nueva_tasa_paro_2] + nueva_categoria_dummies])
    
    # Normalizar el nuevo array utilizando el mismo StandardScaler
    nuevo_dato_normalizado = escalar.transform(nuevo_dato)
    
    # Hacer la predicción
    nueva_prediccion = model.predict(nuevo_dato_normalizado)
    
    return nueva_prediccion[0]

# Ejemplo de uso de la función de predicción
nuevo_precio_m2_2 = 14.7  # Ejemplo de nuevo valor para precio por metro cuadrado
nueva_tasa_paro_2 = 4.27  # Ejemplo de nuevo valor para tasa de paro
nueva_categoria_2 = 'Medio-Alto'  # Ejemplo de nueva categoría

prediccion_2 = predecir_tasa_emancipacion(nuevo_precio_m2_2, nueva_tasa_paro_2, nueva_categoria_2)
print(f"Predicción de la tasa de emancipación para precio_m2 = {nuevo_precio_m2_2}, tasa_paro = {nueva_tasa_paro_2} y categoria = {nueva_categoria_2} con una regresión bayesiana: {prediccion_2:.4f}")

"""
Coeficiente desnormalizado de precio_m2: -0.1301
Coeficiente desnormalizado de tasa de paro: -1.9024

Intercepto: 50.49
Coeficiente de precio_m2: -0.1301 (Por cada incremento de 1 euro en el precio por metro cuadrado,
    la tasa de emancipación aumenta en aproximadamente -0.1301 puntos porcentuales.)
Coeficiente de tasa_paro: -1.9024 (Por cada incremento de 1 punto porcentual en la tasa de paro,
    la tasa de emancipación disminuye en aproximadamente -1.9024 puntos porcentuales.)

Al meter la variable "categoria" el coeficiente de "precio_m2" se vuelve negativo.
Esto puede ser por un problema de multicolinealidad y que "categoría" capture parte de la variabilidad explicada por "precio_m2".
"""

# VER LA CORRELACIÓN ENTRE VARIABLES

# Analizar la correlación entre precio_m2 y las variables categóricas
data_without_fecha = data.drop(columns=["fecha", "distrito"])
correlations = data_without_fecha.corr()

# Seleccionar las variables de interés
corr_precio_m2 = correlations["precio_m2"]
corr_tasa_emancipacion = correlations["tasa_emancipacion"]

# Filtrar las correlaciones con las variables categóricas
corr_precio_m2_categorias = corr_precio_m2.filter(like='categoria_')
corr_tasa_emancipacion_categorias = corr_tasa_emancipacion.filter(like='categoria_')

# Combinar las correlaciones en un solo DataFrame para su visualización
corr_combined = pd.DataFrame({
    "Correlación con precio_m2": corr_precio_m2_categorias,
    "Correlación con tasa_emancipacion": corr_tasa_emancipacion_categorias
})

corr_combined

"""
	                         Correlación con precio_m2       Correlación con tasa_emancipacion
categoria_Alto	             0.776533                        0.458321
categoria_Bajo	             -0.631451	                     -0.616080
categoria_Medio	             -0.079921	                     0.091538
categoria_Medio-Alto	     0.226996	                     0.231048
categoria_Medio-Bajo	    -0.282460	                     -0.156412

Las áreas categorizadas como “Alto” tienden a tener un precio por metro cuadrado más alto.
Las áreas “Bajo” tienen precios por metro cuadrado más bajos.
Las áreas “Alto” tienden a tener tasas de emancipación más altas.
Las áreas “Bajo” tienden a tener tasas de emancipación más bajas.

	1.	Categoria Alto: La tasa de emancipación en la categoría “Alto” es, en promedio, 0.72 puntos porcentuales mayor
        que en la categoría de referencia. Sin embargo, debido a la variabilidad en los datos, este resultado no es concluyente y no podemos estar seguros de que esta diferencia sea significativa.
	2.	Categoria Bajo: En la categoría “Bajo”, la tasa de emancipación es, en promedio, 0.39 puntos porcentuales menor
        que en la categoría de referencia. Al igual que con la categoría “Alto”, esta diferencia no es estadísticamente significativa, lo que significa que no podemos afirmar con certeza que exista una verdadera diferencia.
	3.	Categoria Medio: La categoría “Medio” muestra una tasa de emancipación casi igual a la de la categoría de referencia,
        con una diferencia insignificante de solo -0.07 puntos porcentuales. Esta diferencia tampoco es estadísticamente significativa.
	4.	Categoria Medio-Alto: En la categoría “Medio-Alto”, la tasa de emancipación es, en promedio, 0.31 puntos porcentuales mayor
        que en la categoría de referencia. No obstante, esta diferencia no es significativa desde el punto de vista estadístico, lo que indica que podría ser debida al azar.
	5.	Categoria Medio-Bajo: La categoría “Medio-Bajo” tiene una tasa de emancipación 0.56 puntos porcentuales menor
        que la categoría de referencia. Sin embargo, al igual que las otras categorías,
        esta diferencia no es estadísticamente significativa.
"""
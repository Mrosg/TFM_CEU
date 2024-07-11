import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
import scipy.stats as stats

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

# MODELO 1.

# Aislar las variables numéricas
numericas = data.select_dtypes(include=["float64", "int64"]).columns

# Estandarizar los datos
escalar = StandardScaler()
data_estandar = data.copy()
data_estandar[numericas] = escalar.fit_transform(data[numericas])

# Verificar la estandarización de una variable al azar
media_estandar_preciom2 = data_estandar["precio_m2"].mean()
sd_estandar_preciom2 = data_estandar["precio_m2"].std()

print("La media estandarizada de precio_m2 es: ", media_estandar_preciom2)
print("La desviación típica estandarizada de precio_m2 es: ", sd_estandar_preciom2)

# Normalizar los datos estandarizados
normalizar = MinMaxScaler()
data_normal = data_estandar.copy()
data_normal[numericas] = normalizar.fit_transform(data_estandar[numericas])

## Definir los predictores (x) y la variable dependiente (y)
x = data_normal[["precio_m2", "paro_total", "poblacion"]]
y = data_normal["tasa_emancipacion"]

## Dividir el conjunto de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

## Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(x_train, y_train)

## Hacer predicciones con el conjunto de prueba
y_pred = model.predict(x_test)

## Evaluar el modelo
ecm = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio: {ecm}")
print(f"R-Cuadrado: {r2}")

## Visualizar los resultados
plt.scatter(y_test, y_pred)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m * y_test + b, color='red', label='Recta de Regresión')
plt.title("Regresión lineal con tres variables: valores reales vs predicciones")
plt.show()

## Validación cruzada con 5.000 simulaciones
n_splits = 10  # Número de pliegues
n_repeats = 500  # Número de repeticiones
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Validación cruzada para el modelo de regresión lineal
cv_scores = cross_val_score(model, x, y, cv=rkf, scoring='r2')
print(f"Media de R-Cuadrado en validación cruzada (Regresión Lineal): {cv_scores.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (Regresión Lineal): {cv_scores.std()}")

# Modelos Ridge y Lasso para evitar el sobreajuste

## Modelo de Ridge
modelo_ridge = Ridge(alpha=3)
modelo_ridge.fit(x_train, y_train)
y_ridge = modelo_ridge.predict(x_test)
ecm_ridge = mean_squared_error(y_test, y_ridge)
r2_ridge = r2_score(y_test, y_ridge)

# Validación cruzada para el modelo Ridge
cv_scores_ridge = cross_val_score(modelo_ridge, x, y, cv=rkf, scoring='r2')
print(f"Ridge - Error cuadrático medio: {ecm_ridge}")
print(f"Ridge - R-Cuadrado: {r2_ridge}")
print(f"Media de R-Cuadrado en validación cruzada (Ridge): {cv_scores_ridge.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (Ridge): {cv_scores_ridge.std()}")

## Modelo Lasso
modelo_lasso = Lasso(alpha=3)
modelo_lasso.fit(x_train, y_train)
y_pred_lasso = modelo_lasso.predict(x_test)
ecm_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Validación cruzada para el modelo Lasso
cv_scores_lasso = cross_val_score(modelo_lasso, x, y, cv=rkf, scoring='r2')
print(f"Lasso - Error cuadrático medio: {ecm_lasso}")
print(f"Lasso - R-Cuadrado: {r2_lasso}")
print(f"Media de R-Cuadrado en validación cruzada (Lasso): {cv_scores_lasso.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (Lasso): {cv_scores_lasso.std()}")

## Visualizo de los resultados
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_ridge)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones de Ridge")
plt.title("Regresión de Ridge con tres variables: valores reales vs predicciones")
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lasso)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones de Lasso")
plt.title("Regresión de Lasso con tres variables: valores reales vs predicciones")
plt.show()

# MODELO 2.

# Aislar las variables numéricas
numericas = data.select_dtypes(include=["float64", "int64"]).columns

# Estandarizar los datos
escalar = StandardScaler()
data_estandar = data.copy()
data_estandar[numericas] = escalar.fit_transform(data[numericas])

# Verificar la estandarización de una variable al azar
media_estandar_preciom2 = data_estandar["precio_m2"].mean()
sd_estandar_preciom2 = data_estandar["precio_m2"].std()

print("La media estandarizada de precio_m2 es: ", media_estandar_preciom2)
print("La desviación típica estandarizada de precio_m2 es: ", sd_estandar_preciom2)

# Normalizar los datos estandarizados
normalizar = MinMaxScaler()
data_normal = data_estandar.copy()
data_normal[numericas] = normalizar.fit_transform(data_estandar[numericas])

## Definir los predictores (x) y la variable dependiente (y)
x = data_normal[["precio_m2", "paro_total", "poblacion"]]
y = data_normal["tasa_emancipacion"]

## Dividir el conjunto de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

## Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(x_train, y_train)

## Hacer predicciones con el conjunto de prueba
y_pred = model.predict(x_test)

## Evaluar el modelo
ecm = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio: {ecm}")
print(f"R-Cuadrado: {r2}")

## Visualizar los resultados
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m * y_test + b, color='red', label='Recta de Regresión')
plt.title("Regresión lineal con tres variables: valores reales vs predicciones")
plt.show()

## Validación cruzada con 500 simulaciones
n_splits = 10  # Número de pliegues
n_repeats = 500  # Número de repeticiones
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Validación cruzada para el modelo de regresión lineal
cv_scores = cross_val_score(model, x, y, cv=rkf, scoring='r2')
print(f"Media de R-Cuadrado en validación cruzada (Regresión Lineal): {cv_scores.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (Regresión Lineal): {cv_scores.std()}")

# Modelos Ridge y Lasso para evitar el sobreajuste

## Modelo de Ridge
modelo_ridge = Ridge(alpha=3)
modelo_ridge.fit(x_train, y_train)
y_ridge = modelo_ridge.predict(x_test)
ecm_ridge = mean_squared_error(y_test, y_ridge)
r2_ridge = r2_score(y_test, y_ridge)

# Validación cruzada para el modelo Ridge
cv_scores_ridge = cross_val_score(modelo_ridge, x, y, cv=rkf, scoring='r2')
print(f"Ridge - Error cuadrático medio: {ecm_ridge}")
print(f"Ridge - R-Cuadrado: {r2_ridge}")
print(f"Media de R-Cuadrado en validación cruzada (Ridge): {cv_scores_ridge.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (Ridge): {cv_scores_ridge.std()}")

## Modelo Lasso
modelo_lasso = Lasso(alpha=3)
modelo_lasso.fit(x_train, y_train)
y_pred_lasso = modelo_lasso.predict(x_test)
ecm_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Validación cruzada para el modelo Lasso
cv_scores_lasso = cross_val_score(modelo_lasso, x, y, cv=rkf, scoring='r2')
print(f"Lasso - Error cuadrático medio: {ecm_lasso}")
print(f"Lasso - R-Cuadrado: {r2_lasso}")
print(f"Media de R-Cuadrado en validación cruzada (Lasso): {cv_scores_lasso.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (Lasso): {cv_scores_lasso.std()}")

## Visualizar de los resultados
plt.figure(figsize=(14, 6))

# Regresión de Ridge
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_ridge)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones de Ridge")
plt.title("Ridge con tres variables: valores reales vs predicciones")

# Regresión de Lasso
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lasso)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones de Lasso")
plt.title("Lasso con tres variables: valores reales vs predicciones")

plt.show()

## Calcular los residuos
residuos_lineal = y_test - y_pred
residuos_ridge = y_test - y_ridge
residuos_lasso = y_test - y_pred_lasso

## Visualizar los residuos
plt.figure(figsize=(18, 6))

# Residuals para el modelo de regresión lineal
plt.subplot(1, 3, 1)
plt.scatter(y_pred, residuos_lineal)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicciones")
plt.ylabel("Residuos")
plt.title("Residuos vs. Predicciones (Regresión Lineal)")

# Residuals para el modelo Ridge
plt.subplot(1, 3, 2)
plt.scatter(y_ridge, residuos_ridge)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicciones de Ridge")
plt.ylabel("Residuos")
plt.title("Residuos vs. Predicciones (Ridge)")

# Residuals para el modelo Lasso
plt.subplot(1, 3, 3)
plt.scatter(y_pred_lasso, residuos_lasso)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicciones de Lasso")
plt.ylabel("Residuos")
plt.title("Residuos vs. Predicciones (Lasso)")

plt.show()

# MODELO TRES.

# MODELO TRES.

# Preparar los datos con "categoria"
x_with_categoria = pd.get_dummies(data[['precio_m2', 'tasa_paro', 'categoria']], drop_first=True)
y = data['tasa_emancipacion']

# Dividir los datos
x_train_with_cat, x_test_with_cat, y_train, y_test = train_test_split(x_with_categoria, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal con la variable "categoria"
model_with_categoria = LinearRegression().fit(x_train_with_cat, y_train)
y_pred_with_categoria = model_with_categoria.predict(x_test_with_cat)
ecm_with_categoria = mean_squared_error(y_test, y_pred_with_categoria)
r2_with_categoria = r2_score(y_test, y_pred_with_categoria)

# Entrenar el modelo Ridge con la variable "categoria"
ridge_model = Ridge(alpha=3)
ridge_model.fit(x_train_with_cat, y_train)
y_pred_ridge = ridge_model.predict(x_test_with_cat)
ecm_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Entrenar el modelo Lasso con la variable "categoria"
lasso_model = Lasso(alpha=3)
lasso_model.fit(x_train_with_cat, y_train)
y_pred_lasso = lasso_model.predict(x_test_with_cat)
ecm_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Resultados
print(f"Modelo con 'categoria': ECM = {ecm_with_categoria}, R2 = {r2_with_categoria}")
print(f"Modelo Ridge: ECM = {ecm_ridge}, R2 = {r2_ridge}")
print(f"Modelo Lasso: ECM = {ecm_lasso}, R2 = {r2_lasso}")

# Validación cruzada con 5.000 simulaciones
n_splits = 10  # Número de pliegues
n_repeats = 500
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Validación cruzada para el modelo con "categoria"
cv_scores_with_cat = cross_val_score(model_with_categoria, x_with_categoria, y, cv=rkf, scoring='r2')
print(f"Media de R-Cuadrado en validación cruzada (con 'categoria'): {cv_scores_with_cat.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (con 'categoria'): {cv_scores_with_cat.std()}")

# Validación cruzada para el modelo Ridge
cv_scores_ridge = cross_val_score(ridge_model, x_with_categoria, y, cv=rkf, scoring='r2')
print(f"Media de R-Cuadrado en validación cruzada (Ridge): {cv_scores_ridge.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (Ridge): {cv_scores_ridge.std()}")

# Validación cruzada para el modelo Lasso
cv_scores_lasso = cross_val_score(lasso_model, x_with_categoria, y, cv=rkf, scoring='r2')
print(f"Media de R-Cuadrado en validación cruzada (Lasso): {cv_scores_lasso.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (Lasso): {cv_scores_lasso.std()}")

# Visualizar los resultados de la regresión lineal
plt.figure(figsize=(14, 6))

# Gráfico de regresión lineal con "categoria"
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_with_categoria)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Regresión lineal con 'categoría': valores reales vs predicciones")
m_with_cat, b_with_cat = np.polyfit(y_test, y_pred_with_categoria, 1)
plt.plot(y_test, m_with_cat * y_test + b_with_cat, color='red')
plt.legend()

# Calcular los residuos
residuos_with_categoria = y_test - y_pred_with_categoria
residuos_ridge = y_test - y_pred_ridge
residuos_lasso = y_test - y_pred_lasso

# Visualizar los residuos
plt.figure(figsize=(18, 6))

# Residuos para el modelo de regresión lineal
plt.subplot(1, 3, 1)
plt.scatter(y_pred_with_categoria, residuos_with_categoria)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicciones")
plt.ylabel("Residuos")
plt.title("Residuos vs. Predicciones (Regresión Lineal)")

# Residuos para el modelo Ridge
plt.subplot(1, 3, 2)
plt.scatter(y_pred_ridge, residuos_ridge)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicciones de Ridge")
plt.ylabel("Residuos")
plt.title("Residuos vs. Predicciones (Ridge)")

# Residuos para el modelo Lasso
plt.subplot(1, 3, 3)
plt.scatter(y_pred_lasso, residuos_lasso)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicciones de Lasso")
plt.ylabel("Residuos")
plt.title("Residuos vs. Predicciones (Lasso)")

plt.show()

# Calcular los residuos
residuos_with_categoria = y_test - y_pred_with_categoria
residuos_ridge = y_test - y_pred_ridge
residuos_lasso = y_test - y_pred_lasso

# Histograma de Residuos
plt.figure(figsize=(18, 6))

# Histograma de residuos (Regresión Lineal)
plt.subplot(1, 3, 1)
plt.hist(residuos_with_categoria, bins=30, edgecolor='k')
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.title("Histograma de Residuos (Regresión Lineal)")

# Histograma de residuos (Ridge)
plt.subplot(1, 3, 2)
plt.hist(residuos_ridge, bins=30, edgecolor='k')
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.title("Histograma de Residuos (Ridge)")

# Histograma de residuos (Lasso)
plt.subplot(1, 3, 3)
plt.hist(residuos_lasso, bins=30, edgecolor='k')
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.title("Histograma de Residuos (Lasso)")

plt.show()

# QQ-Plot de Residuos
import scipy.stats as stats

plt.figure(figsize=(18, 6))

# QQ-Plot de residuos (Regresión Lineal)
plt.subplot(1, 3, 1)
stats.probplot(residuos_with_categoria, dist="norm", plot=plt)
plt.title("QQ-Plot de Residuos (Regresión Lineal)")

# QQ-Plot de residuos (Ridge)
plt.subplot(1, 3, 2)
stats.probplot(residuos_ridge, dist="norm", plot=plt)
plt.title("QQ-Plot de Residuos (Ridge)")

# QQ-Plot de residuos (Lasso)
plt.subplot(1, 3, 3)
stats.probplot(residuos_lasso, dist="norm", plot=plt)
plt.title("QQ-Plot de Residuos (Lasso)")

plt.show()

# Residuos vs. Predictores
plt.figure(figsize=(18, 6))

# Residuos vs. precio_m2 (Regresión Lineal)
plt.subplot(1, 3, 1)
plt.scatter(x_test_with_cat['precio_m2'], residuos_with_categoria)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("precio_m2")
plt.ylabel("Residuos")
plt.title("Residuos vs. precio_m2 (Regresión Lineal)")

# Residuos vs. tasa_paro (Ridge)
plt.subplot(1, 3, 2)
plt.scatter(x_test_with_cat['tasa_paro'], residuos_ridge)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("tasa_paro")
plt.ylabel("Residuos")
plt.title("Residuos vs. tasa_paro (Ridge)")

# Residuos vs. precio_m2 (Lasso)
plt.subplot(1, 3, 3)
plt.scatter(x_test_with_cat['precio_m2'], residuos_lasso)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("precio_m2")

# PREDICCIÓN CON MODELO 3.

# Preparar los datos con "categoria"
x_with_categoria = pd.get_dummies(data[['precio_m2', 'tasa_paro', 'categoria']], drop_first=True)
y = data['tasa_emancipacion']

# Dividir los datos
x_train_with_cat, x_test_with_cat, y_train, y_test = train_test_split(x_with_categoria, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler_x = StandardScaler().fit(x_train_with_cat)
scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))

x_train_with_cat = scaler_x.transform(x_train_with_cat)
x_test_with_cat = scaler_x.transform(x_test_with_cat)
y_train = scaler_y.transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Entrenar los modelos
model_with_categoria = LinearRegression().fit(x_train_with_cat, y_train)
ridge_model = Ridge(alpha=3).fit(x_train_with_cat, y_train)
lasso_model = Lasso(alpha=3).fit(x_train_with_cat, y_train)

# Evaluar los modelos
y_pred_with_categoria = model_with_categoria.predict(x_test_with_cat)
y_pred_ridge = ridge_model.predict(x_test_with_cat)
y_pred_lasso = lasso_model.predict(x_test_with_cat)

# Resultados iniciales
print(f"Modelo Lineal con 'categoria': ECM = {mean_squared_error(y_test, y_pred_with_categoria)}, R2 = {r2_score(y_test, y_pred_with_categoria)}")

# Validación cruzada con 50 simulaciones
n_splits = 10  # Número de pliegues
n_repeats = 5  # Número de repeticiones
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Validación cruzada para el modelo Lineal
cv_scores_with_cat = cross_val_score(model_with_categoria, scaler_x.transform(x_with_categoria), scaler_y.transform(y.values.reshape(-1, 1)).flatten(), cv=rkf, scoring='r2')
print(f"Media de R-Cuadrado en validación cruzada (con 'categoria'): {cv_scores_with_cat.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (con 'categoria'): {cv_scores_with_cat.std()}")

# Validación cruzada para el modelo Ridge
cv_scores_ridge = cross_val_score(ridge_model, scaler_x.transform(x_with_categoria), scaler_y.transform(y.values.reshape(-1, 1)).flatten(), cv=rkf, scoring='r2')
print(f"Media de R-Cuadrado en validación cruzada (Ridge): {cv_scores_ridge.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (Ridge): {cv_scores_ridge.std()}")

# Validación cruzada para el modelo Lasso
cv_scores_lasso = cross_val_score(lasso_model, scaler_x.transform(x_with_categoria), scaler_y.transform(y.values.reshape(-1, 1)).flatten(), cv=rkf, scoring='r2')
print(f"Media de R-Cuadrado en validación cruzada (Lasso): {cv_scores_lasso.mean()}")
print(f"Desviación estándar de R-Cuadrado en validación cruzada (Lasso): {cv_scores_lasso.std()}")

# Función para hacer predicciones con nuevos datos
def predecir_tasa_emancipacion(nuevo_precio_m2, nueva_tasa_paro, nueva_categoria, modelo):
    # Crear el array con los nuevos datos
    nuevas_variables = pd.get_dummies(pd.DataFrame({'precio_m2': [nuevo_precio_m2], 'tasa_paro': [nueva_tasa_paro], 'categoria': [nueva_categoria]}), drop_first=True)
    nuevas_variables = nuevas_variables.reindex(columns=x_with_categoria.columns, fill_value=0)

    # Normalizar el nuevo array utilizando el mismo StandardScaler
    nuevo_dato_normalizado = scaler_x.transform(nuevas_variables)
    
    # Hacer la predicción
    nueva_prediccion_normalizada = modelo.predict(nuevo_dato_normalizado)
    
    # Desnormalizar la predicción
    nueva_prediccion = scaler_y.inverse_transform(nueva_prediccion_normalizada.reshape(-1, 1))
    
    return nueva_prediccion[0, 0]

# Ejemplo de uso de la función de predicción
nuevo_precio_m2 = 14.7  # Introducir precio del m2
nueva_tasa_paro = 4.27   # Introducir tasa de paro
nueva_categoria = 'Medio-Alto'  # Introducir categoría

prediccion = predecir_tasa_emancipacion(nuevo_precio_m2, nueva_tasa_paro, nueva_categoria, model_with_categoria)
print(f"Predicción de la tasa de emancipación con un precio_m2 = {nuevo_precio_m2}, tasa_paro = {nueva_tasa_paro} y categoria = {nueva_categoria} con una regresión lineal: {prediccion:.2f}")
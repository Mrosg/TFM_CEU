import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# REGRESIÓN LINEAL CON OPTIMIZACIÓN DE GRADIENTE.

# Cargar el dataset
ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
dataset = pd.read_csv(ruta)

# Seleccionar las columnas necesarias
columns = ['tasa_emancipacion', 'tasa_paro', 'precio_m2', 'categoria']
data = dataset[columns].dropna()

# Convertir la variable categórica en variables dummy
data = pd.get_dummies(data, columns=['categoria'], drop_first=True)

# Separar características (X) y variable objetivo (y)
X = data.drop('tasa_emancipacion', axis=1)
y = data['tasa_emancipacion']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementar la regresión lineal con descenso de gradiente
model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_train_scaled, y_train)

# Obtener los coeficientes del modelo
coefficients = model.coef_
intercept = model.intercept_

# Desnormalizar los coeficientes
coefficients_descaled = coefficients / scaler.scale_
intercept_descaled = intercept - np.sum(coefficients * scaler.mean_ / scaler.scale_)

# Predicción en el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Resultados detallados
model_details = {
    'coeficientes_normalizados': coefficients,
    'intercepto_normalizado': intercept,
    'coeficientes_desnormalizados': coefficients_descaled,
    'intercepto_desnormalizado': intercept_descaled,
    'mean_squared_error': mse,
    'r_squared': r2, 
}

# Imprimir los resultados detallados
print("Detalles del modelo con descenso de gradiente:")
for key, value in model_details.items():
    print(f"{key}: {value}")

# Crear un DataFrame con los coeficientes del modelo
coef_data = {
    'Característica': X.columns,
    'Coeficiente Normalizado': coefficients,
    'Coeficiente Desnormalizado': coefficients_descaled
}

coef_df = pd.DataFrame(coef_data)

# Agregar el intercepto al DataFrame
intercept_data = pd.DataFrame({
    'Característica': ['Intercepto'],
    'Coeficiente Normalizado': [intercept],
    'Coeficiente Desnormalizado': [intercept_descaled]
})

grad_coef = pd.concat([intercept_data, coef_df], ignore_index=False)

# Mostrar la tabla de coeficientes
print(grad_coef)

"""
Detalles del modelo con optimización de gradiente:

coeficientes_normalizados: [-20.11412999  -0.54028169  -0.66354214  -0.51140344  -0.21638234
  -0.92011873]

intercepto_normalizado: [50.49511014]

coeficientes_desnormalizados: [-10.56622124  -0.18581136  -1.63877195  -1.24557296  -0.54763548
  -2.34432898]

intercepto_desnormalizado: [115.14898276]

mean_squared_error: 4.352432379427895

r_squared: 0.9899087584243209

         Característica Coeficiente Normalizado Coeficiente Desnormalizado
0            Intercepto     [50.49511014078888]       [115.14898275539441]
0             tasa_paro               -20.11413                 -10.566221
1             precio_m2               -0.540282                  -0.185811
2        categoria_Bajo               -0.663542                  -1.638772
3       categoria_Medio               -0.511403                  -1.245573
4  categoria_Medio-Alto               -0.216382                  -0.547635
5  categoria_Medio-Bajo               -0.920119                  -2.344329
"""

# Visualización de valores predichos vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.xlabel('Valores Reales de Tasa de Emancipación')
plt.ylabel('Valores Predichos de Tasa de Emancipación')
plt.title('Valores Reales vs. Predichos de Tasa de Emancipación')
plt.grid(True)
plt.show()

# Visualización de la distribución de errores (residuos)
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='purple', edgecolors='k', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Distribución de Errores (Residuos)')
plt.grid(True)
plt.show()

"""
Los residuos están distribuidos alrededor de cero sin un patrón claro, sugiriendo que el modelo es adecuado.
¿Hay un patrón? Podría haber multicolinealidad.
"""

# Histograma de los residuos
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='green', edgecolor='k', alpha=0.7)
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Histograma de Residuos')
plt.grid(True)
plt.show()

"""
La mayoría de los residuos están cerca de cero, indicando que los errores del modelo son pequeños y aleatorios.
"""

# Gráficos de dispersión individuales
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Scatter plot for tasa_paro
axs[0].scatter(X_test['tasa_paro'], y_test, color='blue', edgecolors='k', alpha=0.6, label='Reales')
axs[0].scatter(X_test['tasa_paro'], y_pred, color='red', edgecolors='k', alpha=0.6, label='Predichos')
axs[0].set_xlabel('Tasa de Paro')
axs[0].set_ylabel('Tasa de Emancipación')
axs[0].set_title('Tasa de Paro vs Tasa de Emancipación')
axs[0].legend()
axs[0].grid(True)

# Scatter plot for precio_m2
axs[1].scatter(X_test['precio_m2'], y_test, color='blue', edgecolors='k', alpha=0.6, label='Reales')
axs[1].scatter(X_test['precio_m2'], y_pred, color='red', edgecolors='k', alpha=0.6, label='Predichos')
axs[1].set_xlabel('Precio por m²')
axs[1].set_ylabel('Tasa de Emancipación')
axs[1].set_title('Precio por m² vs Tasa de Emancipación')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Curva de aprendizaje
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Error de Entrenamiento')
plt.plot(train_sizes, test_scores_mean, 'o-', color='red', label='Error de Validación')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Error Cuadrático Medio')
plt.title('Curva de Aprendizaje')
plt.legend(loc='best')
plt.grid(True)
plt.show()

"""
Muestra cómo varía el error del modelo con respecto al tamaño del conjunto de entrenamiento.
La pequeña diferencia entre el error de entrenamiento y el error de validación indica que el modelo generaliza bien.
"""

# Crear un DataFrame con los coeficientes del modelo
coef_data = {
    'Característica': X.columns,
    'Coeficiente Normalizado': coefficients,
    'Coeficiente Desnormalizado': coefficients_descaled
}

coef_df = pd.DataFrame(coef_data)

# Agregar el intercepto al DataFrame
intercept_data = pd.DataFrame({
    'Característica': ['Intercepto'],
    'Coeficiente Normalizado': [intercept],
    'Coeficiente Desnormalizado': [intercept_descaled]
})

dataframe_coef = pd.concat([intercept_data, coef_df], ignore_index=True)

# Mostrar la tabla de coeficientes
print(dataframe_coef)

# Implementar la regresión Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Implementar la regresión Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)

# Predicción en el conjunto de prueba
ridge_y_pred = ridge_model.predict(X_test_scaled)
lasso_y_pred = lasso_model.predict(X_test_scaled)

# Evaluar los modelos
ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_r2 = r2_score(y_test, ridge_y_pred)

lasso_mse = mean_squared_error(y_test, lasso_y_pred)
lasso_r2 = r2_score(y_test, lasso_y_pred)

ridge_model_details = {
    'mean_squared_error': ridge_mse,
    'r_squared': ridge_r2,
}

lasso_model_details = {
    'mean_squared_error': lasso_mse,
    'r_squared': lasso_r2,
}

print("Detalles del modelo Ridge:")
for key, value in ridge_model_details.items():
    print(f"{key}: {value}")

print("\nDetalles del modelo Lasso:")
for key, value in lasso_model_details.items():
    print(f"{key}: {value}")

"""
Detalles del modelo Ridge:
mean_squared_error: 4.352432379427895
r_squared: 0.9899087584243209

Detalles del modelo Lasso:
mean_squared_error: 4.352432379427895
r_squared: 0.9899087584243209
"""

# Validación cruzada
from sklearn.model_selection import cross_val_score

# Validación cruzada para el modelo Ridge
ridge_cv_scores = cross_val_score(ridge_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
ridge_cv_mse = -ridge_cv_scores.mean()

# Validación cruzada para el modelo Lasso
lasso_cv_scores = cross_val_score(lasso_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
lasso_cv_mse = -lasso_cv_scores.mean()

print("MSE promedio del modelo Ridge (cross-validation):", ridge_cv_mse)
print("MSE promedio del modelo Lasso (cross-validation):", lasso_cv_mse)

"""
MSE promedio del modelo Ridge (cross-validation): 4.752843287317453
MSE promedio del modelo Lasso (cross-validation): 4.803584923705156

La validación cruzada proporciona una evaluación más robusta de los modelos.
El MSE promedio es ligeramente superior al MSE obtenido en el conjunto de prueba,
lo cual es esperable ya que la validación cruzada evalúa el modelo en diferentes subconjuntos de datos.
"""

# Análisis de residuos
import scipy.stats as stats

# Residuos del modelo Ridge
ridge_residuals = y_test - ridge_y_pred

# Residuos del modelo Lasso
lasso_residuals = y_test - lasso_y_pred

# Gráficos de residuos para el modelo Ridge
plt.figure(figsize=(10, 6))
plt.scatter(ridge_y_pred, ridge_residuals, color='purple', edgecolors="k", alpha=0.6)
plt.axhline(y=0, color="red", linestyle='-')
plt.xlabel("Valores Predichos (Ridge)")
plt.ylabel("Residuos")
plt.title("Distribución de Errores (Residuos) - Ridge")
plt.grid(True)
plt.show()

# Histograma de los residuos (Ridge)
plt.figure(figsize=(10, 6))
plt.hist(ridge_residuals, bins=30, color="green", edgecolor="k", alpha=0.7)
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.title("Histograma de Residuos - Ridge")
plt.grid(True)
plt.show()

# Gráficos de residuos para el modelo Lasso
plt.figure(figsize=(10, 6))
plt.scatter(lasso_y_pred, lasso_residuals, color="purple", edgecolors="k", alpha=0.6)
plt.axhline(y=0, color="red", linestyle='-')
plt.xlabel("Valores Predichos (Lasso)")
plt.ylabel("Residuos")
plt.title("Distribución de Errores (Residuos) - Lasso")
plt.grid(True)
plt.show()

# Histograma de los residuos (Lasso)
plt.figure(figsize=(10, 6))
plt.hist(lasso_residuals, bins=30, color="green", edgecolor="k", alpha=0.7)
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.title("Histograma de Residuos - Lasso")
plt.grid(True)
plt.show()

# Prueba de normalidad (Q-Q plot) - Ridge
plt.figure(figsize=(10, 6))
stats.probplot(ridge_residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot - Ridge")
plt.show()

# Prueba de normalidad (Q-Q plot) - Lasso
plt.figure(figsize=(10, 6))
stats.probplot(lasso_residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot - Lasso")
plt.show()

"""
Compara los cuantiles de residuos con los cuantiles de una distribución normal.
Se comporta de forma normal en las partes centrales.
"""

# Crear nuevas características
data["precio_m2_log"] = np.log(data["precio_m2"] + 1) # Logaritmo natural de precio_m2 para tratar la asimetría y valores atípicos.
data["tasa_paro_cuadrado"] = data["tasa_paro"] ** 2

# Separar características (X) y variable objetivo (y) nuevamente
X = data.drop("tasa_emancipacion", axis=1)
y = data["tasa_emancipacion"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementar la regresión lineal con descenso de gradiente
model.fit(X_train_scaled, y_train)

# Predicción en el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Evaluar el modelo con las nuevas características
mse_new_features = mean_squared_error(y_test, y_pred)
r2_new_features = r2_score(y_test, y_pred)

print("MSE con nuevas características:", mse_new_features)
print("R² con nuevas características:", r2_new_features)

# Visualización de valores predichos vs valores reales con nuevas características
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", edgecolors="k", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k-", lw=2, color="red")
plt.xlabel("Valores Reales de Tasa de Emancipación")
plt.ylabel("Valores Predichos de Tasa de Emancipación")
plt.title("Valores Reales vs. Predichos de Tasa de Emancipación (Nuevas Características)")
plt.grid(True)
plt.show()

# Visualización de la distribución de errores (residuos) con nuevas características
residuals_new_features = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals_new_features, color="purple", edgecolors="k", alpha=0.6)
plt.axhline(y=0, color="red", linestyle='-')
plt.xlabel("Valores Predichos")
plt.ylabel("Residuos")
plt.title("Distribución de Errores (Residuos) - Nuevas Características")
plt.grid(True)
plt.show()

# Histograma de los residuales con nuevas características
plt.figure(figsize=(10, 6))
plt.hist(residuals_new_features, bins=30, color="green", edgecolor="k", alpha=0.7)
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.title("Histograma de Residuos - Nuevas Características")
plt.grid(True)
plt.show()

# Análisis de Interacción entre Variables
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Crear interacciones de segundo grado entre las variables
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba nuevamente
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler_poly = StandardScaler()
X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
X_test_poly_scaled = scaler_poly.transform(X_test_poly)

# Implementar la regresión lineal con descenso de gradiente
model_poly = SGDRegressor(max_iter=1000, tol=1e-3)
model_poly.fit(X_train_poly_scaled, y_train)

# Predicción en el conjunto de prueba
y_pred_poly = model_poly.predict(X_test_poly_scaled)

# Evaluar el modelo con las interacciones
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("MSE con interacciones:", mse_poly)
print("R² con interacciones:", r2_poly)

"""
PolynomialFeatures: Crea nuevas características que son interacciones de segundo grado
    entre las variables existentes.
Separación y Normalización: Los datos se dividen en conjuntos de entrenamiento y prueba,
    y se normalizan.
Entrenamiento y Evaluación: Se entrena y evalúa el modelo de regresión lineal
    con descenso de gradiente usando las características polinomiales.
"""

# Visualización de valores predichos vs valores reales con interacciones
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_poly, color="blue", edgecolors="k", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '-', lw=2, color="red")
plt.xlabel("Valores Reales de Tasa de Emancipación")
plt.ylabel("Valores Predichos de Tasa de Emancipación")
plt.title("Valores Reales vs. Predichos de Tasa de Emancipación (Interacciones)")
plt.grid(True)
plt.show()

# Visualización de la distribución de errores (residuos) con interacciones
residuals_poly = y_test - y_pred_poly

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_poly, residuals_poly, color="purple", edgecolors="k", alpha=0.6)
plt.axhline(y=0, color="red", linestyle='-')
plt.xlabel("Valores Predichos")
plt.ylabel("Residuos")
plt.title("Distribución de Errores (Residuos) - Interacciones")
plt.grid(True)
plt.show()

# Histograma de los residuales con interacciones

plt.figure(figsize=(10, 6))
plt.hist(residuals_poly, bins=30, color="green", edgecolor="k", alpha=0.7)
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.title("Histograma de Residuos - Interacciones")
plt.grid(True)
plt.show()
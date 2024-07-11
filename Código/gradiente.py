import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset
ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
dataset = pd.read_csv(ruta)

# Seleccionar las columnas necesarias
columns = ['tasa_emancipacion', 'paro_total', 'poblacion', 'precio_m2', 'categoria']
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

grad_coef = pd.concat([intercept_data, coef_df], ignore_index=True)

# Mostrar la tabla de coeficientes
print(grad_coef)
# Imprimir el Mean Squared Error (MSE)
print(f"Mean Squared Error (MSE): {mse}")
# Imprimir el coeficiente de determinación (R²)
print(f"R²: {r2}")

# Validación cruzada de 10 pliegues y 5 repeticiones
cv = RepeatedKFold(n_splits=10, n_repeats=500, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='neg_mean_squared_error')
cv_scores = -cv_scores  # Convertir las puntuaciones negativas a positivas para interpretación

# Calcular la media y la desviación estándar de los errores
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# Resultados de la validación cruzada
cv_results = {
    'MSE promedio': cv_mean,
    'Desviación estándar del MSE': cv_std
}

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

# Histograma de los residuos
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='green', edgecolor='k', alpha=0.7)
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Histograma de Residuos')
plt.grid(True)
plt.show()

# Gráficos de dispersión individuales
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Scatter plot for paro_total
axs[0].scatter(X_test['paro_total'], y_test, color='blue', edgecolors='k', alpha=0.6, label='Reales')
axs[0].scatter(X_test['paro_total'], y_pred, color='red', edgecolors='k', alpha=0.6, label='Predichos')
axs[0].set_xlabel('Paro total')
axs[0].set_ylabel('Tasa de Emancipación')
axs[0].set_title('Paro total vs Tasa de Emancipación')
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

# Mostrar resultados de validación cruzada
cv_results
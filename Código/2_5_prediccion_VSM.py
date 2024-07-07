import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

"""
SVM. Support Vector Machine.

Conjunto de métodos supervisados de aprendizaje usados para clasificación y regresión.

Su idea principal es encontrar un hiperplano en un espacio de alta dimensión que separe (para la clasificación)
o prediga (para la regresión) los datos de manera óptima.

Para regresión, se trata de encontrar una función que tenga un máximo margen de desviación permitido,
es decir, predice valores que están lo más cerca posible de los valores reales dentro de un margen de error tolerable.
"""

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
dataset = pd.read_csv(ruta)

# SUPPORT VECTOR MACHINE.

# One-hot encoding de la columna 'categoria'
# Esto convierte la variable categórica en variables dummy (binarias)
encoder = OneHotEncoder(sparse_output=False)
categoria_encoded = encoder.fit_transform(dataset[['categoria']])

# Añadir las columnas codificadas al DataFrame original
categoria_df = pd.DataFrame(categoria_encoded, columns=encoder.get_feature_names_out(['categoria']))
data = pd.concat([dataset, categoria_df], axis=1)

# Calcular la 'tasa_paro' como el porcentaje de 'paro_total' sobre 'poblacion'
data['tasa_paro'] = (data['paro_total'] / data['poblacion']) * 100

# Seleccionar las columnas relevantes para el modelo
# Variables independientes
features = ['precio_m2', 'tasa_paro'] + list(categoria_df.columns)
# Variable dependiente (target)
target = 'tasa_emancipacion'

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión SVM
# Usamos el kernel radial basis function (RBF)
svm_regressor = SVR(kernel='rbf')
svm_regressor.fit(X_train, y_train)

# Realizar predicciones con el conjunto de entrenamiento y prueba
y_pred_train = svm_regressor.predict(X_train)
y_pred_test = svm_regressor.predict(X_test)

# Evaluar el modelo usando el Error Cuadrático Medio (MSE) y el Coeficiente de Determinación (R²)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Mostrar los resultados de la evaluación del modelo
print("Error Cuadrático Medio en Entrenamiento:", train_mse)
print("Error Cuadrático Medio en Prueba:", test_mse)
print("R² en Entrenamiento:", train_r2)
print("R² en Prueba:", test_r2)

# Crear un gráfico de dispersión para comparar los valores reales vs predichos en el conjunto de prueba
plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred_test, color='blue', edgecolor='k', alpha=0.6, label='Datos de prueba')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Línea ideal')

plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Valores Reales vs. Predichos en el Conjunto de Prueba')
plt.legend()
plt.grid(True)
plt.show()

"""
Resultados del modelo SVM.

Error Cuadrático Medio en Entrenamiento: 0.7843551135507546
Error Cuadrático Medio en Prueba: 0.8566426062639546
R² en Entrenamiento: 0.99808987334876
R² en Prueba: 0.9980138491008641
"""

# ANÁLISIS DE IMPORTANCIA DE VARIABLES.

# Entrenar un modelo de regresión lineal con los datos de entrenamiento
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Obtener las coeficientes del modelo
coefficients = linear_regressor.coef_
importance_df = pd.DataFrame({'Variable': features, 'Importancia': coefficients})

# Visualizar la importancia de las variables
importance_df = importance_df.sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(importance_df['Variable'], importance_df['Importancia'], color='skyblue')
plt.xlabel('Impacto')
plt.title('Impacto de las variables en la predicción de Tasa de Emancipación')
plt.gca().invert_yaxis()  # Invertir el eje y para que la variable más importante esté en la parte superior
plt.grid(True)
plt.show()

"""
1.	Alto: Es la variable más importante y tiene un impacto positivo en la predicción de la tasa de emancipación.
2.	Medio-Alto: También tiene un impacto positivo significativo.
3.	Medio y precio_m2: Tienen un impacto negativo menor.
4.	Bajo y Medio-Bajo: Tienen un impacto negativo mayor en la predicción.
5.	tasa_paro: Es la variable con el mayor impacto negativo, lo que indica que un aumento en
    la tasa de paro disminuye significativamente la tasa de emancipación.
"""

# VALIDACIÓN CRUZADA.

# Validación cruzada con 5 particiones (5-fold cross-validation)
cv_scores = cross_val_score(svm_regressor, X, y, cv=5, scoring='neg_mean_squared_error')

# Calcular el MSE promedio y su desviación estándar
cv_mse_mean = -cv_scores.mean()
cv_mse_std = cv_scores.std()

(cv_mse_mean, cv_mse_std)

"""
ECM promedio: 1.5940053257335962
Desviación estándar del ECM: 1.4036550845564528

El MSE promedio obtenido mediante validación cruzada es ligeramente mayor que el obtenido en la evaluación inicial,
lo que indica que el modelo puede estar ligeramente sobreajustado a los datos de entrenamiento iniciales.

La desviación estándar del MSE es relativamente alta, lo que sugiere que el rendimiento del modelo
puede variar significativamente entre diferentes particiones del conjunto de datos.
"""

# ANÁLISIS DE RESIDUOS.

# Calcular los residuos (errores) en el conjunto de prueba
residuos = y_test - y_pred_test

# Crear un gráfico de residuos
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuos, color='blue', edgecolor='k', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')
plt.grid(True)
plt.show()

"""
La visibilidad de un patrón puede suponer un problema de multicolinealidad.
"""
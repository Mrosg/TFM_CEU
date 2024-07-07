import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.inspection import permutation_importance

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

"""
MODELO DE REGRESIÓN BAYESIANA
"""

# PREPARACIÓN DE LOS DATOS.

    ## Convierto la columna de fechas a formato datetime.

data["fecha"] = pd.to_datetime(data["fecha"], format="%m-%Y")

    ## Codifico la variable "categoría".

encoder = OneHotEncoder(sparse_output=False)
categorias_encoded = encoder.fit_transform(data[["categoria"]])
categorias_df = pd.DataFrame(categorias_encoded, columns=encoder.get_feature_names_out(["categoria"]))
data = pd.concat([data, categorias_df], axis=1)
data.drop(["categoria"], axis=1, inplace=True)

    ## Actualizo la lista de variables independientes.

variables_independientes = ["precio_m2", "tasa_paro"] + list(categorias_df.columns)
x = data[variables_independientes].values
y = data["tasa_emancipacion"].values

"""
La codificación de categorías en variables dummy permite a los algoritmos de ML procesar datos categóricos correctamente,
convirtiendo las categorías en un formato numérico binario.
"""

    ## Normalizo los datos.

escalar = StandardScaler()
x_scaled = escalar.fit_transform(x)

    ## Divido los datos en entrenamiento y prueba.

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

# MODELO DE REGRESIÓN BAYESIANA.

    ## Defino el modelo de regresión bayesiana.

model = BayesianRidge()

    ## Validación cruzada.

n_splits = 10
n_repeats = 100  # Reducido a 100 repeticiones
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
cv_scores = cross_val_score(model, x_scaled, y, cv=rkf, scoring="neg_mean_squared_error")
cv_mse = -cv_scores.mean()
cv_std = cv_scores.std()

    ## Entrenamiento del modelo.

model.fit(x_train, y_train)

    ## Predicciones.

y_pred = model.predict(x_test)

    ## Calculo el ECM.

mse = mean_squared_error(y_test, y_pred)

    ## Creo un dataframe para comparar las predicciones con los valores reales.

resultados = pd.DataFrame({
    "Real" : y_test,
    "Predicción" : y_pred
})

# EXTRACCIÓN DE DATOS DEL MODELO.

    ## Extraigo los coeficientes e intercepto.

intercept = model.intercept_
coef = model.coef_

    ## Matriz de covarianza de los coeficientes.

cov_matrix = model.sigma_

    ## Desviaciones estándar de los coeficientes.

intercept_std = np.sqrt(model.alpha_)
coef_std = np.sqrt(np.diag(cov_matrix))

    ## Creo un dataframe para los intervalos de confianza del modelo.

intervalos_confianza = pd.DataFrame({
    "Características" : ["Intercept"] + variables_independientes,
    "Coeficiente" : [intercept] + list(coef),
    "Límite Inferior" : [intercept - 1.96 * intercept_std] + list(coef - 1.96 * coef_std),
    "Límite Superior" : [intercept + 1.96 * intercept_std] + list(coef + 1.96 * coef_std)
})

print("Resumen del modelo de regresión bayesiana:")
print(intervalos_confianza)

# VISUALIZACIÓN DEL MODELO.

    ## Visualizo las distribuciones de probabilidad de los coeficientes.

x_vals = np.linspace(-30, 30, 400)
colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray"]

plt.figure(figsize=(12, 6))

    ## Gráficos de las distribuciones de probabilidad de los coeficientes.

for i, feature in enumerate(["Intercept"] + variables_independientes):
    mean = intervalos_confianza["Coeficiente"].iloc[i]
    std = intercept_std if i == 0 else coef_std[i - 1]
    plt.plot(x_vals, stats.norm.pdf(x_vals, mean, std), label=feature, color=colors[i % len(colors)])

plt.legend()
plt.title("Distribuciones de probabilidad de los coeficientes del modelo de regresión bayesiana")
plt.xlabel("Valor del Coeficiente")
plt.ylabel("Densidad de Probabilidad")
plt.show()

    ## Visualizo las predicciones vs los valores reales.

plt.figure(figsize=(12, 6))
plt.plot(resultados["Real"].values, label="Valores reales", color="blue")
plt.plot(resultados["Predicción"].values, label="Predicciones", color="red", linestyle="dashed")
plt.legend()
plt.title("Regresión bayesiana: comparación de valores reales y predicciones")
plt.xlabel("Índice")
plt.ylabel("Tasa de emancipación")
plt.show()

    ## Histograma de los errores cuadráticos medios de la validación cruzada.

plt.figure(figsize=(12, 6))
plt.hist(-cv_scores, bins=30, edgecolor="black", alpha=0.7, color="turquoise")
plt.title("Distribución de los Errores Cuadráticos Medios en la validación cruzada")
plt.xlabel("Error Cuadrático Medio")
plt.ylabel("Frecuencia")
plt.show()

    ## Calculo los residuos.

residuales = y_test - y_pred

    ## Histograma de los residuos.

plt.figure(figsize=(12, 6))
plt.hist(residuales, bins=30, edgecolor="black", alpha=0.7, color="purple")
plt.title("Distribución de los Residuales")
plt.xlabel("Residual")
plt.ylabel("Frecuencia")
plt.show()

    ## Residuos vs. Predicciones.

plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuales, edgecolor="black", alpha=0.7, color="green")
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuales vs. Predicciones")
plt.xlabel("Predicciones")
plt.ylabel("Residuales")
plt.show()

"""
 - Distribución de residuos: la asimetría sugiere que el modelo tiene a subestimar
    la tasa de emancipación.
- Residuos: la forma de patrones indica posible heterocedasticidad. Puede que el modelo
    no capture todas las relaciones entre
"""

# PRUEBA DE NORMALIDAD DEL MODELO.

    ## Test de normalidad (Shapiro-Wilk test).

shapiro_test = stats.shapiro(residuales)
print(f"Shapiro-Wilk Test: {shapiro_test}")

# COMPARACIÓN CON OTROS MODELOS.    
    
    ## Defino los modelos.

modelos = {
    "Regresión Lineal" : LinearRegression(),
    "Regresión Ridge" : Ridge(),
    "LASSO" : Lasso()
}

    ## Validación cruzada.

resultados_mse = {}

for nombre, modelo in modelos.items():
    cv_scores = cross_val_score(model, x_scaled, y, cv=rkf, scoring="neg_mean_squared_error")
    cv_mse = -cv_scores.mean()
    resultados_mse[nombre] = cv_mse

    ## Mostrar los resultados.

for nombre, mse in resultados_mse.items():
    print(f"{nombre}: Cross-Validated MSE = {mse:.4f}")

# IMPORTANCIA DE LAS VARIABLES DEL MODELO.

    ## Calculo la importancia de las variables mediante Permutation Importance.

resultados_importancia = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=42)

    ## Creo un dataframe con los resultados.

importancia_df = pd.DataFrame({
    "Variable" : variables_independientes,
    "Importancia Media" : resultados_importancia.importances_mean,
    "Desviación Estándar" : resultados_importancia.importances_std
}).sort_values(by="Importancia Media", ascending=False)

print(importancia_df)

# AJUSTE DE HIPERPARÁMETROS.

    ## Defino la rejilla de hiperparámetros.

param_grid = {
    "alpha_1" : [1e-6, 1e-5, 1e-4, 1e-3],
    "alpha_2" : [1e-6, 1e-5, 1e-4, 1e-3],
    "lambda_1" : [1e-6, 1e-5, 1e-4, 1e-3],
    "lambda_2" : [1e-6, 1e-5, 1e-4, 1e-3]
}

    ## Configuro el GridSearchCV.

grid_search = GridSearchCV(BayesianRidge(), param_grid, scoring = "neg_mean_squared_error", cv = 5)

    ## Ejecuto el GridSearchCV.

grid_search.fit(x_train, y_train)

    ## Mostrar los mejores hiperparámetros y el MSE.

best_params = grid_search.best_params_
best_mse = -grid_search.best_score_

print(f"Mejores hiperparámetros: {best_params}")
print(f"Mejor MSE: {best_mse}")
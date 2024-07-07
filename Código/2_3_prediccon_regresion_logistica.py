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

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

# MODELO DE REGRESIÓN LOGÍSTICA GENERALIZADA.

    ## Selección de características.

selected_features = ["precio_m2", "tasa_paro"]

    ## Filtrar el DataFrame para las características seleccionadas y la variable objetivo.

x = data[selected_features].values
y = data["tasa_emancipacion"].values

    ## Normalizar las características.

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

    ## Dividir los datos en entrenamiento y prueba.

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 42)

    ## Añadir una columna de unos para el intercepto.

x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

    ## Ajustar el modelo de regresión logística.

model = sm.GLM(y_train, x_train, family=sm.families.Gaussian()).fit()

    ## Resumen de los resultados.

summary = model.summary()
print(summary)

    ## Predicciones.

y_pred = model.predict(x_test)

    ## Calcular el MSE.

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

"""
Intercepto (const): 50.4939
    - Este valor indica que cuando todas las características son cero, la tasa de emancipación promedio es aproximadamente 50.49.

"precio_m2" (x1): 0.1530
    - Un aumento en el precio por metro cuadrado está asociado
    con un aumento en la tasa de emancipación. Este coeficiente es significativo con un valor p de 0.006,
    indicando que hay una relación estadísticamente significativa entre el precio por metro cuadrado y la tasa de emancipación.

"tasa_paro" (x2): -20.1984
    - Un aumento en la tasa de paro está asociado con una disminución
    en la tasa de emancipación. Este coeficiente es altamente significativo con un valor p menor que 0.0001,
    indicando que hay una fuerte relación inversa entre la tasa de paro y la tasa de emancipación.
"""

    ## Quiero visualizar los resultados.

        ### Hago las predicciones.

y_pred = model.predict(x_test)

        ### Creo un dataframe para comparar las predicciones con los valores reales.

results = pd.DataFrame({
    "Real": y_test,
    "Predicción": y_pred
})

        ### Visualizo las predicciones con los valores reales.

plt.figure(figsize = (12, 6))
plt.plot(results["Real"].values, label = "Valores reales", color = "black")
plt.plot(results["Predicción"].values, label = "Predicciones", color = "red", linestyle = "dashed")
plt.legend()
plt.title("Regresión logística. Comparación de valores reales y predicciones")
plt.xlabel("Índice")
plt.ylabel("Tasa de emancipación")
plt.show()

"""
Evaluación del modelo:

	•	MSE (Mean Squared Error): 4.7259
	•	El MSE es una medida del error promedio cuadrático entre las predicciones del modelo y los valores reales.
        Un valor menor indica que las predicciones están más cerca de los valores reales.

Predicciones y visualización:

He realizado predicciones utilizando el conjunto de prueba y visualizado los resultados
comparando las predicciones con los valores reales. El gráfico muestra que el modelo sigue
bastante bien los patrones de los datos reales, aunque hay algunas discrepancias normales en cualquier modelo de predicción.

Conclusión:

El modelo de regresión lineal ajustado proporciona una buena aproximación para predecir la tasa de emancipación
basada en precio_m2 y tasa_paro. Aunque no es una regresión bayesiana verdadera, los resultados son interpretables
y útiles para entender cómo estas características afectan la tasa de emancipación.
"""

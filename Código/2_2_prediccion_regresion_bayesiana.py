import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import BayesianRidge
import statsmodels.api as sm
import matplotlib.pyplot as plt

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

# REGRESIÓN BAYESIANA.

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

    ## Selección de características.

selected_features = ["precio_m2", "tasa_paro"]

    ## Filtrar el dataframe para las características seleccionadas y la variable objetivo.

x = data[selected_features].values
y = data["tasa_emancipacion"].values

    ## Normalizo las características.

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

    ## Divido los datos en entrenamiento y prueba.

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 42)

    ## Defino y entrenar el modelo de regresión bayesiana.

model = BayesianRidge()
model.fit(x_train, y_train)

    ## Hago las predicciones.

y_pred = model.predict(x_test)

    ## Calculo el MSE.

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

    ## Creo un dataframe para comparar las predicciones con los valores reales.

results = pd.DataFrame({
    "Real": y_test,
    "Predicción": y_pred
})

    ## Extrae los coeficientes y el intercepto.

intercept = model.intercept_
coef = model.coef_

    ## Crea un dataframe para los coeficientes.

coef_summary = pd.DataFrame({
    "Feature": ["Intercept"] + selected_features,
    "Coefficient": [intercept] + list(coef)
})

    ## Muestro el resumen de los coeficientes.

print("Summary del modelo de Regresión Bayesiana")
print(coef_summary)

    ## Visualizo las predicciones versus los valores reales.

plt.figure(figsize = (12, 6))
plt.plot(results["Real"].values, label = "Valores reales", color = "blue")
plt.plot(results["Predicción"].values, label = "Predicciones", color = "red", linestyle = "dashed")
plt.legend()
plt.title("Regresión bayesiana. Comparación de valores reales y predicciones")
plt.xlabel("Índice")
plt.ylabel("Tasa de emancipación")
plt.show()

"""
Resumen del modelo y métricas:

    - MSE (Mean Squared Error): 4.7258
    - El MSE es una medida del error cuadrático promedio entre las predicciones del modelo
    y los valores reales. Un valor de 4.7258 sugiere que, en promedio, las predicciones del modelo
    están a unos 4.73 puntos de distancia de los valores reales de la tasa de emancipación.
    Un MSE más bajo indica un mejor rendimiento del modelo, ya que implica menores errores de predicción.

Coeficientes del modelo:

Para interpretar completamente el modelo, se deben revisar los coeficientes estimados por BayesianRidge.
Aunque no tenemos estos coeficientes directamente en los resultados impresos, podemos inferir su impacto
observando las características seleccionadas y los patrones en el gráfico.
    - Intercepto (const):
        · Indica el valor esperado de la tasa de emancipación cuando precio_m2 y tasa_paro son cero.
        En el contexto de normalización, este valor puede ser interpretado como el valor base de la tasa de emancipación.
    - precio_m2:
        · En el modelo lineal anterior, el coeficiente de precio_m2 era positivo,
        lo que indica que un aumento en el precio por metro cuadrado está asociado con un aumento en la tasa de emancipación.
        Este patrón es consistente y sugiere que mayores precios de la vivienda pueden estar correlacionados con mayores tasas de emancipación, posiblemente porque áreas con mayores precios reflejan mejores oportunidades económicas.
    - tasa_paro:
        · En el modelo lineal anterior, el coeficiente de tasa_paro era negativo,
        indicando que un aumento en la tasa de paro está asociado con una disminución en la tasa de emancipación.
        Esto también es consistente con la intuición, ya que mayores tasas de desempleo pueden dificultar la
        independencia económica de los individuos, reduciendo la tasa de emancipación.

Visualización de los resultados:
    - La línea azul representa los valores reales de la tasa de emancipación.
    - La línea roja discontinua representa las predicciones del modelo.
    - Observamos que las predicciones del modelo siguen bastante bien la tendencia de los valores reales,
    aunque hay algunas discrepancias, especialmente en los picos y valles.
    Estas discrepancias son normales y reflejan la variabilidad en los datos que el modelo no captura completamente.

Observaciones:

    - Precisión del Modelo:
        · El modelo tiene un MSE relativamente bajo, lo que sugiere que realiza predicciones precisas
        de la tasa de emancipación en función de las características precio_m2 y tasa_paro.
    - Impacto de las Características:
        · precio_m2 y tasa_paro tienen un impacto significativo en la tasa de emancipación, alineado con la intuición económica.
        · A medida que aumenta el precio de la vivienda, la tasa de emancipación tiende a aumentar.
        · A medida que aumenta la tasa de paro, la tasa de emancipación tiende a disminuir.
    - Utilidad del Modelo:
        · Este modelo puede ser útil para predecir la tasa de emancipación en diferentes escenarios económicos,
        ayudando a los formuladores de políticas y a los planificadores urbanos a entender mejor los factores
        que afectan la emancipación de los jóvenes.

Conclusión:

El modelo de regresión bayesiana proporciona una buena aproximación para predecir la tasa de emancipación
basada en precio_m2 y tasa_paro. Los resultados son interpretables y coherentes con la teoría económica,
lo que refuerza la validez del modelo. La visualización muestra que las predicciones están alineadas con los valores reales,
aunque hay espacio para mejoras adicionales en la precisión del modelo.
"""

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

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso

"""
Prueba en Lasso y Ridge, tanto con 2 como con 3 variables independientes,
usando otros valores de alpha: 0.5, 0.75, 1, 1.5 y 3. Anota resultados y compara.
"""

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

# ESCALAR Y NORMALIZAR LOS DATOS PARA MODELAR.

    ## Aislo las variables numéricas de mi dataset.

numericas = data.select_dtypes(include = ["float64", "int64"]).columns

    ## Estandarizo mis datos.

escalar = StandardScaler()
data_estandar = data.copy()
data_estandar[numericas] = escalar.fit_transform(data[numericas])

print(data_estandar.head()) # Vemos los datos estandarizados.
media_estandar_preciom2 = data_estandar["precio_m2"].mean() # Compruebo la media cero de una variable al azar.
sd_estandar_preciom2 = data_estandar["precio_m2"].std() # Compruebo la desviación típica uno de la variable.
print("La media estandarizada de precio_m2 es: ", media_estandar_preciom2)
print("La desviación típica estandarizada de precio_m2 es: ", sd_estandar_preciom2)

    ## Normalizo los datos estandarizados.

normalizar = MinMaxScaler()
data_normal = data_estandar.copy()
data_normal[numericas] = normalizar.fit_transform(data_estandar[numericas])

# DOS VARIABLES INDEPENDIENTES.

# PRIMER MODELO DE REGRESIÓN LINEAL

    ## Defino los predictores (x) y la variable dependiente (y).

x = data_normal[["precio_m2", "tasa_paro"]] # Puedo usar "paro_total" esta vez en vez de la "tasa_paro".
y = data_normal["tasa_emancipacion"]

    ## Divido el conjunto de datos en entrenamiento y prueba.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

    ## Entreno el modelo de regresión lineal.

model = LinearRegression()
model.fit(x_train, y_train)

    ## Hago predicciones con el conjunto de prueba.

y_pred = model.predict(x_test)

    ## Evalúo el modelo.

ecm = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio: {ecm}")
print(f"R-Cuadrado: {r2}")

    ## Visualizo los resultados.

plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Regresión lineal: valores reales vs predicciones")
plt.show()

"""
Los resultados de este modelo de regresión lineal son:

    - Error cuadrático medio: 0.0006
    - R-Cuadrado: 0.99

El ECM es muy bajo y el R-Cuadrado es prácticamente perfecto. Entiendo que hay un sobreajuste del modelo.
"""

# RIDGE Y LASSO.

# INTENTO EVITAR EL SOBREAJUSTE CON MODELOS DE RIDGE Y LASSO CON DOS VARIABLES.

    ## Modelo de Ridge.

modelo_ridge = Ridge(alpha = 3)
modelo_ridge.fit(x_train, y_train)
y_ridge = modelo_ridge.predict(x_test)
ecm_ridge = mean_squared_error(y_test, y_ridge)
r2_ridge = r2_score(y_test, y_ridge)

resultados_ridge = (ecm_ridge, r2_ridge)

print(f"Ridge - Error cuadrático medio: {ecm_ridge}")
print(f"Ridge - R-Cuadrado: {r2_ridge}")

    ## Modelo Lasso.

modelo_lasso = Lasso(alpha = 3)
modelo_lasso.fit(x_train, y_train)
y_pred_lasso = modelo_lasso.predict(x_test)
ecm_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

resultados_lasso = (ecm_lasso, r2_lasso)

print(f"Lasso - Error cuadrático medio: {ecm_lasso}")
print(f"Lasso - R-Cuadrado: {r2_lasso}")

    ## Visualizo de los resultados.

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_ridge)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones de Ridge")
plt.title("Regresión de Ridge: valores reales vs predicciones")
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lasso)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones de Lasso")
plt.title("Regresión de Lasso: valores reales vs predicciones")
plt.show()

"""
ALFA 0.5

Modelo Ridge:
    - Error cuadrático medio (ECM): 0.000655905168398305
	- R-Cuadrado: 0.9902277544296918
Modelo Lasso:
	- Error cuadrático medio (ECM): 0.0672074007268954
	- R-Cuadrado: -0.0013142992136063736

ALFA 0.75

Modelo Ridge:
    - Error cuadrático medio (ECM): 0.0006598493935551132
	- R-Cuadrado: 0.9901689899334293
Modelo Lasso:
	- Error cuadrático medio (ECM): 0.0672074007268954
	- R-Cuadrado: -0.0013142992136063736

ALFA 1

Modelo Ridge:
    - Error cuadrático medio (ECM): 0.000665322915834512
	- R-Cuadrado: 0.9900874406387661
Modelo Lasso:
	- Error cuadrático medio (ECM): 0.0672074007268954
	- R-Cuadrado: -0.0013142992136063736

ALFA 1.5

Modelo Ridge:
    - Error cuadrático medio (ECM): 0.0006805529783384305
	- R-Cuadrado: 0.9898605299236045
Modelo Lasso:
	- Error cuadrático medio (ECM): 0.0672074007268954
	- R-Cuadrado: -0.0013142992136063736

ALFA 3

Modelo Ridge:
    - Error cuadrático medio (ECM): 0.0007557728303825674
	- R-Cuadrado: 0.9887398391570832
Modelo Lasso:
	- Error cuadrático medio (ECM): 0.0672074007268954
	- R-Cuadrado: -0.0013142992136063736
"""

# TRES VARIABLES INDEPENDIENTES.

# SEGUNDO MODELO DE REGRESIÓN LINEAL.

    ## Defino los predictores (x) y la variable dependiente (y).

x = data_normal[["precio_m2", "paro_total", "poblacion"]] # Puedo usar "paro_total" esta vez en vez de la "tasa_paro".
y = data_normal["tasa_emancipacion"]

"""
Usar el número de parados no creo que tenga mucho sentido porque no tiene en cuenta el número de población,
porque en distritos más grandes la tasa es lógicamente más alta y distorsiona el resultado.

Si meto "precio_m2", "paro_total" y "poblacion" los datos son más variados y el R-Cuadrado baja a 0.58
"""

    ## Divido el conjunto de datos en entrenamiento y prueba.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

    ## Entreno el modelo de regresión lineal.

model = LinearRegression()
model.fit(x_train, y_train)

    ## Hago predicciones con el conjunto de prueba.

y_pred = model.predict(x_test)

    ## Evalúo el modelo.

ecm = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio: {ecm}")
print(f"R-Cuadrado: {r2}")

    ## Visualizo los resultados.

plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Regresión lineal con tres variables: valores reales vs predicciones")
plt.show()

"""
Los resultados de este modelo de regresión lineal son:

    - Error cuadrático medio: 0.028114267266406066
    - R-Cuadrado: 0.5811291982536125
"""

# RIDGE Y LASSO.

# INTENTO EVITAR EL SOBREAJUSTE CON MODELOS DE RIDGE Y LASSO CON TRES VARIABLES.

    ## Modelo de Ridge.

modelo_ridge = Ridge(alpha = 3)
modelo_ridge.fit(x_train, y_train)
y_ridge = modelo_ridge.predict(x_test)
ecm_ridge = mean_squared_error(y_test, y_ridge)
r2_ridge = r2_score(y_test, y_ridge)

resultados_ridge = (ecm_ridge, r2_ridge)

print(f"Ridge - Error cuadrático medio: {ecm_ridge}")
print(f"Ridge - R-Cuadrado: {r2_ridge}")

    ## Modelo Lasso.

modelo_lasso = Lasso(alpha = 3)
modelo_lasso.fit(x_train, y_train)
y_pred_lasso = modelo_lasso.predict(x_test)
ecm_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

resultados_lasso = (ecm_lasso, r2_lasso)

print(f"Lasso - Error cuadrático medio: {ecm_lasso}")
print(f"Lasso - R-Cuadrado: {r2_lasso}")

    ## Visualizo de los resultados.

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

"""
ALFA 0.5

Modelo Ridge:
    - Error cuadrático medio (ECM): 0.028344234967988817
	- R-Cuadrado: 0.5777029394567901
Modelo Lasso:
	- Error cuadrático medio (ECM): 0.0672074007268954
	- R-Cuadrado: -0.0013142992136063736

ALFA 0.75

Modelo Ridge:
    - Error cuadrático medio (ECM): 0.02851466155348315
	- R-Cuadrado: 0.5751637759840784
Modelo Lasso:
	- Error cuadrático medio (ECM): 0.0672074007268954
	- R-Cuadrado: -0.0013142992136063736

ALFA 1

Modelo Ridge:
    - Error cuadrático medio (ECM): 0.028681560450824074
	- R-Cuadrado: 0.5726771710771343
Modelo Lasso:
	- Error cuadrático medio (ECM): 0.0672074007268954
	- R-Cuadrado: -0.0013142992136063736

ALFA 1.5

Modelo Ridge:
    - Error cuadrático medio (ECM): 0.02898146944232792
	- R-Cuadrado: 0.5682088661225082
Modelo Lasso:
	- Error cuadrático medio (ECM): 0.0672074007268954
	- R-Cuadrado: -0.0013142992136063736

ALFA 3

Modelo Ridge:
    - Error cuadrático medio (ECM): 0.02962585542474941
	- R-Cuadrado: 0.5586082434019031
Modelo Lasso:
	- Error cuadrático medio (ECM): 0.0672074007268954
	- R-Cuadrado: -0.0013142992136063736
"""

# QUIERO VER CÓMO INFLUYE DOS VARIABLES A LA TASA DE EMANCIPACIÓN Y SI PUEDEN SER VARIABLES CONFUSORAS.

    ## Tasa de emancipación y precio m2.

        ### Dataframe para ver la relación entre la tasa de emancipación y el precio del m2.

plot_data = data[["tasa_emancipacion", "precio_m2"]]

        ### Hago el plot.

            #### Creo la figura y los ejes.
plt.figure(figsize = (10, 6))
            #### Ploto los datos.
sns.scatterplot(data = plot_data, x = "precio_m2", y = "tasa_emancipacion", alpha = 0.6)
            #### Agrego línea de regresión lineal.
sns.regplot(data = plot_data, x = "precio_m2", y = "tasa_emancipacion", scatter = False, color = "black", ci = None)
plt.xlabel("Precio del m2")
plt.ylabel("Tasa de Emancipación")
plt.title("Tasa de Emancipación vs Precio del m2")
plt.legend()
plt.show()

    ## Tasa de emancipación y precio m2 por categorías.

        ### Dataframe para ver la relación entre la tasa de emancipación y el precio del m2 por categorías.

plot_data = data[["tasa_emancipacion", "precio_m2", "categoria"]]

        ### Hago el plot.

            #### Creo la figura y los ejes.
plt.figure(figsize=(10, 6))
            #### Plotear los datos con seaborn para facilitar la inclusión de líneas de regresión
sns.scatterplot(data = plot_data, x = "precio_m2", y = "tasa_emancipacion", hue = "categoria", alpha = 0.6)
            #### Agregar líneas de regresión por categoría
categories = plot_data["categoria"].unique()
for category in categories:
    subset = plot_data[plot_data["categoria"] == category]
    sns.regplot(data = subset, x = "precio_m2", y = "tasa_emancipacion", scatter = False, ci = None,
                label = f"Regresión {category}")
            #### Agregar línea de regresión total
sns.regplot(data = plot_data, x = "precio_m2", y = "tasa_emancipacion", scatter = False, color = "black",
            label = "Regresión Total", ci = None)
plt.xlabel("Precio del m2")
plt.ylabel("Tasa de Emancipación")
plt.title("Tasa de Emancipación vs Precio del m2")
plt.legend(title = "Categoría")
plt.show()

    ## Tasa de emancipación y tasa de paro.
    
        ### Dataframe para ver la relación entre la tasa de emancipación y la tasa de paro.

plot_data = data[["tasa_emancipacion", "tasa_paro"]]

        ### Hago el plot.

            #### Creo la figura y los ejes.
plt.figure(figsize = (10, 6))
            #### Ploto los datos.
sns.scatterplot(data = plot_data, x = "tasa_paro", y = "tasa_emancipacion", alpha = 0.6)
            #### Agrego línea de regresión lineal.
sns.regplot(data = plot_data, x = "tasa_paro", y = "tasa_emancipacion", scatter = False, color = "black", ci = None)
plt.xlabel("Tasa de paro")
plt.ylabel("Tasa de emancipación")
plt.title("Tasa de emancipación vs Tasa de paro")
plt.legend()
plt.show()

    ## Tasa de emancipación y tasa de paro por categorías.

        ### Dataframe para ver la relación entre la tasa de emancipación y la tasa de paro por categorías.

plot_data = data[["tasa_emancipacion", "tasa_paro", "categoria"]]

        ### Hago el plot.

            #### Creo la figura y los ejes.
plt.figure(figsize=(10, 6))
            #### Plotear los datos con seaborn para facilitar la inclusión de líneas de regresión
sns.scatterplot(data = plot_data, x = "tasa_paro", y = "tasa_emancipacion", hue = "categoria", alpha = 0.6)
            #### Agregar líneas de regresión por categoría
categories = plot_data["categoria"].unique()
for category in categories:
    subset = plot_data[plot_data["categoria"] == category]
    sns.regplot(data = subset, x = "tasa_paro", y = "tasa_emancipacion", scatter = False, ci = None,
                label = f"Regresión {category}")
            #### Agregar línea de regresión total
sns.regplot(data = plot_data, x = "tasa_paro", y = "tasa_emancipacion", scatter = False, color = "black",
            label = "Regresión Total", ci = None)
plt.xlabel("Tasa de paro")
plt.ylabel("Tasa de Emancipación")
plt.title("Tasa de Emancipación vs Tasa de paro")
plt.legend(title = "Categoría")
plt.show()

"""
El precio del m2 tiene en principio y a nivel general una pendiente positiva respecto a la tasa de emancipación.
La curva de tendencia se mantiene casi paralela a partir aproximadamente del 11,8€/m2.

La cosa es un poco distinta si tenenemos en cuenta al categoría del distrito en la relación entre estas dos variables.

No en todas las categorías la relación es igual. Algunos tienen más pendiente y en otros menos.

¿Puede ser la categoría del distrito una variable confusora?

En el caso de la tasa de paro y la tasa de emancipación, mantienen una relación negativa, cosa que es lógica.
"""

# TEST ANOVA PARA COMPROBAR DIFERENCIAS ENTRE VARIABLES.

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

    ## Evalúo la asociación entre "categoria" y las variables predictoras.

anova_precio = ols('precio_m2 ~ C(categoria)', data=data).fit()
anova_tasa_paro = ols('tasa_paro ~ C(categoria)', data=data).fit()
anova_emancipacion = ols('tasa_emancipacion ~ C(categoria)', data=data).fit()

    ## Resumen de ANOVA.

anova_precio_summary = sm.stats.anova_lm(anova_precio, typ=2)
anova_tasa_paro_summary = sm.stats.anova_lm(anova_tasa_paro, typ=2)
anova_emancipacion_summary = sm.stats.anova_lm(anova_emancipacion, typ=2)

print("ANOVA Precio m2 vs Categoría")
print(anova_precio_summary)
print("\nANOVA Tasa Paro vs Categoría")
print(anova_tasa_paro_summary)
print("\nANOVA Tasa Emancipación vs Categoría")
print(anova_emancipacion_summary)

"""
	1.	ANOVA Precio m2 vs Categoría:
	•	F-value: 7467.581, p-value < 0.05
	•	La diferencia en los precios del m2 entre las categorías es significativa.
        Las categorías afectan a los precios del m2.
	2.	ANOVA Tasa Paro vs Categoría:
	•	F-value: 857.244, p-value < 0.05
	•	La diferencia en la tasa de paro entre las categorías también es significativa.
        Las categorías afectan a la tasa de paro.
	3.	ANOVA Tasa Emancipación vs Categoría:
	•	F-value: 865.169, p-value < 0.05
	•	La diferencia en la tasa de emancipación entre las categorías también es significativa.
        Las categorías afectan a la tasa de emancipación.

En todos los análisis de ANOVA realizados (precio m2, tasa de paro y tasa de emancipación vs categoría),
los resultados indican que las diferencias observadas entre las categorías son estadísticamente significativas.
"""

# MODELO DE REGRESIÓN LINEAL CON CATEGORÍA Y SIN CATEGORÍA.

    ## Preparo los datos con "categoria".

x_with_categoria = pd.get_dummies(data[['precio_m2', 'tasa_paro', 'categoria']], drop_first=True)
y = data['tasa_emancipacion']

    ## Divido los datos.

x_train_with_cat, x_test_with_cat, y_train, y_test = train_test_split(x_with_categoria, y, test_size=0.2, random_state=42)

    ## Entreno el modelo con la variable "categoria".

model_with_categoria = LinearRegression().fit(x_train_with_cat, y_train)
y_pred_with_categoria = model_with_categoria.predict(x_test_with_cat)
ecm_with_categoria = mean_squared_error(y_test, y_pred_with_categoria)
r2_with_categoria = r2_score(y_test, y_pred_with_categoria)

    ## Preparo los datos sin "categoria".

x_without_categoria = data[['precio_m2', 'tasa_paro']]
y = data['tasa_emancipacion']

    ## Divido los datos.

x_train_without_cat, x_test_without_cat, y_train, y_test = train_test_split(x_without_categoria, y, test_size=0.2, random_state=42)

    ## Entreno el modelo sin la variable "categoria".

model_without_categoria = LinearRegression().fit(x_train_without_cat, y_train)
y_pred_without_categoria = model_without_categoria.predict(x_test_without_cat)
ecm_without_categoria = mean_squared_error(y_test, y_pred_without_categoria)
r2_without_categoria = r2_score(y_test, y_pred_without_categoria)

    ## Veo los resultados.

print(f"Modelo sin 'categoria': ECM = {ecm_without_categoria}, R2 = {r2_without_categoria}")
print(f"Modelo con 'categoria': ECM = {ecm_with_categoria}, R2 = {r2_with_categoria}")

    ## Visualizo los resultados.

plt.figure(figsize=(14, 6))

    ## Predicciones del modelo sin "categoria".

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_without_categoria, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.title("Modelo sin 'categoria'\nECM = {:.2f}, R2 = {:.2f}".format(ecm_without_categoria, r2_without_categoria))

    ## Predicciones del modelo con "categoria".

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_with_categoria, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.title("Modelo con 'categoria'\nECM = {:.2f}, R2 = {:.2f}".format(ecm_with_categoria, r2_with_categoria))

plt.tight_layout()
plt.show()

"""
Comparación de errores de entrenamiento y prueba:
	•	En ambos modelos, el ECM de prueba es ligeramente mayor que el ECM de entrenamiento.
	•	El modelo con categoria tiene un ECM de prueba ligeramente menor.
	•	El R-Cuadrado es muy alto (0.989).
"""

# QUIERO HACER UN SUMMARY DE UN MODELO CON TODAS LAS VARIABLES PARA VER CÓMO AFECTAN LAS VARIABLES ENTRE SÍ.

    ## Cargo de nuevo los datos.

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

    ## Selecciono las variables numéricas.

numericas = data.select_dtypes(include=["float64", "int64"]).columns

    ## Estandarizo los datos de nuevo.

escalar = StandardScaler()
data_estandar = data.copy()
data_estandar[numericas] = escalar.fit_transform(data[numericas])

    ## Normalizo los datos estandarizados de nuevo.

normalizar = MinMaxScaler()
data_normal = data_estandar.copy()
data_normal[numericas] = normalizar.fit_transform(data_estandar[numericas])

    ## Defino las variables predictoras (X) y la variable dependiente (y).

x = data_normal[numericas].drop(columns=["tasa_emancipacion"])
y = data_normal["tasa_emancipacion"]

    ## Divido los datos en conjuntos de entrenamiento y prueba.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    ## Agrego una constante para el intercepto.

x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

    ## Ajusto el modelo de regresión lineal.

modelo1 = sm.OLS(y_train, x_train).fit()

    ## Veo el summary del modelo.

summary = modelo1.summary()
print(summary)
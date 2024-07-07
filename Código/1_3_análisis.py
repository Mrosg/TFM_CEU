import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# IMPORTO MI BASE DE DATOS.

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

# CALCULO PROMEDIOS DE CADA UNO DE LOS GRUPOS PARA LAS VARIABLES TOTALES DE PARO.

    ## Hago la media de paro en cada grupo.

promedios_paro = data.groupby("categoria")["paro_total"].mean()

print(promedios_paro)

"""
Los resultados son:
    - Alto: 18843.262626
    - Medio-Alto: 20684.208113
    - Medio: 16787.420290
    - Medio-Bajo: 18539.741710
    - Bajo: 12518.009756
El número es más alto en los barrios "Alto" y "Medio-Alto" que en los "Bajo".
Para contextualizar mejor los datos voy a utilizar la cifra de población de cada distrito y no sólo la de paro total.
"""

promedios_tasa_paro = data.groupby("categoria")["tasa_paro"].mean()

print(promedios_tasa_paro)

"""
Ahora los resultados son:
    - Alto: 4.057559
    - Medio-Alto: 4.903951
    - Medio: 5.442576
    - Medio-Bajo: 6.321187
    - Bajo: 8.135659
Estos datos tienen más sentido que los anteriores.
Es más coherente que la tasa de paro sea mayor en barrios "pobres" que en "ricos". De hecho, es el doble.
"""

# DIAGRAMA POLAR CON BARRAS SOBRE LA TASA DE PARO POR CATEGORÍA DE DISTRITO.

    ## Datos para el gráfico.
categorias = promedios_tasa_paro.index
valores = promedios_tasa_paro.values
    ## Calculo los ángulos que tendrán las barras del gráfico.
num_categorias = len(categorias)
angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint = False).tolist()
valores = np.concatenate((valores, [valores[0]]))  # Repito el primer valor al final para cerrar el gráfico.
angulos += angulos[:1]
    ## Personalizo el estilo visual del gráfico.
plt.style.use("ggplot")
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize = (10, 10))
width = 2 * np.pi / num_categorias
bars_bg = ax.bar(x = angulos[:-1], height = [max(valores)]*len(angulos[:-1]), width = width, color = "lightgrey",
                 edgecolor = "white", zorder = 1, alpha = 0.2)
bars = ax.bar(x = angulos[:-1], height = valores[:-1], width = width, 
              edgecolor = "white", zorder = 2, alpha = 0.8, color = "dodgerblue")
    ## Cambio las etiquetas.
etiquetas_personalizadas = ["Alta", "Baja", "Media", "Media-Alta", "Media-Baja"]
    ## Bucle para los ángulos, valores y etiquetas personalizados de cada barra.
for angle, height, categoria in zip(angulos[:-1], valores[:-1], etiquetas_personalizadas):
    rotation_angle = np.degrees(angle)
    if angle < np.pi:
        rotation_angle -= 90
    elif angle == np.pi:
        rotation_angle -= 90
    else:
        rotation_angle += 90
    ax.text(angle, max(valores)*1.1, categoria, 
            ha = "center", va = "center", 
            rotation = rotation_angle, rotation_mode = "anchor", fontsize = 14, color = "black")
    ## Añado título y autor.
ax.set_title("Tasa de paro por clase de distrito en Madrid", va=  "bottom", fontsize = 18, color = "black")
fig.text(0.1, 0.05, "Autor: Miguel Ros García", ha = "left", fontsize = 10, color = "black")
    ## Configuro del eje Y para que incluya el símbolo de porcentaje.
ax.set_yticks(np.arange(0, max(valores) * 1.2, max(valores) * 0.2))
ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, max(valores) * 1.2, max(valores) * 0.2)], color = "black")
ax.set_xticks([])
ax.grid(alpha = 0.3, color = "black", lw = 1.5)
plt.ylim(0, max(valores)*1.2)
plt.show()

# DIAGRAMA POLAR CON BARRAS SOBRE EL TOTAL DE PERSONAS CON DISTINTA CATEGORÍA DE ESTUDIOS.

    ## Defino las variables.
columnas = ["no_estudios_total", "estudios_primarios_incompletos_total",
                "estudios_primarios_total", "programa_fp_total",
                "educacion_general_total", "estudios_tecnico_profesionales_superiores_total",
                "estudios_universitarios_ciclo1_total", "estudios_universitarios_ciclo2y3_total",
                "otros_total"]
    ## Calculo la suma de cada variable.
valores = data[columnas].sum().values
    ## Categorías para el gráfico.
categorias = columnas
    ## Calculate angles for the bars
num_categorias = len(categorias)
angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()
valores = np.concatenate((valores, [valores[0]]))  # Repito el primer valor para cerrar el gráfico.
angulos += angulos[:1]
    ## Personalizo el estilo del gráfico.
plt.style.use("ggplot")
fig, ax = plt.subplots(subplot_kw = {"projection": "polar"}, figsize = (10, 10))
    ## Defino el ancho de cada barra.
width = 2 * np.pi / num_categorias
    ## Hago las barras de fondo.
bars_bg = ax.bar(x = angulos[:-1], height = [max(valores)]*len(angulos[:-1]), width = width, color = "lightgrey",
                 edgecolor = "white", zorder = 1, alpha = 0.2)
    ## Hago los valores reales.
bars = ax.bar(x = angulos[:-1], height = valores[:-1], width = width, edgecolor = "white", zorder = 2, alpha = 0.8, color = "dodgerblue")
    ## Personalizo las etiquetas.
etiquetas = ["No estudios", "Primarios incompletos", "Primarios", "FP", "Educación general",
          "Técnico profesional", "Universitarios ciclo 1", "Universitarios ciclo 2 y 3", "Otros"]
for angle, height, label in zip(angulos[:-1], valores[:-1], etiquetas):
    rotation_angle = np.degrees(angle)
    if angle < np.pi:
        rotation_angle -= 90
    elif angle == np.pi:
        rotation_angle -= 90
    else:
        rotation_angle += 90
    ax.text(angle, max(valores)*1.1, label, ha = "center", va = "center", rotation = rotation_angle, rotation_mode = "anchor",
            fontsize = 12, color = "black")
    ## Añadir título y autor.
ax.set_title("Paro en Madrid por niveles de estudios", va = "bottom", fontsize = 18, color = "black")
fig.text(0.1, 0.05, "Autor: Miguel Ros García", ha = "left", fontsize = 10, color = "black")
    ## Configurar el eje Y.
ax.set_yticks(np.arange(0, max(valores) * 1.2, max(valores) * 0.2))
ax.set_yticklabels([f'{int(y)}' for y in np.arange(0, max(valores) * 1.2, max(valores) * 0.2)], color = "black")
    ## Eliminar las marcas del eje X.
ax.set_xticks([])
    ## Personalizo la cuadrícula y los límites del gráfico.
ax.grid(alpha = 0.3, color = "black", lw = 1.5)
plt.ylim(0, max(valores)*1.2)
    ## Muestro el gráfico.
plt.show()

# TEST ANOVA DE I VÍA CENTRADO EN LA VARIABLE TOTAL DE EGRESADOS EN CADA GRUPO DE DISTRITO.

    ## Selecciono las columnas que voy a analizar.

columnas = ["no_estudios_total", "estudios_primarios_incompletos_total",
                "estudios_primarios_total", "programa_fp_total",
                "educacion_general_total", "estudios_tecnico_profesionales_superiores_total",
                "estudios_universitarios_ciclo1_total", "estudios_universitarios_ciclo2y3_total",
                "otros_total"]

    ## Elimino posibles NAs por si acaso.

data_filtro = data[["categoria"] + columnas].dropna()

    ## Selecciono los grupos de "categoria".

categorias = data_filtro["categoria"].unique()

    ## Creo un diccionario para almacenar los resultados del test ANOVA de I vía.

resultados_anova = {categoria: {} for categoria in categorias}

   ## Agrupo los resultados por categoría y hago el test ANOVA de I vía para cada columna en cada grupo de distrito. 

for categoria in categorias:
    ### Creo un dataframe sin la categoría actual.
    categoria_otra = data_filtro[data_filtro["categoria"] != categoria]
    ### Separo el grupo de la categoría actual.
    categoria_actual = data_filtro[data_filtro["categoria"] == categoria]

    for column in columnas:
        #### Obtengo los datos de la columna actual para el grupo de la categoría actual y los otros grupos.
        samples = [categoria_actual[column].values, categoria_otra[column].values]
        #### Hago el test ANOVA de I vía solo si ambos grupos tienen más de una muestra.
        if all(len(sample) > 1 for sample in samples):
            anova_result = f_oneway(*samples)
            resultados_anova[categoria][column] = anova_result

    ## Convierto los resultados en un dataframe para visualizarlos.
            
test_anova = pd.DataFrame(resultados_anova)
anova_visualizar = test_anova.applymap(lambda x: f"F = {x.statistic:.2f}, p = {x.pvalue:.4f}" if isinstance(x, tuple) else x)
print(anova_visualizar)

# PRUEBA ANOVA I VÍA.

prueba_anova = f_oneway(data["estudios_universitarios_ciclo2y3_total"], data["estudios_universitarios_ciclo1_total"],
                        data["estudios_tecnico_profesionales_superiores_total"], data["educacion_general_total"],
                        data["programa_fp_total"], data["estudios_primarios_total"], data["estudios_primarios_incompletos_total"],
                        data["otros_total"], data["no_estudios_total"])
print(prueba_anova)

# PRUEBA 2 ANOVA I VÍA.

    ## Selecciono los grupos de "categoria".

categorias = data_filtro["categoria"].unique()

    ## Creo un diccionario para almacenar los resultados del test ANOVA de una vía.

resultados_anova = {}

    ## Agrupo los resultados por categoría y hago el test ANOVA de una vía para las columnas de interés.

for categoria in categorias:
        ### Filtrar los datos por la categoría actual.
    current_category_data = data_filtro[data_filtro["categoria"] == categoria]
        ### Obtener los datos de las columnas de interés.
    samples = [current_category_data[column] for column in columnas]
        ### Realizar el test ANOVA de una vía.
    anova_result = f_oneway(*samples)
        ### Almacenar el resultado en el diccionario.
    resultados_anova[categoria] = anova_result

    ## Convertir los resultados en un DataFrame para visualizarlos.

resultados_df = pd.DataFrame(resultados_anova, index=['F-value', 'p-value']).T
print(resultados_df)

"""
P_Valor > 0.05 = aceptamos la hipótesis nula y damos por hecho que no hay diferencia entre variables.
P_Valor < 0.05 = rechazamos la hipótesis nula. Ergo, afirmamos que hay una diferencia significativa entre variables.
"""

# PCA (PRINCIPAL COMPONENT ANALYSIS) DE 4 ELEMENTOS.

    ## Importo las librerías necesarias.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

    ## Cargo de nuevo la base de datos (aunque ya está cargada previamente en el código).

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

    ## Selecciono las columnas que quiero analizar con PCA.

        ### Selecciono todas las columnas menos las de variación del alquiler, ya que tiene números negativos.

columnas_pca = data.iloc[:, 3:]

    ## Estandarizo los datos para hacer el PCA.

scaler = StandardScaler()
data_std = scaler.fit_transform(columnas_pca)

    ## Realizo el PCA.

pca = PCA(n_components = 4)
componentes_principales = pca.fit_transform(data_std)

    ## Resultados.

data_pca = pd.DataFrame(data = componentes_principales, columns = ["PC1", "PC2", "PC3", "PC4"])

    ## Vemos los datos del porcentaje de varianza explicada por cada componente principal.

varianza = pca.explained_variance_ratio_

   ## Visualizo los resultados.

        ### Configuración de estilo.
plt.style.use("ggplot")
        ### Creo la figura y los subplots.
fig, axs = plt.subplots(2, 2, figsize = (14, 10))
        ### Listas para los títulos y los ejes. 
titulos = [f"Componente Principal {i} vs. Componente Principal {(i % 4) + 1}" for i in range(1, 5)]
eje_x = [f"Componente Principal {i}" for i in range(1, 5)]
eje_y = [f"Componente Principal {(i % 4) + 1}" for i in range(1, 5)]
        ### Añado colores para los puntos.
colores = np.linspace(0, 1, len(data_pca))
        ### Represento los datos.
for i, ax in enumerate(axs.flatten(), start = 1):
    sc = ax.scatter(data_pca[f"PC{i}"], data_pca[f"PC{(i % 4) + 1}"], c = colores, cmap = "viridis", alpha = 0.6, edgecolors = "w", s = 50)
    ax.set_title(titulos[i-1], fontsize = 14)
    ax.set_xlabel(eje_x[i-1], fontsize = 12)
    ax.set_ylabel(eje_y[i-1], fontsize = 12)
    ax.grid(True, linestyle = "--", alpha = 0.7)
        ### Ajusto la posición de la figura para hacer espacio para la barra de colores.
fig.subplots_adjust(right = 0.85)
        ### Añado una barra de color a modo de leyenda de colores.
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label("Leyenda de colores", rotation = 270, labelpad = 15)
        ### Añado título y autor.
fig.suptitle("Análisis de Componentes Principales (PCA)", fontsize = 16, fontweight = "bold")
fig.text(0.1, 0.01, "Autor: Miguel Ros García", ha = "left", fontsize = 12, style = "italic", fontweight = "bold")
        ### Ajusto el rectángulo del layout para que no se solape.
plt.tight_layout(rect = [0, 0.03, 0.85, 0.95])
plt.show()

"""
Componentes Principales:

PC1: recoge el 94.67% de la variabilidad de los datos.
    - Este Componente Principal representa bien las diferencias entre los datos.
PC2: recoge el 2.8% de la variabilidad de los datos.
PC3: recoge el 1.8% de la variabilidad de los datos.
PC4: recoge el 0.35% de la variabilidad de los datos.
    - Estos tres Componentes Principales la variabilidad es mucho menor, por lo que es menos relevante para el análisis.

Los colores representan que, cuando más oscuro es el punto, más cerca del final de la muestra es: es más reciente en el tiempo. 
"""

# K-MEANS PREVIO PCA.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

    ## Antes de aplicar los K-Means, quiero ver cuál es el número de clústers óptimo para mis datos.

"""
Método que, mediante la suma de los cuadrados dentro de los clústers,
se ve la inercia de los datos (cómo de bien están agrupados los datos).

Con un gráfico se puede ver cómo disminuye la inercia y cuál es el número óptimo de clústers.
"""
        ### Hago diferentes números de clústers y almaceno la suma de los cuadrados.

suma_cuadrados = []

for i in range(2, 10):
    kmeans = KMeans(n_clusters = i, n_init = 20, random_state = 0)
    kmeans.fit(data_pca)
    suma_cuadrados.append(kmeans.inertia_)
        
        ### Visualizo los datos con diferente número de clústers.
    
    plt.figure(figsize = (6, 4))
    plt.scatter(data_pca.iloc[:, 0], data_pca.iloc[:, 1], c = kmeans.labels_, s = 50, cmap = "viridis")
    centros = kmeans.cluster_centers_
    plt.scatter(centros[:, 0], centros[:, 1], c = "red", s = 200, alpha = 0.75)
    plt.title(f"K-means con K={i}")
    plt.show()

        ### Visualizo cuál es el número óptimo de clústers para mis datos.

plt.figure(figsize = (8, 6))
plt.plot(range(2, 10), suma_cuadrados, "bo-", color = "blue", linewidth = 2)
plt.xlabel("Número de clústers")
plt.ylabel("Total de la suma de los cuadrados")
plt.title("Método del 'codo' para ver el número óptimo de K")
plt.grid(True)
plt.show()

    ## Aplico K-Means con tres clústers a los resultados del PCA.

kmeans = KMeans(n_clusters = 3, random_state = 42)  # Ajusto el parámetro K = 3.
kmeans.fit(data_pca)

    ## Añado las etiquetas de los clústeres al dataframe.

data_pca["Cluster"] = kmeans.labels_

    ## Visualizo los resultados con K-Means.

plt.style.use("ggplot")
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
titulos = [f"Componente Principal {i} vs. Componente Principal {(i % 4) + 1}" for i in range(1, 5)]
eje_x = [f"Componente Principal {i}" for i in range(1, 5)]
eje_y = [f"Componente Principal {(i % 4) + 1}" for i in range(1, 5)]

for i, ax in enumerate(axs.flatten(), start=1):
    sc = ax.scatter(data_pca[f"PC{i}"], data_pca[f"PC{(i % 4) + 1}"], c=data_pca["Cluster"], cmap = "viridis",
                    alpha = 0.6, edgecolors = "w", s = 50)
    ax.set_title(titulos[i-1], fontsize = 14)
    ax.set_xlabel(eje_x[i-1], fontsize = 12)
    ax.set_ylabel(eje_y[i-1], fontsize = 12)
    ax.grid(True, linestyle = "--", alpha = 0.7)

fig.subplots_adjust(right=0.85)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sc, cax = cbar_ax)
cbar.set_label("Clústeres", rotation = 270, labelpad = 15)

fig.suptitle("Análisis de Componentes Principales (PCA) con K-Means", fontsize = 16, fontweight = "bold")
fig.text(0.1, 0.01, "Autor: Miguel Ros García", ha = "left", fontsize = 12, style = "italic", fontweight = "bold")

plt.tight_layout(rect = [0, 0.03, 0.85, 0.95])
plt.show()

# CALCULAR LOS INTERVALOS DE CONFIANZA PARA EL PRECIO ENTRE EL 90%-99%.

import scipy.stats as stats

    ## Añado la columna "Cluster" al dataset.

data["Cluster"] = data_pca["Cluster"]

    ## Hago una función para calcular los intervalos de confianza.

def calcular_intervalo_confianza(grupo, nivel_confianza):
    media = np.mean(grupo)
    sem = stats.sem(grupo)  # Error estándar de la media
    intervalo = stats.t.interval(nivel_confianza, len(grupo)-1, loc = media, scale = sem)
    return intervalo

    ## Niveles de confianza a evaluar.

niveles_confianza = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

    ## Calculo los intervalos de confianza para cada clúster y cada nivel de confianza.

resultados_intervalos = {}
for cluster in data["Cluster"].unique():
    grupo = data[data["Cluster"] == cluster]["precio_m2"]
    intervalos = {nivel: calcular_intervalo_confianza(grupo, nivel) for nivel in niveles_confianza}
    resultados_intervalos[cluster] = intervalos

    ## Compruebo cuál es el mejor intervalo de confianza para cada clúster.

mejores_intervalos = {}
for cluster, intervalos in resultados_intervalos.items():
    mejores_intervalos[cluster] = max(intervalos.items(), key = lambda x: x[0])

    ## Convierto los resultados a un dataframe para una mejor visualización.

df_mejores_intervalos = pd.DataFrame(mejores_intervalos, index = ["Nivel de Confianza", "Intervalo"]).T

    ## Visualizo los intervalos de confianza para cada clúster.

plt.figure(figsize = (12, 6))
colors = ["b", "g", "r"]
for idx, (cluster, data) in enumerate(resultados_intervalos.items()):
    niveles = list(data.keys())
    intervalos = list(data.values())
    medias = [np.mean(intervalo) for intervalo in intervalos]
    lower_bounds = [intervalo[0] for intervalo in intervalos]
    upper_bounds = [intervalo[1] for intervalo in intervalos]

    plt.plot(niveles, medias, color = colors[idx], marker = "o", label = f"Cluster {cluster}")
    plt.fill_between(niveles, lower_bounds, upper_bounds, color = colors[idx], alpha = 0.2)

plt.xlabel("Nivel de confianza")
plt.ylabel("Precio por m2")
plt.title("Intervalos de Confianza del Precio por m2 por clúster")
plt.legend(title = "Clústers")
plt.grid(True)
plt.show()

"""
Un intervalo de confianza es un rango de valores, derivado de los datos de la muestra, que se espera que contenga
el valor verdadero de un parámetro poblacional con un cierto nivel de confianza.
En este caso, estamos interesados en los intervalos de confianza del precio por metro cuadrado (precio_m2) en diferentes clústers.

Resultados:

	1. Clúster 0:
	    • Mejor nivel de confianza: 99%
	    • Intervalo de confianza: (12.35, 12.63)
	    • Interpretación: Con un nivel de confianza del 99%, podemos decir que el verdadero precio promedio por metro cuadrado para los datos en el Cluster 0 está entre 12.35 y 12.63 euros. Esto significa que si repitiéramos el muestreo muchas veces, el 99% de las veces el precio promedio por metro cuadrado caería dentro de este rango.
	2. Clúster 2:
	    • Mejor nivel de confianza: 99%
	    • Intervalo de confianza: (11.40, 11.74)
	    • Interpretación: Con un nivel de confianza del 99%, podemos decir que el verdadero precio promedio por metro cuadrado para los datos en el Cluster 2 está entre 11.40 y 11.74 euros. Esto indica que el 99% de las veces, el precio promedio por metro cuadrado caería dentro de este rango si repitiéramos el muestreo muchas veces.
	3. Clúster 1:
	    • Mejor nivel de confianza: 99%
	    • Intervalo de confianza: (14.84, 15.56)
	    • Interpretación: Con un nivel de confianza del 99%, podemos decir que el verdadero precio promedio por metro cuadrado para los datos en el Cluster 1 está entre 14.84 y 15.56 euros. Esto implica que si repitiéramos el muestreo muchas veces, el 99% de las veces el precio promedio por metro cuadrado caería dentro de este rango.

Estos intervalos de confianza nos dan una idea de la variabilidad y la precisión de nuestras estimaciones
del precio por metro cuadrado en diferentes clusters.
He seleccionado el intervalo de confianza más estrecho con el mayor nivel de confianza (99%) para cada clúster,
lo que proporciona una estimación precisa y fiable del precio promedio por metro cuadrado en cada grupo.

	- Clúster 0: rango de precios más estrecho y centrado alrededor de 12.49 euros.
	- Clúster 2: rango de precios centrado alrededor de 11.57 euros.
	- Clúster 1: rango de precios más alto, centrado alrededor de 15.20 euros.
"""

# KOLMOGOROV-SMIRNOV SOBRE EL PARO EN LA MEDIANA EDAD.

import pandas as pd
from scipy.stats import kstest, norm

    ## Cargo de nuevo el dataset.

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"  # Asegúrate de ajustar la ruta a tu archivo
data = pd.read_csv(ruta)

    ## Me quedo con las columnas relativas a la edad.

edad_paro = ["paro_16_19_total", "paro_20_24_total", "paro_25_29_total", "paro_30_34_total", "paro_35_39_total", "paro_40_44_total",
             "paro_45_49_total", "paro_50_54_total", "paro_55_59_total", "paro_60_mas_total"]

    ## Calculo la mediana de edad para cada fila.

data["mediana_edad_paro"] = data[edad_paro].median(axis=1)

    ## Hago una variable de la mediana de edad del paro.

mediana_edad_paro = data["mediana_edad_paro"]

    ## Calculo la media y la desviación estándar de la variable.

mean_mediana_edad = mediana_edad_paro.mean()
std_mediana_edad = mediana_edad_paro.std()

    ## Realizo la prueba Kolmogorov-Smirnov comparando con una distribución normal.

ks_statistic_mediana, p_value_mediana = kstest(mediana_edad_paro, "norm", args = (mean_mediana_edad, std_mediana_edad))

    ## Vemos los resultados.

print(f"Estadística Kolmogorov-Smirnov: {ks_statistic_mediana}")
print(f"Valor p: {p_value_mediana}")

# KOLMOGOROV-SMIRNOV SOBRE RESULTADOS DE PCA.

from scipy.stats import kstest
import pandas as pd

    ## Compruebo que la columna "Clúster" esté presente en mi dataframe.

if "Cluster" not in data.columns:
    data["Cluster"] = data_pca["Cluster"]

    ## Hago el test de Kolmogorov-Smirnov sobre los componentes principales.

ks_results = {}
for col in data_pca.columns[:-1]:  # Excluyendo la columna 'Cluster'
    data_normalized = (data_pca[col] - data_pca[col].mean()) / data_pca[col].std() # Normalizo los datos antes de aplicar el test KS.
    statistic, p_value = kstest(data_normalized, "norm")
    ks_results[col] = {"Estadístico KS": statistic, "P-Valor": p_value}

    ## Vemos los resultados del test KS para las componentes principales.

df_ks_results = pd.DataFrame(ks_results).T

    ## Vemos gráficamente os resultados del test KS para PCA.

fig, axs = plt.subplots(2, 2, figsize = (14, 10))
componentes = data_pca.columns[:-1]  # Excluyendo la columna 'Cluster'
for i, ax in enumerate(axs.flatten()):
    componente = componentes[i]
    data_comp = data_pca[componente]
    ax.hist(data_comp, bins = 30, density = True, alpha = 0.6, color = "blue")
        ### Ajusto una distribución normal y la superpongo.
    mu, std = norm.fit(data_comp)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, "k", linewidth = 2)
    ax.set_title(f"Distribución de {componente} vs Normal")
        ### Añado el resultado del test KS.
    ks_statistic, p_value = ks_results[componente]["Estadístico KS"], ks_results[componente]["P-Valor"]
    ax.text(0.05, 0.95, f"Estadístico KS: {ks_statistic:.2f}\nP-Valor: {p_value:.2e}", 
            transform = ax.transAxes, verticalalignment = "top", bbox = dict(facecolor = "white", alpha = 0.8))

plt.suptitle("Test Kolmogorov-Smirnov para Componentes Principales del PCA")
plt.tight_layout(rect = [0, 0.03, 1, 0.95])
plt.show()

"""
La visualización de los histogramas del test Kolmogorov-Smirnov sobre los Componentes Principales de PCA muestran:

- Cómo se distribuye los valores de los Componentes Principales.
- Cómo se alinean los datos con una distribución normal teórica ajustada a los datos.
- Valores:
    · Estadística KS: diferencia máxima entre la distribución de los datos y la distribución normal.
    · P-Valor: indica si la diferencia es significativa. Un P-Valor < 0.05 sugiere que los datos no siguen una distribución normal.

Los P-Valores son muy bajos. Los Componentes Princpales no siguen una distribucón normal.
"""

# KOLMOGOROV-SMIRNOV SOBRE RESULTADOS DE K-MEANS.

    ## Hago el test de Kolmogorov-Smirnov para los clústers obtenidos de K-Means.

ks_cluster_results = {}
for cluster in data["Cluster"].unique():
    cluster_data = data[data["Cluster"] == cluster]["precio_m2"]
    cluster_data_normalized = (cluster_data - cluster_data.mean()) / cluster_data.std() # Normalizo los datos antes de aplicar el test KS.
    statistic, p_value = kstest(cluster_data_normalized, "norm")
    ks_cluster_results[cluster] = {"Estadístico KS": statistic, "P-Valor": p_value}

    ## Vemos los resultados del test KS para los clústers.
    
df_ks_cluster_results = pd.DataFrame(ks_cluster_results).T

df_ks_results, df_ks_cluster_results

    ## Vemos gráficamente los resultados del test KS para los clústers de K-Means.

fig, axs = plt.subplots(1, 3, figsize = (18, 6))
for i, ax in enumerate(axs.flatten()):
    cluster = i
    data_cluster = data[data["Cluster"] == cluster]["precio_m2"]
    ax.hist(data_cluster, bins = 30, density=True, alpha = 0.6, color = "blue")
        ### Ajusto una distribución normal y la superpongo.
    mu, std = norm.fit(data_cluster)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, "k", linewidth = 2)
    ax.set_title(f"Distribución del clúster {cluster} vs Normal")
        ### Añado el resultado del test KS.
    ks_statistic, p_value = ks_cluster_results[cluster]["Estadístico KS"], ks_cluster_results[cluster]["P-Valor"]
    ax.text(0.05, 0.95, f"Estadístico KS: {ks_statistic:.2f}\nP-Valor: {p_value:.2e}", 
            transform = ax.transAxes, verticalalignment = "top", bbox = dict(facecolor = "white", alpha = 0.8))

plt.suptitle("Test Kolmogorov-Smirnov para los clústers del K-Means")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""
Los gráficos de resultados del test Kolmogorov-Smirnov sobre los clústers del K-Means muestra:

- La distribución de los precios del suelo de alquiler por m2 en cada uno de los tres clústers: 0, 1 y 2.
- La línea negra representa una distribución normal teórica ajustada a los datos.
- Como en el test KS del PCA, se muestra el Estadístico KS y el P-Valor.

El clúster 0 tiene un P-Valor muy bajo. Los precios por m2 no siguen una distribución normal.
El clúster 1 tiene un P-Valor más alto aunque sigue siendo bajo. Los datos pueden no ser normales aunque no se desvían tanto.
El clúster 2 tiene el P-Valor más alto, pero sigue siendo relativamente bajo. Ligera desviación de la normalidad.
"""
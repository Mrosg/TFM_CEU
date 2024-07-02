import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
column_names = ["no_estudios_total", "estudios_primarios_incompletos_total",
                "estudios_primarios_total", "programa_fp_total",
                "educacion_general_total", "estudios_tecnico_profesionales_superiores_total",
                "estudios_universitarios_ciclo1_total", "estudios_universitarios_ciclo2y3_total",
                "otros_total"]
    ## Calculo la suma de cada variable.
values = data[column_names].sum().values
    ## Categorías para el gráfico
categories = column_names
# Calculate angles for the bars
num_categories = len(categories)
angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
values = np.concatenate((values, [values[0]]))  # Repito el primer valor para cerrar el gráfico.
angles += angles[:1]
    ## Personalizo el estilo del gráfico.
plt.style.use("ggplot")
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 10))
    ## Defino el ancho de cada barra.
width = 2 * np.pi / num_categories
    ## Hago las barras de fondo.
bars_bg = ax.bar(x=angles[:-1], height=[max(values)]*len(angles[:-1]), width=width, color="lightgrey",
                 edgecolor="white", zorder=1, alpha=0.2)
    ## Hago los valores reales.
bars = ax.bar(x=angles[:-1], height=values[:-1], width=width, edgecolor="white", zorder=2, alpha=0.8, color="dodgerblue")
    ## Personalizo las etiquetas.
labels = ["No estudios", "Primarios incompletos", "Primarios", "FP", "Educación general",
          "Técnico profesional", "Universitarios ciclo 1", "Universitarios ciclo 2 y 3", "Otros"]
for angle, height, label in zip(angles[:-1], values[:-1], labels):
    rotation_angle = np.degrees(angle)
    if angle < np.pi:
        rotation_angle -= 90
    elif angle == np.pi:
        rotation_angle -= 90
    else:
        rotation_angle += 90
    ax.text(angle, max(values)*1.1, label, ha="center", va="center", rotation=rotation_angle, rotation_mode="anchor", fontsize=12, color="black")
    ## Añadir título y autor.
ax.set_title("Paro en Madrid por niveles de estudios", va="bottom", fontsize=18, color="black")
fig.text(0.1, 0.05, "Autor: Miguel Ros García", ha="left", fontsize=10, color="black")
    ## Configurar el eje Y.
ax.set_yticks(np.arange(0, max(values) * 1.2, max(values) * 0.2))
ax.set_yticklabels([f'{int(y)}' for y in np.arange(0, max(values) * 1.2, max(values) * 0.2)], color="black")
    ## Eliminar las marcas del eje X.
ax.set_xticks([])
    ## Personalizo la cuadrícula y los límites del gráfico.
ax.grid(alpha=0.3, color="black", lw=1.5)
plt.ylim(0, max(values)*1.2)
    ## Muestro el gráfico.
plt.show()
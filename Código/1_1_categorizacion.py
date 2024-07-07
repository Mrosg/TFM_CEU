import pandas as pd

# IMPORTO EL CSV.

dataset_aux = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET.csv")

# PREAPARACIÓN DE DATAFRAME.

dataset_aux["precio_m2"] = dataset_aux["precio_m2"].str.replace(',', '.').astype(float) # Cambio la variable a tipo float.
dataset_aux["tasa_paro"] = dataset_aux["tasa_paro"].str.replace(',', '.').astype(float) # Cambio la variable a tipo float.
precio_m2 = dataset_aux[["fecha", "distrito", "precio_m2"]] # Hago un dataframe solo con las variables que quiero estudiar.

print(precio_m2.head())

# EXPLORO LOS DATOS.

    ## Veo los datos máximos y mínimos de "precio_m2" y obtengo su índice.

precio_maximo = precio_m2["precio_m2"].max()
precio_minimo = precio_m2["precio_m2"].min()
indice_maximo = precio_m2["precio_m2"].idxmax()
indice_minimo = precio_m2["precio_m2"].idxmin()

    ## Saco los valores correspondientes a "distrito" y "fecha".

distrito_maximo = precio_m2.loc[indice_maximo, "distrito"]
fecha_maximo = precio_m2.loc[indice_maximo, "fecha"]
distrito_minimo = precio_m2.loc[indice_minimo, "distrito"]
fecha_minimo = precio_m2.loc[indice_minimo, "fecha"]
    
    ## Vemos los resultados.

print("Valor máximo:", precio_maximo, "en", distrito_maximo, "el", fecha_maximo)
print("Valor máximo:", precio_minimo, "en", distrito_minimo, "el", fecha_minimo)

    ## Estadísticas descriptivas.

estadisticas = precio_m2["precio_m2"]. describe()
print(estadisticas)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

    ## Boxplot de "precio_m2" por cada distrito.

plt.figure(figsize=(15, 10))
sns.boxplot(x='distrito', y='precio_m2', data=precio_m2)
plt.title('Boxplot de precios por m2 por distrito')
plt.xlabel('Distrito')
plt.ylabel('Precio por m2')
plt.xticks(rotation=45)
plt.grid(True)
#plt.show()

    ## Hago un boxplot interactivo de "precio_m2" por cada distrito.

        ### Hago el gráfico.
boxplot_precio_distrito_interactivo = px.box(precio_m2, x='distrito', y='precio_m2', 
             title='Boxplot de precios por m2 por distrito',
             labels={'distrito': 'Distrito', 'precio_m2': 'Precio por m2'},
             template='plotly_white')
        ### Personalizo el diseño del gráfico.
boxplot_precio_distrito_interactivo.update_traces(marker_color='blue')  # Cambiar el color de los boxplots a azul
boxplot_precio_distrito_interactivo.update_layout(
    xaxis_tickangle=-45,  # Rotar etiquetas de los distritos
    yaxis=dict(title="Precio por m2", showgrid=True, zeroline=True, showline=True, linewidth=1, linecolor='black'),
    xaxis=dict(title="Distrito", showgrid=True, zeroline=True, showline=True, linewidth=1, linecolor='black'),
    title=dict(x=0.5),  # Centrar el título
    showlegend=False,
    font=dict(size = 8))
        ### Visualizamos el gráfico.
boxplot_precio_distrito_interactivo.show()
        ### Exporto el gráfico.
ruta_boxplot_precio_distrito_interactivo = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Gráficos/boxplot_precio_por_distrito.png"
#boxplot_precio_distrito_interactivo.write_image(ruta_boxplot_precio_distrito_interactivo, scale = 2)

# DIVIDO MI DATAFRAME EN BARRIOS POBRES, MEDIOS Y RICOS CON LA MISMA PROPORCIÓN EN BASE A LOS CUARTILES DE "precio_m2".

    ## Calculo los cuartiles.

q1 = dataset_aux["precio_m2"].quantile(0.20)
q2 = dataset_aux["precio_m2"].quantile(0.40)
q3 = dataset_aux["precio_m2"].quantile(0.60)
q4 = dataset_aux["precio_m2"].quantile(0.80)

    ## Defino una función para categorizar los datos.

def categorizar(precio_m2):
    if precio_m2 <= q1:
        return "Bajo"
    elif precio_m2 <= q2:
        return "Medio-Bajo"
    elif precio_m2 <= q3:
        return "Medio"
    elif precio_m2 <= q4:
        return "Medio-Alto"
    else:
        return "Alto"

    ## Aplico la función de categorización al dataframe.

dataset_aux["categoria"] = dataset_aux["precio_m2"].apply(categorizar)

    ## Muestreo aleatorio estratificado sin duplicados.

df_bajo = dataset_aux[dataset_aux["categoria"] == "Bajo"]
df_medio_bajo = dataset_aux[dataset_aux["categoria"] == "Medio-Bajo"]
df_medio = dataset_aux[dataset_aux["categoria"] == "Medio"]
df_medio_alto = dataset_aux[dataset_aux["categoria"] == "Medio-Alto"]
df_alto = dataset_aux[dataset_aux["categoria"] == "Alto"]

    ## Combino las muestras en un solo dataframe.

df_categorizado = pd.concat([df_bajo, df_medio_bajo, df_medio, df_medio_alto, df_alto])

    ## Ordeno por fecha y el orden de la nueva variable.

df_categorizado = df_categorizado.sort_values(by = "fecha", ascending = True)
col = df_categorizado.pop("categoria")
df_categorizado.insert(2, "categoria", col)

# ANALIZO LA NUEVA VARIABLE.

    ## Recuento de la categoría.

recuento_categoria = df_categorizado["categoria"].value_counts()

    ## Calculo los valores mínimo y máximo por categoría.

resumen_categoria = df_categorizado.groupby("categoria").agg(
    valor_minimo = ("precio_m2", "min"),
    valor_maximo = ("precio_m2", "max")).reset_index()

    ## Visualizo los datos de "categoria" en una ventana emergente.

print(resumen_categoria)
print(recuento_categoria)

'''
Los resultados de la categorización de los distritos son los siguientes:
    - Número de registros de cada categoría:
        · Bajo: 615
        · Medio-Bajo: 573
        · Medio: 621
        · Medio-Alto: 573
        · Alto: 594
    - Los límites para ser de cada categoría son los siguientes:
        · Bajo: 7.4€/m2 - 10.0€/m2
        · Medio-Bajo: 10.1€/m2 - 11.4€/m2
        · Medio: 11.5€/m2 - 12.8€/m2
        · Medio-Alto: 12.9€/m2 - 15.0€/m2
        · Alto: 15.1€/m2 - 23.5€/m2
'''

    ## Aplico el formato de fecha que quiero.

df_categorizado["fecha"] = pd.to_datetime(df_categorizado["fecha"], format = "%m-%Y")
df_categorizado = df_categorizado.sort_values(by = "fecha")
df_categorizado["fecha"] = df_categorizado["fecha"].dt.strftime("%m-%Y")

    ## Exporto el CSV.

df_categorizado.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/Auxiliares/DATA_CATEGORIZADO.csv", index = False)
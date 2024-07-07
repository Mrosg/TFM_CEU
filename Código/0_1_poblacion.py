import pandas as pd

# Hago una lista con todas las rutas a los datos mensuales.

rutas = ["/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2013-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2014-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2015-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2016-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2017-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2018-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2019-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2020-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2021-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2022-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-6.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-7.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-8.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-9.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-10.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-11.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2023-12.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2024-1.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2024-2.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2024-3.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2024-4.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2024-5.xlsx",
"/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/Mensuales/2024-6.xlsx"
]


# Hago una lista para almacenar los DataFrames.

dataframes = []

# Establezco una fecha inicial.

fecha_inicial = pd.to_datetime("2013-01", format="%Y-%m")
fecha_final = pd.to_datetime("2024-03", format="%Y-%m")

# Función para limpiar y convertir la columna "poblacion".

def limpiar_poblacion(valor):
    if isinstance(valor, str):
        # Eliminar puntos y convertir a entero si es posible.
        valor = valor.replace('.', '')
        if valor.isdigit():
            return int(valor)
    return None     # Devolver "None" si no es un número válido.

# Abrir cada archivo Excel.

for i, ruta in enumerate(rutas):
    df = pd.read_excel(ruta)
    df = df.iloc[6:] # Elimino las primeras 6 filas.
    df.reset_index(drop=True, inplace=True) # Reseteo el índice.
    df.drop(df.columns[2], axis=1, inplace=True) # Elimino la tercera columna (índice 2).
    df.columns = ["distrito", "barrio", "poblacion"] # Renombro las columnas.
    # Limpio y convierto la columna "poblacion".
    df["poblacion"] = df["poblacion"].apply(limpiar_poblacion)
    df["poblacion"] = df["poblacion"].astype('Int64')
    # Creo la columna de fechas.
    fecha = fecha_inicial + pd.DateOffset(months=i)
    if fecha > fecha_final:
        break  # Terminar el bucle si la fecha supera la fecha final
    df['fecha'] = fecha.strftime("%Y-%m")
    # Agrego el dataframe a la lista
    dataframes.append(df)

# Combino todos los dataframes.

df_combined = pd.concat(dataframes, ignore_index=True)

# Sumo todos los valores de "población" según el distrito y la fecha.

poblacion_por_distrito_fecha = df_combined.groupby(['distrito', 'fecha'])['poblacion'].sum().reset_index()
poblacion_por_distrito_fecha = poblacion_por_distrito_fecha.sort_values(by='fecha')

# Asegurar que la columna 'fecha' sea de tipo datetime
poblacion_por_distrito_fecha['fecha'] = pd.to_datetime(poblacion_por_distrito_fecha['fecha'], format="%Y-%m")

# Añadir columna de mes para ordenación
poblacion_por_distrito_fecha['mes'] = poblacion_por_distrito_fecha['fecha'].dt.month

# Ordenar por fecha y luego por mes
poblacion_por_distrito_fecha = poblacion_por_distrito_fecha.sort_values(by=['fecha', 'mes'])

# Eliminar la columna 'mes' si no es necesaria
poblacion_por_distrito_fecha = poblacion_por_distrito_fecha.drop(columns='mes')

# Reiniciar el índice
poblacion_por_distrito_fecha.reset_index(drop=True, inplace=True)

poblacion_por_distrito_fecha.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/pob_13_23.csv", index = False)

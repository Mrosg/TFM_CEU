import pandas as pd

alquiler = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Alquiler/alquiler_distritos_bruto.csv", encoding='latin1', sep = ";")

# Mapeo los meses abreviados a números.

meses = {
    'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 'may': '05', 'jun': '06',
    'jul': '07', 'ago': '08', 'sept': '09', 'oct': '10', 'nov': '11', 'dic': '12'
}

# Hago una función para convertir la fecha en el formato que tienen las otras tablas.

def convertir_fecha(fecha): # Nombro a la función y le doy un argumento: "fecha".
    mes_abreviado, año = fecha.split("-") # Separo el mes y el año.
    mes = meses[mes_abreviado] # Obtengo el número del mes.
    año = "20" + año # Le agrego el "20" a "24" para que quede "2024".
    return f"{mes}-{año}"

# Hago una variable con el nuevo formato de fecha.

alquiler["fecha_formateada"] = alquiler["Fecha"].apply(convertir_fecha)

# Elimino la columna antigua de fecha que no me sirve.

alquiler = alquiler.drop(["Fecha"], axis = 1)

# Creo un vector con los nombres nuevos que quiero que tenga mi dataframe y se lo asigno.

alquiler_columnas = ["distrito", "precio_m2",
                     "variacion_mensual", "variacion_trimestral",
                     "variacion_anual", "fecha"]

alquiler.columns = alquiler_columnas

# Reordeno las columnas.

alquiler_orden = ["fecha", "distrito", "precio_m2",
                     "variacion_mensual", "variacion_trimestral",
                     "variacion_anual"]

alquiler = alquiler[alquiler_orden]

print(alquiler.info())

#alquiler.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/alquiler.csv", encoding = "utf-8", index = False)
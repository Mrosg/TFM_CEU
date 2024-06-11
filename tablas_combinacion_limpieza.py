import pandas as pd
import numpy as np
import locale

# Establezco mi localización como España para modificar la fecha.

locale.setlocale(locale.LC_TIME, "es_ES.UTF-8")

# Cargo mis dataframes.

paro_sexo_edad = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/paro_sexo_edad_2013_2024.csv")
paro_estudios = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/paro_estudios_2013_2024.csv")
paro_actividad = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/paro_actividad_2013_2024.csv")

# Cambio el tipo de variable de la columna "año" de paro_actividad.

paro_actividad["año"] = paro_actividad["año"].fillna(0)
paro_actividad["año"] = paro_actividad["año"].astype(float).astype(int).astype(str)

# Creo una nueva variable que sea "fecha" a partir de "año" y "mes".

paro_sexo_edad["fecha"] = paro_sexo_edad["mes"] + " " + paro_sexo_edad["año"]
paro_sexo_edad["fecha"] = pd.to_datetime(paro_sexo_edad["fecha"], format = "%B %Y", errors = "coerce")

paro_estudios["fecha"] = paro_estudios["mes"] + " " + paro_estudios["año"]
paro_estudios["fecha"] = pd.to_datetime(paro_estudios["fecha"], format = "%B %Y", errors = "coerce")

paro_actividad["fecha"] = paro_actividad["mes"] + " " + paro_actividad["año"]
paro_actividad["fecha"] = pd.to_datetime(paro_actividad["fecha"], format = "%B %Y", errors = "coerce")

# Le doy el formato adecuado a esa variable.

paro_sexo_edad["fecha"] = paro_sexo_edad["fecha"].dt.strftime("%m-%Y")

paro_estudios["fecha"] = paro_estudios["fecha"].dt.strftime("%m-%Y")

paro_actividad["fecha"] = paro_actividad["fecha"].dt.strftime("%m-%Y")

# Cambio de orden las columnas.

columnas = paro_sexo_edad.columns.tolist()
columnas.insert(2, columnas.pop(columnas.index("fecha")))
paro_sexo_edad = paro_sexo_edad[columnas]

columnas_estudios = paro_estudios.columns.tolist()
columnas_estudios.insert(2, columnas_estudios.pop(columnas_estudios.index("fecha")))
paro_estudios = paro_estudios[columnas_estudios]

print(paro_sexo_edad.head())

print(paro_estudios.head())

print(paro_actividad.head())

# Combino los datasets.

combinado_sexo_edad_estudios = pd.merge(paro_sexo_edad, paro_estudios, on = ["fecha", "distrito"])
                                                                             
paro_completo = pd.merge(combinado_sexo_edad_estudios, paro_actividad, on = ["fecha", "distrito"])

# Quito los números en la variable "distrito" y dejo solo el nombre del distrito.

def nombre_distrito(distrito_completo):
    if isinstance(distrito_completo, str):
        partes = distrito_completo.split(".")
        if len(partes) > 1:
            return partes[1].strip()
    return distrito_completo
    
paro_completo["distrito"] = paro_completo["distrito"].apply(nombre_distrito)

print(paro_completo.info())

paro_completo.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/paro_completo.csv", index = False)

# Importo el dataset "alquiler" para meterlo en la tabla conjunta.

alquiler = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/alquiler.csv")

# Reemplazo los nombres de distritos que tienen acento y aparecen sin ese caracter por los nombres correctos.

alquiler["distrito"] = alquiler["distrito"].replace("Chamartn", "Chamartín", regex=True)
alquiler["distrito"] = alquiler["distrito"].replace("Chamber", "Chamberí", regex=True)
alquiler["distrito"] = alquiler["distrito"].replace("Tetun", "Tetuán", regex=True)
alquiler["distrito"] = alquiler["distrito"].replace("Viclvaro", "Vicálvaro", regex=True)

# Combino todo el dataset y lo exporto.

dataset1 = pd.merge(paro_completo, alquiler, on = ["fecha", "distrito"])

#dataset1.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/dataset1.csv", index = False)

# LIMPIEZA DE DATASET.

import pandas as pd

dataset1 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/dataset1.csv")

# Elimino las columnas que están duplicadas o no necesito.

columnas_eliminar = ["año_x", "mes_x", "año_y", "mes_y",
                     "total_y", "hombres_total_y", "mujeres_total_y",
                     "año", "mes", "total", "total_hombres", "total_mujeres"]

dataset1 = dataset1.drop(columnas_eliminar, axis = 1)

# Almaceno el nombre de todas las columnas y los aplico al dataset.

dataset1_columnas = ["fecha", "distrito", "paro_total", "paro_hombres_total", "paro_mujeres_total", "paro_16_19_total",
                     "paro_16_19_hombres", "paro_16_19_mujeres", "paro_20_24_total", "paro_20_24_hombres",
                     "paro_20_24_mujeres", "paro_25_29_total", "paro_25_29_hombres", "paro_25_29_mujeres",
                     "paro_30_34_total", "paro_30_34_hombres", "paro_30_34_mujeres", "paro_35_39_total",
                     "paro_35_39_hombres", "paro_35_39_mujeres", "paro_40_44_total", "paro_40_44_hombres",
                     "paro_40_44_mujeres", "paro_45_49_total", "paro_45_49_hombres", "paro_45_49_mujeres",
                     "paro_50_54_total", "paro_50_54_hombres", "paro_50_54_mujeres", "paro_55_59_total",
                     "paro_55_59_hombres", "paro_55_59_mujeres", "paro_60_mas_total", "paro_60_mas_hombres",
                     "paro_60_mas_mujeres", "no_estudios_total", "no_estudios_hombres", "no_estudios_mujeres",
                    "estudios_primarios_incompletos_total", "estudios_primarios_incompletos_hombres",
                    "estudios_primarios_incompletos_mujeres", "estudios_primarios_total", "estudios_primarios_hombres",
                    "estudios_primarios_mujeres", "programa_fp_total", "programa_fp_hombres", "programa_fp_mujeres",
                    "educacion_general_total", "educacion_general_hombres", "educacion_general_mujeres",
                    "estudios_tecnico_profesionales_superiores_total", "estudios_tecnico_profesionales_superiores_hombres",
                    "estudios_tecnico_profesionales_superiores_mujeres", "estudios_universitarios_ciclo1_total",
                    "estudios_universitarios_ciclo1_hombres", "estudios_universitarios_ciclo1_mujeres",
                    "estudios_universitarios_ciclo2y3_total", "estudios_universitarios_ciclo2y3_hombres",
                    "estudios_universitarios_ciclo2y3_mujeres", "otros_total", "otros_hombres", "otros_mujeres",
                    "otro_fp_total", "otro_fp_hombres", "otro_fp_mujeres", "agricultura_ganaderia_silvicultura_pesca_total",
                    "industrias_extractivas_total", "industria_manufacturera_total", "suministro_energia_total",
                    "suministro_agua_total", "construccion_total", "comercio_reparacion_motor_total",
                    "transporte_almacenamiento_total", "hosteleria_total", "informacion_comunicaciones_total",
                    "actividades_financieras_total", "inmobiliarias_total", "actividades_profesionales_cientificas_total",
                    "actividades_administrativas_total", "administracion_publica_total", "educacion_total",
                    "actividades_sanitarias_sociales_total", "actividades_artisticas_entretenimiento_total",
                    "otros_servicios_total", "actividades_hogar_total",
                    "actividades_organizaciones_organismos_extraterritoriales_total", "sin_empleo_anterior_total",
                    "agricultura_ganaderia_silvicultura_pesca_hombres", "industrias_extractivas_hombres",
                    "industria_manufacturera_hombres", "suministro_energia_hombres", "suministro_agua_hombres",
                    "construccion_hombres", "comercio_reparacion_motor_hombres", "transporte_almacenamiento_hombres",
                    "hosteleria_hombres", "informacion_comunicaciones_hombres", "actividades_financieras_hombres",
                    "inmobiliarias_hombres", "actividades_profesionales_cientificas_hombres",
                    "actividades_administrativas_hombres", "administracion_publica_hombres", "educacion_hombres",
                    "actividades_sanitarias_sociales_hombres", "actividades_artisticas_entretenimiento_hombres",
                    "otros_servicios_hombres", "actividades_hogar_hombres",
                    "actividades_organizaciones_organismos_extraterritoriales_hombres", "sin_empleo_anterior_hombres",
                    "agricultura_ganaderia_silvicultura_pesca_mujeres", "industrias_extractivas_mujeres",
                    "industria_manufacturera_mujeres", "suministro_energia_mujeres", "suministro_agua_mujeres",
                    "construccion_mujeres", "comercio_reparacion_motor_mujeres", "transporte_almacenamiento_mujeres",
                    "hosteleria_mujeres", "informacion_comunicaciones_mujeres", "actividades_financieras_mujeres",
                    "inmobiliarias_mujeres", "actividades_profesionales_cientificas_mujeres",
                    "actividades_administrativas_mujeres", "administracion_publica_mujeres", "educacion_mujeres",
                    "actividades_sanitarias_sociales_mujeres", "actividades_artisticas_entretenimiento_mujeres",
                    "otros_servicios_mujeres", "actividades_hogar_mujeres",
                    "actividades_organizaciones_organismos_extraterritoriales_mujeres", "sin_empleo_anterior_mujeres",
                    "precio_m2", "variacion_mensual", "variacion_trimestral", "variacion_anual"]

dataset1.columns = dataset1_columnas

# Cambio el orden de las columnas y lo aplico.

dataset1_orden_columnas = ["fecha", "distrito", "precio_m2", "variacion_mensual", "variacion_trimestral", "variacion_anual",
                        "paro_total", "paro_hombres_total", "paro_mujeres_total", "paro_16_19_total", "paro_16_19_hombres",
                        "paro_16_19_mujeres", "paro_20_24_total", "paro_20_24_hombres", "paro_20_24_mujeres",
                        "paro_25_29_total", "paro_25_29_hombres", "paro_25_29_mujeres", "paro_30_34_total",
                        "paro_30_34_hombres", "paro_30_34_mujeres", "paro_35_39_total", "paro_35_39_hombres",
                        "paro_35_39_mujeres", "paro_40_44_total", "paro_40_44_hombres", "paro_40_44_mujeres",
                        "paro_45_49_total", "paro_45_49_hombres", "paro_45_49_mujeres", "paro_50_54_total",
                        "paro_50_54_hombres", "paro_50_54_mujeres", "paro_55_59_total", "paro_55_59_hombres",
                        "paro_55_59_mujeres", "paro_60_mas_total", "paro_60_mas_hombres", "paro_60_mas_mujeres",
                        "no_estudios_total", "no_estudios_hombres", "no_estudios_mujeres",
                        "estudios_primarios_incompletos_total", "estudios_primarios_incompletos_hombres",
                        "estudios_primarios_incompletos_mujeres", "estudios_primarios_total", "estudios_primarios_hombres",
                        "estudios_primarios_mujeres", "programa_fp_total", "programa_fp_hombres", "programa_fp_mujeres",
                        "educacion_general_total", "educacion_general_hombres", "educacion_general_mujeres",
                        "estudios_tecnico_profesionales_superiores_total",
                        "estudios_tecnico_profesionales_superiores_hombres",
                        "estudios_tecnico_profesionales_superiores_mujeres", "estudios_universitarios_ciclo1_total",
                        "estudios_universitarios_ciclo1_hombres", "estudios_universitarios_ciclo1_mujeres",
                        "estudios_universitarios_ciclo2y3_total", "estudios_universitarios_ciclo2y3_hombres",
                        "estudios_universitarios_ciclo2y3_mujeres", "otros_total", "otros_hombres", "otros_mujeres", 
                        "otro_fp_total", "otro_fp_hombres", "otro_fp_mujeres", "agricultura_ganaderia_silvicultura_pesca_total",
                        "industrias_extractivas_total", "industria_manufacturera_total", "suministro_energia_total",
                        "suministro_agua_total", "construccion_total", "comercio_reparacion_motor_total",
                        "transporte_almacenamiento_total", "hosteleria_total", "informacion_comunicaciones_total",
                        "actividades_financieras_total", "inmobiliarias_total", "actividades_profesionales_cientificas_total",
                        "actividades_administrativas_total", "administracion_publica_total", "educacion_total",
                        "actividades_sanitarias_sociales_total", "actividades_artisticas_entretenimiento_total",
                        "otros_servicios_total", "actividades_hogar_total",
                        "actividades_organizaciones_organismos_extraterritoriales_total", "sin_empleo_anterior_total",
                        "agricultura_ganaderia_silvicultura_pesca_hombres", "industrias_extractivas_hombres",
                        "industria_manufacturera_hombres", "suministro_energia_hombres", "suministro_agua_hombres",
                        "construccion_hombres", "comercio_reparacion_motor_hombres", "transporte_almacenamiento_hombres",
                        "hosteleria_hombres", "informacion_comunicaciones_hombres", "actividades_financieras_hombres",
                        "inmobiliarias_hombres", "actividades_profesionales_cientificas_hombres",
                        "actividades_administrativas_hombres", "administracion_publica_hombres", "educacion_hombres",
                        "actividades_sanitarias_sociales_hombres", "actividades_artisticas_entretenimiento_hombres",
                        "otros_servicios_hombres", "actividades_hogar_hombres",
                        "actividades_organizaciones_organismos_extraterritoriales_hombres", "sin_empleo_anterior_hombres",
                        "agricultura_ganaderia_silvicultura_pesca_mujeres", "industrias_extractivas_mujeres",
                        "industria_manufacturera_mujeres", "suministro_energia_mujeres", "suministro_agua_mujeres",
                        "construccion_mujeres", "comercio_reparacion_motor_mujeres", "transporte_almacenamiento_mujeres",
                        "hosteleria_mujeres", "informacion_comunicaciones_mujeres", "actividades_financieras_mujeres",
                        "inmobiliarias_mujeres", "actividades_profesionales_cientificas_mujeres",
                        "actividades_administrativas_mujeres", "administracion_publica_mujeres", "educacion_mujeres",
                        "actividades_sanitarias_sociales_mujeres", "actividades_artisticas_entretenimiento_mujeres",
                        "otros_servicios_mujeres", "actividades_hogar_mujeres",
                        "actividades_organizaciones_organismos_extraterritoriales_mujeres", "sin_empleo_anterior_mujeres"]

dataset1 = dataset1.reindex(columns = dataset1_orden_columnas)

# Exporto el dataset completo.

#dataset1.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/DATASET.csv", index = False)

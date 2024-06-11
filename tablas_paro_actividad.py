import pandas as pd

# LIMPIEZA DE LAS TABLAS DEL PARO POR ACTIVIDAD.

        ## Datos de 2013.

paro_actividad_13 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Actividad/paro_actividad_bruto_2013.csv", sep = ";")

paro_actividad_columnas = ["distrito", "año", "mes", "total", "agricultura_ganaderia_silvicultura_pesca_total", "industrias_extractivas_total",
                           "industria_manufacturera_total", "suministro_energia_total", "suministro_agua_total",
                           "construccion_total", "comercio_reparacion_motor_total", "transporte_almacenamiento_total",
                           "hosteleria_total", "informacion_comunicaciones_total", "actividades_financieras_total",
                           "inmobiliarias_total", "actividades_profesionales_cientificas_total", "actividades_administrativas_total",
                           "administracion_publica_total", "educacion_total", "actividades_sanitarias_sociales_total",
                           "actividades_artisticas_entretenimiento_total", "otros_servicios_total", "actividades_hogar_total",
                           "actividades_organizaciones_organismos_extraterritoriales_total", "sin_empleo_anterior_total",
                           "total_hombres", "agricultura_ganaderia_silvicultura_pesca_hombres", "industrias_extractivas_hombres",
                           "industria_manufacturera_hombres", "suministro_energia_hombres", "suministro_agua_hombres",
                           "construccion_hombres", "comercio_reparacion_motor_hombres", "transporte_almacenamiento_hombres",
                           "hosteleria_hombres", "informacion_comunicaciones_hombres", "actividades_financieras_hombres",
                           "inmobiliarias_hombres", "actividades_profesionales_cientificas_hombres", "actividades_administrativas_hombres",
                           "administracion_publica_hombres", "educacion_hombres", "actividades_sanitarias_sociales_hombres",
                           "actividades_artisticas_entretenimiento_hombres", "otros_servicios_hombres", "actividades_hogar_hombres",
                           "actividades_organizaciones_organismos_extraterritoriales_hombres", "sin_empleo_anterior_hombres",
                           "total_mujeres", "agricultura_ganaderia_silvicultura_pesca_mujeres", "industrias_extractivas_mujeres",
                           "industria_manufacturera_mujeres", "suministro_energia_mujeres", "suministro_agua_mujeres",
                           "construccion_mujeres", "comercio_reparacion_motor_mujeres", "transporte_almacenamiento_mujeres",
                           "hosteleria_mujeres", "informacion_comunicaciones_mujeres", "actividades_financieras_mujeres",
                           "inmobiliarias_mujeres", "actividades_profesionales_cientificas_mujeres", "actividades_administrativas_mujeres",
                           "administracion_publica_mujeres", "educacion_mujeres", "actividades_sanitarias_sociales_mujeres",
                           "actividades_artisticas_entretenimiento_mujeres", "otros_servicios_mujeres", "actividades_hogar_mujeres",
                           "actividades_organizaciones_organismos_extraterritoriales_mujeres", "sin_empleo_anterior_mujeres"]

paro_actividad_13.columns = paro_actividad_columnas

paro_actividad_13 = paro_actividad_13.drop(paro_actividad_13.index[:5])

print(paro_actividad_13.head())

        ## Datos de 2014 a 2015.

paro_actividad_14_15 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Actividad/paro_actividad_bruto_2014-2015.csv", sep = ";")

paro_actividad_14_15.columns = paro_actividad_columnas

paro_actividad_14_15 = paro_actividad_14_15.drop(paro_actividad_14_15.index[:5])

print(paro_actividad_14_15.head())

        ## Datos de 2016 a 2017.

paro_actividad_16_17 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Actividad/paro_actividad_bruto_2016-2017.csv", sep = ";")

paro_actividad_16_17.columns = paro_actividad_columnas

paro_actividad_16_17 = paro_actividad_16_17.drop(paro_actividad_16_17.index[:5])

print(paro_actividad_16_17.head())

        ## Datos de 2018.

paro_actividad_18 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Actividad/paro_actividad_bruto_2018.csv", sep = ";")

paro_actividad_18.columns = paro_actividad_columnas

paro_actividad_18 = paro_actividad_18.drop(paro_actividad_18.index[:5])

print(paro_actividad_18.head())

        ## Datos de 2019 a 2020.

paro_actividad_19_20 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Actividad/paro_actividad_bruto_2019-2020.csv", sep = ";")

paro_actividad_19_20.columns = paro_actividad_columnas

paro_actividad_19_20 = paro_actividad_19_20.drop(paro_actividad_19_20.index[:5])

print(paro_actividad_19_20.head())

        ## Datos de 2011 a 2022.

paro_actividad_21_22 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Actividad/paro_actividad_bruto_2021-2022.csv", sep = ";")

paro_actividad_21_22.columns = paro_actividad_columnas

paro_actividad_21_22 = paro_actividad_21_22.drop(paro_actividad_21_22.index[:5])

print(paro_actividad_21_22.head())

        ## Datos de 2023 a 2024.

paro_actividad_23_24 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Actividad/paro_actividad_bruto_2023-2024.csv", sep = ";")

paro_actividad_23_24.columns = paro_actividad_columnas

paro_actividad_23_24 = paro_actividad_23_24.drop(paro_actividad_23_24.index[:5])

print(paro_actividad_23_24.head())

# HAGO UN DATAFRAME ÚNICO CON TODOS LOS DATOS.

paro_actividad_2013_2024 = pd.concat([paro_actividad_13, paro_actividad_14_15, paro_actividad_16_17, paro_actividad_18,
                                     paro_actividad_19_20, paro_actividad_21_22, paro_actividad_23_24], ignore_index = True)

#paro_actividad_2013_2024.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/paro_actividad_2013_2024.csv", index = False)
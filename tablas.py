import pandas as pd
import numpy as np
import locale

# LIMPIEZA DE LAS TABLAS DEL PARO POR SEXO Y EDAD.

    ## Datos de 2013 a 2017.


paro_sexo_edad_13_17 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por sexo y edad/paro_sexo_edad_bruto_2013-2017.csv", sep = ";")

paro_sexo_edad_13_17.columns = ["año", "mes", "distrito", "total",
                                "hombres_total", "mujeres_total",
                                "16_19_total", "16_19_hombres", "16_19_mujeres",
                                "20_24_total", "20_24_hombres", "20_24_mujeres",
                                "25_29_total", "25_29_hombres", "25_29_mujeres",
                                "30_34_total", "30_34_hombres", "30_34_mujeres",
                                "35_39_total", "35_39_hombres", "35_39_mujeres",
                                "40_44_total", "40_44_hombres", "40_44_mujeres",
                                "45_49_total", "45_49_hombres", "45_49_mujeres",
                                "50_54_total", "50_54_hombres", "50_54_mujeres",
                                "55_59_total", "55_59_hombres", "55_59_mujeres",
                                "60_mas_total", "60_mas_hombres", "60_mas_mujeres"]

paro_sexo_edad_13_17 = paro_sexo_edad_13_17.drop(paro_sexo_edad_13_17.index[:5])

print(paro_sexo_edad_13_17.head())


    ## Datos de 2018 a 2022.


paro_sexo_edad_18_22 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por sexo y edad/paro_sexo_edad_bruto_2018-2022.csv", sep = ";")

paro_sexo_edad_18_22.columns = ["año", "mes", "distrito", "total",
                                "hombres_total", "mujeres_total",
                                "16_19_total", "16_19_hombres", "16_19_mujeres",
                                "20_24_total", "20_24_hombres", "20_24_mujeres",
                                "25_29_total", "25_29_hombres", "25_29_mujeres",
                                "30_34_total", "30_34_hombres", "30_34_mujeres",
                                "35_39_total", "35_39_hombres", "35_39_mujeres",
                                "40_44_total", "40_44_hombres", "40_44_mujeres",
                                "45_49_total", "45_49_hombres", "45_49_mujeres",
                                "50_54_total", "50_54_hombres", "50_54_mujeres",
                                "55_59_total", "55_59_hombres", "55_59_mujeres",
                                "60_mas_total", "60_mas_hombres", "60_mas_mujeres"]

paro_sexo_edad_18_22 = paro_sexo_edad_18_22.drop(paro_sexo_edad_18_22.index[:5])

print(paro_sexo_edad_18_22.head())


    ## Datos de 2023 a 2024.


paro_sexo_edad_23_24 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por sexo y edad/paro_sexo_edad_bruto_2023-2024.csv", sep = ";")

paro_sexo_edad_23_24.columns = ["año", "mes", "distrito", "total",
                                "hombres_total", "mujeres_total",
                                "16_19_total", "16_19_hombres", "16_19_mujeres",
                                "20_24_total", "20_24_hombres", "20_24_mujeres",
                                "25_29_total", "25_29_hombres", "25_29_mujeres",
                                "30_34_total", "30_34_hombres", "30_34_mujeres",
                                "35_39_total", "35_39_hombres", "35_39_mujeres",
                                "40_44_total", "40_44_hombres", "40_44_mujeres",
                                "45_49_total", "45_49_hombres", "45_49_mujeres",
                                "50_54_total", "50_54_hombres", "50_54_mujeres",
                                "55_59_total", "55_59_hombres", "55_59_mujeres",
                                "60_mas_total", "60_mas_hombres", "60_mas_mujeres"]

paro_sexo_edad_23_24 = paro_sexo_edad_23_24.drop(paro_sexo_edad_23_24.index[:5])

print(paro_sexo_edad_23_24.head())

    ## Hago un dataframe único con todos los datos de par, sexo y edad.

paro_sexo_edad_2013_2024 = pd.concat([paro_sexo_edad_13_17, paro_sexo_edad_18_22, paro_sexo_edad_23_24], ignore_index = True)

print(paro_sexo_edad_2013_2024.head())

    ## Cambio los nombres de los distritos para unificar nombres.

paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"01. Centro": "CENTRO"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"02. Arganzuela": "ARGANZUELA"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"03. Retiro": "RETIRO"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"04. Salamanca": "SALAMANCA"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"05. Chamartín": "CHAMARTÍN"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"06. Tetuán": "TETUÁN"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"07. Chamberí": "CHAMBERÍ"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"08. Fuencarral-El Pardo": "FUENCARRAL - EL PARDO"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"09. Moncloa-Aravaca": "MONCLOA - ARAVACA"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"10. Latina": "LATINA"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"11. Carabanchel": "CARABANCHEL"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"12. Usera": "USERA"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"13. Puente de Vallecas": "PUENTE DE VALLECAS"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"14. Moratalaz": "MORATALAZ"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"15. Ciudad Lineal": "CIUDAD LINEAL"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"16. Hortaleza": "HORTALEZA"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"17. Villaverde": "VILLAVERDE"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"18. Villa de Vallecas": "VILLA DE VALLECAS"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"19. Vicálvaro": "VICÁLVARO"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"20. San Blas-Canillejas": "SAN BLAS-CANILLEJAS"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"21. Barajas": "BARAJAS"})
paro_sexo_edad_2013_2024["distrito"] = paro_sexo_edad_2013_2024["distrito"].replace({"Ciudad de Madrid": "MADRID"})


print(paro_sexo_edad_2013_2024.head())

    ## Elimino los puntos separadores de miles.

paro_sexo_edad_2013_2024 = paro_sexo_edad_2013_2024.replace({r'\.': ''}, regex=True)

    ## Exporto el CSV.

paro_sexo_edad_2013_2024.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/PARO_SEXO_EDAD_2013_2024.csv", index = False)


# LIMPIEZA DE LAS TABLAS DEL PARO POR ESTUDIOS.

    ## Datos de 2013 a 2015.

paro_estudios_13_15 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por estudios/paro_estudios_bruto_2013-2015.csv", sep = ";")

nombre_columnas_paro_estudios = ["año", "mes", "distrito",
                               "total", "hombres_total", "mujeres_total",
                               "no_estudios_total", "no_estudios_hombres", "no_estudios_mujeres",
                               "estudios_primarios_incompletos_total", "estudios_primarios_incompletos_hombres", "estudios_primarios_incompletos_mujeres",
                               "estudios_primarios_total", "estudios_primarios_hombres", "estudios_primarios_mujeres",
                               "programa_fp_total", "programa_fp_hombres", "programa_fp_mujeres",
                               "educacion_general_total", "educacion_general_hombres", "educacion_general_mujeres",
                               "estudios_tecnico_profesionales_superiores_total", "estudios_tecnico_profesionales_superiores_hombres", "estudios_tecnico_profesionales_superiores_mujeres",
                               "estudios_universitarios_ciclo1_total", "estudios_universitarios_ciclo1_hombres", "estudios_universitarios_ciclo1_mujeres",
                               "estudios_universitarios_ciclo2y3_total", "estudios_universitarios_ciclo2y3_hombres", "estudios_universitarios_ciclo2y3_mujeres",
                               "otros_total", "otros_hombres", "otros_mujeres",
                               "otro_fp_total", "otro_fp_hombres", "otro_fp_mujeres"]


paro_estudios_13_15.columns = nombre_columnas_paro_estudios

paro_estudios_13_15 = paro_estudios_13_15.drop(paro_estudios_13_15.index[:5])

print(paro_estudios_13_15.head())

    ## Datos de 2016.

paro_estudios_16 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por estudios/paro_estudios_bruto_2016.csv", sep = ";")

paro_estudios_16 = paro_estudios_16.drop(paro_estudios_16.index[:5])

paro_estudios_16.columns = nombre_columnas_paro_estudios

print(paro_estudios_16.head())

    ## Datos de 2017.

paro_estudios_17 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por estudios/paro_estudios_bruto_2017.csv", sep = ";")

paro_estudios_17 = paro_estudios_17.drop(paro_estudios_17.index[:5])

paro_estudios_17.columns = nombre_columnas_paro_estudios

print(paro_estudios_17.head())

    ## Datos de 2018.

paro_estudios_18 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por estudios/paro_estudios_bruto_2018.csv", sep = ";")

paro_estudios_18 = paro_estudios_18.drop(paro_estudios_18.index[:5])

paro_estudios_18.columns = nombre_columnas_paro_estudios

print(paro_estudios_18.head())

    ## Datos de 2019.

paro_estudios_19 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por estudios/paro_estudios_bruto_2019.csv", sep = ";")

paro_estudios_19 = paro_estudios_19.drop(paro_estudios_19.index[:5])

paro_estudios_19.columns = nombre_columnas_paro_estudios

print(paro_estudios_19.head())

    ## Datos de 2020 a 2022.

paro_estudios_20_22 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por estudios/paro_estudios_bruto_2020-2022.csv", sep = ";")

paro_estudios_20_22 = paro_estudios_20_22.drop(paro_estudios_20_22.index[:5])

paro_estudios_20_22.columns = nombre_columnas_paro_estudios

print(paro_estudios_20_22.head())

    # Datos de 2023.
 
paro_estudios_23 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por estudios/paro_estudios_bruto_2023.csv", sep = ";")

paro_estudios_23 = paro_estudios_23.drop(paro_estudios_23.index[:5])

paro_estudios_23.columns = nombre_columnas_paro_estudios

print(paro_estudios_23.head())

    # Datos de 2024.
 
paro_estudios_24 = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Paro/Por estudios/paro_estudios_bruto_2024.csv", sep = ";")

paro_estudios_24 = paro_estudios_24.drop(paro_estudios_24.index[:5])

paro_estudios_24.columns = nombre_columnas_paro_estudios

print(paro_estudios_24.head())

    ## Hago un dataframe único con todos los datos.

paro_estudios_2013_2024 = pd.concat([paro_estudios_13_15, paro_estudios_16, paro_estudios_17,
                                     paro_estudios_18, paro_estudios_19, paro_estudios_20_22,
                                     paro_estudios_23, paro_estudios_24], ignore_index = True)

    ## Cambio los nombres de los distritos para unificar nombres.

paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"01. Centro": "CENTRO"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"02. Arganzuela": "ARGANZUELA"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"03. Retiro": "RETIRO"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"04. Salamanca": "SALAMANCA"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"05. Chamartín": "CHAMARTÍN"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"06. Tetuán": "TETUÁN"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"07. Chamberí": "CHAMBERÍ"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"08. Fuencarral-El Pardo": "FUENCARRAL - EL PARDO"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"09. Moncloa-Aravaca": "MONCLOA - ARAVACA"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"10. Latina": "LATINA"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"11. Carabanchel": "CARABANCHEL"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"12. Usera": "USERA"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"13. Puente de Vallecas": "PUENTE DE VALLECAS"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"14. Moratalaz": "MORATALAZ"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"15. Ciudad Lineal": "CIUDAD LINEAL"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"16. Hortaleza": "HORTALEZA"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"17. Villaverde": "VILLAVERDE"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"18. Villa de Vallecas": "VILLA DE VALLECAS"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"19. Vicálvaro": "VICÁLVARO"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"20. San Blas-Canillejas": "SAN BLAS-CANILLEJAS"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"21. Barajas": "BARAJAS"})
paro_estudios_2013_2024["distrito"] = paro_estudios_2013_2024["distrito"].replace({"Ciudad de Madrid": "MADRID"})


paro_estudios_2013_2024["fecha"] = paro_estudios_2013_2024["mes"] + " " + paro_estudios_2013_2024["año"].astype(str)

#paro_estudios_2013_2024["fecha"] = pd.to_datetime(paro_estudios_2013_2024["fecha"], format = "%B %Y")

print(paro_estudios_2013_2024.head())

paro_estudios_2013_2024 = paro_estudios_2013_2024.replace({r'\.': ''}, regex=True)

        ## Exporto el CSV.

paro_estudios_2013_2024.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/PARO_ESTUDIOS_2013_2024.csv", index = False)

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

    ## Hago un dataframe único con todos los datos.

paro_actividad_2013_2024 = pd.concat([paro_actividad_13, paro_actividad_14_15, paro_actividad_16_17, paro_actividad_18,
                                     paro_actividad_19_20, paro_actividad_21_22, paro_actividad_23_24], ignore_index = True)


    ## Relleno los NAs con ceros y paso la variable a int.

paro_actividad_2013_2024["año"].fillna(0, inplace=True)
paro_actividad_2013_2024["año"] = paro_actividad_2013_2024["año"].astype(int)

    ## Cambio los nombres de los distritos para unificar nombres.

paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"01. Centro": "CENTRO"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"02. Arganzuela": "ARGANZUELA"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"03. Retiro": "RETIRO"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"04. Salamanca": "SALAMANCA"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"05. Chamartín": "CHAMARTÍN"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"06. Tetuán": "TETUÁN"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"07. Chamberí": "CHAMBERÍ"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"08. Fuencarral-El Pardo": "FUENCARRAL - EL PARDO"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"09. Moncloa-Aravaca": "MONCLOA - ARAVACA"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"10. Latina": "LATINA"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"11. Carabanchel": "CARABANCHEL"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"12. Usera": "USERA"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"13. Puente de Vallecas": "PUENTE DE VALLECAS"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"14. Moratalaz": "MORATALAZ"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"15. Ciudad Lineal": "CIUDAD LINEAL"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"16. Hortaleza": "HORTALEZA"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"17. Villaverde": "VILLAVERDE"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"18. Villa de Vallecas": "VILLA DE VALLECAS"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"19. Vicálvaro": "VICÁLVARO"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"20. San Blas-Canillejas": "SAN BLAS-CANILLEJAS"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"21. Barajas": "BARAJAS"})
paro_actividad_2013_2024["distrito"] = paro_actividad_2013_2024["distrito"].replace({"Ciudad de Madrid": "MADRID"})


paro_actividad_2013_2024 = paro_actividad_2013_2024.replace({r'\.': ''}, regex=True)

        ## Exporto el CSV.

paro_actividad_2013_2024.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/PARO_ACTIVIDAD_2013_2024.csv", index = False)

# LIMPIEZA DE TABLAS DE ALQUILER.

alquiler = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Alquiler/alquiler_distritos_bruto.csv", encoding='latin1', sep = ";")

    ## Mapeo los meses abreviados a números.

meses = {
    'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 'may': '05', 'jun': '06',
    'jul': '07', 'ago': '08', 'sept': '09', 'oct': '10', 'nov': '11', 'dic': '12'
}

    ## Hago una función para convertir la fecha en el formato que tienen las otras tablas.

def convertir_fecha(fecha):
    try:
        mes_abreviado, año = fecha.split("-")  # Separar el mes y el año
        mes = meses[mes_abreviado]  # Obtener el número del mes
        año = "20" + año  # Agregar "20" al año
        return f"{mes}-{año}"
    except Exception as e:
        print(f"Error al convertir la fecha: {fecha}, error: {e}")
        return None  # Devuelve None si hay un error

    ## Hago una variable con el nuevo formato de fecha.

alquiler["Fecha"] = alquiler["Fecha"].astype(str)
alquiler["fecha_formateada"] = alquiler["Fecha"].apply(convertir_fecha)

    ## Elimino la columna antigua de fecha que no me sirve.

alquiler = alquiler.drop(["Fecha"], axis = 1)

    ## Creo un vector con los nombres nuevos que quiero que tenga mi dataframe y se lo asigno.

alquiler_columnas = ["distrito", "precio_m2",
                     "variacion_mensual", "variacion_trimestral",
                     "variacion_anual", "fecha"]

alquiler.columns = alquiler_columnas

    ## Reordeno las columnas.

alquiler_orden = ["fecha", "distrito", "precio_m2",
                     "variacion_mensual", "variacion_trimestral",
                     "variacion_anual"]

alquiler = alquiler[alquiler_orden]

    ## Reemplazo los nombres de distritos que tienen acento y aparecen sin ese caracter por los nombres correctos.

alquiler["distrito"] = alquiler["distrito"].replace("Chamartn", "Chamartín", regex=True)
alquiler["distrito"] = alquiler["distrito"].replace("Chamber", "Chamberí", regex=True)
alquiler["distrito"] = alquiler["distrito"].replace("Tetun", "Tetuán", regex=True)
alquiler["distrito"] = alquiler["distrito"].replace("Viclvaro", "Vicálvaro", regex=True)

    ## Cambio los nombres de los distritos para unificar nombres.

alquiler["distrito"] = alquiler["distrito"].replace({"Centro": "CENTRO"})
alquiler["distrito"] = alquiler["distrito"].replace({"Arganzuela": "ARGANZUELA"})
alquiler["distrito"] = alquiler["distrito"].replace({"Retiro": "RETIRO"})
alquiler["distrito"] = alquiler["distrito"].replace({"Salamanca": "SALAMANCA"})
alquiler["distrito"] = alquiler["distrito"].replace({"Chamartín": "CHAMARTÍN"})
alquiler["distrito"] = alquiler["distrito"].replace({"Tetuán": "TETUÁN"})
alquiler["distrito"] = alquiler["distrito"].replace({"Chamberí": "CHAMBERÍ"})
alquiler["distrito"] = alquiler["distrito"].replace({"Fuencarral-El Pardo": "FUENCARRAL - EL PARDO"})
alquiler["distrito"] = alquiler["distrito"].replace({"Moncloa-Aravaca": "MONCLOA - ARAVACA"})
alquiler["distrito"] = alquiler["distrito"].replace({"Latina": "LATINA"})
alquiler["distrito"] = alquiler["distrito"].replace({"Carabanchel": "CARABANCHEL"})
alquiler["distrito"] = alquiler["distrito"].replace({"Usera": "USERA"})
alquiler["distrito"] = alquiler["distrito"].replace({"Puente de Vallecas": "PUENTE DE VALLECAS"})
alquiler["distrito"] = alquiler["distrito"].replace({"Moratalaz": "MORATALAZ"})
alquiler["distrito"] = alquiler["distrito"].replace({"Ciudad Lineal": "CIUDAD LINEAL"})
alquiler["distrito"] = alquiler["distrito"].replace({"Hortaleza": "HORTALEZA"})
alquiler["distrito"] = alquiler["distrito"].replace({"Villaverde": "VILLAVERDE"})
alquiler["distrito"] = alquiler["distrito"].replace({"Villa de Vallecas": "VILLA DE VALLECAS"})
alquiler["distrito"] = alquiler["distrito"].replace({"Vicálvaro": "VICÁLVARO"})
alquiler["distrito"] = alquiler["distrito"].replace({"San Blas-Canillejas": "SAN BLAS-CANILLEJAS"})
alquiler["distrito"] = alquiler["distrito"].replace({"Barajas": "BARAJAS"})
alquiler["distrito"] = alquiler["distrito"].replace({"Madrid": "MADRID"})

print(alquiler.info())

    ## Exporto el CSV.

alquiler.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/ALQUILER.csv", encoding = "utf-8", index = False)

# TRATAMIENTO Y UNIFICACIÓN DE TABLAS.

    ## Establezco mi localización como España para modificar la fecha.

locale.setlocale(locale.LC_TIME, "es_ES.UTF-8")

    ## Cargo mis dataframes.

paro_sexo_edad = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/PARO_SEXO_EDAD_2013_2024.csv")
paro_estudios = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/PARO_ESTUDIOS_2013_2024.csv")
paro_actividad = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/PARO_ACTIVIDAD_2013_2024.csv")
poblacion = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/POBLACION_2013_2024.csv")


    ## Convertir las columnas a int desde la columna 7 en adelante.

for col in paro_sexo_edad.columns[3:]:
    paro_sexo_edad[col] = pd.to_numeric(paro_sexo_edad[col], errors='coerce').fillna(0).astype(int)

for col in paro_estudios.columns[3:]:
    paro_estudios[col] = pd.to_numeric(paro_estudios[col], errors='coerce').fillna(0).astype(int)

for col in paro_actividad.columns[3:]:
    paro_actividad[col] = pd.to_numeric(paro_actividad[col], errors='coerce').fillna(0).astype(int)

for col in poblacion.columns[2:]:
    poblacion[col] = pd.to_numeric(poblacion[col], errors='coerce').fillna(0).astype(int)

    ## Creo una nueva variable que sea "fecha" a partir de "año" y "mes".

paro_sexo_edad["fecha"] = paro_sexo_edad["mes"].astype(str) + " " + paro_sexo_edad["año"].astype(str)
paro_sexo_edad["fecha"] = pd.to_datetime(paro_sexo_edad["fecha"], format="%B %Y", errors="coerce")

paro_estudios["fecha"] = paro_estudios["mes"].astype(str) + " " + paro_estudios["año"].astype(str)
paro_estudios["fecha"] = pd.to_datetime(paro_estudios["fecha"], format="%B %Y", errors="coerce")

paro_actividad["fecha"] = paro_actividad["mes"].astype(str) + " " + paro_actividad["año"].astype(str)
paro_actividad["fecha"] = pd.to_datetime(paro_actividad["fecha"], format="%B %Y", errors="coerce")

    ## Le doy el formato adecuado a esa variable.

paro_sexo_edad["fecha"] = paro_sexo_edad["fecha"].dt.strftime("%m-%Y")

paro_estudios["fecha"] = paro_estudios["fecha"].dt.strftime("%m-%Y")

paro_actividad["fecha"] = paro_actividad["fecha"].dt.strftime("%m-%Y")

    ## Cambio de orden las columnas.

columnas = paro_sexo_edad.columns.tolist()
columnas.insert(2, columnas.pop(columnas.index("fecha")))
paro_sexo_edad = paro_sexo_edad[columnas]

columnas_estudios = paro_estudios.columns.tolist()
columnas_estudios.insert(2, columnas_estudios.pop(columnas_estudios.index("fecha")))
paro_estudios = paro_estudios[columnas_estudios]

print(paro_sexo_edad.head())

print(paro_estudios.head())

print(paro_actividad.head())

    ## Combino los datasets.

combinado_sexo_edad_estudios = pd.merge(paro_sexo_edad, paro_estudios, on = ["fecha", "distrito"])
                                                                             
paro_completo = pd.merge(combinado_sexo_edad_estudios, paro_actividad, on = ["fecha", "distrito"])

paro_poblacion = pd.merge(paro_completo, poblacion, on = ["fecha", "distrito"])

print(paro_poblacion.info())

#paro_poblacion.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/paro_poblacion.csv", index = False)

    ## Importo el dataset "alquiler" para meterlo en la tabla conjunta.

alquiler = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/ALQUILER.csv")

    ## Combino todo el dataset y lo exporto.

previo_dataset = pd.merge(paro_poblacion, alquiler, on = ["fecha", "distrito"])

previo_dataset.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/previo_dataset.csv", index = False)

# LIMPIEZA DE DATASET.

previo_dataset = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/previo_dataset.csv")

    ## Elimino las columnas que están duplicadas o no necesito.

columnas_eliminar = ["año_x", "mes_x", "año_y", "mes_y",
                     "total_y", "hombres_total_y", "mujeres_total_y",
                     "año", "mes", "total", "total_hombres", "total_mujeres"]

previo_dataset = previo_dataset.drop(columnas_eliminar, axis = 1)

    ## Almaceno el nombre de todas las columnas y los aplico al dataset.

previo_dataset_columnas = ["fecha", "distrito", "paro_total", "paro_hombres_total", "paro_mujeres_total", "paro_16_19_total",
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
                    "poblacion", "precio_m2", "variacion_mensual", "variacion_trimestral", "variacion_anual"]

previo_dataset.columns = previo_dataset_columnas

    ## Cambio el orden de las columnas y lo aplico.

previo_dataset_orden_columnas = ["fecha", "distrito", "precio_m2", "variacion_mensual", "variacion_trimestral", "variacion_anual", "poblacion",
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

previo_dataset = previo_dataset.reindex(columns = previo_dataset_orden_columnas)

    ## Exporto el dataset completo.

previo_dataset.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/BRUTO_DATASET.csv", index = False)

# HAGO LA VARIABLE DE TASA DE PARO.

dataset = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/BRUTO_DATASET.csv")

# Calcular el porcentaje de "paro_total" sobre "poblacion" y almacenarlo en la variable "porcentaje_paro"
dataset["porcentaje_paro"] = (dataset["paro_total"] / dataset["poblacion"]) * 100

# Convertir "porcentaje_paro" a float y reemplazar primero los puntos por comas
dataset["porcentaje_paro"] = dataset["porcentaje_paro"].round(2).apply(lambda x: str(x).replace('.', ','))

dataset_columnas = ["fecha", "distrito", "precio_m2", "variacion_mensual", "variacion_trimestral", "variacion_anual", "poblacion",
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
                        "actividades_organizaciones_organismos_extraterritoriales_mujeres", "sin_empleo_anterior_mujeres",
                        "tasa_paro"]

dataset.columns = dataset_columnas

dataset_orden_columnas = ["fecha", "distrito", "precio_m2", "variacion_mensual", "variacion_trimestral", "variacion_anual", "poblacion",
                        "paro_total", "tasa_paro", "paro_hombres_total", "paro_mujeres_total", "paro_16_19_total", "paro_16_19_hombres",
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

dataset = dataset.reindex(columns = dataset_orden_columnas)

dataset.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET.csv", index = False)
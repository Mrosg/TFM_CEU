import pandas as pd

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

# HAGO UN DATAFRAME ÚNICO CON TODOS LOS DATOS.

paro_estudios_2013_2024 = pd.concat([paro_estudios_13_15, paro_estudios_16, paro_estudios_17,
                                     paro_estudios_18, paro_estudios_19, paro_estudios_20_22,
                                     paro_estudios_23, paro_estudios_24], ignore_index = True)


#paro_estudios_2013_2024["fecha"] = paro_estudios_2013_2024["mes"] + " " + paro_estudios_2013_2024["año"].astype(str)

#paro_estudios_2013_2024["fecha"] = pd.to_datetime(paro_estudios_2013_2024["fecha"], format = "%B %Y")

#print(paro_estudios_2013_2024.head())

# paro_estudios_2013_2024.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/paro_estudios_2013_2024.csv", index = False)
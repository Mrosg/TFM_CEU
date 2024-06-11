import pandas as pd

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

# HAGO UN DATAFRAME ÚNICO CON TODOS LOS DATOS.

paro_sexo_edad_2013_2024 = pd.concat([paro_sexo_edad_13_17, paro_sexo_edad_18_22, paro_sexo_edad_23_24], ignore_index = True)

print(paro_sexo_edad_2013_2024.head())

# paro_sexo_edad_2013_2024.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/paro_sexo_edad_2013_2024.csv", index = False)
library(tidyverse)
library(imputeTS)
library(dplyr)


# Importo mi base de datos. -----------------------------------------------

na_pob_13_23 <- read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Brutos/Ayuntamiento de Madrid/Población/pob_13_23.csv")

# Reemplazar ceros con NAs. -----------------------------------------------

na_pob_13_23 <- na_pob_13_23 %>%
  mutate(poblacion = na_if(poblacion, 0))

# Convierto la variable en una serie temporal. ----------------------------

na_pob_st <- na_pob_13_23 %>%
  pivot_wider(names_from = fecha, values_from = poblacion) %>%
  column_to_rownames("distrito") %>%
  t()

# Estimo los valores (incluyendo aquellos que eran NAs). ------------------

pob_estimada <- na_kalman(na_pob_st)
pob_estimada <- round(pob_estimada)

# Paso de nuevo los datos a formato largo. --------------------------------

pob_13_23 <- as.data.frame(pob_estimada) %>%
  rownames_to_column("fecha") %>%
  pivot_longer(cols = -fecha, names_to = "distrito", values_to = "poblacion")

# Convierto la variable "fecha" en tipo Date de nuevo. --------------------

pob_13_23$fecha <- as.Date(pob_13_23$fecha)

# Cambio el formato y quito dígitos extraños. -----------------------------

pob_13_23$poblacion <- as.numeric(formatC(pob_13_23$poblacion, format = "f", digits = 3))

# Formateo los números para asegurar que no se pierden decimales al exportar --------

pob_13_23$poblacion <- format(pob_13_23$poblacion, digits = 10, nsmall = 3)

# Hago gráficos de las estimaciones para ver si hay errores. --------------

plot(pob_estimada[,1], xlab = "Arganzuela")
plot(pob_estimada[,2], xlab = "Barajas")
plot(pob_estimada[,3], xlab = "Carabanchel")
plot(pob_estimada[,4], xlab = "Centro")
plot(pob_estimada[,5], xlab = "Chamartín")
plot(pob_estimada[,6], xlab = "Chamberí")
plot(pob_estimada[,7], xlab = "Ciudad Lineal")
plot(pob_estimada[,8], xlab = "Fuencarral - El Pardo")
plot(pob_estimada[,9], xlab = "Hortaleza")
plot(pob_estimada[,10], xlab = "Latina")
plot(pob_estimada[,11], xlab = "Moncloa - Aravaca")
plot(pob_estimada[,12], xlab = "Moratalaz")
plot(pob_estimada[,13], xlab = "Puente de Vallecas")
plot(pob_estimada[,14], xlab = "Retiro")
plot(pob_estimada[,15], xlab = "Salamanca")
plot(pob_estimada[,16], xlab = "San Blas - Canillejas")
plot(pob_estimada[,17], xlab = "Tetuán")
plot(pob_estimada[,18], xlab = "Usera")
plot(pob_estimada[,19], xlab = "Vicálvaro")
plot(pob_estimada[,20], xlab = "Villa de Vallecas")
plot(pob_estimada[,21], xlab = "Villaverde")


# Añado a "MADRID" cada mes -----------------------------------------------------------------

pob_13_23$year_month <- format(pob_13_23$fecha, "%Y-%m")
pob_13_23$poblacion <- as.numeric(pob_13_23$poblacion)

# Agrupo por año y mes y hago la suma.

monthly_population <- pob_13_23 %>%
  group_by(year_month) %>%
  summarise(poblacion = sum(poblacion))

# Creo el valor "MADRID" en la variable "distrito".

monthly_population <- monthly_population %>%
  mutate(distrito = "MADRID")

# Cambio el orden de las columnas y paso la variable creada a tipo Date.

monthly_population <- monthly_population %>%
  mutate(fecha = as.Date(paste0(year_month, "-01"))) %>%
  select(fecha, distrito, poblacion)

POBLACION_2013_2024 <- bind_rows(pob_13_23, monthly_population)

# Últimos retoques --------------------------------------------------------

# Ordeno el dataframe por fecha.

POBLACION_2013_2024 <- POBLACION_2013_2024 %>%
  arrange(fecha)

# Elimino la última columna que se generó al crear "MADRID".

POBLACION_2013_2024 <- POBLACION_2013_2024 %>%
  select(-4)

# Cambio el formato de la variable "fecha".

POBLACION_2013_2024 <- POBLACION_2013_2024 %>%
  mutate(fecha = format(as.Date(fecha), "%m-%Y"))

# Paso la variable a numérica.

POBLACION_2013_2024$poblacion <- as.numeric(POBLACION_2013_2024$poblacion)

# Lo exporto. -------------------------------------------------------------

ruta_poblacion <- "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/POBLACION_2013_2024.csv"
write.csv(POBLACION_2013_2024, file = ruta_poblacion, row.names = FALSE, fileEncoding = "UTF-8")

import pandas as pd
import numpy as np
from scipy.stats import binom

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/Auxiliares/DATA_CATEGORIZADO.csv"

data = pd.read_csv(ruta)

# Calcular la probabilidad de encontrar empleo
data['probabilidad_empleo'] = 1 - (data['tasa_paro'] / 100)

# Tamaño de la muestra
n = 50
# Umbral de éxito (al menos 1/3)
k = n // 3

# Calcular la probabilidad de que al menos 1/3 tenga empleo
data['tasa_emancipacion'] = data['probabilidad_empleo'].apply(lambda p: binom.cdf(n, n, p) - binom.cdf(k-1, n, p))

col = data.pop("tasa_emancipacion")
data.insert(2, "tasa_emancipacion", col)

data.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/Auxiliares/PRUEBA_BINOMIAL.csv", index = False)

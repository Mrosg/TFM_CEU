import pandas as pd
import numpy as np
from scipy.stats import norm
import math

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/Auxiliares/DATA_CATEGORIZADO.csv"

data = pd.read_csv(ruta)

# Calcular la probabilidad de encontrar empleo.

probabilidad_paro = (data["tasa_paro"]/100)
probabilidad_empleo = 1 - (data["tasa_paro"]/100)

n = 50
mu = probabilidad_paro.mean() # La media.
pq_n = ((probabilidad_empleo * probabilidad_paro) / n)
sigma = math.sqrt(pq_n.mean()) # Desviación estándar.

# Calcular la CDF para una nueva variable x
cdf_values = norm.cdf(probabilidad_paro, mu, sigma)
data["tasa_emancipacion"] = ((1 - cdf_values)*100).round(2)


col = data.pop("tasa_emancipacion")
data.insert(9, "tasa_emancipacion", col)

data.to_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv", index = False)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ruta = "/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Datasets/Definitivos/DATASET_FINAL.csv"
data = pd.read_csv(ruta)

data["fecha"] = pd.to_datetime(data["fecha"], format="%m-%Y")

# Codifico la variable "categoría".

encoder = OneHotEncoder(sparse_output=False)
categorias_encoded = encoder.fit_transform(data[["categoria"]])
categorias_df = pd.DataFrame(categorias_encoded, columns=encoder.get_feature_names_out(["categoria"]))
data = pd.concat([data, categorias_df], axis=1)
data.drop(["categoria"], axis=1, inplace=True)

# Análisis Temporal
# Calcular la tasa de emancipación media por mes
tasa_emancipacion_mensual = data.groupby(data["fecha"].dt.to_period("M"))["tasa_emancipacion"].mean().reset_index()

# Convertir la columna 'fecha' a formato timestamp para una mejor visualización
tasa_emancipacion_mensual["fecha"] = tasa_emancipacion_mensual["fecha"].dt.to_timestamp()

# Visualizar la tasa de emancipación media a lo largo del tiempo
plt.figure(figsize=(12, 6))

# Graficar la tasa de emancipación media por mes
plt.plot(tasa_emancipacion_mensual["fecha"], tasa_emancipacion_mensual["tasa_emancipacion"], marker="o", linestyle="-", color="b")
plt.title("Tasa de Emancipación Media a lo Largo del Tiempo")
plt.xlabel("Fecha")
plt.ylabel("Tasa de Emancipación Media")
plt.grid(True)
plt.show()

# Análisis geográfico
# Calcular la tasa de emancipación media por distrito
tasa_emancipacion_distrito = data.groupby("distrito")["tasa_emancipacion"].mean().reset_index()

# Visualizar la tasa de emancipación media por distrito
plt.figure(figsize=(12, 6))

# Graficar la tasa de emancipación media por distrito
plt.bar(tasa_emancipacion_distrito["distrito"], tasa_emancipacion_distrito["tasa_emancipacion"], color="skyblue")
plt.title("Tasa de Emancipación Media por Distrito")
plt.xlabel("Distrito")
plt.ylabel("Tasa de Emancipación Media")
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# Mostrar la tasa de emancipación media por distrito
tasa_emancipacion_distrito
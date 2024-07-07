import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

comparacion = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Parte I/salario_alquiler.csv", sep = ";")

print(comparacion.head())

def integer_formatter(x, pos): # Pongo al eje X como número entero para que no salgan decimales.
    return f'{int(x)}'

def euros(x, pos): # Pongo sufijo % al eye Y.
    return f"{int(x)}€"

fuente_texto = "Fuente: idealista y Encuesta Condiciones de Vida 2023 - INE"

plt.plot(comparacion["año"], comparacion["renta_media_joven"], marker="o", linestyle="-",
         color="red", markersize = 4, linewidth = 1, label = "Salario medio en euros")
plt.plot(comparacion["año"], comparacion["medio_alquiler"], marker="o", linestyle="-",
         color="blue", markersize = 4, linewidth = 1, label = "Precio medio del alquiler (80m2) en euros")
plt.gca().xaxis.set_major_formatter(FuncFormatter(integer_formatter)) # Aplicamos el eje X formateado.
plt.gca().yaxis.set_major_formatter(FuncFormatter(euros)) # Aplicamos el eje Y formateado.
plt.xticks(comparacion["año"])
plt.title("Precio medio del alquiler vs salario medio de jóvenes españoles (16-29 años)",
          fontsize = 14, fontweight = "bold")
plt.figtext(0.01, 0.05, "Autor: Miguel Ros García", verticalalignment = "bottom",
            horizontalalignment ="left", fontsize=8, style ="italic", fontweight = "bold")
plt.figtext(0.01, 0.01, fuente_texto, horizontalalignment="left", fontsize = 8)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
plt.ylim(400, 1200)
plt.legend()
plt.show()
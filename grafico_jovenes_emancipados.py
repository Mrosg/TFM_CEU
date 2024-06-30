import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

emancipados = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Parte I/Datos/tasa_jovenes_emancipados.csv", sep = ";")

def integer_formatter(x, pos): # Pongo al eje X como número entero para que no salgan decimales.
    return f'{int(x)}'

def percentage_formatter(x, pos): # Pongo sufijo % al eye Y.
    return f"{x}%"

fuente_texto = "Fuente: Juventud Vulnerable y Democracia en España y Observatorio de Emancipación del Consejo de la Juventud"

plt.plot(emancipados["fechas"], emancipados["tasa"], marker="o", linestyle="-",
         color="red", markersize = 4, linewidth = 1)
plt.title("Tasa de jóvenes emancipados en España (16-29 años)", fontsize = 14, fontweight = "bold")
plt.gca().xaxis.set_major_formatter(FuncFormatter(integer_formatter)) # Aplicamos el eje X formateado.
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter)) # Aplicamos el eje Y formateado.
plt.xticks(emancipados["fechas"])
plt.figtext(0.01, 0.05, "Autor: Miguel Ros García", verticalalignment = "bottom",
            horizontalalignment="left", fontsize=8, style="italic", fontweight = "bold")
plt.figtext(0.01, 0.01, fuente_texto, horizontalalignment="left", fontsize = 8)
plt.grid(False)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
#plt.savefig("grafico_emancipacion.png", dpi = 1000)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator

viviendas = pd.read_csv("/Users/miguelrosgarcia/Desktop/Máster/Curso/TFM/Parte I/Datos/viviendas_completadas.csv", sep = ";")

print(viviendas.head())

def format_y(value, tick_number):
    return f'{value:,.0f}'.replace(',', '.')

fuente_texto = "Fuente: Asociación Hipotecaria Española"

plt.plot(viviendas["fechas"], viviendas["viviendas"], marker="o", linestyle="-",
         color="red", markersize = 4, linewidth = 1)
plt.xticks(viviendas["fechas"])
plt.title("Viviendas nuevas completadas en España",
          fontsize = 14, fontweight = "bold")
plt.figtext(0.01, 0.05, "Autor: Miguel Ros García", verticalalignment = "bottom",
            horizontalalignment="left", fontsize=8, style="italic", fontweight = "bold")
plt.figtext(0.01, 0.01, fuente_texto, horizontalalignment="left", fontsize = 8)
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
plt.ylim(0, 700000)
plt.legend()
plt.show()
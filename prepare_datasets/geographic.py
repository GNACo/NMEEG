import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Diccionario con los datos a visualizar
data = {
    "Chile": 13,
    "Argentina": 19,
    "Cuba": 247,
    "Colombia": 216
}

# Cargar el mapa de Latinoamérica
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
latam = world[world['continent'] == 'South America']

# Incluir Centroamérica si es necesario
extra_countries = ['Cuba']
latam = pd.concat([latam, world[world['name'].isin(extra_countries)]])

# Agregar la columna de valores
latam['value'] = latam['name'].map(data)

# Crear el gráfico
fig, ax = plt.subplots(figsize=(10, 6))
latam.plot(column='value', cmap='OrRd', linewidth=0.8, edgecolor='black', legend=True, ax=ax)

# Configurar títulos y estilos
ax.set_title('Distribución de Cantidades por País', fontsize=14)
ax.set_axis_off()

# Mostrar la gráfica
plt.show()

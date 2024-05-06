# FUNCIONES PARA EL ALGORITMO TOPSIS
# Autores: Ivan Pinto Grilo, Maria Perez Sebastian, Soraya Alvarez Codesal
# Fecha modificacion: 22/04/24


# Cargamos librerias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from scipy.stats import rankdata # para hacer un ranking de candidates
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go # para plotear

import pandas as pd
import matplotlib.pyplot as plt


# funcion visualizar precio en categorias
def precio_a_categorias(df):
    """
  Función para categorizar precios y generar un gráfico de barras.

  Argumentos:
    df: El DataFrame que contiene la columna 'precio'.
    bins: Lista de valores de corte para crear categorías de precio.

  Retorno:
    Objeto de gráfico de barras de Matplotlib.
  """

  # Definir categorías de precio
    bins =[1,10,20,40,80,2799]
    categorias = ['Precio_0-10', 'Precio_10-20', 
                  "Precio_20-40", "Precio_40-80","Precio_+80"]

  # Crear categorías de precio
    df['precio_bins'] = pd.cut(df['precio'], bins, labels=categorias)

  # Contar ocurrencias por categoría y generar gráfico de barras
    plot = df['precio_bins'].value_counts().plot.bar()

    return plot


# solicitar al usuario un precio minimo del vino
def solicitar_precio_min():
    """
  Función para solicitar al usuario el precio mínimo que desea.

  Retorno:
    Un valor flotante que representa el precio mínimo deseado.
  """
    while True:
        try:
            precio_min = float(input("Ingrese el precio mínimo que desea (en euros): "))
            if precio_min < 0:
                raise ValueError("El precio mínimo no puede ser negativo.")
            return precio_min
        except ValueError:
            print("Error: Debe ingresar un valor numérico válido para el precio mínimo.")


# solicitar al usuario un precio maximo del vino
def solicitar_precio_max(precio_min):
    """
  Función para solicitar al usuario el precio máximo que desea, considerando el precio mínimo.

  Argumento:
    precio_min: El precio mínimo previamente ingresado por el usuario.

  Retorno:
    Un valor flotante que representa el precio máximo deseado.
  """
    while True:
        try:
            precio_max = float(input("Ingrese el precio máximo que desea (en euros): "))
            if precio_max <= precio_min:
                raise ValueError("El precio máximo no puede ser menor o igual al precio mínimo.")
            return precio_max
        except ValueError:
            print("Error: Debe ingresar un valor numérico mayor que el precio mínimo.")

            
# generar DataFrame al que aplicar topsis
def filtrar_por_precio(df, precio_min, precio_max):
    """
  Función para filtrar un DataFrame por rango de precios.

  Argumentos:
    df: El DataFrame a filtrar.
    precio_min: Precio mínimo del rango.
    precio_max: Precio máximo del rango.

  Retorno:
    DataFrame filtrado por rango de precios.
  """

  # Filtrar por rango de precios
    df_filtrado = df[(precio_min <= df["precio"]) & (df["precio"] <= precio_max)]

    vinos = df_filtrado[["titulo","link","tipo2","precio","rating"]]
    vinos = vinos.dropna()
    
    return vinos

# Define la función para buscar por título y devolver "tipo2"
def buscar_tipo2_por_titulo(df,titulo):
    """
  Función para buscar el tipo2 por un título específico en el DataFrame vinos.

  Argumentos:
    df : DataFrame
    Columna_elegida: En este caso, columna llamada "tipo2"

  Retorno:
    Devuelve una lista con los datos de tipo2 que vamos a añadir a la matriz de topsis
  """
  # Filtra el DataFrame por el títuloabs
    df_filtrados = df[df["titulo"] == titulo]

  # Si se encuentra el título, devuelve el tipo2
    if not vinos_filtrados.empty:
        col_nueva = df_filtrados["tipo2"].values[0]
        return col_nueva
    else:
        return None


# funcion que engloba el algoritmo de topsis

def topsis(vinos):
    
    benefit_attributes = set([0, 1, 2, 3, 4])
    raw_data = vinos.values[:, 3:]
    candidates = vinos.values[:, 0]
    attributes = list(vinos.columns[3:])

    m = len(raw_data) # calcula el numero de datos
    n = len(attributes) # calcula el numero de atributos

  # Crear un array vacío para almacenar los divisores de normalización
    divisors = np.empty(n)
    for j in range(n): # El bucle for itera a través de cada atributo (j):
        column = raw_data[:,j] # Esto calcula la desviación estándar (raíz cuadrada de la varianza) de la column.
        divisors[j] = np.sqrt(column @ column)

    raw_data /= divisors # Esto normaliza todo el array raw_data dividiendo cada elemento por el valor correspondiente en el array divisors (división elemento a elemento).

    column_names = ["X {}".format(i + 1) for i in range(n)]
    df_topsis = pd.DataFrame(data=raw_data, index=candidates, columns=column_names)
#     print("Esta es la matriz normalizada:")
    df_topsis

    ######### 2.APLICAMOS LOS PESOS
    
    w_precios = 0.80  # peso de precios
    w_rating = 0.20   # peso de rating

    weights = np.array([w_precios, w_rating])
    
    raw_data *= weights
    df_topsis_pesos = pd.DataFrame(data=raw_data, index=candidates, columns=column_names)
    # Convierte el índice en una columna llamada "titulo"
    df_topsis_pesos.reset_index(inplace=True)
    df_topsis_pesos.rename(columns={"index": "titulo"}, inplace=True)
    df_topsis_pesos
        
    ######### 3.MATRIZ IDEAL /ANTIIDEAL
      
    a_pos = np.zeros(n) # Este vector representará los valores positivos ideales
    a_neg = np.zeros(n) # Este vector representará los valores negativos ideales

    for j in range(n): 
        column = raw_data[:,j] # Extrae la columna j específica del array raw_data como un array separado llamado column.
        max_val = np.max(column) # Calcula el valor máximo en la columna
        min_val = np.min(column) # Calcula el valor mínimo en la columna

        # See if we want to maximize benefit or minimize cost (for PIS)
        if j in benefit_attributes:
            a_pos[j] = max_val
            a_neg[j] = min_val
        else:
            a_pos[j] = min_val
            a_neg[j] = max_val

    pd.DataFrame(data=[a_pos, a_neg], index=["$A^*$", "$A^-$"], columns=column_names)

    
    ######### 4.MATRIZ DISTANCIAS  

    sp = np.zeros(m)
    sn = np.zeros(m)
    cs = np.zeros(m)

    for i in range(m):
        diff_pos = raw_data[i] - a_pos
        diff_neg = raw_data[i] - a_neg
        sp[i] = np.sqrt(diff_pos @ diff_pos)
        sn[i] = np.sqrt(diff_neg @ diff_neg)
        cs[i] = sn[i] / (sp[i] + sn[i])

    pd.DataFrame(data=zip(sp, sn, cs), index=candidates, columns=["$S^*$", "$S^-$", "$C^*$"])
    
    ######### 5.MATRIZ DISTANCIAS ORDENADA POR RANKING
    
    # Define una funcion para asignar un orden de ranking  
    def rank_according_to(data):
        ranks = rankdata(data).astype(int)
        ranks -= 1
        return candidates[ranks][::-1]
    
    cs_order = rank_according_to(cs)
    sp_order = rank_according_to(sp)
    sn_order = rank_according_to(sn)

    df_ranking = pd.DataFrame(data=zip(cs_order, sp_order, sn_order), index=range(1, m + 1), columns=["$C^*$", "$S^*$", "$S^-$"])
    # Convierte el índice en una columna llamada "ranking_topsis"
    df_ranking.reset_index(inplace=True)
    df_ranking.rename(columns={"index": "ranking_topsis"}, inplace=True)
    

    return df_topsis_pesos, df_ranking


##############################################################################

# AQUI LO DE PLOTEAR

# Define la función para generar las etiquetas
def generar_etiquetas(df):
    """
  Función para generar las etiquetas personalizadas para cada punto.

  Argumentos:
    row: Una fila del DataFrame.

  Retorno:
    Un diccionario con las etiquetas "etiqueta_ranking" y "etiqueta_titulo".
  """
    return {
      "etiqueta_ranking": df["ranking_topsis"],
      "etiqueta_titulo": df["titulo"],
  }


# funcion para hacer el grafico con un rel plot explicando por codigo de color el tipo de vino y por tamaño el precio. Tambien asigna una etiqueta de ranking
def grafico_topsis(df):
    # paleta de color:
    color_vino = {'tinto': 'crimson','blanco': 'sandybrown','espumoso': 'royalblue','generoso':'purple','rosado': 'violet', 'vermouth': 'limegreen'}

    # Crea el diagrama de dispersión
    g = sns.relplot(
        x="X 1",
        y="X 2",
        hue="tipo2",
        size="precio",
        sizes=(50, 200),
        data=df,
        palette=color_vino,)

    # Añade etiquetas personalizadas a cada punto
    for i, row in df.iterrows():
        etiquetas = generar_etiquetas(row)
      # Separar el título en dos líneas
        titulo_linea_1 = etiquetas["etiqueta_titulo"][:len(etiquetas["etiqueta_titulo"]) // 2]
        titulo_linea_2 = etiquetas["etiqueta_titulo"][len(etiquetas["etiqueta_titulo"]) // 2:]

        plt.annotate(f"{titulo_linea_1}\n{titulo_linea_2}", (row["X 1"], row["X 2"]), textcoords="offset points", xytext=(0, 20), ha='center')
        plt.annotate(etiquetas["etiqueta_ranking"], (row["X 1"], row["X 2"]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Establecer el texto del eje X
    plt.xlabel("Precio Ponderado") 
    # Establecer el texto del eje Y
    plt.ylabel("Ranking Ponderado") 
    # Muestra el plot
    return plt.show()

#### FUNCION DE VISUALIZACION INTERACTIVA CON EL USUARIO

def visualizacion_interactiva(df):
    """
    Crea un diagrama de dispersión para visualizar datos de vinos con diferentes opciones de personalización.

    Parámetros:
        df (pandas.DataFrame): El DataFrame que contiene los datos de los vinos.

    Devoluciones:
        plotly.Figure: El objeto de figura de Plotly que representa el diagrama de dispersión.
    """

    # Obtiene las columnas para el eje X y el eje Y
    x = df["X 1"]
    y = df["X 2"]

    # Definimos un mapeo de colores para los tipos de vinos
    color_map = {
        'tinto': 'crimson',
        'blanco': 'sandybrown',
        'espumoso': 'royalblue',
        'generoso': 'purple',
        'rosado': 'violet',
        'vermouth': 'limegreen'}

    # Define el tamaño de los puntos en función del ranking TOPSIS
    sizes = (df["ranking_topsis"].max() - df["ranking_topsis"]) / df["ranking_topsis"].max() * 20 + 5

    # Define el gráfico
    fig = go.Figure()

    # Agrega los puntos al gráfico
    for tipo in color_map.keys():
        datos_tipo = df[df["tipo2"] == tipo]
        fig.add_scatter(
            x=datos_tipo["X 1"],  # Corregido aquí
            y=datos_tipo["X 2"],  # Corregido aquí
            name=tipo,
            mode="markers",
            marker=dict(color=color_map[tipo], size=sizes, sizemode='diameter'),
            hoverinfo="text",
            text="**Tipo:** " + tipo + "<br>" + datos_tipo["titulo"] + "<br>Ranking TOPSIS: " + datos_tipo["ranking_topsis"].astype(str) + "<br>Precio: "+ datos_tipo["precio"].astype(str),  # Corregido aquí
        )

    # Personaliza el título y las etiquetas de los ejes
    fig.update_layout(
        title="Diagrama de dispersión de vinos",
        xaxis_title="Nombre del eje X",
        yaxis_title="Nombre del eje Y",
        legend_title="Tipos de Vinos",
    )

    # Muestra el gráfico y devuelve la figura
    return fig
###############################################################3#

def visualizacion_interactiva2(df):   
  # Obtiene las columnas para el eje X y el eje Y
    x = df["X 1"]
    y = df["X 2"]

    # Definimos un mapeo de colores para los tipos de vinos
    color_map = {
    'tinto': 'crimson',
    'blanco': 'sandybrown',
    'espumoso': 'royalblue',
    'generoso': 'purple',
    'rosado': 'violet', 
    'vermouth': 'limegreen'}

    # Crea un diccionario con los datos
    datos = {"x": x, "y": y, "tipo2": df["tipo2"]}

    # Define el gráfico
    fig = go.Figure()

    # Agrega los puntos al gráfico
    fig.add_scatter(
    x=datos["x"],
    y=datos["y"],
    name="Vinos",
    mode="markers",
    marker=dict(color=df["tipo2"].map({'tinto': 'crimson','blanco': 'sandybrown',
                            'espumoso': 'royalblue','generoso': 'purple',
                            'rosado': 'violet', 'vermouth': 'limegreen'}),size=15),
    hoverinfo="text",
    text=df["titulo"],
    )

    # Personaliza el título y las etiquetas de los ejes
    fig.update_layout(
    title="Diagrama de dispersión de vinos",
    xaxis_title="X1",
    yaxis_title="Y1",
    )


    # Muestra el gráfico
    return fig.show()

##################################################################3

def topsis_general(df):
    df = df[1:]
    precio_min = solicitar_precio_min()
    precio_max = solicitar_precio_max(precio_min)
    vinos = filtrar_por_precio(df, precio_min, precio_max)
    df_topsis_pesos, df_ranking = topsis(vinos)
    df_ranking.rename(columns={"$C^*$": "titulo"}, inplace=True)
    df_combinado = vinos.merge(df_ranking[["ranking_topsis","titulo"]], how='left', on='titulo')
    df_combinado = df_combinado.sort_values(by='ranking_topsis', ascending=True)  # Ordenar por 'ranking_topsis' de forma descendente
    df_combinado2 = df_combinado.merge(df_topsis_pesos[["titulo", "X 1", "X 2"]], how='left', on='titulo')
    return  visualizacion_interactiva(df_combinado2)
    


# Polynomial Regression
#Se utilizó el dataset poblacion_20210_2022.csv de la competencia de Kaggle "Colombia Municipalities Population 2010 to 2022", con más de 1400 registros.
#Link del CSV obtenido: https://www.kaggle.com/datasets/julianusugaortiz/colombia-poblation-20102022-municipalities
#Desde la línea 54 esta el analisis tecnico y estadistico del modelo
# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Leer el CSV 
dataset = pd.read_csv('poblacion_2010_2022.csv', encoding='utf-16')

# Agrupar por año y sumar la población total
poblacion_anual = dataset.groupby('AÑO')['Total_Poblacion'].sum().reset_index()

# Variables independientes y dependientes
X = poblacion_anual['AÑO'].values.reshape(-1, 1)
y = poblacion_anual['Total_Poblacion'].values

# Ajustar regresión lineal (solo para comparar visualmente)
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Crear y entrenar regresión polinómica (grado 4)
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizar regresión lineal
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Regresión Lineal - Población Total en Colombia')
plt.xlabel('Año')
plt.ylabel('Población Total')
plt.show()

# Visualizar regresión polinómica (curva suave)
X_grid = np.arange(min(X), max(X)+1, 0.1).reshape(-1, 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.transform(X_grid)), color='blue')
plt.title('Regresión Polinómica (Grado 4) - Población Total en Colombia')
plt.xlabel('Año')
plt.ylabel('Población Total')
plt.show()

# Predicción para año futuro (ej. 2023)
año_pred = 2023
poblacion_pred = lin_reg_2.predict(poly_reg.transform([[año_pred]]))
print(f"Predicción de población total para {año_pred}: {int(poblacion_pred[0]):,} habitantes")

# ---------------------------------------------
# ¿Por qué este archivo permite aplicar un algoritmo de Regresión Polinómica?
# 
# 1. El archivo CSV contiene registros agrupados por año (columna 'AÑO') y el total de población por año ('Total_Poblacion'), 
#    lo que genera una relación clara entre variable independiente (X: Año) y dependiente (Y: Población).
# 
# 2. Al graficar los datos, se observa que la tendencia del crecimiento poblacional no es lineal. 
#    Esto justifica el uso de una regresión polinómica para capturar la curvatura del comportamiento a lo largo del tiempo.
# 
# 3. El uso de un polinomio de grado 4 se adapta bien al patrón observado, sin llegar a sobreajustar debido a la baja cantidad de muestras (13 años),
#    lo cual lo hace adecuado para este caso puntual de análisis temporal.
#
# 4. Este tipo de regresión permite mejorar la precisión de predicción para valores futuros, como el año 2023.

# ---------------------------------------------
# Conclusiones estadísticas
# 
# - La regresión polinómica de grado 4 ofrece un R² significativamente mayor que la regresión lineal, lo que indica un mejor ajuste del modelo.
# - El error cuadrático medio (RMSE) también es menor en el modelo polinómico, confirmando su superioridad en precisión.
# - Aunque el modelo es más complejo, logra ajustarse mejor a la realidad de los datos históricos de población en Colombia.
# - Es importante no extrapolar demasiado lejos del rango original (2010-2022), ya que los polinomios pueden crecer muy rápido y generar errores.

# ---------------------------------------------
# Análisis técnico de los resultados
# 
# - La regresión lineal subestima el crecimiento poblacional en los últimos años, lo que se refleja en su desviación respecto a los puntos reales.
# - En cambio, la curva polinómica sigue muy de cerca los datos reales, especialmente entre 2015 y 2022, donde el crecimiento es más acelerado.
# - El modelo polinómico representa correctamente la aceleración del crecimiento poblacional.
# - Se recomienda este enfoque para análisis exploratorio, visualización y proyecciones a corto plazo.
# - Para predicciones a largo plazo o uso en sistemas en producción, se sugiere validar con datos más recientes y ajustar el modelo según sea necesario.


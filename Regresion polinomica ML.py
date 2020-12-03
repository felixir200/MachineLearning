# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 23:37:16 2020

@author: Jumper
"""

#PLANTILLA DE PREPROCESADO

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv('historical_data-Whiting Petroleum.csv')
X = dataset.iloc[:, 0:3].values #casi siempre se queda igual
y = dataset.iloc[:,3].values 

"""
# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)
"""

"""
# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
-------------------------------------------------------------------------

# Ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Ajustar la regresion polinomica con el dataset
from  sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree = 10) #crea un array de cuadrados de x
X_poly = poly_reg.fit_transform(X) #aplica la transformacion a X
lin_reg_2 =LinearRegression() 
lin_reg_2.fit(X_poly, y)

# Visualizacion de los resultados del modelo lineal
plt.scatter(X,y, color="red")
plt.plot(X,lin_reg.predict(X), color="blue")
plt.title("Modelo lineal")
plt.xlabel("posicion del empleado")
plt.xlabel("sueldo")
plt.show()


#Visualizacion de los resultados del modelo polinomico
X_grid = np.arange(min(X), max(X),0.1) #crea matriz cuadricula
X_grid = X_grid.reshape(len(X_grid),1) #transpone la matriz
plt.scatter(X,y, color="red")
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Modelo Polinomico")
plt.xlabel("posicion del empleado")
plt.xlabel("sueldo")
plt.show()

#Prediccion del modelo
lin_reg.predict([[6.5]]) #escribir en terminal
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
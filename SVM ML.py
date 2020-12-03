# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:29:29 2020

@author: Jumper
"""

# SUPPORT VECTOR REGRESSION


#PLANTILLA DE PREPROCESADO

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv('D:/no prioritarios/COMPU/machinelearning-az-master/datasets/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #casi siempre se queda igual
y = dataset.iloc[:,2].values 


"""
# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

"""


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))



#Ajustar la regresion con el dataser SVR
from sklearn.svm import SVR
regression =SVR(kernel = "rbf") #Maquina de soporte vectorial
regression.fit(X, y)


#Prediccion del modelo SVR
y_pred= sc_y.inverse_transform(
    regression.predict(sc_X.transform([[6.5]]))) #Poner doble corchete



#Visualizacion de los resultados del modelo polinomico
X_grid = np.arange(min(X), max(X),0.1) #crea matriz cuadricula o parrilla
X_grid = X_grid.reshape(len(X_grid),1) #transpone la matriz
plt.scatter(X,y, color="red")
plt.plot(X_grid,regression.predict(X_grid), color="blue")
plt.title("Modelo Polinomico")
plt.xlabel("posicion del empleado")
plt.xlabel("sueldo")
plt.show()



# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:44:45 2020

@author: Jumper
"""



#PLANTILLA DE PREPROCESADO

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv('D:/no prioritarios/COMPU/machinelearning-az-master/datasets/Part 2 - Regression/Section 9 - Random Forest Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #casi siempre se queda igual
y = dataset.iloc[:,2].values 


# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Ajustar el Random Forest con el dataset
from sklearn.ensemble import RandomForestRegressor
regresor = RandomForestRegressor(n_estimators = 300, random_state = 0 ) #crea el bosque
regresor.fit(X,y) #Aplica a nuestros datos

#Prediccion del modelo con Random Forest
y_pred = regresor.predict([[6.5]]) #prediccion de los valores de la lista

#Visualizacion de los resultados del modelo de Random Forest
X_grid = np.arange(min(X), max(X),0.0001) #crea matriz cuadricula
X_grid = X_grid.reshape(len(X_grid),1) #transpone la matriz
plt.scatter(X,y, color="red")
plt.plot(X_grid, regresor.predict(X_grid), color="blue")
plt.title("Modelo con Random Forest")
plt.xlabel("posicion del empleado")
plt.xlabel("sueldo")
plt.show()

print("Hello world")
print("ERES ARTE BABY")


# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 21:52:29 2020

@author: Jumper
"""

# PLANTILLA DE PREPROCESADO

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv("D:/no prioritarios/COMPU/machinelearning-az-master/datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,1].values 

# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 0)

# MODELO DE REGRESION LINEAL SIMPLE

# Crear modelo de regresion lineal simple con entrenamiento
from sklearn.linear_model import LinearRegression
regresion=LinearRegression()
regresion.fit(X_train, y_train)
regresion.summary()

# Predecir el conjunto de Test
y_pred=regresion.predict(X_test)

# Visualizar los datos de Entrenamiento
plt.plot(X_train,y_train,linestyle="None", marker="o", color="red") #nube de puntos
plt.plot(X_train, regresion.predict(X_train),linestyle=None, color="blue") #funcion lineal
plt.title("Sueldo vs Experiencia (Entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario (en $)")
plt.show()

# Visualizar los datos de Test
plt.plot(X_test,y_test,linestyle="None", marker="o", color="red") #nube de puntos
plt.plot(X_train, regresion.predict(X_train),linestyle=None, color="blue") #funcion lineal
plt.title("Sueldo vs Experiencia (Test)")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario (en $)")
plt.show()

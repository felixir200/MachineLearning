# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:20:23 2020

@author: Jumper
"""
#PLANTILLA DE PREPROCESADO

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv("D:/no prioritarios/COMPU/machinelearning-az-master/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv")
X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,4].values 

#Tomar en cuenta no tener multicolinealidad en las variables dummys

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) #Crea la etiqueta de la variable dummy

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
) #transforma la variable original a la variable dummy

X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

# Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regresion = LinearRegression()
regresion.fit(X_train, y_train)

#Predecir el conjunto test
y_pred=regresion.predict(X_test)

#Contruir el modelo optimo de RLM con eliminacion hacia atras
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
SL=0.05

X_opti = X[: ,[0,1,2,3,4,5]]
regresion_OLS=sm.OLS(endog=y, exog = X_opti.tolist()).fit()
regresion_OLS.summary() #los P-VALORES los sacaremos de jupyter

#Como la variable numero 2 (NewYork) es la menos indispensable la eliminamos

X_opti = X[: ,[0,1,3,4,5]]
regresion_OLS=sm.OLS(endog=y, exog = X_opti.tolist()).fit()
regresion_OLS.summary() #los P-VALORES los sacaremos de jupyter

X_opti = X[: ,[0,3,4,5]]
regresion_OLS=sm.OLS(endog=y, exog = X_opti.tolist()).fit()
regresion_OLS.summary() #los P-VALORES los sacaremos de jupyter

X_opti = X[: ,[0,3,5]]
regresion_OLS=sm.OLS(endog=y, exog = X_opti.tolist()).fit()
regresion_OLS.summary() #los P-VALORES los sacaremos de jupyter


#Para eliminacion automatica de las variables
"""
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regresion_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regresion_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regresion_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regresion_OLS.summary()    
    return x 
 
SL = 0.05
X_Modeled = backwardElimination(X_opt, SL)
"""

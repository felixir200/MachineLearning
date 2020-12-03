# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 01:39:44 2020

@author: Jumper
"""
# RED NEURONAL ARTIFICAL RNA
# Parte 1 - Preprocesado

#Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:] #se elimina una columna para evitar la multicoolinealidad

#Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Parte 2 - Creacion y entrenamiento de la Red Neuronal

#Importar librerias
import keras
from keras.models import Sequential
from keras.layers import Dense #Conexion entre neuronas


#Inicializar la RNA
clasificador = Sequential()

#Añadir las capas de entrada y las capas ocultas
clasificador.add(Dense(units = 6, kernel_initializer ='uniform', activation = 'relu', input_dim = 11)) #1° capa oculta
                 
clasificador.add(Dense(units = 6, kernel_initializer ='uniform', activation = 'relu')) #2° capa oculta

#Añadir la capa de salida
clasificador.add(Dense(units = 1, kernel_initializer ='uniform', activation = 'relu')) #capa de salida

#Compilar la RNA
clasificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Ajustar la RNA al conjunto de Entrenamiento
clasificador.fit(X_train, y_train, batch_size =10, epochs = 100)


# Parte 3 - Evaluacion del modelo y calculo de predicciones finales

#Prediccion de resultados con el conjunto de Testing
y_pred = clasificador.predict(X_test)
y_pred = (y_pred>0.5) #depende del objetivo y de la empresa
#Elaboracion de la matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
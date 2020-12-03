# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 20:24:38 2020

@author: Jumper
"""

#PLANTILLA DE PREPROCESADO

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el Data set
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values #las variables que se mapaearan
#y = dataset.iloc[:,:].values 

#Busqueda del numero optimo de Clusters (Codo)
from sklearn.cluster import KMeans
wcss =[] #se crea un array para las distancia
for i in range(1,11): #se inicia el bucle
    kmeans =KMeans(n_clusters= i, init = 'k-means++', n_init = 10, max_iter=300, random_state=0) 
    kmeans.fit(X) #ajusta la matriz de caracteristicas
    wcss.append(kmeans.inertia_) #agrega a wcss la suma de los cuadrados de las distancia
plt.plot(range(1,11), wcss) 
plt.title("Metodo del codo")  
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS")
plt.show()

#Aplicar el metodo K-Means para segmentar el dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300, random_state=0) #ejecutamos el clustering
y_kmeans = kmeans.fit_predict(X) #se hace el ajuste y la prediccion


#Visualizacion de los clusters
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0, 1],s=100, c='red', label='Cautos')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1, 1],s=100, c='green', label='Adictos')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2, 1],s=100, c='blue', label='Capitalistas')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3, 1],s=100, c='cyan', label='Anarcos')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4, 1],s=100, c='magenta', label='Tacaños')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200, c='yellow', label='baricentros')
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales")
plt.ylabel("Puntuacion de gastos")
plt.legend()
plt.show()





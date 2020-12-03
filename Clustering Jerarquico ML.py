# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:11:56 2020

@author: Jumper
"""


#PLANTILLA DE PREPROCESADO

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el Data set
dataset = pd.read_csv("D:/no prioritarios/COMPU/machinelearning-az-master/datasets/Part 4 - Clustering/Section 25 - Hierarchical Clustering/Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values #las variables que se mapaearan
#y = dataset.iloc[:,:].values 

#Buscar el umero optimo de clusters (Dendograma)
import scipy.cluster.hierarchy as sch
dendograma=sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("Dendograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia")
plt.show()

#Aplicar el clustering jerarquico al dataset
from sklearn.cluster import AgglomerativeClustering
ch =AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage= "ward")
y_ch =ch.fit_predict(X)

#visualizar los clusters
plt.scatter(X[y_ch == 0,0], X[y_ch ==0,1], s=100, c="red", label="precavidos")
plt.scatter(X[y_ch == 1,0], X[y_ch ==1,1], s=100, c="blue", label="Cluster 2")
plt.scatter(X[y_ch == 2,0], X[y_ch ==2,1], s=100, c="green", label="Cluster 3")
plt.scatter(X[y_ch == 3,0], X[y_ch ==3,1], s=100, c="cyan", label="Cluster 4")
plt.scatter(X[y_ch == 4,0], X[y_ch ==4,1], s=100, c="magenta", label="Cluster 5")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales")
plt.ylabel("Puntuacion de gastos")
plt.legend()
plt.show()

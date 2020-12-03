# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:49:17 2020

@author: Jumper
"""


#PLANTILLA DE PREPROCESADO

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values #casi siempre se queda igual
y = dataset.iloc[:,13].values 


# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)



# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Reduccion de la dimension con PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


"""
#Reduccion de la dimension con kernalPCA
from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components = 2, kernel ="rbf")
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
"""

#PARTE 2- algoritmo de clasificacion 

#Crear el clasificador del conjunto de entrenamiento
from sklearn.svm import SVC
clasificador = SVC(kernel = 'rbf', degree=7, random_state = 0 )
clasificador.fit(X_train, y_train)

#Prediccion de resultados de testing
y_pred=clasificador.predict(X_test)

#Crear matriz de confusion
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

#Representacion grafica de los resultados train
from matplotlib.colors import ListedColormap
X_set, y_set =X_train, y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:,0].max() +1, step=0.01), 
                    np.arange(start = X_set[:, 1].min() -1, stop = X_set[:,1].max() +1, step=0.01))
plt.contourf(X1,X2, clasificador.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                                          alpha =0.75,cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_test)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1],
                c= ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('SVM del conjunto train')
plt.xlabel('Edad')
plt.ylabel('Sueldo')
plt.legend()
plt.show

#Representacion grafica de los resultados test
X_set, y_set =X_test, y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:,0].max() +1, step=0.01), 
                    np.arange(start = X_set[:, 1].min() -1, stop = X_set[:,1].max() +1, step=0.01))
plt.contourf(X1,X2, clasificador.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                                          alpha =0.75,cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_test)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1],
                c= ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('SVM del conjunto test')
plt.xlabel('Edad')
plt.ylabel('Sueldo')
plt.legend()
plt.show

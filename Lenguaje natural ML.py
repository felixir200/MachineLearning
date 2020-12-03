# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:11:56 2020

@author: Jumper
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar el dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

#Limpieza de texto
import re
import nltk
nltk.download('stopwords') #descarga diccionario de palabras irrelevates
from nltk.corpus import stopwords #creo lista de palabras stopwords
from nltk.stem.porter import PorterStemmer #extrae raices de palabras 
corpus = []
for i in range(0,1000): #limpiamos cada una de las filas i
    
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) #reemplaza los caracteres por espacios
    review = review.lower() #Pasa todo a minusculas 
    review = review.split() #Separa la oracion en una lista de caracteres
    ps = PorterStemmer() #elimina conjugaciones de una determinada palabra
    review = [ps.stem(word) for word in review if not  word in set(stopwords.words('english'))] #reemplaza palabra por palabra que esten en stopwords y las stemmiza
    review = ' '.join(review) #regresa el conjunto a ser una oracion
    corpus.append(review) #agrego a corpus cada review
    
#Crear el bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features =1500)
X = cv.fit_transform(corpus).toarray() #crea la matriz de caracteristicas y la convierte en array
y = dataset.iloc[:, 1].values


#USAR ALGUN METODO DE CLASIFICACION

# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)


#Crear el clasificador del conjunto de entrenamiento
from sklearn.ensemble import RandomForestClassifier
clasificadorRF=RandomForestClassifier(n_estimators=500, criterion ="entropy",min_samples_split=2, random_state=0)
clasificadorRF.fit(X_train, y_train)

#indice 'gini' mide la dispersion de los datos o impureza 
#indice 'entropy' mide el caos, es decir la discrepancia cuando
#los datos estan juntos o separados para minimizar la entropia de cada nodo hoja


#Prediccion de resultados de testing
y_pred=clasificadorRF.predict(X_test)

#Crear matriz de confusion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
cm= confusion_matrix(y_test, y_pred)
cmplot= plot_confusion_matrix(clasificadorRF, X_test, y_pred, values_format='.3g')

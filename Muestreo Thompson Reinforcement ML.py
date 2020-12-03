# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 23:46:14 2020

@author: Jumper
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar el dataset
dataset =pd.read_csv("D:/no prioritarios/COMPU/machinelearning-az-master/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

#Algoritmo de Muestreo Thompson
import random
N=10000
d=10
numero_de_recompenzas_1 = [0]*d
numero_de_recompenzas_0 = [0]*d
ads_seleccionado=[] 
recompenza_total=0 #guarda la recompenza total y se va actualizando
#ronda = usuario
for n in range(0,N): #bucle para individuos
    max_random =0 #inicializa la variable
    ad=0 #inicializa la variable ad
    for i in range(0,d): #bucle para cada anuncios
        random_beta = random.betavariate(numero_de_recompenzas_1[i]+1, numero_de_recompenzas_0[i]+1) #calcula la funcion beta
        if random_beta > max_random: #Verificamos si el valor de esa ronda es el maximo
            max_random =random_beta #si es asi, lo guardamos en variable max
            ad=i #guarda el anuncio i
    ads_seleccionado.append(ad) #selecciona el mejor de los anuncios
    
    recompenza=dataset.values[n,ad] #extrae el valor del dataset
    if recompenza ==1:
        numero_de_recompenzas_1[ad] = numero_de_recompenzas_1[ad] +1 #Suma las recompenzas 1
    else:
        numero_de_recompenzas_0[ad] = numero_de_recompenzas_0[ad] +1 #Suma las recompenzas 0
        
    recompenza_total = recompenza_total + recompenza
    
#histograma de resultados
plt.hist(ads_seleccionado)
plt.title("histograma de anuncios")
plt.xlabel("Id del anuncio")
plt.ylabel("Frecuencia del anuncio")
plt.show()

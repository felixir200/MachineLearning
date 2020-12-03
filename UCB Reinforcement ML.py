# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 22:16:03 2020

@author: Jumper
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar el dataset
dataset =pd.read_csv("D:/no prioritarios/COMPU/machinelearning-az-master/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

#Algoritmo de Upper Confidence Bound
import math
N=10000
d=10
numero_de_selecciones=[0]*d #inicializa la variable en 0s
suma_de_recompenzas=[0]*d
ads_seleccionado=[] 
recompenza_total=0 #guarda la recompenza total y se va actualizando
#ronda = usuario
for n in range(0,N): #bucle para individuos
    max_upper_bound=0 #inicializa la variable
    ad=0 #inicializa la variable ad
    for i in range(0,d): #bucle para cada anuncios
        if (numero_de_selecciones[i]>0):
            recompenza_media= suma_de_recompenzas[i] / numero_de_selecciones[i] #calculo de recompenza media
            delta_i=math.sqrt(3/2*math.log(n+1)/numero_de_selecciones[i]) #calculo de intervalo de confianza  se le suma +1 para no dividir entre zero
            upper_bound =recompenza_media + delta_i #Calculo del limite superior
        else:
            upper_bound = 1e400
            
        if upper_bound > max_upper_bound: #Verificamos si el valor de esa ronda es el maximo
            max_upper_bound =upper_bound #si es asi, lo guardamos en variable max
            ad=i #guarda el anuncio
    ads_seleccionado.append(ad) #selecciona el mejor de los anuncios
    numero_de_selecciones[ad]=numero_de_selecciones[ad] + 1 #actualiza las selecciones
    recompenza=dataset.values[n,ad] #extrae el valor del dataset
    suma_de_recompenzas[ad]=suma_de_recompenzas[ad] + recompenza 
    recompenza_total = recompenza_total + recompenza
    
#histograma de resultados
plt.hist(ads_seleccionado)
plt.tilte("histograma de anuncios")
plt.xlabel("Id del anuncio")
plt.ylabel("Frecuencia del anuncio")
plt.show
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:58:00 2020

@author: Jumper
"""


#K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator =clasificador, X = X_train, y=y_train, cv = 10)
accuracies.mean() #promedio de exactitudes
accuracies.std() #DESVIACION ESTANDAR DE LOS DATOS

#Aplicar Grid Search para optimizar los parametros
from sklearn.model_selection import GridSearchCV
parametros =[{'C':[1,10,100,1000], 'kernel':['linear']}, #explora los parametros del kernel lineal
             {'C':[1,10,100,1000], 'kernel':['rbf'],'gamma': [0.5,0.1,0.001,0.0001]}] #parametros para kernel rbf
grid_search =GridSearchCV(estimator = clasificador,
                          param_grid = parametros, 
                          scoring = 'accuracy', #metrica
                          cv =10, #hace la validacion cruzada
                          n_jobs = -1) 
grid_search = grid_search.fit(X_train, y_train) # se ajusta al modelo
best_accuracy = grid_search.best_score_ 
best_parameters = grid_search.best_params_ # da los mejores palametros

#conocidos los parametros podemos volver a hacer el grid search con parametros mas cercanos a lo que nos marco como los optimos 
#para buscar optimizar aun mas los parametros
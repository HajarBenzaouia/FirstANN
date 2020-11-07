# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:23:29 2020

@author: BENZAOUIA Hajar
"""
import numpy as np
from sklearn.model_selection import train_test_split
# determiner un nbr de seed pour reproduire l'execution
seed = 7
np.random.seed(seed)
# charger la base pima-indians-diabetes
dataset = np.loadtxt("D:\Reims\DL\TP1\diabetes.csv", delimiter=",")
# recupérer la matrice des données dans X, et le vecteur des classes dans Y
X = dataset[:,0:8]
Y = dataset[:,8]
# 33% test et 67% train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)


from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
# Fonction pour creer le modele Keras
def define_model():
 # definir le modele
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer= 'uniform' , activation=
    'relu' ))
    model.add(Dense(8, kernel_initializer= 'uniform' , activation= 'relu' ))
    model.add(Dense(1, kernel_initializer= 'uniform' , activation= 'sigmoid' ))
 # compiler le modele
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=['accuracy' ])
    return model
# utiliser le modele avec KerasClassifier
model = KerasClassifier(build_fn=define_model)


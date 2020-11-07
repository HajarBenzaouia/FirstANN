# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 14:34:07 2020

@author: BENZAOUIA Hajar
"""

import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

seed = 7 
np.random.seed(seed)

#Fonction pour charger la base
def load_data():
    dataset=np.loadtxt("seeds.csv", delimiter=",")
    return dataset

#Fonction pour créer le modèle Keras
def define_model():
    
    #definir le modèle
    model =  Sequential()
    model.add(Dense(12,input_dim = 7, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8,kernel_initializer='uniform', activation='relu'))
    model.add(Dense(3,kernel_initializer='uniform', activation='softmax'))          
    
    #compiler modele
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

#Recupérer la matrice des données dans X, et le vecteur des classes dans Y
data = load_data()
X = data[:,0:7]
Y = data [:,7]

#Utiliser le modèle avec KerasClassifier pour un problème de classification
model = KerasClassifier(build_fn=define_model, epochs=150, batch_size=10, verbose=1)

#Validation croisee avec 10-fold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


#Appliquer la validation croisee sur le modele
results =  cross_val_score(model, X, Y, cv=kfold)

#Afficher les moyennes des scores sur les 10-fold
print(results.mean())
    


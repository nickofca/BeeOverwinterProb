# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:35:01 2018

@author: Anderson Lab
"""

import pandas
from keras.models import Model
from keras.layers import Input, Dense
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
import numpy
import seaborn 
import matplotlib.pyplot as plt

inputFile = inpF = "data/Nick file.txt"

def modeler(inp):
    ##Hyperparameters
    survivalshipThreshold = 6 #[Frames of bees]
    probabilityThreshold = .5
    randomSeed = 95
    
    ##Import and preprocess
    inputData = inpD = pandas.read_csv(inp, sep='\t', header = 0)
    inpD.columns = ["SeptBees","SeptMites","JanuaryBees"]
    inpD["Survived"] = (inpD["JanuaryBees"] >= survivalshipThreshold).astype(int)
    train, test = model_selection.train_test_split(inpD, random_state = randomSeed)

    ##Build Neural Network (simple architecture)
    inp = Input(shape=(2,))
    hid = Dense(8,activation="relu")(inp)
    hid = Dense(8,activation="relu" )(hid)
    hid = Dense(1,activation="sigmoid" )(hid)
    model = Model(inputs=inp, outputs=hid)
    
    ##Train NN
    model.compile(optimizer="sgd",loss="mse")
    model.fit(x=train[["SeptBees","SeptMites"]],y=train["Survived"],epochs=400)
    testPrediction = model.predict(x=test[["SeptBees","SeptMites"]])
    
    ##Measure model performance     
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(test["Survived"].values,testPrediction)
    roc_auc = auc(fpr, tpr)
    
    ##Do a grid run with the model
    xmin = int(numpy.floor(inpD["SeptBees"].min()))
    xmax = int(numpy.ceil(inpD["SeptBees"].max()))
    miteAxis = range(20)
    heatmapData = hmD = pandas.DataFrame(index=miteAxis,columns=range(xmin,xmax))
    for x in range(xmin,xmax):
        for y in miteAxis:
            #numpy.asscalar to turn numpy float64 into native python float
            hmD.loc[y,x] = model.predict(numpy.array([[x,y]]))[0][0]
        #Inner elements are stored as objects; converted to numpy.float64
        hmD[x]=pandas.to_numeric(hmD[x])
        
    ##Create heat map
    fig, ax = plt.subplots(figsize=(15,15)) 
    seaborn.heatmap(hmD.T,vmin=0,vmax=1,cbar_kws={"shrink": 0.35},linewidths = .6, square=True, ax=ax, annot=True)
    plt.xlabel("Mites per 100 bees")
    plt.ylabel("Frames of Bees")
    plt.tight_layout()
    plt.savefig("heatmap.png")
    plt.show()
    
    ##Plot ROC curve
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    plt.figure()
    lw=2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([  0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig("ROC_curve.png")
    plt.show()
    
    return model

    
    
    
model= modeler(inpF)

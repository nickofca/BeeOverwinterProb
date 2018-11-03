# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:36:37 2018

@author: Anderson Lab
"""

import pandas
from sklearn import model_selection, linear_model
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
    
    ##Predict outcomes
    model = linear_model.LogisticRegression(random_state=randomSeed).fit(train[["SeptBees","SeptMites"]],train["Survived"])
    testPrediction = model.predict_proba(test[["SeptBees","SeptMites"]])
    testPredRound = (testPrediction[:,1] > probabilityThreshold)
    testDif = testPredRound == test["Survived"]
    testAccuracy = testDif.sum()/test.shape[0]
    testPrecision = (testPredRound == test["Survived"]).sum()/testPredRound.sum()
    testRecall = (testPredRound == test["Survived"]).sum()/test["Survived"].sum()
    
    ##Do a grid run with the model
    xmin = int(numpy.floor(inpD["SeptBees"].min()))
    xmax = int(numpy.ceil(inpD["SeptBees"].max()))
    miteAxis = range(20)
    heatmapData = hmD = pandas.DataFrame(index=miteAxis,columns=range(xmin,xmax))
    for x in range(xmin,xmax):
        for y in miteAxis:
            #numpy.asscalar to turn numpy float64 into native python float
            hmD.loc[y,x] = model.predict_proba([[x,y]])[0][1]
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
    
    return(model,testAccuracy, testPrecision, testRecall)

model, testAccuracy, testPrecision, testRecall = modeler(inpF)
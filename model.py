# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:36:37 2018

@author: Anderson Lab
"""

import pandas
from sklearn import model_selection, linear_model
import numpy
import seaborn 

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
    model = linear_model.LogisticRegression(random_state=randomSeed).fit(test[["SeptBees","SeptMites"]],test["Survived"])
    testPrediction = model.predict_proba(train[["SeptBees","SeptMites"]])
    testPredRound = (testPrediction[:,1] > probabilityThreshold)
    testDif = testPredRound == train["Survived"]
    testAccuracy = testDif.sum()/train.shape[0]
    
    
    ##Do a grid run with the model
    xmin = int(numpy.floor(inpD["SeptBees"].min()))
    xmax = int(numpy.ceil(inpD["SeptBees"].max()))
    miteAxis = range(20)
    heatmapData = hmD = pandas.DataFrame(index=miteAxis,columns=range(xmin,xmax))
    for x in range(xmin,xmax):
        for y in miteAxis:
            #numpy.asscalar to turn numpy float64 into native python float
            hmD.loc[y,x] = numpy.asscalar(model.predict_proba([[x,y]])[0][1])
    
    ##Create heat map
    seaborn.heatmap(hmD.values)
    
    return(model,testAccuracy)

model, testAccuracy = modeler(inpF)
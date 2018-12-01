# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:36:37 2018

@author: Anderson Lab
"""

import pandas
import numpy
import seaborn 
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm

inputFile = inpF = "data/Nick file.txt"

def modeler(inp):
    ##Hyperparameters
    #proportion of data withheld to to test
    survivalshipThreshold = 6 #[Frames of bees]
    randomSeed = 95
    
    ##Import and preprocess
    inputData = inpD = pandas.read_csv(inp, sep='\t', header = 0)
    inpD.columns = ["SeptBees","SeptMites","JanuaryBees"]
    inpD["Survived"] = (inpD["JanuaryBees"] >= survivalshipThreshold).astype(int)
    x = inpD[["SeptBees","SeptMites"]]
    y = inpD["Survived"]
    
    ##Train model statsmodels 
    #Use of default MLE method
    model = sm.Logit(y,x).fit()
    coeff = model.params.values
    odds_ratio = numpy.exp(coeff)
    print(model.wald_test_terms())
    
    ##Do a grid run with the model
    xmin = int(numpy.floor(inpD["SeptBees"].min()))
    xmax = int(numpy.ceil(inpD["SeptBees"].max()))
    miteAxis = range(20)
    heatmapData = hmD = pandas.DataFrame(index=miteAxis,columns=range(xmin,xmax))
    for x in range(xmin,xmax):
        for y in miteAxis:
            #numpy.asscalar to turn numpy float64 into native python float
            hmD.loc[y,x] = model.predict([[x,y]])[0]
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
    
    return model, odds_ratio, pvalues, chi2

model, oddsRatio, pvalues, chi2 = modeler(inpF)

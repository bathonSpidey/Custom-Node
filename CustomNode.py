# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 10:13:13 2021

@author: batho
"""
import numpy as np

class Node():
    
    def dataReshape(self,dataset):
        return dataset.reshape(dataset.shape[0],-1).T
        
    def sigmoid(self,linearFunctionResult):
        return 1./(1+np.exp(-linearFunctionResult))
    
    def initializeWeightAndBias(self,totalFeatures):
        return np.zeros((totalFeatures,1)) ,0
    
    def forwardPropagate(self,weights,bias,trainingSet, targetSet):
        trainingSet=self.dataReshape(trainingSet)
        activatedResult=self.getActivatedResult(weights, bias,trainingSet)
        gradients, cost=self.getGradients(trainingSet, targetSet, activatedResult)
        return activatedResult, gradients, cost
    
    def getActivatedResult(self, weights, bias, dataset):
        linearFunctionResult=np.dot(weights.T,dataset)+bias
        activatedResult=self.sigmoid(linearFunctionResult)
        return activatedResult
    
    def getGradients(self,trainingSet, targetSet, activatedResult):
        totalEntries=trainingSet.shape[1]
        cost=self.getCost(activatedResult,targetSet, totalEntries)
        error=activatedResult-targetSet
        gradientWeights=(np.dot(trainingSet,error.T))/totalEntries
        gradientBias=(np.sum(error))/totalEntries
        gradients={"gradientWeight":gradientWeights,"gradientBias": gradientBias}
        return gradients, cost
    
    def getCost(self, activatedResult,targetSet, totalEntries):
        cost=-(np.dot(targetSet,np.log(activatedResult).T)+
               np.dot(1-targetSet,np.log(1-activatedResult).T))/totalEntries
        cost = np.squeeze(cost)
        return cost
    
    def gradientDescent(self, nodeParameters, trainingSet, targetSet, iterations, learningRate):
        costs=[]
        weights, bias=nodeParameters
        for i in range(iterations):
            _,gradients,cost =self.forwardPropagate(weights,bias,trainingSet, targetSet)
            weightGradient=gradients["gradientWeight"]
            biasGradient=gradients["gradientBias"]
            weights=weights-learningRate*weightGradient
            bias=bias-learningRate*biasGradient
            if i%100==0:
                costs.append(cost)
        parameters = {"weights": weights,"bias": bias}
        gradients = {"gradientWeight": weightGradient,"gradientBias": biasGradient}
        return parameters, gradients
    
    def predict(self,nodeParameters,testSet):
        weights,bias=nodeParameters
        testSet=self.dataReshape(testSet)
        weights=weights.reshape(testSet.shape[0],1)
        totalTestEntries=testSet.shape[1]
        predictions=np.zeros((1,totalTestEntries))
        activatedResults=self.getActivatedResult(weights,bias,testSet)
        for i in range(activatedResults.shape[1]):
            if activatedResults[0,i]>0.5:
                predictions[0,i]=1
            else:
                predictions[0,i]=0
        return predictions


        
        
        


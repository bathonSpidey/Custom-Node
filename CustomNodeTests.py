# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:00:01 2021

@author: batho
"""

import unittest
import CustomNode as nd
import numpy as np

class TestNode(unittest.TestCase):
    
    
    def setUp(self):
        self.dataset=np.array([(0,0),(0,1),(1,0),(1,1)])
        self.testDataset=self.dataset
        self.targetSet=np.array([0,1,1,1])
        self.node=nd.Node()
        self.weights,self.bias=self.node.initializeWeightAndBias(2)
        
    def test_reshape(self):
        self.assertEqual(self.dataset.shape[0], 4)
        trainingSet=self.node.dataReshape(self.dataset)
        self.assertEqual(trainingSet.shape[0], 2)
        self.assertEqual(trainingSet.shape[1], 4)
        
    def test_sigmoid(self):
        self.assertEqual(self.node.sigmoid(0), .5)
        self.assertEqual(round(self.node.sigmoid(1),3), .731)
    
    def test_initialization(self):
        self.assertEqual(self.weights.shape[0], 2)
        self.assertEqual(self.weights.shape[1], 1)
        self.assertEqual(self.bias, 0)
    
    def test_activation(self):
        weights,bias=self.node.initializeWeightAndBias(self.dataset.shape[1])
        result, gradients, cost= self.node.forwardPropagate(weights,bias,
                                                            self.dataset,
                                                            self.targetSet)
        self.assertEqual(result.shape[0],1)
        self.assertEqual(result.shape[1],4)
        for i in range(4):
            self.assertEqual(result[0][i], .5)
        return gradients, cost
          
    def test_gradients(self):
        gradients, cost= self.test_activation()
        weights=gradients["gradientWeight"]
        bias=gradients["gradientBias"]
        self.assertEqual(weights.shape[0],2)
        self.assertEqual(weights[0], weights[1])
        self.assertEqual(weights[0], -0.25)
        self.assertEqual(bias,-0.25)
        return cost
    
    def test_cost(self):
        cost=self.test_gradients()
        self.assertEqual(cost.shape,())
        self.assertEqual(round(cost.item(),3) ,0.693)
        
    def test_gradientDescentOneIteration(self):
        parameters, gradients=self.node.gradientDescent((self.weights, self.bias),
                                                        self.dataset,
                                                        self.targetSet,
                                                        1,0.1)
        weights=parameters["weights"]
        bias=parameters["bias"]
        self.assertEqual(weights.shape[0],2)
        self.assertEqual(weights.shape[1],1)
        self.assertEqual(weights[0][0],weights[1][0])
        self.assertEqual(weights[0][0],.025)
        self.assertEqual(bias,.025)
        return parameters
    
    def test_gradientDescentTwoIteration(self):
        parameters, gradients=self.node.gradientDescent((self.weights, self.bias),
                                                        self.dataset,
                                                        self.targetSet,
                                                        2,0.1)
       
        weights=parameters["weights"]
        bias=parameters["bias"]
        self.assertEqual(round(weights[0][0],3),.049)
        self.assertEqual(round(weights[1][0],3),.049)
        self.assertEqual(round(bias,3),.049)
        
    def test_predict(self):
        parameters=self.test_gradientDescentOneIteration()
        nodeParameters=(parameters["weights"],parameters["bias"])
        prediction=self.node.predict(nodeParameters,self.testDataset)
        self.assertEqual(prediction[0][0], 1)
        
if __name__ == '__main__':
    unittest.main()
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 16:49:44 2016

@author: v-yuewng
"""


import random
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from environment.memory import Memory
# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
# import theano


class tableq:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_size = inputs
        self.output_size = outputs
        self.memory = Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate
   
    def initNetworks(self):
        self.model = np.zeros([self.input_size,self.output_size])
        self.targetModel = np.zeros([self.input_size,self.output_size])
        
        
    def updateTargetNetwork(self):
         self.targetModel = self.model.copy()

    # predict Q values for all the actions
    def getQValues(self, state):
        
        return self.model[state,:]

    def getTargetQValues(self, state):
        
        return self.targetModel[state,:]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else : 
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action



    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=False):
        # Do not learn until we've got self.learnStart samples    
        
        
        
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            minibatch = self.memory.getMiniBatch(1)
            sample = minibatch[0]
            isFinal = sample['isFinal']
            state = sample['state']
            action = sample['action']
            reward = sample['reward']
            newState = sample['newState']

            qValues = self.getQValues(state)
            if useTargetNetwork:
                qValuesNewState = self.getTargetQValues(newState)
            else :
                qValuesNewState = self.getQValues(newState)
            targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)
            
            self.model[state,action] = self.model[state,action]+self.learningRate*(targetValue-qValues[action])
            self.losshistory = ((self.model-self.targetModel)**2).sum()
           
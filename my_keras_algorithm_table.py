# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 21:53:43 2016

@author: v-yuewng
"""

from environment import env_keras as ek
from environment.memory import Memory
#from environment.learner import DeepQ
import random
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from environment.tableq import tableq
# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
# import theano


envir = ek.matenv('mygenerate')
# envir.monitor.start('/tmp/mountaincar-experiment-1', force=True)

# Exploring the new environment observations and actions:
#
# >>> import gym
# env = gym.make('MountainCar-v0')>>> env = gym.make('MountainCar-v0')
# [2016-06-19 17:37:12,780] Making new env: MountainCar-v0
# >>> print env.observation_space
# Box(2,)
# >>> print env.action_space
# Discrete(3)


epochs = 100000
steps = 5000
updateTargetNetwork = 100
explorationRate = 1
minibatch_size = 1
learnStart = 1
learningRate = 0.0025
discountFactor =0.99
memorySize = 1000000

last100Scores = [0] * 10
last100ScoresIndex = 0
last100Filled = False

statenum = envir.statenum
featurenum = envir.featurenum
#deepQ = DeepQ(featurenum, 2, memorySize, discountFactor, learningRate, learnStart)
# deepQ.initNetworks([30,30,30])
#deepQ.initNetworks([])
# deepQ.initNetworks([300,300])


deepQ = tableq(statenum, 2, memorySize, discountFactor, learningRate, learnStart)
deepQ.initNetworks()


stepCounter = 0
average = [0]
rewardsum = [0]
rewradsumepoch = [0]
loss = []
# number of reruns
for epoch in xrange(epochs):
    while True:
        
        envir.setstate()
        observation = envir.currentstate
        if envir.isfinal == False:
            break
    
    # number of timesteps
    rewardsum.append(rewradsumepoch[-1])
    rewradsumepoch = [0]
    for t in xrange(steps):
        qValues = deepQ.getQValues(observation)
        
        action = deepQ.selectAction(qValues, explorationRate)
      
        newObservation, reward, done= envir.step(action,2)
        rewradsumepoch.append(reward+rewradsumepoch[-1])
        if epoch>=0:
            pass
            
#           
        
        if (t >= 499):
            print "Failed. Time out"
            print t
            done = True
            # reward = 200            

        if done and t < 499:
            print "Sucess!"
            # reward -= 200
        deepQ.addMemory(observation, action, reward, newObservation, done)

        if stepCounter >= learnStart:
            if stepCounter <= updateTargetNetwork:
                deepQ.learnOnMiniBatch(minibatch_size, False)
                loss.append(deepQ.losshistory)
            else :
                deepQ.learnOnMiniBatch(minibatch_size, True)
                loss.append(deepQ.losshistory)

        observation = newObservation

        if done:
            last100Scores[last100ScoresIndex] = t
            last100ScoresIndex += 1
            if last100ScoresIndex >= 0:
                last100Filled = True
                last100ScoresIndex = 0
            if not last100Filled:
                print "Episode ",epoch," finished after {} timesteps".format(t+1)
            else :
                print "Episode ",epoch," finished after {} timesteps".format(t+1)," last 100 average: ",(sum(last100Scores)/len(last100Scores))
                average.append((sum(last100Scores)/len(last100Scores)))
            break

        stepCounter += 1
        if stepCounter % updateTargetNetwork == 0:
            deepQ.updateTargetNetwork()
            print "updating target network"
        

    explorationRate *= 0.95
    # explorationRate -= (2.0/epochs)
    explorationRate = max (0.05, explorationRate)

# env.monitor.close()


def getallqvalue(envir,deepQ):
    allqvalue = -100*np.ones([envir.statenum,2])
    for ii in xrange(envir.statenum):
        allqvalue[ii,:] = deepQ.getQValues(ii)
    return allqvalue
    
res = getallqvalue(envir,deepQ)
plot(res,'.')
plot(res[:,0]-res[:,1],'*')

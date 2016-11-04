# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 14:54:12 2016

@author: v-yuewng
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 21:53:43 2016

@author: v-yuewng
"""

from environment import env_keras as ek
from environment.memory import Memory
from environment.learner import DeepQ
import random
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
# import theano


envir = ek.matenv('my')
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


epochs = 1
steps = 5000
updateTargetNetwork = 10
explorationRate = 1
minibatch_size = 40
learnStart = 80
learningRate = 0.00025
discountFactor =0.99
memorySize = 1000000

last100Scores = [0] * 10
last100ScoresIndex = 0
last100Filled = False

deepQ = DeepQ(2, 2, memorySize, discountFactor, learningRate, learnStart)
# deepQ.initNetworks([30,30,30])
deepQ.initNetworks([30,30,30])
# deepQ.initNetworks([300,300])

stepCounter = 0

# number of reruns
for epoch in xrange(epochs):
    envir.setstat()
    observation = envir.getstat()
    
    # number of timesteps
    for t in xrange(steps):
        qValues = deepQ.getQValues(observation)

        action = deepQ.selectAction(qValues, explorationRate)
      
        newObservation, reward, done= envir.step(action)
        if epoch>=0:
            
#            print '@@@@@@@@@@  '+str(action) +  '@@@@@@@@@@  ' + str(reward) +'@@@@@@@@@@  ' + str(envir.currentstat)
#            print observation
#            print '\n'
            print deepQ.getQValues(np.array([1,0]))
            print deepQ.getQValues(np.array([0,1]))
            print '\n\n'
            
            
            
            
#            print '\n'
#            print observation
#            print '\n'
#            print action
#            print '\n'
#            print reward
#            print '\n'
#            
#            print newObservation
#            print '\n'
        
        if (t >= 199):
            print "Failed. Time out"
            print t
            done = False
            # reward = 200            

        if done and t < 199:
            print "Sucess!"
            # reward -= 200
        deepQ.addMemory(observation, action, reward, newObservation, done)

        if stepCounter >= learnStart:
            if stepCounter <= updateTargetNetwork:
                deepQ.learnOnMiniBatch(minibatch_size, False)
            else :
                deepQ.learnOnMiniBatch(minibatch_size, True)

        observation = newObservation

        if done:
            last100Scores[last100ScoresIndex] = t
            last100ScoresIndex += 1
            if last100ScoresIndex >= 10:
                last100Filled = True
                last100ScoresIndex = 0
            if not last100Filled:
                print "Episode ",epoch," finished after {} timesteps".format(t+1)
            else :
                print "Episode ",epoch," finished after {} timesteps".format(t+1)," last 100 average: ",(sum(last100Scores)/len(last100Scores))
            break

        stepCounter += 1
        if stepCounter % updateTargetNetwork == 0:
            deepQ.updateTargetNetwork()
            print "updating target network"
        

    explorationRate *= 0.9995
    # explorationRate -= (2.0/epochs)
    explorationRate = max (0.05, explorationRate)

# env.monitor.close()
import os
import numpy as np
import pandas as pd
import random

def txt2mat(filename):
    f = open(filename)
    filetxt = f.readlines()
    
    for ii in filetxt:
        if ii[0]=='#':
            dimenv = ii.split(',')
            xdim = int(dimenv[1])
            ydim = int(dimenv[2])
            transmat = np.zeros((xdim,ydim))            
            y=0
        else:
            transpro = ii.split(',')
            transmat[y,:]=np.array(transpro,dtype='float')
            y+=1
    return transmat

    
def stat2num(filename):
    f = open(filename)
    filetxt = f.readlines()
    res = dict()
    nn=0
    for ii in filetxt:
        if ii[0]=='#':
            dimstat = ii.split(',')
            statdim = int(dimstat[1])
            featuredim = int(dimstat[2])
        else:
            
            res[nn]=np.array(ii.split(','),dtype='float')
            nn+=1
    return res
            
    
def simulation(transmat,currstat):
    thisrand = random.random()
    statnum = transmat.shape[0]
    prob = 0
    for jj in range(statnum):
        prob += transmat[currstat,jj]
        if thisrand < prob:
            return jj
        
        


class matenv:
##     newObservation, reward, done, info = env.step(action)  
    def __init__(self,filename):
        
        self.transmat = txt2mat(filename+'trans.csv')
        self.rewardmat = txt2mat(filename+'reward.csv')
        self.stats = stat2num(filename + 'stat.csv')
        self.statnum = len(self.stats)
        self.isfinal = False
        self.currentstat  =random.randint(1,self.statnum-2)
        
    def setstat(self,stat= None):
        if stat ==None:
            self.currentstat  =random.randint(1,self.statnum-2)
        else:
            self.currentstat = stat

    def getstat(self):
        return self.stats[self.currentstat]
            
    def step(self,action):
        
        
        
        newstat = self.currentstat+2*action-1
        reward = self.rewardmat[self.currentstat,self.currentstat+2*action-1]
        
        if newstat==0 or newstat == self.statnum:
            self.isfinal = True
        
        return self.stats[newstat],reward,self.isfinal
 
        


    

    
#    
#class agent:
#    
#    def __init__(self,filename):
#        self.env = matenv(filename)
#        self.actions = self.env.statnum
#        
#        
#    def setstat(self,stat= None):
#        if stat ==None:
#            self.currentstat  =random.randint(0,self.actions-1)
#        else:
#            self.currentstat = stat
#            
#    def selectionaction():
#        pass
#    
#    def 
#        
        
        
        
    
            
    
    
        

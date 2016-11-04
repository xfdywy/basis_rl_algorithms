import os
import numpy as np
import pandas as pd
import random
from environment.memory import Memory
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
            transpro = ii.split(',')[:xdim]
            transmat[y,:]=np.array(transpro,dtype='float')
            y+=1
    return transmat

    
def state2num(filename):
    f = open(filename)
    filetxt = f.readlines()
    res = dict()
    isfinal = dict()
    nn=0
    for ii in filetxt:
        if ii[0]=='#':
            dimstate = ii.split(',')
            statedim = int(dimstate[1])
            featuredim = int(dimstate[2])
        else:
            if ii.split(',')[0] == 'T' :
                isfinal[nn] = True
            else:
                isfinal[nn] = False
            res[nn]=np.array(ii.split(',')[1:featuredim+1],dtype='float')
            nn+=1
    return res,isfinal,featuredim
            
    
def simulation(transmat,currstate):
    thisrand = random.random()
    statenum = transmat.shape[0]
    prob = 0
    for jj in range(statenum):
        prob += transmat[currstate,jj]
        if thisrand < prob:
            return jj
        
        


class matenv:
##     newObservation, reward, done, info = env.step(action)  
    def __init__(self,filename):
        
        ##self.transmat = txt2mat('./model/'+filename+'/'+filename+'trans.csv')
        self.rewardmat = txt2mat('./model/'+filename+'/'+filename+'reward.csv')
        self.states,self.isfinals,self.featurenum = state2num('./model/'+filename+'/'+filename + 'state.csv')
        self.statenum = len(self.states)
        
        self.currentstate  =random.randint(1,self.statenum-1)
        self.isfinal = self.isfinals[self.currentstate]
        
    def setstate(self,state= None):
        if state ==None:
            self.currentstate  =random.randint(0,self.statenum-1)
        else:
            self.currentstate = state
        self.isfinal = self.isfinals[self.currentstate]

    def getstate(self):
        return self.states[self.currentstate]
            
    def step(self,action,lors=1):
        
        
        
        newstate = self.currentstate +2*action-1
        reward = self.rewardmat[self.currentstate,newstate]
        self.currentstate = newstate        
        self.isfinal = self.isfinals[self.currentstate]
        
        if lors == 1:
            return self.states[newstate],reward,self.isfinal
        else:
            return newstate,reward,self.isfinal
 
        


    

    
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
        
        
        
    
            
    
    
        

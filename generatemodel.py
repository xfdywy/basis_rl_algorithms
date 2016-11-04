# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 15:35:19 2016

@author: v-yuewng
"""

import numpy as np
import pandas as pd
import os

nstate = 30
nfeature = 30
modelname = 'mygenerate'

finalstate = [0,nstate-1]

if not os.path.exists('./model/'+modelname):
    os.mkdir('./model/'+modelname)
os.chdir('./model/'+modelname)


f = open(modelname+'state.csv','w+')
f.write('#,'+str(nstate)+','+str(nfeature)+'\n')
for ii in xrange(nstate):
    if ii in finalstate:
        f.write('T')
    else:
        f.write('F')
    for jj in xrange(nfeature):
        if ii==jj:
            f.write(',1')
        else:
            f.write(',0')
    f.write('\n')
f.close()


f = open(modelname+'reward.csv','w+')
f.write('#,'+str(nstate)+','+str(nstate)+'\n')
for ii in xrange(nstate):
    if ii in finalstate:
        for jj in xrange(nstate):
            if ii == jj:
                f.write('100,')
            else:
                f.write('0,')
    else:
        for jj in xrange(nfeature):
            if jj== ii-1 or jj== ii+1:
                
                if  0 in np.abs(jj - np.array(finalstate)):
                    f.write('100,')
                else:
                    f.write('-10.5,')
            else:
                f.write('0,')
            
           
    f.write('0\n')
f.close()
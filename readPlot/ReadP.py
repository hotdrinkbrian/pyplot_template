import pickle
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
from   hm import *

path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/temp/'
pklN = 'result_v9_withSelection.pickle'







f1     = open(path + pklN,'read')
ret    = pickle.load(f1)

maxAOC = 0
maxKey = ''
"""
for key in ret:
    tempAOC = ret[key]['aoc']
    if tempAOC > maxAOC:
        maxAOC = tempAOC
        maxKey = key 
    #print key + ': ' + str( ret[key]['aoc'] )
print 'optimal combi: ' + maxKey
print 'AOC: '+ str( maxAOC )
"""

"""
#for grid search##################################################
for keyi, dicti in ret.iteritems():
    for keyj, dictj in dicti.iteritems():
        tempAOC = dictj['aoc']
    if tempAOC > maxAOC:
        maxAOC = tempAOC
        maxKey = {'lr':keyi,'ne':keyj}
    #print key + ': ' + str( ret[key]['aoc'] )
print 'optimal combi: ' 
print maxKey
print 'AOC: '+ str( maxAOC )
#for grid search##################################################
"""

#for random search##################################################
for keyi, dicti in ret.iteritems():
    #for keyj, dictj in dicti.iteritems():
    tempAOC = dicti['aoc']
    if tempAOC > maxAOC:
        maxAOC = tempAOC
        maxKey = dicti['params']
    #print key + ': ' + str( ret[key]['aoc'] )
print 'optimal combi: ' 
print maxKey
print 'AOC: '+ str( maxAOC )
#for random search##################################################





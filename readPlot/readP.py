import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hm import *

selected = 1#1
attrKin  = 0#1
testOn   = 0#1
version  = '5'
if selected == 1:
    versionStr = version+'_withSelection'
else:
    versionStr = version+'_noSelection'
#path = '../qcd_with_jumps/'
#path = '../'
#path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/'
#path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/'
path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/temp/'

if attrKin == 1:
    pklN = 'result_with_pt_mass_energy_v' + versionStr + '.pickle'
else:
    pklN = 'result_v' + versionStr + '.pickle'

if testOn == 0:
    attrL = ['cHadE','nHadE','cHadEFrac','nHadEFrac','nEmE','nEmEFrac','cEmE','cEmEFrac','cmuE','cmuEFrac','muE','muEFrac','eleE','eleEFrac','eleMulti','photonE','photonEFrac','photonMulti','cHadMulti','npr','cMulti','nMulti']#,'FracCal']
else:
    attrL=['cHadE','nHadE']
    attrL=['cHadE']


f1 = open(path + pklN,'read')
ret = pickle.load(f1)
#print ret
#for key in ret:
#    print key
maxAOC = 0
maxKey = ''
for key in ret:
    tempAOC = ret[key]['aoc']
    if tempAOC > maxAOC:
        maxAOC = tempAOC
        maxKey = key 
    #print key + ': ' + str( ret[key]['aoc'] )
print 'optimal combi: ' + maxKey
print 'AOC: '+ str( maxAOC )
#print '~~~~~~~~~~~~~~~~~~~~~~~~~'
#print ret['photonMulti,cHadEFrac']['aoc']
#print ret['cHadE,cHadE']['aoc']

"""
m1 = np.zeros( (len(attrL),len(attrL)) )
#print m1
for i in enumerate(attrL):
    for j in enumerate(attrL):
        a = i[1] + ',' + j[1]
        b = j[1] + ',' + i[1]
        for key in ret:
            if a == key or b == key:
                m1[i[0]][j[0]] = ret[key]['aoc'] 
print m1
"""

imDict = {}
for i in attrL:
    imDict[i] = {}

for i in attrL:
    for j in attrL:
        imDict[i][j] = 0.

LL = []
for i in attrL:
    for j in attrL:
        c = i+','+j
        d = j+','+i
        for key in ret:  
            if c == key or d == key:   
                imDict[i][j] = ret[key]['aoc'] 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~2bDel
#imDict['cHadE']['cHadE'] = 0.
#imDict['muE']['muEFrac'] = 0.54027419
#imDict['muEFrac']['muE'] = 0.54027419
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~2bDel

df = pd.DataFrame(imDict)
print df
LL = df.columns.values.tolist()



fig, ax = plt.subplots()

im, cbar = heatmap(df, LL, LL, ax=ax,
                   cmap="YlGn", cbarlabel="AOC")
texts = annotate_heatmap(im, valfmt="{x:.3f} t", fsize=6)

fig.tight_layout()
plt.show()
 
    




f1.close()

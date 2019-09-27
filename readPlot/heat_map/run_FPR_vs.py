from os import system as act
from time import sleep as slp

cnn = 0#1#0#1

A = 'python '
B = '/beegfs/desy/user/hezhiyua/git/pyplot_template/readPlot/heat_map/'
C = 'FPR_vs.py'

#D = ' --trnm '  + '50'
#E = ' --trnl '  + '2000'
#F  = ' --mode '

G = ' --valm '  + '40'#'30'
H = ' --vall '  + '1000'#'2000'

if   cnn == 0:    I = ' --model ' + 'lola'#'cnn'#'lola'
elif cnn == 1:    I = ' --model ' + 'cnn'

J = ' --baseline ' + 'cut'#'bdt'

# For cnn:
if   cnn == 1:
    #mass_list = ['50']
    #ctau_list = ['500']
    mass_list = ['30']
    ctau_list = ['500']
elif cnn == 0:
    mass_list = [20,30,40,50]
    ctau_list = [500,1000,2000,5000]
    mass_list = [30]#[20]#[40]
    ctau_list = [500]#[500]#[2000]
    mass_list = ['cb']
    ctau_list = ['cb']

mode_list = ['val']#['val','err']

for mm in mass_list:
    mm = str(mm)
    D  = ' --trnm '  + mm
    for ct in ctau_list:
        ct = str(ct)
        E  = ' --trnl '  + ct
        for md in mode_list:
            F = ' --mode ' + md

            act_str = A+B+C+D+E+F +G+H +I+J
            print act_str
            act(act_str)
            #slp(1*1)


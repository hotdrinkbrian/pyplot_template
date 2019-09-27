from os import system as act
from time import sleep as slp

A = 'python '
B = '/beegfs/desy/user/hezhiyua/git/pyplot_template/readPlot/heat_map/'+'MA/lola/'
C = 'lola_vs_cut.py'#'readP.py'

#D = ' --trnm '  + '50'
#E = ' --trnl '  + '2000'
#F  = ' --mode '

mass_list = [20,30,40,50]
ctau_list = [500,1000,2000,5000]
mass_list = [30]#[20]
ctau_list = [500]#[1000]
mass_list = ['cb']
ctau_list = ['cb']


mode_list = ['val','err']
#mode_list = ['val']
#mode_list = ['err']

for mm in mass_list:
    mm = str(mm)
    D  = ' --trnm '  + mm
    for ct in ctau_list:
        ct = str(ct)
        E  = ' --trnl '  + ct
        for md in mode_list:
            F = ' --mode ' + md

            act_str = A+B+C+D+E+F
            print act_str
            act(act_str)
            #slp(1)


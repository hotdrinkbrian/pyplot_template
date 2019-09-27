from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hm import *

import matplotlib
matplotlib.use('Agg') #prevent the plot from showing
import argparse as agp

pars = agp.ArgumentParser()
pars.add_argument('--trnm',action='store',type=str,help='train mass')
pars.add_argument('--trnl',action='store',type=str,help='train lifetime')
pars.add_argument('--mode',action='store',type=str,help='mode: value or error')

args  = pars.parse_args()
trn_m = args.trnm
trn_l = args.trnl
mode  = args.mode


#nn_pth  = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/all_in_1/nn_format/2jets/DPG/lola/Results/'
nn_pth  = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/all_in_1/nn_format/2jets/DPG/lola/Results/'+'trn30_500val50_1000/'

"""
mean_auc = []
for i in range(4):
    nn_name_i = 'lola_auc_'+str(i)+'.pkl'
    nn_dic_i  = joblib.load(nn_pth+nn_name)
    mean_auc.append(nn_dic_i)
"""
i       = 0
nn_name = 'lola_auc_'+str(i)+'.pkl'
nn_dic  = joblib.load(nn_pth+nn_name)



path    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/train_on_selected_QCD_bug/test/'
pth_out = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/train_on_selected_QCD_bug/auc_map/' 

def read_pkl(pth):
    pkls = joblib.load(pth)
    return pkls['data']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

#trn_m = 40
#trn_l = 5000
trn_m = int(trn_m)
trn_l = int(trn_l)

val = mode
#val = 'val'
#val = 'err'

#mass_list    = [20,30,40,50,60]
mass_list    = [20,30,40,50]
ctau_list    = [500,1000,2000,5000]

cut_type     = ['hard_cut']#['loose_cut']#['hard_cut']#['loose_cut','hard_cut']
inputs       = ['2best']#['full']#['2best']#['full']#['2best','full']
kin_var      = ['kin1']#['kin0']#['kin0','kin1']

combi_dict   = {}
cc           = 0
for i in inputs:
    for j in kin_var:
        tmp_list = []
        tmp_list.append(i)
        tmp_list.append(j)
        combi_dict[cc] = '_'.join(tmp_list)
        cc            += 1

empty_log = []
out_dict  = {}

if 1:
    if 1:
	out_dict = {}
	for key, item in combi_dict.iteritems():
	    out_dict[item] = {}
	    for m_trn in [trn_m]:
		out_dict[item][m_trn]            = {}
		for l_trn in [trn_l]: 
		    out_dict[item][m_trn][l_trn] = {}
		    for m_tst in mass_list:
			out_dict[item][m_trn][l_trn][m_tst] = {} 
			for l_tst in ctau_list:
			    out_dict[item][m_trn][l_trn][m_tst][l_tst] = {}
			    trn_part = str(m_trn)+'GeV_'+str(l_trn)+'mm_'
			    tst_part = 'tst'+'_'+str(m_tst)+'GeV_'+str(l_tst)
			    file_to_look = 'RS_'+'trn'+'_'+trn_part+tst_part+'mm_slct1_attr_'+item+'_v0.pkl'
			    print file_to_look
			    path_tot     = path + file_to_look
			    in_dict      = read_pkl(path_tot)
			    if 1:#in_dict:             !!!!!!!
				roc_dict = in_dict['roc']
                                auc_bdt  = in_dict['aoc']
				#auc_bdt  = roc_dict['aoc']

                                phsp_str = str(m_tst)+'_'+str(l_tst)
                                auc_nn   = nn_dic[phsp_str]

				#out_dict[item][m_trn][l_trn][m_tst][l_tst]['auc'] = auc_nn / float(auc_bdt) 
			        #out_dict[item][m_trn][l_trn][m_tst][l_tst]['auc'] = auc_nn
                                out_dict[item][m_trn][l_trn][m_tst][l_tst]['auc'] = auc_bdt
			 

    #print out_dict
    ####################################################################
    ####################################################################







imDict   = {}
err_dict = {}
for i in mass_list:
    imDict[i]   = {}
    err_dict[i] = {}  
    for j in ctau_list:
        imDict[i][j]   = 0.
        err_dict[i][j] = 0.


#input_string = 'full_kin0'
#input_string = '2best_kin0'
input_string = '2best_kin1'
#input_string = 'full_kin1'
LL = []
for mmi in mass_list:
    for lli in ctau_list:  
        imDict[mmi][lli]   = out_dict[input_string][trn_m][trn_l][mmi][lli]['auc'] 
        #err_dict[mmi][lli] = out_dict[input_string][trn_m][trn_l][mmi][lli]['auc--err']


df_val = pd.DataFrame(imDict)
#df_err = pd.DataFrame(err_dict)

if val == 'val':
    df        = df_val
    val_label = 'AUC ratio'
#elif val == 'err':
#    df        = df_err
#    val_label = '(1/FPR_BDT)/(1/FPR_cut) at cut TPR -- errors'


m_L = df.columns.values.tolist()
c_L = [500,1000,2000,5000]


fig, ax  = plt.subplots()
im, cbar = heatmap(df, c_L, m_L, ax=ax, cmap="YlGn", cbarlabel=val_label)
texts    = annotate_heatmap(im, valfmt="{x:.3f}", fsize=6)
fig.tight_layout()
#plt.show()
 
outName  = 'phase_space_map_bdt_vs_lola'+'_trn_'+str(trn_m)+'_'+str(trn_l)+'_'+val 
#outName  = 'phase_space_map_auc_lola'+'_trn_'+str(trn_m)+'_'+str(trn_l)+'_'+val
#outName  = 'phase_space_map_auc_bdt'+'_trn_'+str(trn_m)+'_'+str(trn_l)+'_'+val

fig.savefig(pth_out + outName + '.png', bbox_inches='tight')    



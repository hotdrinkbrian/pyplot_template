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

pars.add_argument('--valm',action='store',type=str,help='val mass')
pars.add_argument('--vall',action='store',type=str,help='val lifetime')

pars.add_argument('--model',action='store',type=str,help='model: lola or cnn')
pars.add_argument('--baseline',action='store',type=str,help='model: bdt or cut')

pars.add_argument('--mode',action='store',type=str,help='mode: value or error')

args  = pars.parse_args()
trn_m = args.trnm
trn_l = args.trnl

val_m = args.valm
val_l = args.vall

model = args.model
baseline = args.baseline

mode  = args.mode

if model == 'cnn' :
    if   baseline == 'bdt':    val_label = r'$\frac{ \frac{1}{FPR_{CNN}} }{ \frac{1}{FPR_{BDT}} }|TPR_{cut}$'
    elif baseline == 'cut':    val_label = r'$\frac{ \frac{1}{FPR_{CNN}} }{ \frac{1}{FPR_{cut}} }|TPR_{cut}$'
    decimal_p = 2
if model == 'lola':
    if   baseline == 'bdt':    val_label = r'$\frac{ \frac{1}{FPR_{LoLa}} }{ \frac{1}{FPR_{BDT}} }|TPR_{cut}$'
    elif baseline == 'cut':    val_label = r'$\frac{ \frac{1}{FPR_{LoLa}} }{ \frac{1}{FPR_{cut}} }|TPR_{cut}$'
    decimal_p = 2#1

n_jets       = 4#2
v_production = 'v6'

#tv_str = 'trn50_500val30_2000/'#'trn30_500val50_1000/'
#tv_str  = 'trn'+str(trn_m)+'_'+str(trn_l)+'val'+'30_2000'+'/'

if not trn_m == 'cb':
    tv_str  = 'trn'+str(trn_m)+'_'+str(trn_l)+'val'+'40_1000'+'/'
else:
    tv_str  = 'trn'+'_combined_'+'val'+'_combined_'+'/'  

PTH = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/allInOne/nn_format/'+str(n_jets)+'jets/'+v_production+'/'

#nn_pth  = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/all_in_1/nn_format/2jets/DPG/lola/Results/'+tv_str
#nn_pth  = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/allInOne/nn_format/2jets/DPG/lola/Results/'+tv_str
#nn_pth  = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/allInOne/nn_format/'+str(n_jets)+'jets/'+v_production+'/lola/Results/'+tv_str
nn_pth  = PTH+'/lola/Results/'+tv_str

path_nn = nn_pth





#path    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/train_on_selected_QCD_bug/test/'
#path    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/DPG/'
#path    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/DPG_new/'
#path    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/DPG_post/'
#path    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/DPG_post/'
#path    = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/'+v_production+'/'
path    = PTH+'bdt/'

#pth_out = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/train_on_selected_QCD_bug/auc_map/'
#pth_out = path_nn
pth_out = path + 'heat/'

def read_pkl(pth):
    pkls = joblib.load(pth)
    return pkls['data']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

#trn_m = 40
#trn_l = 5000
if not trn_m == 'cb':
    trn_m = int(trn_m)
    trn_l = int(trn_l)

val = mode
#val = 'val'
#val = 'err'

#VV = '1'
#VV = '0'
VV = '01'
#VV = '0123'

mass_list    = [20,30,40,50]
ctau_list    = [500,1000,2000,5000]

cut_type     = ['hard_cut']#['loose_cut']#['hard_cut']#['loose_cut','hard_cut']
inputs       = ['6best']#['2best']#['full']#['2best','full']
kin_var      = ['kin1']#['kin1']#['kin0']#['kin0','kin1']

#jet_lst      = ['jet0']
#jet_lst      = ['jet1']
jet_lst      = ['jet01']
#ExcLimit     = ['ExLim1']
ExcLimit     = ['ExLim0']

input_string = '_'.join([inputs[0],kin_var[0],jet_lst[0],ExcLimit[0]])



combi_dict   = {}
cc           = 0
for i in inputs:
    for j in kin_var:
        for k in jet_lst:
            for l in ExcLimit:
        #if True:
        #    if True:
                tmp_list = []
                tmp_list.append(i)
                tmp_list.append(j)
                tmp_list.append(k)
                tmp_list.append(l)
                combi_dict[cc] = '_'.join(tmp_list)
                cc            += 1

empty_log = []
out_dict  = {}

if True:
    for ci in cut_type:
	out_dict[ci] = {}
	for key, item in combi_dict.iteritems():
	    out_dict[ci][item] = {}
	    for m_trn in [trn_m]:
		out_dict[ci][item][m_trn]            = {}
		for l_trn in [trn_l]: 
		    out_dict[ci][item][m_trn][l_trn] = {}
		    for m_tst in mass_list:
			out_dict[ci][item][m_trn][l_trn][m_tst] = {} 
			for l_tst in ctau_list:
			    out_dict[ci][item][m_trn][l_trn][m_tst][l_tst] = {}
			    trn_part = str(m_trn)+'GeV_'+str(l_trn)+'mm_'
			    tst_part = 'tst'+'_'+str(m_tst)+'GeV_'+str(l_tst)
			    file_to_look = 'RS_'+'trn'+'_'+trn_part+tst_part+'mm_slct1_attr_'+item+'_v0.pkl'
			    print file_to_look
			    path_tot     = path + file_to_look
			    in_dict      = read_pkl(path_tot)

                            file_to_look_nn = 'RS_'+'trn'+'_'+trn_part+tst_part+'mm_slct1_attr_'+model+'_v'+str(VV)+'.pkl'#str(0)
                            path_tot_nn     = path_nn + file_to_look_nn
                            in_dict_nn      = read_pkl(path_tot_nn) 

			    if 1:#in_dict:             !!!!!!!
				roc_dict = in_dict['roc']
				fpr_bdt  = roc_dict['fpr']
				tpr_bdt  = roc_dict['tpr']
				#e_tpr_l  = roc_dict['e_tpr_l']
				e_fpr_l  = roc_dict['e_fpr_l']
				#e_tpr_h  = roc_dict['e_tpr_h']
				e_fpr_h  = roc_dict['e_fpr_h']
                
                                roc_dict_nn = in_dict_nn['roc']
                                fpr_nn      = roc_dict_nn['fpr']
                                tpr_nn      = roc_dict_nn['tpr']
                                #e_tpr_l_nn  = roc_dict_nn['e_tpr_l']
                                e_fpr_l_nn  = roc_dict_nn['e_fpr_l']
                                #e_tpr_h_nn  = roc_dict_nn['e_tpr_h']
                                e_fpr_h_nn  = roc_dict_nn['e_fpr_h']

      
                                #"""
				cut_dict = in_dict['cut_based']
				dicti    = cut_dict[ci]
				sgn_eff  = dicti['tpr']
				fls_eff  = dicti['fpr']
				#tpr_e_l  = dicti['tpr_e_l']
				fpr_e_l  = dicti['fpr_e_l']
				#tpr_e_h  = dicti['tpr_e_h']
				fpr_e_h  = dicti['fpr_e_h']
                                #"""  

                                ##################### test 
                                #print sgn_eff
                                #sgn_eff  = 0.1#0.4#0.9#0.2#0.66#0.2#0.4#0.5#0.6  #0.9#0.8#0.7   #0.1#0.4     
                                #####################          

				if sgn_eff != 0:
				    tmp_tpr, indx = find_nearest(tpr_bdt, sgn_eff)
				    tmp_fpr       = fpr_bdt[indx]

                                    tmp_tpr_nn, indx_nn = find_nearest(tpr_nn, sgn_eff)
                                    tmp_fpr_nn          = fpr_nn[indx_nn]
      
                                    count     = 1   
                                    if_done   = False
                                    #print 'sgn_eff', sgn_eff
                                    while not if_done:
                                        if   tmp_tpr > sgn_eff:
                                            idx_h     = indx
                                            idx_l     = indx+count
                                            tmp_tpr_h = tmp_tpr
                                            tmp_tpr_l = tpr_bdt[idx_l]
                                            if tmp_tpr_l < sgn_eff:    if_done = True     
                                        elif tmp_tpr < sgn_eff:
                                            idx_l     = indx
                                            idx_h     = indx-count
                                            tmp_tpr_l = tmp_tpr
                                            tmp_tpr_h = tpr_bdt[idx_h]  
                                            idx_l     = indx
                                            if tmp_tpr_h > sgn_eff:    if_done = True 
                                        elif tmp_tpr == sgn_eff:  
                                            idx_h     = indx
                                            idx_l     = indx
                                            tmp_tpr_l = tmp_tpr
                                            tmp_tpr_h = tmp_tpr       
                                            if_done   = True   
                                        if file_to_look == 'RS_trn_30GeV_500mm_tst_20GeV_5000mm_slct1_attr_full_kin0_v0.pkl':
                                            print 'sgn_eff', sgn_eff  
                                            print 'tmp_tpr_l', tmp_tpr_l
                                            print 'tmp_tpr_h', tmp_tpr_h
                                            print 'tmp_fpr', tmp_fpr
                                        count += 1     

                                    #delta_fpr_l   = np.abs( tmp_fpr-(fpr_bdt[idx_l]-e_fpr_l[idx_l]) )
                                    #delta_fpr_h   = np.abs( (fpr_bdt[idx_h]+e_fpr_h[idx_h])-tmp_fpr )  
                                    #err_fpr       = np.max([delta_fpr_l,delta_fpr_h,e_fpr_h[indx],e_fpr_l[indx]])   
                                    #fpr_err       = np.maximum(fpr_e_l,fpr_e_h)

                                    """
                                    count_nn     = 1
                                    if_done_nn   = False
                                    #print 'sgn_eff', sgn_eff
                                    while not if_done_nn:
                                        if   tmp_tpr_nn > sgn_eff:
                                            idx_h_nn     = indx_nn
                                            idx_l_nn     = indx_nn+count_nn
                                            tmp_tpr_h_nn = tmp_tpr_nn
                                            tmp_tpr_l_nn = tpr_nn[idx_l_nn]
                                            if tmp_tpr_l_nn < sgn_eff:    if_done_nn = True
                                        elif tmp_tpr_nn < sgn_eff:
                                            idx_l_nn     = indx_nn
                                            idx_h_nn     = indx_nn-count_nn
                                            tmp_tpr_l_nn = tmp_tpr_nn
                                            tmp_tpr_h_nn = tpr_nn[idx_h_nn]
                                            idx_l_nn     = indx_nn
                                            if tmp_tpr_h_nn > sgn_eff:    if_done_nn = True
                                        elif tmp_tpr_nn == sgn_eff:
                                            idx_h_nn     = indx_nn
                                            idx_l_nn     = indx_nn
                                            tmp_tpr_l_nn = tmp_tpr_nn
                                            tmp_tpr_h_nn = tmp_tpr_nn
                                            if_done_nn   = True
                                        count_nn += 1

                                     
                                    delta_fpr_l   = np.abs( tmp_fpr-(fpr_bdt[idx_l]-e_fpr_l[idx_l]) )
                                    delta_fpr_h   = np.abs( (fpr_bdt[idx_h]+e_fpr_h[idx_h])-tmp_fpr )  
                                    err_fpr       = np.max([delta_fpr_l,delta_fpr_h,e_fpr_h[indx],e_fpr_l[indx]])   
                                    fpr_err       = np.maximum(fpr_e_l,fpr_e_h)
                                    """
                                    

				    if tmp_fpr != 0:
                                        print 'TPR:' 
                                        print 'BDT: ', tmp_tpr
                                        print 'NN: ', tmp_tpr_nn  
                                        if baseline == 'bdt': 
					    inv_fpr     = tmp_fpr/float(tmp_fpr_nn)#fls_eff/float(tmp_fpr)
                                        elif baseline == 'cut':
                                            inv_fpr     = fls_eff/float(tmp_fpr_nn)  

                                        #inv_fpr_err = np.sqrt(np.square(fpr_err/fls_eff) + np.square(err_fpr/tmp_fpr))

				    else           :
					print '>>>>>>>>>>>>>>>>>>>> Zero devision!'
					inv_fpr     = 0
					inv_fpr_err = 0
				else:    pass # !!!!!!!!!!!!!!!!!!!!!!!!!!!! hard coded cut TPR!!!!!
 
				out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['FPR_ratio']      = inv_fpr 
				#out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['FPR_ratio_err'] = inv_fpr_err 

			    else:
				print '>>>>>>>>>>>>>>>>>>>> no in_dict!!'
				out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['FPR_ratio']     = 0 
				out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['FPR_ratio_err'] = 0 

				empty_log.append(file_to_look)


    #path_out = './plot/'
    


    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Empty_log:', empty_log

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

LL = []
for mmi in mass_list:
    for lli in ctau_list:  
        imDict[mmi][lli]   = out_dict['hard_cut'][input_string][trn_m][trn_l][mmi][lli]['FPR_ratio'] 
        #err_dict[mmi][lli] = out_dict['hard_cut'][input_string][trn_m][trn_l][mmi][lli]['FPR_ratio_err']


df_val = pd.DataFrame(imDict)
#df_err = pd.DataFrame(err_dict)

if val == 'val':
    df        = df_val

#elif val == 'err':
#    df         = df_err
#    val_label += ' -- errors' 

m_L = df.columns.values.tolist()
c_L = [500,1000,2000,5000]


fig, ax  = plt.subplots()
im, cbar = heatmap(df, c_L, m_L, ax=ax, cmap="YlGn", cbarlabel=val_label)
texts    = annotate_heatmap(im, valfmt="{x:."+str(decimal_p)+"f}", fsize=16)
fig.tight_layout()
#plt.show()
 
outName  = '2Dmap_'+model+'_vs_'+baseline +'_trn_'+str(trn_m)+'_'+str(trn_l)+'_val_'+str(val_m)+'_'+str(val_l) +'_'+input_string+'_'+val +'_v'+str(VV)

fig.savefig(pth_out + outName + '.png', bbox_inches='tight')    







from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hm import *

#path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/generalization_bdt/rs/'
path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/generalization_bdt/find2b/'


attr_list = ['J1cHadEFrac','J1nHadEFrac','J1nEmEFrac','J1cEmEFrac','J1cmuEFrac','J1muEFrac','J1eleEFrac','J1eleMulti','J1photonEFrac','J1photonMulti','J1cHadMulti','J1nHadMulti','J1npr','J1cMulti','J1nMulti','J1nSelectedTracks','J1ecalE']

def combi_2ofN(lst):
    kL = []
    for a1 in lst:
        for a2 in lst:
            if not ([a1,a2] in kL) | ([a2,a1] in kL):    kL.append([a1,a2])
    return kL

def read_pkl(pth):
    pkls = joblib.load(pth)
    return pkls['data']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


attr_2combi_list = combi_2ofN(attr_list)
a_c_list         = []
for i in attr_2combi_list:
    a_c_list.append('_'.join(i))
print a_c_list    

counting = 0
out_dict  = {}
#print attr_2combi_list
for a1 in attr_list:
    a1_s = a1[2:]
    out_dict[a1_s] = {}
    for a2 in attr_list:
        a2_s = a2[2:] 
        a_c1 = a1+'_'+a2
        a_c2 = a2+'_'+a1
        if   a_c1 in a_c_list:    a_c = a_c1
        elif a_c2 in a_c_list:    a_c = a_c2
        file_name = 'RS_trn_60GeV_5000mm_tst_60GeV_5000mm_slct1_attr_'+a_c+'_kin0_v0.pkl'
        counting += 1 
        print counting
        print file_name
        in_dict   = read_pkl(path+file_name)
        
        roc_dict = in_dict['roc']
        fpr_bdt  = roc_dict['fpr']
        tpr_bdt  = roc_dict['tpr']
        #e_tpr_l  = roc_dict['e_tpr_l']
        e_fpr_l  = roc_dict['e_fpr_l']
        #e_tpr_h  = roc_dict['e_tpr_h']
        e_fpr_h  = roc_dict['e_fpr_h']
        
        cut_dict = in_dict['cut_based']
        dicti    = cut_dict['hard_cut']
        sgn_eff  = dicti['tpr']
        fls_eff  = dicti['fpr']
        #tpr_e_l  = dicti['tpr_e_l']
        fpr_e_l  = dicti['fpr_e_l']
        #tpr_e_h  = dicti['tpr_e_h']
        fpr_e_h  = dicti['fpr_e_h']

        tmp_tpr, indx = find_nearest(tpr_bdt, sgn_eff)
        tmp_fpr       = fpr_bdt[indx]
        
        count     = 1   
        if_done   = False
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
            #print 'tmp_tpr_l', tmp_tpr_l
            #print 'tmp_tpr_h', tmp_tpr_h
            count += 1     
       
        delta_fpr_l = np.abs( tmp_fpr-(fpr_bdt[idx_l]-e_fpr_l[idx_l]) )
        delta_fpr_h = np.abs( (fpr_bdt[idx_h]+e_fpr_h[idx_h])-tmp_fpr )  
        err_fpr     = np.max([delta_fpr_l,delta_fpr_h,e_fpr_h[indx],e_fpr_l[indx]])   
        fpr_err     = np.maximum(fpr_e_l,fpr_e_h)
        
       
        inv_fpr     = fls_eff/float(tmp_fpr)
        inv_fpr_err = np.sqrt(np.square(fpr_err/fls_eff) + np.square(err_fpr/tmp_fpr))

        #inv_fpr = 1./float(fpr)
        out_dict[a1_s][a2_s] = {}
        out_dict[a1_s][a2_s]['score'] = inv_fpr
        out_dict[a1_s][a2_s]['err'  ] = inv_fpr_err


print out_dict








exit()







"""


def read_pkl(pth):
    pkls = joblib.load(pth)
    return pkls['data']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

trn_m = 20#30#60#30#60#20#30#60#50#40
trn_l = 1000#5000#2000#5000#500

val = 'err'
#val = 'val'

mass_list    = [20,30,40,50,60]
ctau_list    = [500,1000,2000,5000]

cut_type     = ['hard_cut']#['loose_cut','hard_cut']
inputs       = ['full']#['2best','full']
kin_var      = ['kin0']#['kin0','kin1']

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
			    if in_dict:
				roc_dict = in_dict['roc']
				fpr_bdt  = roc_dict['fpr']
				tpr_bdt  = roc_dict['tpr']
				#e_tpr_l  = roc_dict['e_tpr_l']
				e_fpr_l  = roc_dict['e_fpr_l']
				#e_tpr_h  = roc_dict['e_tpr_h']
				e_fpr_h  = roc_dict['e_fpr_h']
                 
				cut_dict = in_dict['cut_based']
				dicti    = cut_dict[ci]
				sgn_eff  = dicti['tpr']
				fls_eff  = dicti['fpr']
				#tpr_e_l  = dicti['tpr_e_l']
				fpr_e_l  = dicti['fpr_e_l']
				#tpr_e_h  = dicti['tpr_e_h']
				fpr_e_h  = dicti['fpr_e_h']

				if sgn_eff != 0:
				    tmp_tpr, indx = find_nearest(tpr_bdt, sgn_eff)
				    tmp_fpr       = fpr_bdt[indx]

      
                                    count     = 1   
                                    if_done   = False
                                    print 'sgn_eff', sgn_eff
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
                                        print 'tmp_tpr_l', tmp_tpr_l
                                        print 'tmp_tpr_h', tmp_tpr_h
                                        count += 1     

                                    delta_fpr_l   = np.abs( tmp_fpr-(fpr_bdt[idx_l]-e_fpr_l[idx_l]) )
                                    delta_fpr_h   = np.abs( (fpr_bdt[idx_h]+e_fpr_h[idx_h])-tmp_fpr )  
                                    err_fpr       = np.max([delta_fpr_l,delta_fpr_h,e_fpr_h[indx],e_fpr_l[indx]])   
                                    fpr_err       = np.maximum(fpr_e_l,fpr_e_h)

				    if tmp_fpr != 0:
					inv_fpr     = fls_eff/float(tmp_fpr)
                                        inv_fpr_err = np.sqrt(np.square(fpr_err/fls_eff) + np.square(err_fpr/tmp_fpr))

				    else           :
					print '>>>>>>>>>>>>>>>>>>>> Zero devision!'
					inv_fpr     = 0
					inv_fpr_err = 0
				else:    pass # !!!!!!!!!!!!!!!!!!!!!!!!!!!! hard coded cut TPR!!!!!
 
				out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['(1/FPR_BDT)/(1/FPR_C)|cut_TPR']      = inv_fpr 
				out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['(1/FPR_BDT)/(1/FPR_C)|cut_TPR--err'] = inv_fpr_err 

			    else:
				print '>>>>>>>>>>>>>>>>>>>> no in_dict!!'
				out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['(1/FPR_BDT)/(1/FPR_C)|cut_TPR']      = 0 
				out_dict[ci][item][m_trn][l_trn][m_tst][l_tst]['(1/FPR_BDT)/(1/FPR_C)|cut_TPR--err'] = 0 

				empty_log.append(file_to_look)


    path_out = './plot/'
    


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
        imDict[mmi][lli]   = out_dict['hard_cut']['full_kin0'][trn_m][trn_l][mmi][lli]['(1/FPR_BDT)/(1/FPR_C)|cut_TPR'] 
        err_dict[mmi][lli] = out_dict['hard_cut']['full_kin0'][trn_m][trn_l][mmi][lli]['(1/FPR_BDT)/(1/FPR_C)|cut_TPR--err']


df_val = pd.DataFrame(imDict)
df_err = pd.DataFrame(err_dict)

if val == 'val':
    df        = df_val
    val_label = '(1/FPR_BDT)/(1/FPR_cut) at cut TPR'
elif val == 'err':
    df        = df_err
    val_label = '(1/FPR_BDT)/(1/FPR_cut) at cut TPR -- errors'


m_L = df.columns.values.tolist()
c_L = [500,1000,2000,5000]


fig, ax  = plt.subplots()
im, cbar = heatmap(df, c_L, m_L, ax=ax, cmap="YlGn", cbarlabel=val_label)
texts    = annotate_heatmap(im, valfmt="{x:.3f} t", fsize=6)
fig.tight_layout()
plt.show()
 
    

"""

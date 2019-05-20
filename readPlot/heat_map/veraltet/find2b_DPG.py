from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hm import *

#test_self = 1
h_n = 0#2

val = 'val'
#val = 'err'

#path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/DPG/'
path = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/DPG_new/'+'find2best/'


#attr_list = ['J1cHadEFrac','J1nHadEFrac','J1nEmEFrac','J1cEmEFrac','J1cmuEFrac','J1muEFrac','J1eleEFrac','J1eleMulti','J1photonEFrac','J1photonMulti','J1cHadMulti','J1nHadMulti','J1npr','J1cMulti','J1nMulti','J1nSelectedTracks','J1ecalE']
#attr_list = ['J1cHadEFrac','J1nHadEFrac','J1nEmEFrac','J1cEmEFrac','J1cHadMulti','J1nHadMulti','J1npr','J1cMulti','J1nMulti','J1ecalE']
attr_list = ['cHadEFrac','nHadEFrac','nEmEFrac','cHadMulti','nHadMulti','npr','cMulti','ecalE','photonEFrac']#,'nSelectedTracks'] 

not_there = ['cEmEFrac']


score = 'aoc'
#score = 'fpr'

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
for a1 in attr_list:
    a1_s = a1[h_n:]
    out_dict[a1_s] = {}
    for a2 in attr_list:
        a2_s = a2[h_n:] 
        a_c1 = a1+'_'+a2
        a_c2 = a2+'_'+a1
        if   a_c1 in a_c_list:    a_c = a_c1
        elif a_c2 in a_c_list:    a_c = a_c2

        #file_name = 'RS_trn_30GeV_500mm_tst_30GeV_500mm_slct1_attr_'+a_c+'_kin1_v0.pkl'  
        file_name = 'RS_trn_50GeV_500mm_tst_50GeV_5000mm_slct1_attr_'+a_c+'_kin1_v0.pkl'  

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
     
        aoc      = in_dict['aoc']
   
        """ 
        cut_dict = in_dict['cut_based']
        dicti    = cut_dict['hard_cut']
        sgn_eff  = dicti['tpr']
        fls_eff  = dicti['fpr']
        #tpr_e_l  = dicti['tpr_e_l']
        fpr_e_l  = dicti['fpr_e_l']
        #tpr_e_h  = dicti['tpr_e_h']
        fpr_e_h  = dicti['fpr_e_h']
        

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> temp!!!! 
        #sgn_eff      = 0.2          
      
        tmp_tpr, indx = find_nearest(tpr_bdt, sgn_eff)
        tmp_fpr       = fpr_bdt[indx]
        if tmp_fpr == 0.:    nanb = 1       
        else            :    nanb = 0

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
        
        if   nanb == 0:
            inv_fpr     = fls_eff/float(tmp_fpr)
            inv_fpr_err = np.sqrt(np.square(fpr_err/fls_eff) + np.square(err_fpr/tmp_fpr))
        elif nanb == 1:
            inv_fpr     = 0 
            inv_fpr_err = 0
        """


        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> temp!!!!
        """
        if nanb == 0:
            inv_fpr = 1./float(tmp_fpr)
        if nanb == 1: 
            inv_fpr = 0
        """
 

        out_dict[a1_s][a2_s] = {}
        #out_dict[a1_s][a2_s]['inv_fpr'] = inv_fpr
        #out_dict[a1_s][a2_s]['err'  ] = inv_fpr_err

        out_dict[a1_s][a2_s]['aoc'] = aoc 

#print out_dict




imDict   = {}
err_dict = {}
for a1 in attr_list:
    a1_s = a1[h_n:]
    imDict[a1_s]   = {}
    err_dict[a1_s] = {}
    for a2 in attr_list:
        a2_s = a2[h_n:]
        imDict[a1_s][a2_s]   = 0.
        err_dict[a1_s][a2_s] = 0.

LL = []
for a1 in attr_list:
    a1_s = a1[h_n:]
    for a2 in attr_list:
        a2_s = a2[h_n:]

        if score == 'aoc':
            imDict[a1_s][a2_s] = out_dict[a1_s][a2_s]['aoc']
        elif score == 'fpr':
            imDict[a1_s][a2_s]   = out_dict[a1_s][a2_s]['inv_fpr']
        #err_dict[a1_s][a2_s] = out_dict[a1_s][a2_s]['err']

df_val = pd.DataFrame(imDict)
#df_err = pd.DataFrame(err_dict)
print df_val


if val == 'val':
    df        = df_val
    if score == 'aoc': 
        val_label = 'AUC'#'AOC'
    elif score == 'fpr':
        val_label = '(1/FPR_BDT)/(1/FPR_cut) at cut TPR'
elif val == 'err':
    df        = df_err
    if score == 'aoc':
        val_label = 'AOC -- errors'
    elif score == 'fpr':
        val_label = '(1/FPR_BDT)/(1/FPR_cut) at cut TPR -- errors'


attr_L = df.columns.values.tolist()


fig, ax  = plt.subplots()
im, cbar = heatmap(df, attr_L, attr_L, ax=ax, cmap="YlGn", cbarlabel=val_label)
texts    = annotate_heatmap(im, valfmt='{x:.2f}', fsize=8) # valfmt="{x:.3f} t"
fig.tight_layout()
#plt.show()

pth_out = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Brian/DPG_new/find2best/'
out_name = '2best.png'
fig.savefig(pth_out+out_name, bbox_inches='tight')
plt.show()








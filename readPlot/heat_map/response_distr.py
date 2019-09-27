from sklearn.externals import joblib
import pandas as pd

from hm import *


from matplotlib import pyplot as plt


pth     = '/home/hezhiyua/desktop/DeepTop/LLP/Limits/'
#pth_out = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/'+'v6'+'/'+'punzi/'
pth_out = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/allInOne/nn_format/4jets/v6/punzi/'
in_name = 'store_min_response.pkl'

trn_m = '30'
trn_l = '500'

mass_list = [20,30,40,50]
ctau_list = [500,1000,2000,5000]

n_bins  = 50#30
Bin     = np.linspace(0,1,n_bins)
if_norm = False#True
cumul   = 0#-1
log_on  = True

my_ticks = np.arange(0,1,0.02)

in_dic   = joblib.load(pth+in_name)

#print in_dic

models = ['cut_nhf','LoLa']


punzi_dic = {}
for mi in mass_list:
    punzi_dic[mi] = {}  
    for li in ctau_list:
        punzi_dic[mi][li] = {} 
        stri  = str(mi)+'_'+str(li) 
        for key in models:
            tpli = in_dic[stri][key]
            punzi_dic[mi][li][key] = tpli
   

for li in ctau_list:

    for mi in mass_list:
        for mdl in models:
            if mdl == 'cut_nhf':    styl = 'dashed'
            else               :    styl = None 
            label_b_i = str(mi)+'GeV_'+mdl
            r_b = punzi_dic[mi][li][mdl][0]
            w_b = punzi_dic[mi][li][mdl][2]
            #plt.plot(punzi_dic[mi][li][mdl][0], punzi_dic[mi][li][mdl][1], linestyle=styl, label=str(mi)+'GeV_'+mdl)
            plt.hist(x=r_b, bins=Bin, weights=w_b, linestyle=styl, histtype='step', cumulative=cumul, normed=if_norm, log=log_on, label=label_b_i)
            #plt.hist(x=r_s, bins=Bin, weights=w_s, histtype='step', cumulative=cumul , normed=if_norm, log=log_on, label=label_s_i)




    #plt.xticks(my_ticks)
    plt.title('bkg')
    plt.xlabel('response')
    plt.ylabel('a.u.')

    plt.legend(loc='lower left')
    plt.savefig(pth_out+'bkg_stacked_non_cumu_'+str(li)+'mm.png')
    #plt.show()
    plt.close()

















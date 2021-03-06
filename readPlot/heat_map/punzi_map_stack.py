from sklearn.externals import joblib
import pandas as pd

from hm import *


from matplotlib import pyplot as plt


pth     = '/home/hezhiyua/desktop/DeepTop/LLP/Limits/'+'MA/'+'bdt/'
#pth_out = '/beegfs/desy/user/hezhiyua/LLP/bdt_output/result/Lisa/'+'v6'+'/'+'punzi/'
pth_out = '/beegfs/desy/user/hezhiyua/2bBacked/skimmed/LLP/allInOne/nn_format/4jets/v6/punzi/'
in_name = 'store_punzi.pkl'

trn_m = '30'
trn_l = '500'

mass_list = [20,30,40,50]
ctau_list = [500,1000,2000,5000]


#n_digits = '.2f'

in_dic   = joblib.load(pth+in_name)

#print in_dic

models = ['cut_nhf','LoLa']

models = ['cut_nhf','BDT']
models = ['cut_nhf']


plt.title('Punzi.')
plt.xlabel(r'$c\tau$')
plt.ylabel('Punzi.') 
punzi_dic = {}
for mi in mass_list:
    punzi_dic[mi] = {}
  
    for li in ctau_list:
        punzi_dic[mi][li] = {} 
        stri  = str(mi)+'_'+str(li) 
        for key in models:
            #print key
            #print stri
            #print type(in_dic[stri]['plt'])
            tpli = in_dic[stri]['plt'][key]
            punzi_dic[mi][li][key] = tpli
   

for li in ctau_list:

    for mi in mass_list:
        for mdl in models:
            if mdl == 'cut_nhf':
                styl = 'dashed'
                MD = 'cut-based (nHadEFrac)'
            else               :
                styl = None 
                MD = mdl
            plt.plot(punzi_dic[mi][li][mdl][0], punzi_dic[mi][li][mdl][1], linestyle=styl, label=str(mi)+' GeV '+MD)

    plt.yscale('log')
    plt.title(r'Punzi. (c$\tau$ = '+str(li)+' mm)')
    plt.xlabel('classification output')
    plt.ylabel('Punzi.')

    cutL = 0.45
    #cutL = 0.9
    #utL = 0.5
    #plt.axvline(x=cutL,label=r'$cut_{LoLa}$ = '+str(cutL))
    #plt.axvline(x=cutL,label=r'$cut_{BDT}$ = '+str(cutL))
    plt.axvline(x=cutL,label=r'$cut_{nHadEFrac}$ = '+str(cutL))

    plt.legend()
    plt.savefig(pth_out+'punzi_stacked__'+str(li)+'mm.png')
    #plt.show()
    plt.close()

















exit()

imDict = {}
for i in mass_list:
    imDict[i]   = {}
    #err_dict[i] = {}
    for j in ctau_list:
        imDict[i][j]   = 0.
        #err_dict[i][j] = 0.

for mmi in mass_list:
    for lli in ctau_list:
        tmp_str          = str(mmi)+'_'+str(lli) 
        imDict[mmi][lli] = in_dic[tmp_str]['impr']
        print in_dic[tmp_str]['impr']


df_val    = pd.DataFrame(imDict)
df        = df_val
val_label = r'$\frac{ punzi_{BDT} }{ punzi_{cut_nhf} }$'


m_L = df.columns.values.tolist()
c_L = [500,1000,2000,5000]



plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax  = plt.subplots()
im, cbar = heatmap(df, c_L, m_L, ax=ax, cmap="YlGn", cbarlabel=val_label)
texts    = annotate_heatmap(im, valfmt='{x:'+n_digits+'}', fsize=16)#6)
fig.tight_layout()
#plt.show()

outName  = 'punzi_2Dmap_bdt_vs_'+'_trn_'+str(trn_m)+'_'+str(trn_l)#+'_'+input_string+'_'+val
fig.savefig(pth_out + outName + '.png', bbox_inches='tight')








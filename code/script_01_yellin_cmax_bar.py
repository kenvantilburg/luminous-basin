dir_home = '/mnt/home/kvantilburg/'
dir_ceph = dir_home+'ceph/luminous-basin/'
dir_mc = dir_ceph+'yellin_mc_vols/k10/'

name = 'vols_k10'

import sys
from my_units import *
from functions_model import * 
from functions_load import *
from functions_yellin import *

i_mu = int(sys.argv[1]) # mu to compute

df = pd.read_csv(dir_mc+name+'_imu_'+f'{i_mu:03d}'+'.csv',
                 names=['mu']+list(range(0,int(2e2),1)),skiprows=[0],index_col=False) #set no. of columns large


#df = df.replace(1.,np.NaN) #set all volumes=1.0 equal to nan
#df = df.dropna(axis=1,how='all') #drop columns with all nan

df = df.replace(np.nan,1.0) #set all nan volumes to 1
nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
df = df.drop(cols_to_drop[1:], axis=1)
mu = df.iloc[0]['mu']
vols_mc = df.to_numpy()[:,1:] #get volumes
arr_C_n = []
for n in range(vols_mc.shape[1]):
    res = stats.cumfreq(vols_mc[:,n],numbins=1001,defaultreallimits=(0,1))
    x = res.lowerlimit + np.linspace(0,res.binsize*res.cumcount.size,res.cumcount.size)
    res_int = interp1d(x,res.cumcount/vols_mc.shape[0])
    arr_C_n.append(res_int)
list_C_max = np.zeros(vols_mc.shape[0])
for i in range(vols_mc.shape[0]):
    for n,vol in enumerate(vols_mc[i]):
        if 0.999 > vol > 0: #if not nan
            C_n = arr_C_n[n](vol)
            list_C_max[i] = np.max([C_n,list_C_max[i]])
list_C_max[list_C_max==0] = 1.0
C_max_bar = np.quantile(list_C_max,0.9)

df_out = pd.DataFrame(data=[[mu,C_max_bar,vols_mc.shape[0]]],columns=['mu','C_max_bar','N_MC'])

with open(dir_mc+name+'_cmax_bar.csv','a') as f: #open data file, in 'append mode'
    df_out.to_csv(f, header=f.tell()==0, index=False)
    
print('mu = '+str(mu)[0:8]+', 1-C_max_bar = '+str(1-C_max_bar))
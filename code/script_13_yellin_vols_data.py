dir_home = '/mnt/home/kvantilburg/'
dir_ceph = dir_home+'ceph/luminous-basin/'
dir_source = dir_ceph+'data/products_80610202001_orbit1_chu12_spatialARF/'
dir_events = dir_source+'event_lists/'
dir_production = dir_ceph+'data/production/'

import sys
#sys.path.insert(0, dir_notebook)

from my_units import *
from functions_model import * 
from functions_load import *
from functions_yellin import *

i_p = int(sys.argv[1]) # solar position integer
i_m = int(sys.argv[2]) # axion mass integer
k = int(sys.argv[3]) # number of bins for yellin binning
print('i_p =',str(i_p),'| i_m =',str(i_m))

dir_res = dir_ceph+'results/data/'
dir_proj = dir_res+'proj/proj_'+str(i_p)+'/'
dir_vols = dir_res+'vols/vols_'+str(i_p)+'/'

file_proj = dir_proj+'proj_'+str(i_p)+'_'+str(i_m)+'.csv'

df_proj = pd.read_csv(file_proj)
N_proj = len(df_proj)
i_down_max = np.max([0,int(np.ceil(np.log(N_proj/100)/np.log(2)))])

for i_down in range(0,1+i_down_max):
    file_vol = dir_vols+'vols_k'+str(k)+'_'+str(i_p)+'_'+str(i_m)+'_d_'+f'{i_down:02d}'+'.csv'
    
    df_proj_sample = df_proj.sample(frac=2**(-i_down))

    A, edges = np.histogramdd(sample=df_proj_sample.to_numpy(),
                              bins=4*[np.linspace(0,1,k+1)]);

    vols = maximal_cuboid_volumes(A)

    df_vols = pd.DataFrame([vols],columns=['V_'+str(i) for i in range(len(vols))])

    with open(file_vol,'w') as f: #open data file, in 'write mode'
        df_vols.to_csv(f, index=False)
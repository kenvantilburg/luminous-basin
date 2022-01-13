############## preamble #################################
dir_home = '/mnt/home/kvantilburg/'
dir_notebook = dir_home+'luminous-basin/code/'
dir_ceph = dir_home+'ceph/luminous-basin/yellin_vols/k10_1/'

import sys
sys.path.insert(0, dir_notebook)

from functions_load import *
from functions_model import *
from function_yellin import *

############## monte carlo #################################
list_mu = np.logspace(0.2,2.11,201)

i_mu = np.int(sys.argv[1])
mu = list_mu[i_mu]
print('mu =',mu)
N_MC = 3*10**4
N_chunk = 1
k = 10

for i_MC in range(N_MC//N_chunk):
    vols = monte_carlo_volumes(N=N_chunk,mu=mu,k1=k,k2=k,k3=k,k4=k)
    df = pd.DataFrame(vols)
    df.insert(0, 'mu', N_chunk*[mu])
    with open(dir_ceph+'vols_k'+f'{k:02d}'+'_imu_'+f'{i_mu:03d}'+'.csv','a') as f: #open data file, in 'append mode'
        df.to_csv(f, header=f.tell()==0, index=False)




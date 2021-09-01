dir_home = '/mnt/home/kvantilburg/'
dir_ceph = dir_home+'ceph/LuminousBasin/'
dir_source = dir_ceph+'data/products_80610202001_orbit1_chu12_spatialARF/'
dir_events = dir_source+'event_lists/'
dir_production = dir_ceph+'data/production/'


import sys
#sys.path.insert(0, dir_notebook)

from my_units import *
from model_functions import * 
from load_functions import *
from yellin_functions import *

i_m = np.int(sys.argv[1]) # axion mass integer

dir_proj = dir_ceph+'yellin_projections/data/'


file_rho0 = dir_proj+'rho0_'+str(i_m)+'_fid.csv'
file_proj = dir_proj+'proj_'+str(i_m)+'_fid.csv'

# time intervals [seconds]
good_time_ints = [
(  3.37603080e+08,   3.37603330e+08),
(  3.37603330e+08,   3.37603380e+08),
(  3.37603380e+08,   3.37603380e+08),
(  3.37603580e+08,   3.37603581e+08),
(  3.37603630e+08,   3.37605080e+08),
(  3.37605081e+08,   3.37605230e+08)]
exposure = np.sum([interval[1]-interval[0] for interval in good_time_ints])
livetime = np.asarray([1501.16599845754, 1481.86081041239])/exposure # effective fractional livetime of A and B
duration = np.max(good_time_ints)- np.min(good_time_ints)

# energy bins [units = keV]
sigma_E = 0.166 # energy resolution [keV]
n_sigma_E = 3
width_E = 0.04 # energy bin width is 40 keV
bins_E = np.arange(1.6,200.01,width_E)
list_m = np.arange(3,40,0.1)
m = list_m[i_m]  # axion mass [keV]

# fiducial solar position 
ra_sun_fid = 170.66855149 * degree
dec_sun_fid = 4.02092024 * degree
# error on solar position 
sigma_sun = 2 * arcmin
# shift in solar position over duration starting from t_min
delta_ra_sun = 0.01962028 * degree
delta_dec_sun = -0.00835105 * degree
t_min = 3.37603341e+08
sigma_sun = 2*arcmin # error on solar position

# initial solar position
ra_sun_0 = ra_sun_fid
dec_sun_0 = dec_sun_fid

list_file_events = np.sort([dir_events+file for file in listdir(dir_events)])
file_box_centers = dir_source+'box_centers.txt'
list_file_arf = [dir_source+'arfs/'+file for file in listdir(dir_source+'arfs/')]

##### load data #####
print('m = '+str(m)[0:8]+': initialized, loading data...')

df_box = load_box(file_box_centers)
rotation = df_box['rotation'].iloc[0] * degree
df_arf = load_arf_m(list_file_arf,bins_E,df_box,m,sigma_E,n_sigma_E)
df_events_m = load_events_m(list_file_events,m,sigma_E,n_sigma_E)

t = df_events_m['t'].to_numpy()
E = df_events_m['E'].to_numpy()
ra = df_events_m['ra'].to_numpy()
dec = df_events_m['dec'].to_numpy()
x,y = map_x_y_from_ra_dec(ra,dec,rotation)

print('m = '+str(m)[0:8]+': data files loaded, interpolating arf...')

##### yellin/poisson projection #####
int_arf, x_min, x_max, y_min, y_max = load_int_arf(m,df_arf,rotation,bins_E,width_E) # arf interpolation function + boundaries

print('m = '+str(m)[0:8]+': arf interpolated, projecting onto unit cuboid...')

r_1, r_2, r_3, r_4, rho_0 = proj_unit_cuboid(t,E,x,y,m,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration, good_time_ints,livetime,bins_E,sigma_E,n_sigma_E,x_min,x_max,y_min,y_max,rotation,df_arf,int_arf,N_x=26,N_y=26) # project onto unit cuboid

print('m = '+str(m)[0:8]+': projection complete, writing out data...')

df_rho0 = pd.DataFrame(data=[[m,ra_sun_0,dec_sun_0,rho_0]],columns=['m','ra_sun_0','dec_sun_0','rho_0'])
df_proj = pd.DataFrame(data=np.transpose([r_1,r_2,r_3,r_4]),columns=['r_1','r_2','r_3','r_4'])

with open(file_rho0,'w') as f: #open data file, in 'write mode'
    df_rho0.to_csv(f, index=False)
with open(file_proj,'w') as f: #open data file, in 'write mode'
    df_proj.to_csv(f, index=False)
    
print('m = '+str(m)[0:8]+': done.')
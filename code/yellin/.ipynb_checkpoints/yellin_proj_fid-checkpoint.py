dir_home = '/Users/kvantilburg/Dropbox/projects/LuminousBasin/LuminousBasin/luminous-basin/'
dir_events = dir_home+'mocks/event_lists/'
dir_source = dir_home+'data/likelihood_yellin_data/products_80610202001_orbit1_chu12_spatialARF/'
dir_production = dir_home+'data/production/'

from my_units import *
from model_functions import * 
from load_functions import *
from yellin_functions import *

i_mock = '6' # mock file number
m = 8 # axion mass [keV]

# time intervals [seconds]
good_time_ints = np.asarray([
    [3.37603341e+08, 3.376033795e+08],
    [3.376036305e+08, 3.3760522972e+08]
])
exposure = np.sum([interval[1]-interval[0] for interval in good_time_ints])
livetime = np.asarray([1501.16599845754, 1481.86081041239])/exposure # effective fractional livetime of A and B
duration = np.max(good_time_ints)- np.min(good_time_ints)

# energy bins [units = keV]
sigma_E = 0.166 # energy resolution [keV]
n_sigma_E = 3
width_E = 0.04 # energy bin width is 40 keV
bins_E = np.arange(1.6,200.01,width_E)

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

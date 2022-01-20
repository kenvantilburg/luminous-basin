import sys
from my_units import *
from functions_model import * 
from functions_load import *
from functions_yellin import *
from functions_likelihood import *

i_mock = int(sys.argv[1]) # mock file number
i_m = int(sys.argv[2]) # axion mass integer

list_m = np.arange(3,40,0.1)
m = list_m[i_m]  # axion mass [keV]

dir_home = '/mnt/home/kvantilburg/'
dir_ceph = dir_home+'ceph/luminous-basin/'
dir_events = dir_ceph+'mocks/event_lists/'
dir_source = dir_ceph+'data/products_80610202001_orbit1_chu12_spatialARF/'
dir_production = dir_ceph+'data/production/'
dir_background = dir_home+'/luminous-basin/data/backgrounds/'

dir_res = dir_ceph+'results/mocks/mock_'+str(i_mock)+'/'
dir_mcmc = dir_res+'mcmc/'
file_mcmc = dir_mcmc+'samples_'+str(i_mock)+'_'+str(i_m)+'.npy'

dir_proj = dir_ceph+'results/mocks/mock_'+str(i_mock)+'/proj/'
file_rho0 = dir_proj+'rho0_'+str(i_mock)+'_'+str(i_m)+'_fid.csv'
file_proj = dir_proj+'proj_'+str(i_mock)+'_'+str(i_m)+'_fid.csv'

file_LL_lim = dir_res+'results_LL_'+str(i_mock)+'.csv'

# time intervals [seconds]
good_time_ints = np.asarray([
    [3.37603341e+08, 3.376033795e+08],
    [3.376036305e+08, 3.3760522972e+08]
])
exposure = np.sum([interval[1]-interval[0] for interval in good_time_ints])
livetime = np.asarray([1501.16599845754, 1481.86081041239])/exposure # effective fractional livetime of A and B
duration = np.max(good_time_ints)- np.min(good_time_ints)
# time bins
N_bins_t = 10
bins_t = np.linspace(good_time_ints[0][0], good_time_ints[-1][1], N_bins_t+1)

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

list_file_events = np.sort([dir_events+file for file in listdir(dir_events) if file[0:len(str(i_mock))+1]==str(i_mock)+'_'])
file_box_centers = dir_source+'box_centers.txt'
list_file_arf = [dir_source+'arfs/'+file for file in listdir(dir_source+'arfs/')]

##### load data #####
print('m = '+str(m)[0:8]+': initialized, loading data...')

df_data = load_data(m,sigma_E,good_time_ints,livetime,bins_t,bins_E,n_sigma_E,list_file_events,file_box_centers,list_file_arf)
t = df_data['t'].to_numpy()
E = df_data['E'].to_numpy()
ra = df_data['ra'].to_numpy()
dec = df_data['dec'].to_numpy()
exp = df_data['exp'].to_numpy()
arf = df_data['arf'].to_numpy()
eps = exp * arf * (df_data['Omega'].to_numpy() * arcmin**2 / (degree**2)) * width_E
counts = df_data['counts'].to_numpy()

df_data_not_m = load_data_not_m(m,sigma_E,good_time_ints,livetime,bins_t,bins_E,n_sigma_E,list_file_events,file_box_centers,list_file_arf)
E_not_m = df_data_not_m['E'].to_numpy()
exp_not_m = df_data_not_m['exp'].to_numpy()
counts_not_m = df_data_not_m['counts'].to_numpy()

df_tmp1 = pd.read_csv(file_rho0)
N_data = len(pd.read_csv(file_proj))
N_sig_lim_poisson = poisson_limit(N_data)
S_0_lim_poisson = N_sig_lim_poisson/df_tmp1['rho_0'][0]
S0_guess = S_0_lim_poisson
print('S0_guess = '+str(S0_guess)+' sec^{-1} degree^{-2} cm^{-2}')

df_background = pd.read_csv(dir_background+'backgrounds.txt',delimiter=" ",skiprows=3,header=None)
df_background.columns = ["Energy [keV]","unknown","Total", "aCXB", "Internal","fXCB","Continuum"]

E_bg = df_background['Energy [keV]'].to_numpy()
rate_total_bg = df_background['Total'].to_numpy()
rate_aCXB_bg = df_background['aCXB'].to_numpy()
rate_internal_bg = df_background['Internal'].to_numpy()
rate_fXCB_bg = df_background['fXCB'].to_numpy()
rate_continuum_bg = df_background['Continuum'].to_numpy()

int_rate_total_bg = interp1d(E_bg,rate_total_bg, fill_value='extrapolate')
int_rate_aCXB_bg = interp1d(E_bg, rate_aCXB_bg, fill_value='extrapolate')
int_rate_internal_bg = interp1d(E_bg, rate_internal_bg, fill_value='extrapolate')
int_rate_continuum_bg = interp1d(E_bg, rate_continuum_bg, fill_value='extrapolate')

##### MCMC #####
# Background guesses from global fit:
B1_guess = 30.73054844
B2_guess = 26.84442745
B3_guess = 20.16336952
B1_sigma = 2.2184273926203124
B2_sigma = 2.6849045247214165
B3_sigma = 2.947270654559794

p0_init = np.asarray([[B1_guess, B2_guess, B3_guess, S0_guess, ra_sun_fid, dec_sun_fid]])
nwalkers = 32
ndim = 6 # B1,B2,B3,S0,alpha0,delta0
p0_B1 = np.abs(np.random.normal(B1_guess,B1_sigma,nwalkers))
p0_B2 = np.abs(np.random.normal(B2_guess,B2_sigma,nwalkers))
p0_B3 = np.abs(np.random.normal(B3_guess,B3_sigma,nwalkers))
p0_S0 = np.random.normal(S0_guess,0.1*S0_guess,nwalkers)
p0_ra_sun_0 = np.random.normal(ra_sun_fid,sigma_sun,nwalkers)
p0_dec_sun_0 = np.random.normal(dec_sun_fid,sigma_sun,nwalkers)
p0 = np.transpose([p0_B1,p0_B2,p0_B3,p0_S0,p0_ra_sun_0,p0_dec_sun_0])

# Set up the backend
# Don't forget to clear it in case the file already exists
#backend = emcee.backends.HDFBackend(file_mcmc)
#backend.reset(nwalkers, ndim)

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, LL, #backend=backend,
                                   args=[m,t,E,ra,dec,exp,eps,counts,exposure,width_E,sigma_E,
                                      t_min,delta_ra_sun,delta_dec_sun,sigma_sun,ra_sun_fid,dec_sun_fid,duration,
                                      E_not_m,exp_not_m,counts_not_m,
                                      int_rate_aCXB_bg,int_rate_internal_bg,int_rate_continuum_bg])

max_n = int(4e2) # maximum length of MCMC chains
old_tau = np.inf # used to test convergence
# Now we'll sample for up to max_n steps
for sample in sampler.sample(p0, iterations=max_n, progress=False):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
# save raw chains
samples = sampler.get_chain()
np.save(file_mcmc,samples)

# get output
tau = sampler.get_autocorr_time(tol=0)
burnin = int(4 * np.max(tau))
thin = int(0.25 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))

##### limit #####
# get CDF of S0
cdf_samples = stats.cumfreq(samples[:,3],numbins=10**6)
S0_cdf_array = cdf_samples.lowerlimit + np.linspace(0, cdf_samples.binsize*cdf_samples.cumcount.size,cdf_samples.cumcount.size)
cdf_S0_interpolation = interp1d(S0_cdf_array,cdf_samples.cumcount/samples_units.shape[0],bounds_error=False,fill_value=(1e-20,1))

# functions to set 1-alpha (default = 90% CL) limit
def fun_intersect_unconstrained(S0,alpha=0.1):
    return (1-cdf_S0_interpolation(S0))-alpha
def fun_intersect_CLs(S0,alpha=0.1):
    return (1-cdf_S0_interpolation(S0))/(1-cdf_S0_interpolation(0))-alpha

alpha_median = 0.5
S0_fit = optimize.brentq(fun_intersect_unconstrained,S0_cdf_array[0],S0_cdf_array[-1],args=(alpha_median)) # median (best-fit)
S0_lim_unconstrained = optimize.brentq(fun_intersect_unconstrained,S0_cdf_array[0],S0_cdf_array[-1])
S0_lim_CLs = optimize.brentq(fun_intersect_unconstrained,S0_cdf_array[0],S0_cdf_array[-1])

##### write out limit and best fit #####
df = pd.DataFrame(data = [[m,S0_fit,S0_lim_unconstrained,S0_lim_CLs]], 
                  columns=['m','S_0_fit','S_0_lim_unconstrained','S_0_lim_CLs'])
with open(file_LL_lim,'a') as f: #open data file, in 'append mode'
    df.to_csv(f, header=f.tell()==0, index=False)


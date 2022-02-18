import sys
from my_units import *
from functions_model import * 
from functions_load import *
from functions_yellin import *
from functions_likelihood import *

i_p = 160 # this projection number corresponds to fiducial solar position
i_m = int(sys.argv[1]) # axion mass integer

list_m = np.arange(3,40,0.1)
m = list_m[i_m]  # axion mass [keV]

dir_home = '/mnt/home/kvantilburg/'
dir_ceph = dir_home+'ceph/luminous-basin/'
dir_source = dir_ceph+'data/products_80610202001_orbit1_chu12_spatialARF/'
dir_events = dir_source+'event_lists/'
dir_production = dir_ceph+'data/production/'
dir_background = dir_home+'/luminous-basin/data/backgrounds/'
dir_res = dir_ceph+'results/data/'
dir_mcmc = dir_res+'mcmc/'

file_mcmc = dir_mcmc+'samples_'+str(i_m)+'.npy'
file_LL_lim = dir_res+'results_LL.csv'

dir_proj = dir_res+'proj/proj_'+str(i_p)+'/'

file_rho0 = dir_proj+'rho0_'+str(i_p)+'_'+str(i_m)+'.csv'
file_proj = dir_proj+'proj_'+str(i_p)+'_'+str(i_m)+'.csv'

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

# time bins
N_bins_t = 1
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

list_file_events = np.sort([dir_events+file for file in listdir(dir_events)])
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
B1_guess = 38.531322099323305
B2_guess = 19.769783925331748
B3_guess = 8.752452853548448
B1_sigma = 1.27120913
B2_sigma = 1.10136636
B3_sigma = 3.88785303

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

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, LL, #backend=backend,
                                   args=[m,t,E,ra,dec,exp,eps,counts,exposure,width_E,sigma_E,
                                      t_min,delta_ra_sun,delta_dec_sun,sigma_sun,ra_sun_fid,dec_sun_fid,duration,
                                      E_not_m,exp_not_m,counts_not_m,
                                      int_rate_aCXB_bg,int_rate_internal_bg,int_rate_continuum_bg])

max_n = int(1e5)
# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(p0, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
    
tau = sampler.get_autocorr_time(tol=0)
burnin = int(4 * np.max(tau))
thin = int(0.25 * np.min(tau))
samples_f = sampler.get_chain(discard=burnin, flat=True, thin=thin)
samples_w = sampler.get_chain(discard=burnin, flat=False, thin=thin)
print("mean tau: "+str(np.mean(tau))[0:6])
print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("chain shape: {0}".format(samples_w.shape))

p_out_f = np.asarray([np.percentile(samples_f[:,i],[14,50,86]) for i in range(np.shape(samples_f)[1])])
p_out_w = np.asarray([[np.percentile(samples_w[:,i,j],[14,50,86]) for j in range(np.shape(samples_w)[2])] for i in range(np.shape(samples_w)[1])])
flag_keep_w = np.asarray(nwalkers * [True])
for i_w in range(nwalkers):
    for i_d in range(ndim):
        if (p_out_w[i_w,i_d,1] < p_out_f[i_d,0]) or (p_out_w[i_w,i_d,1] > p_out_f[i_d,2]):
            flag_keep_w[i_w] = False
            print('stray walker: '+'i_w = '+str(i_w)+', i_d = '+str(i_d))
print('shape(samples_w) = '+str(samples_w.shape))
print('shape(samples_f) = '+str(samples_f.shape))
samples = samples_w[:,flag_keep_w,:]
samples = samples.reshape((samples.shape[0]*samples.shape[1], samples.shape[2]))
print('shape(samples) = '+str(samples.shape))
# save chains
np.save(file_mcmc,samples)

p_out = np.asarray([np.percentile(samples[:,i],[1,16,50,84,99]) for i in range(np.shape(samples)[1])])
[B1_fit, B2_fit, B3_fit, S0_fit, ra_sun_fit, dec_sun_fit] = p_out[:,2]
print('[B1_guess, B2_guess, B3_guess, S0_guess, ra_sun_fid, dec_sun_fid] =',
      str([B1_guess, B2_guess, B3_guess, S0_guess, ra_sun_fid, dec_sun_fid]))
print('[B1_fit, B2_fit, B3_fit, S0_fit, ra_sun_fit, dec_sun_fit] =',
      str([B1_fit, B2_fit, B3_fit, S0_fit, ra_sun_fit, dec_sun_fit]))
print('sigmas =',str((p_out[:,3]-p_out[:,1])/2))

##### limit #####
# get CDF of S0
cdf_samples = stats.cumfreq(samples[:,3],numbins=10**6)
S0_cdf_array = cdf_samples.lowerlimit + np.linspace(0, cdf_samples.binsize*cdf_samples.cumcount.size,cdf_samples.cumcount.size)
cdf_S0_interpolation = interp1d(S0_cdf_array,cdf_samples.cumcount/samples.shape[0],bounds_error=False,fill_value=(1e-20,1))

# functions to set 1-alpha (default = 90% CL) limit
def fun_intersect_unconstrained(S0,alpha=0.1):
    return (1-cdf_S0_interpolation(S0))-alpha
def fun_intersect_CLs(S0,alpha=0.1):
    return (1-cdf_S0_interpolation(S0))/(1-cdf_S0_interpolation(0))-alpha

alpha_median = 0.5
S0_fit = optimize.brentq(fun_intersect_unconstrained,S0_cdf_array[0],S0_cdf_array[-1],args=(alpha_median)) # median (best-fit)
S0_lim_unconstrained = optimize.brentq(fun_intersect_unconstrained,S0_cdf_array[0],S0_cdf_array[-1])
S0_lim_CLs = optimize.brentq(fun_intersect_CLs,S0_cdf_array[0],S0_cdf_array[-1])

##### write out limit and best fit #####
df = pd.DataFrame(data = [[m,S0_fit,S0_lim_unconstrained,S0_lim_CLs]], 
                  columns=['m','S_0_fit','S_0_lim_unconstrained','S_0_lim_CLs'])
with open(file_LL_lim,'a') as f: #open data file, in 'append mode'
    df.to_csv(f, header=f.tell()==0, index=False)
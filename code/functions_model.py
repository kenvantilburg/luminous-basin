import numpy as np
import pandas as pd
from my_units import *
from scipy import optimize
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

# dir_home = '/mnt/home/kvantilburg/'
# dir_production = dir_home+'ceph/luminous-basin/data/production/'

########### units #####################
degree = np.pi/180 # degree in units of radians
tau_SS = 4.6e9 * Year

########## model functions ##########

def z_min(theta):
    """Returns z_min (returns zero if theta > theta_sun) used in the spatial signal template function"""
    theta_sun = np.arcsin(0.004636733) # RSolar/AU
    theta = np.asarray(theta)
    vec_theta_sun = np.ones(theta.shape) * theta_sun
    vec_arg = np.minimum(theta,vec_theta_sun)
    return(np.sqrt(np.sin(theta_sun)**2 - np.sin(vec_arg)**2))

def T_flux_template(t,ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration):
    """Calculate spatial template function for signal."""
    theta_sun = np.arcsin(0.004636733) # RSolar/AU
    ra_sun = ra_sun_0 + delta_ra_sun * (t-t_min)/(duration)
    dec_sun = dec_sun_0 + delta_dec_sun *(t-t_min)/(duration)
    theta = np.sqrt((dec*degree-dec_sun)**2 + np.cos((dec*degree+dec_sun)/2)**2 * (ra*degree-ra_sun)**2) # small-angle approx    
    theta = np.asarray(theta+1e-20); #(shift up to avoid dividing by zero)
    T = np.zeros(theta.shape)
    T += (theta > theta_sun) * 3 * np.pi / 2 * np.sin(theta_sun)**3 * np.sin(theta)**-3
    T += (theta <= theta_sun) * 3 / 4 * np.sin(theta_sun)**3 * np.sin(theta)**-2 * (4 * z_min(theta) / (-1 - 2*z_min(theta)**2 + np.cos(2*theta)) + (np.pi - 2 * np.arctan(z_min(theta) * np.sin(theta)**-1)) * np.sin(theta)**-1)
    return T 

def solar_disk_mask(t,ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration,fudge=1.0):
    """Inverse of a solar disk mask. Heaviside theta at solar limb with 1 inside, 0 outside."""
    theta_sun = np.arcsin(0.004636733) # RSolar/AU
    ra_sun = ra_sun_0 + delta_ra_sun * (t-t_min)/(duration)
    dec_sun = dec_sun_0 + delta_dec_sun *(t-t_min)/(duration)
    theta = np.sqrt((dec*degree-dec_sun)**2 + np.cos((dec*degree+dec_sun)/2)**2 * (ra*degree-ra_sun)**2) # small-angle approx    
    theta = np.asarray(theta+1e-20); #(shift up to avoid dividing by zero)
    T = np.zeros(theta.shape)
    T += (theta <= fudge*theta_sun) 
    return T 

def Gamma_rad(m,gagg,gaee):
    """Decay rate in units of second^-1. Units: Gamma_rad, m [keV], gagg [GeV^-1]."""
    return (gagg)**2 * m**3 / (64 * np.pi) + AlphaEM**2 * gaee**2 / (9216 * np.pi**3) * m**7 / MElectron**6

tab_e = np.asarray(pd.read_csv('../data/production/tabe.csv'))
tab_gamma = np.asarray(pd.read_csv('../data/production/tabgamma.csv'))

int_rho_dot_over_m_e = interp1d(np.log10(tab_e[:,0]),np.log10(tab_e[:,1]/tab_e[:,0] * (GeV/keV) * (RSolar/AU)**4),
                                bounds_error=False,fill_value=(1e-100,1e-100))
int_rho_dot_over_m_gamma = interp1d(np.log10(tab_gamma[:,0]),np.log10(tab_gamma[:,1]/tab_gamma[:,0] * (GeV/keV) * (RSolar/AU)**4),
                                bounds_error=False,fill_value=(1e-100,1e-100))

def fn_rho_dot_over_m(m, gaee, gagg):
    """Returns number density injection rate at R = 1 AU.
    Units: m [keV], gagg [GeV^-1]"""
    return (CentiMeter**-3 * Year**-1) * (gaee**2 * 10**int_rho_dot_over_m_e(np.log10(m/keV)) + (gagg * GeV)**2 * 10**int_rho_dot_over_m_gamma(np.log10(m/keV)))

def S0_signal(m,gagg,gaee):
    """Signal count rate per unit area and per solid angle on the celestial sphere, in the center of the Sun (theta = 0). 
    Units: m [keV], gagg [GeV^-1]"""
    factor_prod = fn_rho_dot_over_m(m,gaee,gagg) / (6*np.pi) * AU**4 * RSolar**-3
    factor_decay = (1 - np.exp(-Gamma_rad(m,gagg,gaee) * tau_SS))
    return factor_prod * factor_decay

def N_signal(m,gagg,gaee,rho_0):
    """Total expected signal counts"""
    return S0_signal(m,gaee,gagg) * rho_0 * Second *  CentiMeter**2 * degree**2
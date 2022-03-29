import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.integrate import quad, nquad
from scipy import optimize
from scipy import stats
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

import emcee

import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib import font_manager
from matplotlib import rcParams
from matplotlib import rc

from time import time as tictoc

from functions_model import *

################################################

def LL_prior_sun(ra_sun_0,dec_sun_0,sigma_sun,ra_sun_fid,dec_sun_fid):
    return - (np.cos(dec_sun_fid)**2 * (ra_sun_0 - ra_sun_fid)**2 + (dec_sun_0 - dec_sun_fid)**2) / (2 * sigma_sun**2)

def LL_prior_B(B1,B2,B3,B4,gamma4):
    LL_prior_B1 = np.piecewise(B1,[B1<=0,B1>0],[-np.inf,0])
    LL_prior_B2 = np.piecewise(B2,[B2<=0,B2>0],[-np.inf,0])
    LL_prior_B3 = np.piecewise(B3,[B3<=0,B3>0],[-np.inf,0])
    LL_prior_B4 = np.piecewise(B3,[B3<=0,B3>0],[-np.inf,0])
    LL_prior_gamma4 = np.piecewise(gamma4,[gamma4<=0,gamma4>0],[0,-np.inf])
    return LL_prior_B1 + LL_prior_B2 + LL_prior_B3 + LL_prior_B4 + LL_prior_gamma4

def LL_m(model_inputs,m,t,E,ra,dec,exp,eps,counts,exposure,width_E,sigma_E,
         t_min,delta_ra_sun,delta_dec_sun,duration,
         int_rate_aCXB_bg,int_rate_internal_bg,int_rate_continuum_bg):
    B1 = model_inputs[0]
    B2 = model_inputs[1]
    B3 = model_inputs[2]
    B4 = model_inputs[3]
    gamma4 = model_inputs[4]
    S0 = model_inputs[5]
    ra_sun_0 = model_inputs[6]
    dec_sun_0 = model_inputs[7]
    
    rate_bg = B1*int_rate_aCXB_bg(E) + B2*int_rate_internal_bg(E) + B3*int_rate_continuum_bg(E)\
        + (B4*np.power(E,gamma4) * solar_disk_mask(t,ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration))
    counts_bg = rate_bg * exp * width_E / (13**2)# exp * width_E / (13**2) ~ 0.3
    
    if m > 0:
        counts_sig = S0 * eps * T_flux_template(t,ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration)*np.exp(-(E-m/2)**2/(2*sigma_E**2)) / np.sqrt(2 * np.pi * sigma_E**2)
    elif m < 0:
        m = -m
        counts_sig = 0
    else:
        print("Error: m is not a non-zero float.")
        
    mu = counts_bg + counts_sig
    
    if np.min(mu) <= 0:
        return -np.inf
    else:
        return np.sum(counts*np.log(mu) - mu)
    
def LL_not_m(model_inputs,m,t,E,ra,dec,exp,eps,counts,exposure,width_E,sigma_E,
         t_min,delta_ra_sun,delta_dec_sun,duration,
         int_rate_aCXB_bg,int_rate_internal_bg,int_rate_continuum_bg):
    B1 = model_inputs[0]
    B2 = model_inputs[1]
    B3 = model_inputs[2]
    B4 = model_inputs[3]
    gamma4 = model_inputs[4]
    S0 = model_inputs[5]
    ra_sun_0 = model_inputs[6]
    dec_sun_0 = model_inputs[7]
    
    rate_bg = B1*int_rate_aCXB_bg(E) + B2*int_rate_internal_bg(E) + B3*int_rate_continuum_bg(E)\
        + (B4*np.power(E,gamma4) * solar_disk_mask(t,ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration))
    counts_bg = rate_bg * exp * width_E / (13**2)# exp * width_E / (13**2) ~ 0.3
    counts_sig = S0 * eps * T_flux_template(t,ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration)*np.exp(-(E-m/2)**2/(2*sigma_E**2)) / np.sqrt(2 * np.pi * sigma_E**2)
    mu = counts_bg + counts_sig
    
    if np.min(mu) <= 0:
        return -np.inf
    else:
        return np.sum(counts*np.log(mu) - mu)
    
# def LL_not_m(model_inputs,E_not_m,exp_not_m,counts_not_m,exposure,width_E,
#              int_rate_aCXB_bg,int_rate_internal_bg,int_rate_continuum_bg):
#     B1 = model_inputs[0]
#     B2 = model_inputs[1]
#     B3 = model_inputs[2]
#     B4 = model_inputs[3]
#     gamma4 = model_inputs[4]
#     S0 = model_inputs[5]
#     ra_sun_0 = model_inputs[6]
#     dec_sun_0 = model_inputs[7]
#     # fiducial solar position 
#     # ra_sun_0 = 170.66855149 * degree # fixed at fiducial value
#     # dec_sun_0 = 4.02092024 * degree # fixed at fiducial value
#     t_not_m = 3.37603580e+08 # fixed to some intermediate value for initial testing
    
#     rate_bg = B1*int_rate_aCXB_bg(E_not_m) + B2*int_rate_internal_bg(E_not_m) + B3*int_rate_continuum_bg(E_not_m)\
#         + (B4*np.power(E_not_m,gamma4) * solar_disk_mask(t_not_m,ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration))
#     counts_bg = rate_bg * exp_not_m * width_E
#     counts_sig = 0
#     mu = counts_bg + counts_sig
    
#     if np.min(mu) <= 0:
#         return -np.inf
#     else:
#         return np.sum(counts_not_m*np.log(mu) - mu)
    
def LL(model_inputs,m,
       t,E,ra,dec,exp,eps,counts,exposure,width_E,sigma_E,
       t_min,delta_ra_sun,delta_dec_sun,sigma_sun,ra_sun_fid,dec_sun_fid,duration,
       t_not_m,E_not_m,ra_not_m,dec_not_m,exp_not_m,eps_not_m,counts_not_m,
       int_rate_aCXB_bg,int_rate_internal_bg,int_rate_continuum_bg):
    B1 = model_inputs[0]
    B2 = model_inputs[1]
    B3 = model_inputs[2]
    B4 = model_inputs[3]
    gamma4 = model_inputs[4]
    S0 = model_inputs[5]
    ra_sun_0 = model_inputs[6]
    dec_sun_0 = model_inputs[7]

    LL_0 = LL_prior_sun(ra_sun_0,dec_sun_0,sigma_sun,ra_sun_fid,dec_sun_fid)
    LL_1 = LL_prior_B(B1,B2,B3,B4,gamma4)
    LL_2 = LL_m(model_inputs,m,t,E,ra,dec,exp,eps,counts,exposure,width_E,sigma_E,
                t_min,delta_ra_sun,delta_dec_sun,duration,
                int_rate_aCXB_bg,int_rate_internal_bg,int_rate_continuum_bg)
    LL_3 = LL_m(model_inputs,-m,t_not_m,E_not_m,ra_not_m,dec_not_m,exp_not_m,eps_not_m,counts_not_m,exposure,width_E,sigma_E,
                t_min,delta_ra_sun,delta_dec_sun,duration,
                int_rate_aCXB_bg,int_rate_internal_bg,int_rate_continuum_bg)
    # LL_3 = LL_not_m(model_inputs,E_not_m,exp_not_m,counts_not_m,exposure,width_E,
    #                 int_rate_aCXB_bg,int_rate_internal_bg,int_rate_continuum_bg)
    
    return LL_0 + LL_1 + LL_2 + LL_3
    #return LL_0 + LL_1 + LL_3
        
        


    
import numpy as np

########## model functions ##########

def z_min(theta):
    """z_min function (returns zero if theta > theta_sun)"""
    theta_sun = np.arcsin(0.004636733) # RSolar/AU
    theta = np.asarray(theta)
    vec_theta_sun = np.ones(theta.shape) * theta_sun
    vec_arg = np.minimum(theta,vec_theta_sun)
    return(np.sqrt(np.sin(theta_sun)**2 - np.sin(vec_arg)**2))

def T_flux_template(ra,dec,t,duration,alpha0,delta0):
    v_ra = 0.01962028
    v_dec = -0.00835105
    t_min = 3.37603341e+08
    theta_sun = np.arcsin(0.004636733) # RSolar/AU
    
    alpha = alpha0 + v_ra*(t-t_min)/(duration)
    delta = delta0 + v_dec*(t-t_min)/(duration)
    
    theta = np.sqrt((dec-delta)**2 * degree**2 + np.cos((dec+delta)/2 * degree)**2 * (ra-alpha)**2 * degree**2) # using small angle approx
    theta = np.asarray(theta+1e-20); #(shift up to avoid dividing by zero)
    T = np.zeros(theta.shape)
    T += (theta > theta_sun) * 3 * np.pi / 2 * np.sin(theta_sun)**3 * np.sin(theta)**-3
    T += (theta <= theta_sun) * 3 / 4 * np.sin(theta_sun)**3 * np.sin(theta)**-2 * (4 * z_min(theta) / (-1 - 2*z_min(theta)**2 + np.cos(2*theta)) + (np.pi - 2 * np.arctan(z_min(theta) * np.sin(theta)**-1)) * np.sin(theta)**-1)
 
    return T 
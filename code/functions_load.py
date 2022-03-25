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
import corner

import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib import font_manager
from matplotlib import rcParams
from matplotlib import rc
from shapely import geometry


from time import time as tictoc

from functions_model import * 

plt.rcdefaults()
fontsize = 14
rcParams['font.family'] = 'sans-serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':fontsize})

########### units #####################
degree = np.pi/180 # degree in units of radians

########### general-usage functions #####################
factorial_vec = np.vectorize(np.math.factorial) # vectorized version of numpy's factorial function


def findIntersection(contour1,contour2):
  p1 = contour1.collections[0].get_paths()[0]
  v1 = p1.vertices

  p2 = contour2.collections[0].get_paths()[0]
  v2 = p2.vertices

  poly1 = geometry.LineString(v1)
  poly2 = geometry.LineString(v2)

  intersection = poly1.intersection(poly2)

  return intersection

def poisson_limit(N_dat,CL=0.9):
    """Given 'N_dat' data points, return CL='CL' limit on number of signal counts."""
    N_sig = np.logspace(0,8,int(1e5))
    idx = np.argwhere(np.diff(np.sign((1-CL) - stats.poisson.cdf(N_dat,N_sig)))).flatten()[0]
    return N_sig[idx]

########### mapping functions #####################
def map_ra_dec_from_i1_i2(detector,i1,i2,df_box):
    """Given a data frame "df_box" with pixels [i1,i2] in different detectors, returns [ra,dec] coordinates for pixel centers in df_box."""
    return df_box[(df_box['detector']==detector)&(df_box['i1']==i1)&(df_box['i2']==i2)][['ra','dec']].to_numpy()[0]

def map_x_y_from_ra_dec(ra,dec,rotation):
    """Returns x,y from ra,dec coordinates which are (inversely) rotated by 'rotation'."""
    mat_rot = np.asarray([[np.cos(-rotation),np.sin(-rotation)],[-np.sin(-rotation),np.cos(-rotation)]])
    x,y = mat_rot @ np.asarray([ra,dec])
    return x,y

########### load files #####################
def load_events(list_file_events):
    """Loads a data frame with a photon counts labeled by time 't' [second], 'E' [keV], 'ra' [degree], 'dec' [degree]. """
    df_A = pd.read_csv(list_file_events[0],skiprows=1,names=['t','E','ra','dec'])
    df_A.insert(0,'detector','A')
    df_B = pd.read_csv(list_file_events[1],skiprows=1,names=['t','E','ra','dec'])
    df_B.insert(0,'detector','B')
    df_events = df_A.append(df_B)
    return df_events

def load_events_m(list_file_events,m,sigma_E,n_sigma_E=2):
    """Loads a data frame with a photon counts labeled by time 't' [second], 'E' [keV], 'ra' [degree], 'dec' [degree]. Only photon counts with energy |E-m/2| < n_sigma_E * sigma_E are returned."""
    df_events = load_events(list_file_events)
    df_events_m = df_events[np.abs(df_events['E']-m/2) < n_sigma_E*sigma_E]
    return df_events_m

def load_events_not_m(list_file_events,m,sigma_E,n_sigma_E):
    """Loads a data frame with a photon counts labeled by time 't' [second], 'E' [keV], 'ra' [degree], 'dec' [degree]. Only photon counts with energy |E-m/2| > n_sigma_E * sigma_E are returned."""
    #E_cut = 3. # energy cutoff below which calibration cannot be trusted
    df_events = load_events(list_file_events)
    df_events_not_m = df_events[np.abs(df_events['E']-m/2) >= n_sigma_E*sigma_E]
    #df_events_not_m = df_events_not_m[df_events_not_m['E']>E_cut]
    return df_events_not_m

# def load_exp(bins_t,good_time_ints):
#     """Loads exposures of time bins (bins_t) over the "good time intervals" (good_time_ints) of the detectors. These exposures are not corrected for livetime < 1."""
#     df_exp = pd.DataFrame(columns=['idx_t','exp'])
#     for idx_t in range(len(bins_t)-1):
#         t0 = bins_t[idx_t]
#         t1 = bins_t[idx_t+1]
#         exp = 0
#         for (T0,T1) in good_time_ints[:]:
#             if (t0 < T0 < t1):
#                 if (T1 < t1):
#                     exp += T1 - T0
#                 elif (t1 <= T1):
#                     exp += t1 - T0
#             elif (T0 <= t0):
#                 if (t0 < T1 < t1):
#                     exp += T1 - t0
#                 elif (t1 <= T1):
#                     exp += t1 - t0
#         df_exp = df_exp.append(pd.DataFrame([[idx_t,exp]],columns['idx_t','exp']),ignore_index=True)
#         print(df_exp)
#     return df_exp

def load_exp(bins_t,good_time_ints):
    """Loads exposures of time bins (bins_t) over the "good time intervals" (good_time_ints) of the detectors. These exposures are not corrected for livetime < 1."""
    list_exp = []
    for idx_t in range(len(bins_t)-1):
        t0 = bins_t[idx_t]
        t1 = bins_t[idx_t+1]
        exp = 0
        for (T0,T1) in good_time_ints[:]:
            if (t0 < T0 < t1):
                if (T1 < t1):
                    exp += T1 - T0
                elif (t1 <= T1):
                    exp += t1 - T0
            elif (T0 <= t0):
                if (t0 < T1 < t1):
                    exp += T1 - t0
                elif (t1 <= T1):
                    exp += t1 - t0
        list_exp.append([idx_t,exp])
    df_exp = pd.DataFrame(list_exp, columns=['idx_t','exp'])
    return df_exp

def load_box(file_box_centers):
    """Loads a dataframe with 'tile' locations labeled by 'i1' and 'i2' and equatorial coordinates 'ra' and 'dec' [degree]. Also returns the relative 'rotation' [degree] of the 'x,y' coordinates (of the tile lattice) relative to the 'ra,dec' coordinates, and the tile dimensions 'delta_x' and 'delta_y' [arcmin]."""
    df_box = pd.read_csv(file_box_centers,header=None,sep=r' |,',engine='python',
                         names=['detector','i1','nan1','i2','nan2','ra','dec','delta_x','delta_y','rotation'])
    df_box = df_box.replace('[()]','',regex=True)
    df_box.drop(['nan1','nan2'],axis=1,inplace=True)      
    df_box = df_box.astype({'i1': int,'i2': int})
    return df_box

def load_arf(list_file_arf,bins_E,df_box):
    width_E = bins_E[1] - bins_E[0]
    E_cut = 3. # energy cutoff below which ARF cannot be trusted (conservatively set ARF = 0 below)
    df_arf = pd.DataFrame(columns=['detector','idx_E','i1','i2','ra','dec','arf'])
    for i_f,file in enumerate(list_file_arf):
        try:
            file_name = file.split('/')[-1]
            detector = file_name.split('_')[0]
            i1 = int(file_name.split('_')[1])
            i2 = int(file_name.split('_')[2])
            [ra,dec]=map_ra_dec_from_i1_i2(detector,i1,i2,df_box)
            df = pd.read_csv(file)
            list_E = df['# Start of Energy Bin (keV)'].to_numpy()
            idx_E = (np.round((list_E-bins_E[0])/width_E)).astype(int)
            list_arf = df[' Effective Area (cm^2)'].to_numpy()
            list_arf[list_E < E_cut] = 0. # set ARF to vanish below E_cut (conservative) 
            df_tmp = pd.DataFrame(data=np.transpose([len(idx_E)*[detector],idx_E,len(idx_E)*[i1],
                                                     len(idx_E)*[i2],len(idx_E)*[ra],len(idx_E)*[dec],list_arf]),
                                  columns=['detector','idx_E','i1','i2','ra','dec','arf'])
            df_arf = pd.concat([df_arf,df_tmp],ignore_index=True)
            #print(detector, i1, i2)
        except Exception:
            pass
    df_arf = df_arf.astype({'detector': str, 'idx_E': int, 'i1': int,'i2': int, 
                            'ra': float, 'dec': float, 'arf': float})
    return df_arf

def load_arf_m(list_file_arf,bins_E,df_box,m,sigma_E,n_sigma_E=2):
    df_arf = load_arf(list_file_arf,bins_E,df_box)
    df_arf_m = df_arf[np.abs(bins_E[df_arf['idx_E']]-m/2) < n_sigma_E*sigma_E]
    df_arf_m = df_arf_m.sort_values(by=['detector','idx_E','i1','i2'],ignore_index=True)
    return df_arf_m


########## binning functions ##########

def indexed_events(dat,bins_t,bins_E,df_box,m=None,sigma_E=0.166,n_sigma_E=3):
    """Bins events (DataFrame = 'dat') in time bins, energy bins, and pixelated source region, for detectors A and B.
    If m=None, then all data are indexed. If the axion mass 'm' is specified, then only data with m/2 - 2*sigma_E < E < m/2 + 2*sigma_E are indexed.
    Returns data frame with columns = [detector, idx_t, idx_E, pix]."""
    
    df_indexed = pd.DataFrame(columns=['detector','idx_t','idx_E','i1','i2'])
    
    for detector in ['A','B']:
        dat_det = dat[dat['detector']==detector]
        if m==None:
            pass
        else:
            dat_det = dat_det[np.abs(dat_det['E']-m/2)<n_sigma_E*sigma_E]
        box_RA = df_box[df_box['detector']==detector]['ra'].to_numpy()
        box_DEC = df_box[df_box['detector']==detector]['dec'].to_numpy()
    
        idx_t = np.digitize(dat_det['t'], bins_t)-1 #index of t bin for each photon
        idx_E = np.digitize(dat_det['E'], bins_E)-1 #index of E bin for each photon
        
        i1 = np.zeros(len(dat_det),dtype=int)
        i2 = np.zeros(len(dat_det),dtype=int)
        for i in range(len(dat_det)):
            ra = dat_det['ra'].iloc[i]
            dec = dat_det['dec'].iloc[i]
            cos_dec = np.cos(dec*degree)
            dist2 = cos_dec**2 * (ra - box_RA)**2 + (dec - box_DEC)**2
            pix = np.argmin(dist2)
            i1[i] = df_box['i1'][pix]
            i2[i] = df_box['i2'][pix]
            
        df_det = pd.DataFrame(
            data = np.transpose([len(dat_det)*[detector],idx_t,idx_E,i1,i2]),
            columns=['detector','idx_t','idx_E','i1','i2'])
                
        df_indexed = pd.concat([df_indexed,df_det],ignore_index=True)
    
    df_indexed = df_indexed.astype({'detector': str,'idx_t': int, 'idx_E': int, 'i1': int, 'i2': int})
                
    return df_indexed

def indexed_events_not_m(dat,bins_E):
    df_indexed = pd.DataFrame(columns=['detector','idx_E'])
    
    for detector in ['A','B']:
        dat_det = dat[dat['detector']==detector]
        
        idx_E = np.digitize(dat_det['E'], bins_E)-1 #index of E bin for each photon    
        df_det = pd.DataFrame(
            data = np.transpose([len(dat_det)*[detector],idx_E]),
            columns=['detector','idx_E'])
        df_indexed = pd.concat([df_indexed,df_det],ignore_index=True)
    
    df_indexed = df_indexed.astype({'detector': str,'idx_E': int})
    return df_indexed

def binned_events(dat,bins_t,bins_E,df_box,m,sigma_E=0.166,n_sigma_E=3):
    """Bins events (DataFrame = 'dat') in time bins, energy bins, and pixelated source region, for detectors A and B.
    If m=None, then all data are binned. If the axion mass 'm' is specified, then only data with m/2 - 2*sigma_E < E < m/2 + 2*sigma_E are binned.
    Returns data frame with columns = [detector, idx_t, idx_E, pix, counts], only for bins with nonzero counts."""
    
    df_indexed = indexed_events(dat,bins_t,bins_E,df_box,m,sigma_E,n_sigma_E)
    
    df_bin = df_indexed.groupby(df_indexed.columns.tolist(),as_index=False).size().reset_index()
    df_bin = df_bin.rename(columns={"size": "counts"})
    
    df_bin = df_bin.sort_values(by=['idx_t','detector','idx_E','i1','i2'])#,ignore_index=True)
    df_bin = df_bin.reset_index()
    df_bin = df_bin.drop(columns=['index'])
    
    return df_bin

def binned_events_not_m(dat,bins_E):
    df_indexed = indexed_events_not_m(dat,bins_E)
    
    df_bin = df_indexed.groupby(df_indexed.columns.tolist(),as_index=False).size().reset_index()
    df_bin = df_bin.rename(columns={"size": "counts"})
    
    df_bin = df_bin.sort_values(by=['detector','idx_E'])#,ignore_index=True)
    df_bin = df_bin.reset_index()
    df_bin = df_bin.drop(columns=['index'])
    
    return df_bin

########## generate input data frame ##########
def load_data(m,sigma_E,good_time_ints,livetime,bins_t,bins_E,n_sigma_E,list_file_events,file_box_centers,list_file_arf):
    df_events = load_events_m(list_file_events,m,sigma_E,n_sigma_E)
    df_exp = load_exp(bins_t,good_time_ints)
    df_box = load_box(file_box_centers)
    df_arf_m = load_arf_m(list_file_arf,bins_E,df_box,m,sigma_E,n_sigma_E)
    df_events_bin = binned_events(df_events,bins_t,bins_E,df_box,m,sigma_E,n_sigma_E)
    
    ## adding in non-event data ##
    df_data = pd.DataFrame()
    for idx_t,t in enumerate(bins_t[:-1]):
        exp = df_exp[df_exp['idx_t']==idx_t]['exp'].to_numpy()[0]
        tmp = df_arf_m.copy(deep=True)
        tmp.insert(1,'idx_t',idx_t)
        tmp.insert(7,'exp',exp)
        df_data = df_data.append(tmp,ignore_index=True)
    df_data.insert(5,'t',bins_t[df_data['idx_t']])
    df_data.insert(6,'E',bins_E[df_data['idx_E']])
    df_data.insert(11,'Omega',1.0)
    df_data.insert(12,'counts',0)
    
    ## correct for livetime
    df_data.loc[df_data['detector']=='A','exp'] = livetime[0] * df_data.loc[df_data['detector']=='A','exp']
    df_data.loc[df_data['detector']=='B','exp'] = livetime[1] * df_data.loc[df_data['detector']=='B','exp']
    
    ## adding in event data
    df_merge = df_data.append(df_events_bin) # merge source data and event data frames
    idx_first = df_merge.duplicated(subset=['detector','idx_t','idx_E','i1','i2'],keep='first') # row indices of events
    idx_last = df_merge.duplicated(subset=['detector','idx_t','idx_E','i1','i2'],keep='last') # row indices of source
    df_data.loc[idx_last[0:len(df_data)],'counts'] = np.asarray(df_merge.iloc[np.where(idx_first)[0]]['counts'],dtype=int) # add events
    
    ## throw out data below E_cut_min because of lack of calibration, and above E_cut_max
    E_cut_min = 2.999
    df_data = df_data[df_data['E']>E_cut_min]
    E_cut_max = 150.001
    df_data = df_data[df_data['E']<E_cut_max]
    
    return df_data

def load_data_not_m(m,sigma_E,good_time_ints,livetime,bins_t,bins_E,n_sigma_E,list_file_events,file_box_centers,list_file_arf):
    df_events_not_m = load_events_not_m(list_file_events,m,sigma_E,n_sigma_E)
    df_events_bin_not_m = binned_events_not_m(df_events_not_m,bins_E)   
    #df_events_bin_not_m = df_events_bin_not_m.groupby(by=['idx_E'],as_index=False).sum()
    
    exposure = np.sum([interval[1]-interval[0] for interval in good_time_ints])
    
    ## adding in non-event data ##
    df_data = pd.DataFrame()
    df_data.insert(0,'detector',np.concatenate([['A']*len(bins_E),['B']*len(bins_E)]))
    df_data.insert(1,'idx_E',np.concatenate([range(len(bins_E)),range(len(bins_E))]))
    df_data.insert(2,'E',bins_E[df_data['idx_E']])
    df_data.insert(3,'exp',exposure)
    df_data.insert(4,'counts',0)
    
    ## adding in event data
    df_merge = df_data.append(df_events_bin_not_m) # merge source data and event data frames
    idx_first = df_merge.duplicated(subset=['detector','idx_E'],keep='first') # row indices of events
    idx_last = df_merge.duplicated(subset=['detector','idx_E'],keep='last') # row indices of source
    
    df_data.loc[idx_last[0:len(df_data)],'counts'] = np.asarray(df_merge.iloc[np.where(idx_first)[0]]['counts'].fillna(0),dtype=int) # add events
    
    ## correct for livetime
    df_data.loc[df_data['detector']=='A','exp'] = livetime[0] * df_data.loc[df_data['detector']=='A','exp']
    df_data.loc[df_data['detector']=='B','exp'] = livetime[1] * df_data.loc[df_data['detector']=='B','exp']
    
    ## throw out data below E_cut because of lack of calibration
    E_cut = 2.999
    df_data = df_data[df_data['E']>E_cut]
                   
    return df_data

########## generate arf interpolation function ##########
def load_int_arf(m,df_arf,rotation,bins_E,width_E):
    """Returns as first output an interpolation function 'int_arf(E,x,y,livetime)' of the combined, livetime-corrected arf function of the detectors as a function of energy 'E' [keV], and 'x,y' coordinates (those of the tiled arf lattice). Subsequent output are the boundaries x_min, x_max, y_min, y_max over which the ARF interpolation is to be trusted."""
    mat_rot = np.asarray([[np.cos(-rotation),np.sin(-rotation)],[-np.sin(-rotation),np.cos(-rotation)]])
    df_arf_A = df_arf[df_arf['detector']=='A']
    df_arf_B = df_arf[df_arf['detector']=='B']
    arr_keys_A = df_arf_A[['idx_E','i1','i2']].to_numpy()
    arr_keys_B = df_arf_B[['idx_E','i1','i2']].to_numpy()
    arr_arf_A = df_arf_A['arf'].to_numpy()
    arr_arf_B = df_arf_B['arf'].to_numpy()
    arr_arf_A = np.reshape(arr_arf_A,(len(np.unique(arr_keys_A[:,0])),len(np.unique(arr_keys_A[:,1])),len(np.unique(arr_keys_A[:,2]))))
    arr_arf_B = np.reshape(arr_arf_B,(len(np.unique(arr_keys_B[:,0])),len(np.unique(arr_keys_B[:,1])),len(np.unique(arr_keys_B[:,2]))))
    
    idx_E = np.round((m/2-bins_E[0])/width_E).astype(int)
    RA_A, DEC_A = np.transpose(df_arf[(df_arf['idx_E']==idx_E) & (df_arf['detector']=='A')][['ra','dec']].to_numpy())
    RA_B, DEC_B = np.transpose(df_arf[(df_arf['idx_E']==idx_E) & (df_arf['detector']=='B')][['ra','dec']].to_numpy())
    
    X_A, Y_A = mat_rot @ np.asarray([RA_A,DEC_A])
    X_B, Y_B = mat_rot @ np.asarray([RA_B,DEC_B])
    
    N_X = len(np.unique(df_arf['i1'])); 
    N_Y = len(np.unique(df_arf['i2']));
    X_A_min = np.min(X_A); X_A_max = np.max(X_A)
    Y_A_min = np.min(Y_A); Y_A_max = np.max(Y_A)
    X_B_min = np.min(X_B); X_B_max = np.max(X_B)
    Y_B_min = np.min(Y_B); Y_B_max = np.max(Y_B)
    X_A_spacing = (X_A_max - X_A_min)/N_X
    Y_A_spacing = (Y_A_max - Y_A_min)/N_Y
    X_B_spacing = (X_B_max - X_B_min)/N_X
    Y_B_spacing = (Y_B_max - Y_B_min)/N_Y
    vec_X_A = np.linspace(X_A_min,X_A_max,N_X)
    vec_Y_A = np.linspace(Y_A_min,Y_A_max,N_Y)
    vec_X_B = np.linspace(X_B_min,X_B_max,N_X)
    vec_Y_B = np.linspace(Y_B_min,Y_B_max,N_Y)
    vec_E_A = bins_E[np.min(df_arf_A['idx_E'])] + np.asarray(range(len(np.unique(df_arf_A['idx_E'])))) * width_E
    vec_E_B = bins_E[np.min(df_arf_B['idx_E'])] + np.asarray(range(len(np.unique(df_arf_B['idx_E'])))) * width_E
    
    int_arf_A = rgi( points=(vec_E_A,vec_X_A,vec_Y_A), values=arr_arf_A, method='linear', bounds_error=False, fill_value=None )
    int_arf_B = rgi( points=(vec_E_B,vec_X_B,vec_Y_B), values=arr_arf_B, method='linear', bounds_error=False, fill_value=None )

    def int_arf(E,x,y,livetime):
        return livetime[0] * int_arf_A((E,x,y)) + livetime[1] * int_arf_B((E,x,y)) 
    
    # find boundaries
    x_A_min = X_A_min - 0.5 * X_A_spacing
    x_A_max = X_A_max + 0.5 * X_A_spacing
    y_A_min = Y_A_min - 0.5 * Y_A_spacing
    y_A_max = Y_A_max + 0.5 * Y_A_spacing
    x_B_min = X_B_min - 0.5 * X_B_spacing
    x_B_max = X_B_max + 0.5 * X_B_spacing
    y_B_min = Y_B_min - 0.5 * Y_B_spacing
    y_B_max = Y_B_max + 0.5 * Y_B_spacing
    x_min = np.max([x_A_min,x_B_min])
    x_max = np.min([x_A_max,x_B_max])
    y_min = np.max([y_A_min,y_B_min])
    y_max = np.min([y_A_max,y_B_max])
    
    return int_arf, x_min, x_max, y_min, y_max

def plot_xy_ARF(m,ra,dec,df_arf,livetime,rotation,bins_E,width_E,sigma_E):
    """Returns two matplotlib figures, fig1 and fig2
    *** fig1: of the centers of the tiles on which the ARF is computed, and overlays the observed events in x,y space. Also draws rectangles of the fields of view of the detectors, as well as the rectangle (yellow) where all FOVs overlap.
    *** fig2: color density plot of the ARFs at -2,0,+2 sigma_E away from m/2."""
    mat_rot = np.asarray([[np.cos(-rotation),np.sin(-rotation)],[-np.sin(-rotation),np.cos(-rotation)]])
    df_arf_A = df_arf[df_arf['detector']=='A']
    df_arf_B = df_arf[df_arf['detector']=='B']
    arr_keys_A = df_arf_A[['idx_E','i1','i2']].to_numpy()
    arr_keys_B = df_arf_B[['idx_E','i1','i2']].to_numpy()
    arr_arf_A = df_arf_A['arf'].to_numpy()
    arr_arf_B = df_arf_B['arf'].to_numpy()
    arr_arf_A = np.reshape(arr_arf_A,(len(np.unique(arr_keys_A[:,0])),len(np.unique(arr_keys_A[:,1])),len(np.unique(arr_keys_A[:,2]))))
    arr_arf_B = np.reshape(arr_arf_B,(len(np.unique(arr_keys_B[:,0])),len(np.unique(arr_keys_B[:,1])),len(np.unique(arr_keys_B[:,2]))))
    
    idx_E = np.round((m/2-bins_E[0])/width_E).astype(int)
    RA_A, DEC_A = np.transpose(df_arf[(df_arf['idx_E']==idx_E) & (df_arf['detector']=='A')][['ra','dec']].to_numpy())
    RA_B, DEC_B = np.transpose(df_arf[(df_arf['idx_E']==idx_E) & (df_arf['detector']=='B')][['ra','dec']].to_numpy())
    
    X_A, Y_A = mat_rot @ np.asarray([RA_A,DEC_A])
    X_B, Y_B = mat_rot @ np.asarray([RA_B,DEC_B])
    
    N_X = len(np.unique(df_arf['i1'])); 
    N_Y = len(np.unique(df_arf['i2']));
    X_A_min = np.min(X_A); X_A_max = np.max(X_A)
    Y_A_min = np.min(Y_A); Y_A_max = np.max(Y_A)
    X_B_min = np.min(X_B); X_B_max = np.max(X_B)
    Y_B_min = np.min(Y_B); Y_B_max = np.max(Y_B)
    X_A_spacing = (X_A_max - X_A_min)/N_X
    Y_A_spacing = (Y_A_max - Y_A_min)/N_Y
    X_B_spacing = (X_B_max - X_B_min)/N_X
    Y_B_spacing = (Y_B_max - Y_B_min)/N_Y
    vec_X_A = np.linspace(X_A_min,X_A_max,N_X)
    vec_Y_A = np.linspace(Y_A_min,Y_A_max,N_Y)
    vec_X_B = np.linspace(X_B_min,X_B_max,N_X)
    vec_Y_B = np.linspace(Y_B_min,Y_B_max,N_Y)
    vec_E_A = bins_E[np.min(df_arf_A['idx_E'])] + np.asarray(range(len(np.unique(df_arf_A['idx_E'])))) * width_E
    vec_E_B = bins_E[np.min(df_arf_B['idx_E'])] + np.asarray(range(len(np.unique(df_arf_B['idx_E'])))) * width_E
    
    int_arf_A = rgi( points=(vec_E_A,vec_X_A,vec_Y_A), values=arr_arf_A, method='linear', bounds_error=False, fill_value=None )
    int_arf_B = rgi( points=(vec_E_B,vec_X_B,vec_Y_B), values=arr_arf_B, method='linear', bounds_error=False, fill_value=None )

    def int_arf(E,x,y,livetime):
        return livetime[0] * int_arf_A((E,x,y)) + livetime[1] * int_arf_B((E,x,y)) 
    
    # find boundaries
    x_A_min = X_A_min - 0.5 * X_A_spacing
    x_A_max = X_A_max + 0.5 * X_A_spacing
    y_A_min = Y_A_min - 0.5 * Y_A_spacing
    y_A_max = Y_A_max + 0.5 * Y_A_spacing
    x_B_min = X_B_min - 0.5 * X_B_spacing
    x_B_max = X_B_max + 0.5 * X_B_spacing
    y_B_min = Y_B_min - 0.5 * Y_B_spacing
    y_B_max = Y_B_max + 0.5 * Y_B_spacing
    x_min = np.max([x_A_min,x_B_min])
    x_max = np.min([x_A_max,x_B_max])
    y_min = np.max([y_A_min,y_B_min])
    y_max = np.min([y_A_max,y_B_max])
    
    # rotate data
    x,y = mat_rot @ np.asarray([ra,dec])
    
    fig1, ax1 = plt.subplots(1,1,figsize=(6,6))
    ax1.scatter(x,y,s=4,color=(0.2,0.2,0.8,0.3));
    ax1.scatter(X_A,Y_A,color=(0.8,0.2,0))
    ax1.scatter(X_B,Y_B,color=(0.2,0.8,0))
    ax1.scatter(np.meshgrid(vec_X_A,vec_Y_A,indexing='ij')[0],np.meshgrid(vec_X_A,vec_Y_A,indexing='ij')[1],s=2,color='black')
    ax1.scatter(np.meshgrid(vec_X_B,vec_Y_B,indexing='ij')[0],np.meshgrid(vec_X_B,vec_Y_B,indexing='ij')[1],s=2,color='gray')
    rectangle_A = plt.Rectangle((x_A_min,y_A_min), x_A_max-x_A_min, y_A_max-y_A_min,ec=(0.8,0.2,0),fc=(1,1,1,0))
    rectangle_B = plt.Rectangle((x_B_min,y_B_min), x_B_max-x_B_min, y_B_max-y_B_min,ec=(0.2,0.8,0),fc=(1,1,1,0))
    rectangle = plt.Rectangle((x_min,y_min), x_max-x_min, y_max-y_min,ec=(1,1,0),fc=(1,1,1,0))
    plt.gca().add_patch(rectangle_A); plt.gca().add_patch(rectangle_B); plt.gca().add_patch(rectangle);
    ax1.set_xlim(x_min-0.02,x_max+0.02)
    ax1.set_ylim(y_min-0.02,y_max+0.02)
    ax1.set_xlabel('$x$ [degree]'); ax1.set_ylabel('$y$ [degree]');
    
    fig2, ax2 = plt.subplots(3,3,figsize=(18,15))
    for i,En in enumerate(m/2 + sigma_E * np.asarray([-2,0,+2])):
        idx_E = np.round((En-vec_E_A[0])/width_E).astype(int)
        vec_x_plot = np.linspace(x_min,x_max,200)
        vec_y_plot = np.linspace(y_min,y_max,200)
        arr_xy_plot = np.meshgrid(vec_x_plot,vec_y_plot)
        arr_arf_plot = int_arf(En,arr_xy_plot[0],arr_xy_plot[1],livetime)
        plot = ax2[i,0].pcolormesh(vec_x_plot,vec_y_plot,arr_arf_plot,shading='nearest'); fig2.colorbar(plot,ax=ax2[i,0],fraction=0.04)
        plot = ax2[i,1].pcolormesh(vec_X_A,vec_Y_A,np.transpose(arr_arf_A[idx_E]),shading='nearest'); fig2.colorbar(plot,ax=ax2[i,1],fraction=0.04)
        plot = ax2[i,2].pcolormesh(vec_X_B,vec_Y_B,np.transpose(arr_arf_B[idx_E]),shading='nearest'); fig2.colorbar(plot,ax=ax2[i,2],fraction=0.04)
    for i,En in enumerate(m/2 + sigma_E * np.asarray([-2,0,+2])):
        ax2[i,0].set_title(r'interpolated ARF (livetime weighted), $E =$'+str(En))
        ax2[i,1].set_title(r'ARF_A, $E =$'+str(En))
        ax2[i,2].set_title(r'ARF_B, $E =$'+str(En))
        for j in range(3):
            ax2[i,j].set_xlabel('$x$'); ax2[i,j].set_ylabel('$y$')
            ax2[i,j].set_xlim(x_min-0.02,x_max+0.02)
            ax2[i,j].set_ylim(y_min-0.02,y_max+0.02)
            ax2[i,j].set_aspect('equal')
    fig2.tight_layout()
    
    return fig1,fig2

########## project onto unit cuboid ##########
def proj_unit_cuboid(t,E,x,y,m,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration,
                     good_time_ints,livetime,bins_E,sigma_E,n_sigma_E,x_min,x_max,y_min,y_max,rotation,df_arf,int_arf,N_x=20,N_y=20):
    """Returns (array of) unit-cuboid-projected r_1,r_2,r_3,r_4 coordinates, as well as rho_0,
    given an array (t,E,x,y) of photon counts, axion mass 'm', and initial solar position and shift parameters."""
    
    mat_rot = np.asarray([[np.cos(-rotation),np.sin(-rotation)],[-np.sin(-rotation),np.cos(-rotation)]])
    
    df_arf_A = df_arf[df_arf['detector']=='A']
    df_arf_B = df_arf[df_arf['detector']=='B']
    
    width_E = bins_E[1]-bins_E[0];
    vec_E_A = bins_E[np.min(df_arf_A['idx_E'])] + np.asarray(range(len(np.unique(df_arf_A['idx_E'])))) * width_E
    vec_E_B = bins_E[np.min(df_arf_B['idx_E'])] + np.asarray(range(len(np.unique(df_arf_B['idx_E'])))) * width_E
    
    bounds_E = [np.max([vec_E_A[0],vec_E_B[0],m/2 - n_sigma_E*sigma_E]), np.min([vec_E_A[-1],vec_E_B[-1],m/2 + n_sigma_E*sigma_E])]
    bounds_x = [x_min,x_max]
    bounds_y = [y_min,y_max]
    
    idx_c = (x_min < x) & (x < x_max) & (y_min < y) & (y < y_max) & (bounds_E[0] < E) & (E < bounds_E[1])
    t = t[idx_c]
    E = E[idx_c]
    x = x[idx_c]
    y = y[idx_c]
    
    vec_E_proj = np.linspace(bounds_E[0],bounds_E[1],len(vec_E_A))
    vec_x_proj = np.linspace(x_min,x_max,N_x)
    vec_y_proj = np.linspace(y_min,y_max,N_y)
    arr_Exy_proj = np.meshgrid(vec_E_proj,vec_x_proj,vec_y_proj,indexing='ij')
    arr_xy_proj = np.meshgrid(vec_x_proj,vec_y_proj,indexing='ij')

    ## rho functions
    def rho_4(t,E,x,y):
        ra, dec = np.linalg.inv(mat_rot) @ np.asarray([x,y])
        arf_livetime = int_arf(E,x,y,livetime)
        lineshape = np.exp(-(E-m/2)**2 / (2 * sigma_E**2)) / np.sqrt(2 * np.pi * sigma_E**2)
        T = T_flux_template(t,ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration)
        return arf_livetime * lineshape * T
    def rho_3_quad(E,x,y):
        ra, dec = np.linalg.inv(mat_rot) @ np.asarray([x,y])
        arf_livetime = int_arf(E,x,y,livetime)
        lineshape = np.exp(-(E-m/2)**2 / (2 * sigma_E**2)) / np.sqrt(2 * np.pi * sigma_E**2)
        integral_t = np.sum([
            quad(T_flux_template,interval[0],interval[1],args=(ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration),limit=100,epsrel=1e-5,epsabs=0)[0] 
            for interval in good_time_ints])
        return arf_livetime * lineshape * integral_t
    rho_3_quad = np.vectorize(rho_3_quad)

    def rho_2_quad(x,y):
        ra, dec = np.linalg.inv(mat_rot) @ np.asarray([x,y])
        integral_t = np.sum([
            quad(T_flux_template,interval[0],interval[1],args=(ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration),limit=100,epsrel=1e-5,epsabs=0)[0] 
            for interval in good_time_ints])    
        def fun_E(E):
            return int_arf(E,x,y,livetime) * np.exp(-(E-m/2)**2 / (2 * sigma_E**2)) / np.sqrt(2 * np.pi * sigma_E**2)
        integral_E = quad(fun_E,bounds_E[0],bounds_E[1],epsrel=1e-6,epsabs=0)[0]
        return integral_E * integral_t
    rho_2_quad = np.vectorize(rho_2_quad)
    
    #print('Computing rho_3 table...')
    tic3 = tictoc()
    arr_rho_3 = rho_3_quad(arr_Exy_proj[0],arr_Exy_proj[1],arr_Exy_proj[2])
    toc3 = tictoc()
    #print('time for rho_3 table =',toc3 - tic3)
    #print('Computing rho_2 table...')
    tic2 = tictoc()
    arr_rho_2 = rho_2_quad(arr_xy_proj[0],arr_xy_proj[1])
    toc2 = tictoc()
    #print('time for rho_2 table =',toc2 - tic2)
    
    int_rho_3_tuple = rgi(points=(vec_E_proj,vec_x_proj,vec_y_proj),values=arr_rho_3,method='linear',bounds_error=False,fill_value=None)
    int_rho_2_tuple = rgi(points=(vec_x_proj,vec_y_proj),values=arr_rho_2,method='linear',bounds_error=False,fill_value=None)
    def rho_3(E,x,y):
        return int_rho_3_tuple((E,x,y))
    def rho_2(x,y):
        return int_rho_2_tuple((x,y))
    
    def rho_1_quad(y):
        return quad(rho_2,bounds_x[0],bounds_x[1],args=y,epsrel=1e-4,epsabs=0,limit=100)[0]
    rho_1_quad = np.vectorize(rho_1_quad)
    
    #print('Computing rho_1 table...')
    arr_rho_1 = rho_1_quad(vec_y_proj)
    rho_1 = interp1d(vec_y_proj,arr_rho_1,kind='linear',bounds_error=False,fill_value='extrapolate')
    
    
    def rho_0():
        return quad(rho_1,bounds_y[0],bounds_y[1],epsrel=1e-4,epsabs=0,limit=100)[0]
    #print('Computing rho_0...')
    rho_0_val = rho_0()
    
    print('m = '+str(m)[0:8]+': rho functions computed in '+str(toc2-tic3)+' seconds.')
    
    def map_r_4(t,E,x,y):
        ra, dec = np.linalg.inv(mat_rot) @ np.asarray([x,y])
        arf_livetime = int_arf(E,x,y,livetime)
        lineshape = np.exp(-(E-m/2)**2 / (2 * sigma_E**2)) / np.sqrt(2 * np.pi * sigma_E**2)
        integral_t = np.sum([
            quad(T_flux_template,interval[0],np.min([t,interval[1]]),args=(ra,dec,ra_sun_0,dec_sun_0,delta_ra_sun,delta_dec_sun,t_min,duration),limit=100,epsrel=1e-5,epsabs=0)[0] 
            for interval in good_time_ints if t > interval[0]])
        return arf_livetime * lineshape * integral_t / rho_3(E,x,y)
    map_r_4 = np.vectorize(map_r_4)
    
    def map_r_3_quad(E,x,y):
        return quad(rho_3,bounds_E[0],E,args=(x,y),limit=100,epsrel=1e-3,epsabs=0)[0] / rho_2(x,y)
    map_r_3 = np.vectorize(map_r_3_quad)
    def map_r_2_quad(x,y):
        return quad(rho_2,bounds_x[0],x,args=(y),limit=100,epsrel=1e-3,epsabs=0)[0] / rho_1(y)
    map_r_2 = np.vectorize(map_r_2_quad)
    def map_r_1_quad(y):
        return quad(rho_1,bounds_y[0],y,limit=100,epsrel=1e-3,epsabs=0)[0] / rho_0_val
    map_r_1 = np.vectorize(map_r_1_quad)
    
    #print('Computing r_4 coordinates...')
    tic4 = tictoc()
    r_4 = map_r_4(t,E,x,y)
    toc4 = tictoc()
    #print('time for r_4 array =',toc4 - tic4)
    
    #print('Computing r_3 coordinates...')
    tic3 = tictoc()
    r_3 = map_r_3(E,x,y)
    toc3 = tictoc()
    #print('time for r_3 array =',toc3 - tic3)
    
    #print('Computing r_2 coordinates...')
    tic2 = tictoc()
    r_2 = map_r_2(x,y)
    toc2 = tictoc()
    #print('time for r_2 array =',toc2 - tic2)
    
    #print('Computing r_1 coordinates...')
    tic1 = tictoc()
    r_1 = map_r_1(y)
    toc1 = tictoc()
    #print('time for r_1 array =',toc1 - tic1)
    
    print('m = '+str(m)[0:8]+': r coordinates computed in '+str(toc1-tic4)+' seconds.')
    
    r_1[r_1<0] = 1e-4
    r_2[r_2<0] = 1e-4
    r_3[r_3<0] = 1e-4
    r_4[r_4<0] = 1e-4
    r_1[1<r_1] = 1-1e-4
    r_2[1<r_2] = 1-1e-4
    r_3[1<r_3] = 1-1e-4
    r_4[1<r_4] = 1-1e-4
    
    return r_1, r_2, r_3, r_4, rho_0_val
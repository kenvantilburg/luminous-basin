import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import scipy.stats as stats
import pandas as pd
from os import listdir
import math
import csv
from scipy import optimize
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.misc import derivative
import numdifftools
from mpl_toolkits import mplot3d
from tqdm import tqdm, tqdm_notebook

from matplotlib import font_manager
from matplotlib import rcParams
from matplotlib import rc

plt.rcdefaults()
fontsize = 14
rcParams['font.family'] = 'sans-serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':fontsize})

########### units #####################
degree = np.pi/180 # degree in units of radians

########### general-usage functions #####################
factorial_vec = np.vectorize(np.math.factorial) # vectorized version of numpy's factorial function

########### mapping functions #####################
def fn_ra_dec_from_i1_i2(detector,i1,i2,df_box):
    """Given a data frame "df_box" with pixels [i1,i2] in different detectors, returns [ra,dec] coordinates for pixel centers in df_box."""
    return df_box[(df_box['detector']==detector)&(df_box['i1']==i1)&(df_box['i2']==i2)][['ra','dec']].to_numpy()[0]

########### load files #####################
def load_events(list_file_events):
    df_A = pd.read_csv(list_file_events[0],skiprows=1,names=['t','E','ra','dec'])
    df_A.insert(0,'detector','A')
    df_B = pd.read_csv(list_file_events[1],skiprows=1,names=['t','E','ra','dec'])
    df_B.insert(0,'detector','B')
    df_events = df_A.append(df_B)
    return df_events

def load_exp(bins_t,good_time_ints):
    df_exp = pd.DataFrame(columns=['idx_t','exp'])
    for idx_t in range(len(bins_t)-1):
        t0 = bins_t[idx_t]
        t1 = bins_t[idx_t+1]
        exp = 0
        for (T0,T1) in good_time_ints[1:]:
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
        df_exp = df_exp.append(pd.DataFrame([[idx_t,exp]],columns=['idx_t','exp']),ignore_index=True)
    return df_exp

def load_box(file_box_centers):
    df_box = pd.read_csv(file_box_centers,header=None,sep=r' |,',engine='python',
                         names=['detector','i1','nan1','i2','nan2','ra','dec','delta_x','delta_y','rotation'])
    df_box = df_box.replace('[()]','',regex=True)
    df_box.drop(['nan1','nan2'],axis=1,inplace=True)      
    df_box = df_box.astype({'i1': int,'i2': int})
    return df_box

def load_arf(list_file_arf,bins_E,df_box):
    width_E = bins_E[1] - bins_E[0]
    df_arf = pd.DataFrame(columns=['detector','idx_E','i1','i2','ra','dec','arf'])
    for i_f,file in tqdm(enumerate(list_file_arf)):
        try:
            file_name = file.split('/')[-1]
            detector = file_name.split('_')[0]
            i1 = np.int(file_name.split('_')[1])
            i2 = np.int(file_name.split('_')[2])
            [ra,dec]=fn_ra_dec_from_i1_i2(detector,i1,i2,df_box)
            df = pd.read_csv(file)
            list_E = df['# Start of Energy Bin (keV)'].to_numpy()
            idx_E = (np.round((list_E-bins_E[0])/width_E)).astype(int)
            list_arf = df[' Effective Area (cm^2)'].to_numpy()
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

def load_arf_m(list_file_arf,bins_E,df_box,m,sigma_E):
    df_arf = load_arf(list_file_arf,bins_E,df_box)
    df_arf_m = df_arf[np.abs(bins_E[df_arf['idx_E']]-m/2) < 2*sigma_E]
    df_arf_m = df_arf_m.sort_values(by=['detector','idx_E','i1','i2'],ignore_index=True)
    return df_arf_m


########## binning functions ##########

def indexed_events(dat,bins_t,bins_E,df_box,m=None,sigma_E=0.166):
    """Bins events (DataFrame = 'dat') in time bins, energy bins, and pixelated source region, for detectors A and B.
    If m=None, then all data are indexed. If the axion mass 'm' is specified, then only data with m/2 - 2*sigma_E < E < m/2 + 2*sigma_E are indexed.
    Returns data frame with columns = [detector, idx_t, idx_E, pix]."""
    
    df_indexed = pd.DataFrame(columns=['detector','idx_t','idx_E','i1','i2'])
    
    for detector in ['A','B']:
        dat_det = dat[dat['detector']==detector]
        if m==None:
            pass
        else:
            dat_det = dat_det[np.abs(dat_det['E']-m/2)<2*sigma_E]
        box_RA = df_box[df_box['detector']==detector]['ra'].to_numpy()
        box_DEC = df_box[df_box['detector']==detector]['dec'].to_numpy()
    
        idx_t = np.digitize(dat_det['t'], bins_t)-1 #index of t bin for each photon
        idx_E = np.digitize(dat_det['E'], bins_E)-1 #index of E bin for each photon
        
#        pix = np.zeros(len(dat_det),dtype=int)
        i1 = np.zeros(len(dat_det),dtype=int)
        i2 = np.zeros(len(dat_det),dtype=int)
        for i in range(len(dat_det)):
            ra = dat_det['ra'].iloc[i]
            dec = dat_det['dec'].iloc[i]
            cos_dec = np.cos(dec*degree)
            dist2 = cos_dec**2 * (ra - box_RA)**2 + (dec - box_DEC)**2
#            pix[i] = np.argmin(dist2)
            pix = np.argmin(dist2)
            i1[i] = df_box['i1'][pix]
            i2[i] = df_box['i2'][pix]
            
        df_det = pd.DataFrame(
            data = np.transpose([len(dat_det)*[detector],idx_t,idx_E,i1,i2]),
            columns=['detector','idx_t','idx_E','i1','i2'])
                
        df_indexed = pd.concat([df_indexed,df_det],ignore_index=True)
    
    df_indexed = df_indexed.astype({'detector': str,'idx_t': int, 'idx_E': int, 'i1': int, 'i2': int})
                
    return df_indexed

def binned_events(dat,bins_t,bins_E,df_box,m,sigma_E=0.166):
    """Bins events (DataFrame = 'dat') in time bins, energy bins, and pixelated source region, for detectors A and B.
    If m=None, then all data are binned. If the axion mass 'm' is specified, then only data with m/2 - 2*sigma_E < E < m/2 + 2*sigma_E are binned.
    Returns data frame with columns = [detector, idx_t, idx_E, pix, counts], only for bins with nonzero counts."""
    
    df_indexed = indexed_events(dat,bins_t,bins_E,df_box,m,sigma_E)
    
    df_bin = df_indexed.groupby(df_indexed.columns.tolist(),as_index=False).size()
    df_bin = df_bin.rename(columns={"size": "counts"})
    
    df_bin = df_bin.sort_values(by=['idx_t','detector','idx_E','i1','i2'],ignore_index=True)
    
    return df_bin

########## generate input data frame ##########
def load_data(m,sigma_E,good_time_ints,livetime,bins_t,bins_E,list_file_events,file_box_centers,list_file_arf):
    df_events = load_events(list_file_events)
    df_exp = load_exp(bins_t,good_time_ints)
    df_box = load_box(file_box_centers)
    df_arf_m = load_arf_m(list_file_arf,bins_E,df_box,m,sigma_E)
    df_events_bin = binned_events(df_events,bins_t,bins_E,df_box,m,sigma_E)
    
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
    return df_data
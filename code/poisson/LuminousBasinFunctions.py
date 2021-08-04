import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from preamble import * 
from myUnits import *

########### Useful Numbers ##########
theta_sun = np.arcsin(0.004636733) # RSolar/AU

########### Conversion Functions ##########
def XY_to_thetaphi(X,Y):
    """convert X,Y (arcsec) to theta,phi (radians)"""
    phi = np.arctan2(Y * arcsec,X * arcsec)
    theta = arcsec * np.sqrt(X**2 + Y**2)
    return theta, phi

########### Count Volumes ###########

def CountArrayOriginCuboid(A):
    """Takes a 4D array A, and returns the total count S[n,m,l,k] in A for each hypercube defined by the origin [0,0,0,0] and the index [n,m,l,k]. Returns an array S that is larger than A by 1 along each axis."""
    N,M,L,K = np.shape(A)
    S = np.zeros([N+1,M+1,L+1,K+1])
    for k in range(1,K+1):
        for l in range(1,L+1):
            for m in range(1,M+1):
                for n in range(1,N+1):
                    S[n,m,l,k] = S[n-1,m,l,k]+S[n,m-1,l,k]+S[n,m,l-1,k]+S[n,m,l,k-1]-S[n-1,m-1,l,k]-S[n-1,m,l-1,k]-S[n-1,m,l,k-1]-S[n,m-1,l-1,k]-S[n,m-1,l,k-1]-S[n,m,l-1,k-1]+S[n-1,m-1,l-1,k]+S[n-1,m-1,l,k-1]+S[n-1,m,l-1,k-1]+S[n,m-1,l-1,k-1]-S[n-1,m-1,l-1,k-1]+A[n-1,m-1,l-1,k-1]
    return(S)

def CountsInCuboid(S,a1,b1,c1,d1,a2,b2,c2,d2): 
    """Given a count array S = CountsOriginRectangles(A) for hypercubes defined by the origin [0,0,0,0] and the index [n,m,l,k], finds the number of counts in a hypercube defined by points [a1-1,b1-1,c1-1,d1-1] and [a2-1,b2-1,c2-1,d2-1]."""
    count = S[a2,b2,c2,d2]-S[a1-1,b2,c2,d2]-S[a2,b1-1,c2,d2]-S[a2,b2,c1-1,d2]-S[a2,b2,c2,d1-1]+S[a1-1,b1-1,c2,d2] + S[a1-1,b2,c1-1,d2] + S[a1-1,b2,c2,d1-1] + S[a2,b1-1,c1-1,d2] + S[a2,b1-1,c2,d1-1] +S[a2,b2,c1-1,d1-1] - S[a1-1,b1-1,c1-1,d2] - S[a1-1,b1-1,c2,d1-1] - S[a1-1,b2,c1-1,d1-1] -S[a2,b1-1,c1-1,d1-1] + S[a1-1,b1-1,c1-1,d1-1]
    return(count)

#finds max rectangles with {0,1,2,..n} points in 4D array A with unit size, 
def MaximalCuboidVolumes(A):
    """For a 4D array A with n points, finds the maximal-volume cuboids with j or fewer points, and returns an array of length n+1 of their volumes V_j for j = [0,1,2,...,n]. The array A is assumed to be a discretization of the unit cube. By construction, therefore, V_n = 1, and all volumes V_j are smaller. V_0 is the largest empty cuboid."""
    n = np.int(np.sum(A)); # number of events in A
    S = CountArrayOriginCuboid(A)
    N,M,L,K = np.shape(A)
    Vj = np.zeros(n+1);
    total_ncell = N*M*L*K # use to normalize
    for a1 in range(1,N+1):
        for a2 in range(a1,N+1):
            for b1 in range(1,M+1):
                for b2 in range(b1,M+1):
                    for c1 in range(1,L+1):
                        for c2 in range(c1,L+1):
                            for d1 in range(1,K+1):
                                for d2 in range(d1,K+1):
                                    count = int(CountsInCuboid(S,a1,b1,c1,d1,a2,b2,c2,d2))
                                    vol = np.abs((a2-a1+1)*(b2-b1+1)*(c2-c1+1)*(d2-d1+1))/(total_ncell)
                                    if count<(n+1):
                                        if vol>Vj[count]:
                                            Vj[count] = vol
    for i in range(n):
        if Vj[i]>Vj[i+1]:
            Vj[i+1] = Vj[i]
    return(Vj)

########### Signal Monte Carlo ###########
def GenerateSamples(N,mu,k1,k2,k3,k4):
    """Generates N Monte Carlo samples with mean number of events mu identically distributed over the discretized 4D unit cube with k1*k2*k3*k4 cells. Returns np.array of length (N,k1,k2,k3,k4)."""
    samples = np.zeros((N,k1,k2,k3,k4))
    n_cells = k1*k2*k3*k4
    for n in range(N):
        samples[n] = np.random.poisson(mu/n_cells,size=(k1,k2,k3,k4))
    return(samples)

def MonteCarloVolumes(N,mu,k1,k2,k3,k4):
    """Generates N Monte Carlo samples with mean number of events mu identically distributed over the discretized 4D unit cube with k1*k2*k3*k4 cells, and returns the list of volumes V_j for each Monte Carlo sample. Returns array of shape N x dim_max where dim_max is the largest number of points over all samples (+1)."""
    samples = np.zeros((N,k1,k2,k3,k4))
    n_cells = k1*k2*k3*k4
    
    arr_V = []
    for n in range(N):
        A = np.random.poisson(mu/n_cells,size=(k1,k2,k3,k4))
        V_n = MaximalCuboidVolumes(A)
        arr_V.append(V_n)
    dim_max = np.max([len(V_n) for V_n in arr_V])
    arr_V_pad = np.nan * np.ones((N,dim_max))
    for i in range(N):
        for j in range(len(arr_V[i])):
            arr_V_pad[i,j] = arr_V[i][j]
    arr_V_pad[arr_V_pad>1-1e-10]=np.nan #set all V_n = 1 to V_n = np.nan
    return arr_V_pad

def MonteCarloCDFVolumes(N,mu,k1,k2,k3,k4,numbins=1001):
    """Returns list of volumes (shape = (N,max_n)) and interpolated cumulative distribution functions (shape = max_n)."""
    vols = MonteCarloVolumes(N,mu,k1,k2,k3,k4)
    arr_C_n = []
    for i in range(vols.shape[1]):
        res = stats.cumfreq(vols[:,i],numbins=numbins,defaultreallimits=(0,1))
        x = res.lowerlimit + np.linspace(0,res.binsize*res.cumcount.size,res.cumcount.size)
        res_int = interp1d(x,res.cumcount/N)
        arr_C_n.append(res_int)
    return vols, arr_C_n

def MonteCarloCmax(N,mu,k1,k2,k3,k4,numbins=1001):
    """Returns CDF of C_max = max_n{ C_n }."""
    vols, arr_C_n = MonteCarloCDFVolumes(N,mu,k1,k2,k3,k4,numbins=numbins)
    list_C_max = np.zeros(N)
    for i in range(N):
        for n,vol in enumerate(vols[i]):
            if vol > 0: #if not nan
                C_n = arr_C_n[n](vol)
                list_C_max[i] = np.max([C_n,list_C_max[i]])
    return(list_C_max)

def MonteCarloCmaxBar(N,mu,k1,k2,k3,k4,numbins=1001):
    """Estimates 90th percentile of C_max for given mu and k binning."""
    list_C_max = MonteCarloCmax(N,mu,k1,k2,k3,k4,numbins=numbins)
    C_max_bar = np.quantile(list_C_max,0.9)
    return C_max_bar
    
            
            
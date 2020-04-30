from numba import njit
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

EPS = 1e-7 
EPS2 = -1 

@njit('float64(float64[:])',cache=True)
def _logsumexp(a):
    a_max = np.max(a)

    tmp = np.exp(a - a_max)

    s = np.sum(tmp)
    out = np.log(s)

    out += a_max
    return out

@njit('float64(float64[:],float64[:])',cache=True)
def _logsumexpb(a,b):

    a_max = np.max(a)

    tmp = b * np.exp(a - a_max)

    s = np.sum(tmp)
    out = np.log(s)

    out += a_max
    return out

@njit('float64[:](float64,float64[:],float64[:],float64[:],int64)',cache=True)
def _frequency_memos(x0,epochs,selEpochPtr,N,anc=0):
    if True:
        bit = EPS 
    else:
        bit = 1
    if anc:
        selEpoch = -np.array(list(selEpochPtr)) - bit
        x0 = 1-x0
    else:
        selEpoch = np.array(list(selEpochPtr)) + bit
    
    freqs = np.zeros(len(epochs))
    freqs[0] = x0
    for ie in range(1,len(epochs)):
        t0 = epochs[ie-1]
        t1 = epochs[ie]
        s = selEpoch[ie-1]
        x0 = x0 * (x0 + (1-x0) * np.exp(s * (t1-t0)))**(-1)
        freqs[ie] = x0
    return freqs

@njit('float64(float64,float64[:],float64[:],float64[:],float64[:],int64)',cache=True)
def _freq_using_memos(t,epochs,selEpochPtr,freqMemosPtr,N,anc=0):
    iEpoch = int(np.digitize(np.array([t]),epochs)[0]-1)
    if True:
        bit = EPS 
    else:
        bit = 1
    if anc:
        selEpoch = -np.array(list(selEpochPtr)) - bit
        freqMemos = 1-freqMemosPtr
    else:
        selEpoch = np.array(list(selEpochPtr)) + bit
        freqMemos = freqMemosPtr

    s = selEpoch[iEpoch]
    t1 = epochs[iEpoch]
    x0 = freqMemos[iEpoch]
    f = x0*(x0 + (1-x0)*np.exp(s*(t-t1)))**-1
    return f


@njit('float64[:](float64[:],float64[:],float64[:],float64[:],int64)',cache=True)
def _coal_intensity_memos(epochs,selEpochPtr,freqMemosPtr,N,anc=0):
    '''
    returns the intensity function evaluated at each epoch (for 
        faster likelihood and gradient calculations)
    '''
    if True:
        bit = EPS 
    else:
        bit = 1
    if anc:
        selEpoch = -np.array(list(selEpochPtr)) - bit
        freqMemos = 1-freqMemosPtr
    else:
        selEpoch = np.array(list(selEpochPtr)) + bit
        freqMemos = freqMemosPtr

    Lambda = np.zeros(len(epochs))
    N0 = N[0]
    for ie in range(1,len(epochs)):
        x0 = freqMemos[ie-1]
        t0 = epochs[ie-1]
        t1 = epochs[ie]
        s = selEpoch[ie-1]
        if np.abs(s) > EPS2:
            Lambda[ie] = (t1-t0) + (1-x0)/(x0*s)*(np.exp(s*(t1-t0)) - 1)
        else:
            Lambda[ie] = 1/x0 * (t1-t0)
        Lambda[ie] *= N0/N[ie-1]
        Lambda[ie] += Lambda[ie-1]
    return Lambda

@njit('float64(float64,float64[:],float64[:],float64[:],float64[:],float64[:],int64)',cache=True)
def _coal_intensity_using_memos(t,epochs,selEpochPtr,freqMemosPtr,intensityMemos,N,anc=0):
    iEpoch = int(np.digitize(np.array([t]),epochs)[0]-1)
    if True:
        bit = EPS 
    else:
        bit = 1
    if anc:
        selEpoch = -np.array(list(selEpochPtr)) - bit
        freqMemos = 1-freqMemosPtr
    else:
        selEpoch = np.array(list(selEpochPtr)) + bit
        freqMemos = freqMemosPtr


    s = selEpoch[iEpoch]
    t1 = epochs[iEpoch]
    x0 = freqMemos[iEpoch]
    N0 = N[0] 
    Lambda = intensityMemos[iEpoch]
    if np.abs(s) > EPS2:
        Lambda += N0/N[iEpoch] * ((t-t1) + (1-x0)/(x0*s)*(np.exp(s*(t-t1)) - 1))
    else:
        Lambda += N0/N[iEpoch] * 1/x0 * (t-t1)
    return Lambda
    
@njit('float64(float64[:],int64,float64[:],float64[:],float64,float64[:],int64,float64)',cache=True)
def _log_coal_density(times,n,epochs,selEpoch,x0,N,anc=0,tCutoff=5000.0):
    logp = 0
    prevt = 0
    prevLambda = 0
    times = times[times < tCutoff]
    times = times[times >= 0]
    mySelEpoch = selEpoch
    N0 = N[0]    
    # memoize frequency and intensity    
    myFreqMemos = _frequency_memos(x0,epochs,mySelEpoch,N,anc=0)
    myIntensityMemos = _coal_intensity_memos(epochs,mySelEpoch,myFreqMemos,N,anc=anc)
    for i,t in enumerate(times):
        k = n-i
        kchoose2 = k*(k-1)/(4*N0)
        Lambda = _coal_intensity_using_memos(t,epochs,mySelEpoch,myFreqMemos,myIntensityMemos,N,anc=anc)
        logpk = -np.log(_freq_using_memos(t,epochs,mySelEpoch,myFreqMemos,N,anc=anc)) \
                - kchoose2 * ( Lambda - prevLambda)
        logp += logpk
        
        prevt = t
        prevLambda = Lambda
    ## now add the probability of lineages not coalescing by tCutoff
    k -= 1
    kchoose2 = k*(k-1)/(4*N0)
    logPk = - kchoose2 * (_coal_intensity_using_memos(tCutoff,epochs,mySelEpoch,myFreqMemos,myIntensityMemos,N,anc=anc) \
                                - prevLambda)

    logp += logPk
    return logp

@njit('float64(float64[:,:,:],int64[:],float64[:],float64[:],float64,float64[:],float64)',cache=True)
def log_l_importance_sampler(times,ns,epochs,selEpoch,x0,N,tCutoff=5000.0):
    M = times.shape[0]
    logls = np.zeros(M)
    neuSelEpoch = np.zeros(len(selEpoch))
    tCutoffNeu = 2000000
    for i in range(M):
        times_m = times[i,:,:]
        n = ns[0]
        m = ns[1]
        derTimes = times_m[:,0]
        ancTimes = times_m[:,1]
        val = _log_coal_density(derTimes,n,epochs,selEpoch,x0,N,anc=0,tCutoff=tCutoff)
        val += _log_coal_density(ancTimes,m,epochs,selEpoch,x0,N,anc=1,tCutoff=tCutoff)

        neuTimes = np.sort(np.concatenate((derTimes,ancTimes)))
        val -= _log_coal_density(neuTimes,n+m,epochs,neuSelEpoch,1.0-EPS,N,anc=0,tCutoff=tCutoffNeu)

        logls[i] = val
    logl = _logsumexp(logls) - np.log(M) 

    if np.isnan(logl):
        logl = -np.inf
    return logl

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 00:30:59 2021

@author: Tapas Tripura
"""

import numpy as np
from scipy import linalg as LA
from sklearn.metrics import mean_squared_error as MSE

from numpy import linalg as LA2
from numpy.random import gamma as IG
from numpy.random import beta
from numpy.random import binomial as bern
from numpy.random import multivariate_normal as mvrv

from scipy.special import loggamma as LG


"""
Theta: Multivariate Normal distribution
"""
def sigmu(z, D, vs, xdts):
    index = np.array(np.where(z != 0))
    index = np.reshape(index,-1) # converting to 1-D array, 
    Dr = D[:,index] 
    Aor = np.eye(len(index)) # independent prior
    # Aor = np.dot(len(Dr), LA2.inv(np.matmul(Dr.T, Dr))) # g-prior
    BSIG = LA2.inv(np.matmul(Dr.T,Dr) + np.dot(pow(vs,-1), LA2.inv(Aor)))
    mu = np.matmul(np.matmul(BSIG,Dr.T),xdts)
    return mu, BSIG, Aor, index

"""
P(Y|zi=(0|1),z-i,vs)
"""
def pyzv(D, ztemp, vs, N, xdts, asig, bsig):
    rind = np.array(np.where(ztemp != 0))[0]
    rind = np.reshape(rind, -1) # converting to 1-D array,   
    Sz = sum(ztemp)
    Dr = D[:, rind] 
    Aor = np.eye(len(rind)) # independent prior
    # Aor = np.dot(N, LA2.inv(np.matmul(Dr.T, Dr))) # g-prior
    BSIG = np.matmul(Dr.T, Dr) + np.dot(pow(vs, -1),LA2.inv(Aor))
    
    (sign, logdet0) = LA2.slogdet(LA2.inv(Aor))
    (sign, logdet1) = LA2.slogdet(LA2.inv(BSIG))
    
    PZ = LG(asig + 0.5*N) -0.5*N*np.log(2*np.pi) - 0.5*Sz*np.log(vs) \
        + asig*np.log(bsig) - LG(asig) + 0.5*logdet0 + 0.5*logdet1
    denom1 = np.eye(N) - np.matmul(np.matmul(Dr, LA2.inv(BSIG)), Dr.T)
    denom = (0.5*np.matmul(np.matmul(xdts.T, denom1), xdts))
    PZ = PZ - (asig+0.5*N)*(np.log(bsig + denom))
    return PZ

"""
P(Y|zi=0,z-i,vs)
"""
def pyzv0(xdts, N, asig, bsig):
    PZ0 = LG(asig + 0.5*N) - 0.5*N*np.log(2*np.pi) + asig*np.log(bsig) - LG(asig) \
        + np.log(1) - (asig+0.5*N)*np.log(bsig + 0.5*np.matmul(xdts.T, xdts))
    return PZ0
    

"""
# Residual variance:
"""
def res_var(D, xdts):
    theta1 = np.dot(LA.pinv(D), xdts)
    error = xdts - np.matmul(D, theta1)
    err_var = np.var(error)
    
    return err_var
    
"""
# Initial latent vector finder:
"""
def latent(nl, D, xdts):
    # Forward finder:
    zint = np.zeros(nl)
    theta = np.matmul(LA.pinv(D), xdts)
    index = np.array(np.where(zint != 0))[0]
    index = np.reshape(index,-1) # converting to 1-D array,
    Dr = D[:, index]
    thetar = theta[index]
    err = MSE(xdts, np.dot(Dr, thetar))
    for i in range(0, nl):
        index = i
        Dr = D[:, index]
        thetar = theta[index]
        err = np.append(err, MSE(xdts, np.dot(Dr, thetar)) )
        if err[i+1] <= err[i]:
            zint[index] = 1
        else:
            zint[index] = 0
    
    # Backward finder:
    index = np.array(np.where(zint != 0))
    index = np.reshape(index,-1) # converting to 1-D array,
    # gg = index.flatten()
    # gg = np.ravel(index)
    Dr = D[:, index]
    thetar = theta[index]
    err = MSE(xdts, np.dot(Dr, thetar))
    ind = 0
    for i in range(nl-1, -1, -1):
        index = ind
        Dr = D[:, index]
        thetar = theta[index]
        err = np.append(err, MSE(xdts, np.dot(Dr, thetar)) )
        if err[ind+1] <= err[ind]:
            zint[index] = 1
        else:
            zint[index] = 0
        ind = ind + 1
    
    # for the states
    zint[[0, 1]] = [1, 1]
    return zint
    
    
"""
Sparse regression with Normal Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def sparse(xdts, Dict, MCMC, burn_in):
    # Library creation:
    D = Dict
    nl = Dict.shape[1]

    # Residual variance:
    err_var = res_var(D, xdts)

    """
    # Gibbs sampling:
    """
    # Hyper-parameters
    ap, bp = 0.1, 1 # for beta prior for p0
    av, bv = 0.5, 0.5 # inverge gamma for vs
    asig, bsig = 1e-4, 1e-4 # invese gamma for sig^2

    # Parameter Initialisation:
    p0 = np.zeros(MCMC)
    vs = np.zeros(MCMC)
    sig = np.zeros(MCMC)
    p0[0] = 0.1
    vs[0] = 10
    sig[0] = err_var

    N = len(xdts)

    # Initial latent vector
    zval = np.zeros(nl)
    zint  = latent(nl, D, xdts)
    zstore = np.transpose(np.vstack([zint]))
    zval = zint

    zval0 = zval
    vs0 = vs[0]
    mu, BSIG, Aor, index = sigmu(zval0, D, vs0, xdts)
    Sz = sum(zval)

    # Sample theta from Normal distribution
    thetar = mvrv(mu, np.dot(sig[0], BSIG))
    thetat = np.zeros(nl)
    thetat[index] = thetar
    theta = np.vstack(thetat)

    for i in range(1, MCMC):    
        if i % 50 == 0:
            print(i)
        # sample z from the Bernoulli distribution:
        zr = np.zeros(nl) # instantaneous latent vector (z_i):
        zr = zval
        for j in range(nl):
            ztemp0 = zr
            ztemp0[j] = 0
            if np.mean(ztemp0) == 0:
                PZ0 = pyzv0(xdts, N, asig, bsig)
            else:
                vst0 = vs[i-1]
                PZ0 = pyzv(D, ztemp0, vst0, N, xdts, asig, bsig)
            
            ztemp1 = zr
            ztemp1[j] = 1      
            vst1 = vs[i-1]
            PZ1 = pyzv(D, ztemp1, vst1, N, xdts, asig, bsig)
            
            zeta = PZ0 - PZ1  
            zeta = p0[i-1]/( p0[i-1] + np.exp(zeta)*(1-p0[i-1]))
            zr[j] = bern(1, p = zeta, size = None)
        
        zval = zr
        zstore = np.append(zstore, np.vstack(zval), axis = 1)
        
        # sample sig^2 from inverse Gamma:
        asiggamma = asig+0.5*N
        temp = np.matmul(np.matmul(mu.T, LA2.inv(BSIG)), mu)
        bsiggamma = bsig+0.5*(np.dot(xdts.T, xdts) - temp)
        sig[i] = 1/IG(asiggamma, 1/bsiggamma) # inverse gamma RVs
        
        # sample vs from inverse Gamma:
        avvs = av+0.5*Sz
        bvvs = bv+(np.matmul(np.matmul(thetar.T, LA2.inv(Aor)), thetar))/(2*sig[i])
        vs[i] = 1/IG(avvs, 1/bvvs) # inverse gamma RVs
        
        # sample p0 from Beta distribution:
        app0 = ap+Sz
        bpp0 = bp+nl-Sz # Here, P=nl (no. of functions in library)
        p0[i] = beta(app0, bpp0)
        # or, np.random.beta()
        
        # Sample theta from Normal distribution:
        vstheta = vs[i]
        mu, BSIG, Aor, index = sigmu(zval, D, vstheta, xdts)
        Sz = sum(zval)
        thetar = mvrv(mu, np.dot(sig[i], BSIG))
        thetat = np.zeros(nl)
        thetat[index] = thetar
        theta = np.append(theta, np.vstack(thetat), axis = 1)

    # Marginal posterior inclusion probabilities (PIP):
    zstoredrift = zstore[:, burn_in:] # discard the first 'burn_in' samples
    Zmeandrift = np.mean(zstoredrift, axis=1)

    # Post processing:
    thetadrift = theta[:, burn_in:] # discard the first 'burn_in' samples
    mutdrift = np.mean(thetadrift, axis=1)
    sigtdrift = np.cov(thetadrift, bias = False)
    
    return zstoredrift, Zmeandrift, thetadrift, mutdrift, sigtdrift
    

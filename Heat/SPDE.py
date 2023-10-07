"""
This is the utility file which contains usefull functions for:
    1. The Bayes regression,
    2. Lasso regression,
    3. SINDy,
    4. Building library,
    5. Differentiation.

Paper: Discovering stochastic partial differential equations fromlimited data
       using variational Bayes inference.
       - Yogesh Chandrakant Mathpati, Tapas Tripura, Rajdip Nayek, Souvik Chakraborty
"""

import numpy as np
import warnings
from numpy import linalg as la
from scipy.special import loggamma


def sindy(lam,D,dxdt,iteration=10): 
    Xi = np.matmul(np.linalg.pinv(D), dxdt.T) # initial guess: Least-squares
    for k in range(iteration):
        smallinds = np.where(abs(Xi) < lam)   # find small coefficients
        Xi[smallinds] = 0
        for ind in range(Xi.shape[1]):
            biginds = np.where(abs(Xi[:,ind]) > lam)
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.matmul(np.linalg.pinv(D[:, biginds[0]]), dxdt[ind, :].T) 
    return Xi


def Lasso(X0, Y, lam, w = np.array([0]), maxit = 100, normalize = 2):
    """
    Uses accelerated proximal gradient (FISTA) to solve Lasso
    argmin (1/2)*||Xw-Y||_2^2 + lam||w||_1
    """
    
    # Obtain size of X
    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    Y = Y.reshape(n,1)
    
    # Create w if none is given
    if w.size != d:
        w = np.zeros((d,1), dtype=np.complex64)
    w_old = np.zeros((d,1), dtype=np.complex64)
    
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0

    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X),2)
    
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
         
        # Update w
        z = w + iters/float(iters+1)*(w - w_old)
        w_old = w
        z = z - X.T.dot(X.dot(z)-Y)/L
        for j in range(d): w[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j])-lam/L,0]))

        # Could put in some sort of break condition based on convergence here.
    
    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(w != 0)[0]
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],Y,rcond=None)[0]

    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w
    
    
def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3
    
    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """
    
    n = u.size
    ux = np.zeros(n, dtype=np.float64)
    
    if d == 1:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-u[i-1]) / (2*dx)
        
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux
    
    if d == 2:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2
        
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux
    
    if d == 3:
        for i in range(2,n-2):
            ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3
        
        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux
    
    if d > 3:
        return FiniteDiff(FiniteDiff(u,dx,3), dx, d-3)
    

def build_linear_system(u, dt, dx, D = 3, P = 3, 
                        lam_t = None, lam_x = None, width_x = None, width_t = None,
                        deg_x = 5, deg_t = None, sigma = 2):
    """
    Constructs a large linear system to use in later regression for finding PDE.  
    This function works when we are not subsampling the data or adding in any forcing.
    Input:
        Required:
            u = data to be fit to a pde
            dt = temporal grid spacing
            dx = spatial grid spacing
        Optional:
            D = max derivative to include in rhs (default = 3)
            P = max power of u to include in rhs (default = 3)
            lam_t = penalization for L2 norm of second time derivative
                    only applies if time_diff = 'TV'
                    default = 1.0/(number of timesteps)
            lam_x = penalization for L2 norm of (n+1)st spatial derivative
                    default = 1.0/(number of gridpoints)
            width_x = number of points to use in polynomial interpolation for x derivatives
                      or width of convolutional smoother in x direction if using FDconv
            width_t = number of points to use in polynomial interpolation for t derivatives
            deg_x = degree of polynomial to differentiate x
            deg_t = degree of polynomial to differentiate t
            sigma = standard deviation of gaussian smoother
                    only applies if time_diff = 'FDconv'
                    default = 2
    Output:
        ut = column vector of length u.size
        R = matrix with ((D+1)*(P+1)) of column, each as large as ut
        rhs_description = description of what each column in R is
    """

    n, m = u.shape

    if width_x == None: width_x = n//10
    if width_t == None: width_t = m//10
    if deg_t == None: deg_t = deg_x

    # If we're using polynomials to take derviatives, then we toss the data around the edges.
    m2 = m
    offset_t = 0
    n2 = n
    offset_x = 0

    ########################
    # First take the time derivaitve for the left hand side of the equation
    ########################
    ut = np.zeros((n2,m2), dtype=np.float64)
    for i in range(n2):
        ut[i,:] = FiniteDiff(u[i + offset_x,:],dt,1)
    
    ut = np.reshape(ut, (n2*m2,1), order='F')

    ########################
    # Now form the rhs one column at a time, and record what each one is
    ########################
    u2 = u[offset_x:n-offset_x,offset_t:m-offset_t]
    Theta = np.zeros((n2*m2, (D+1)*(P+1)), dtype=np.float64)
    ux = np.zeros((n2,m2), dtype=np.float64)
    rhs_description = ['' for i in range((D+1)*(P+1))]

    for d in range(D+1):

        if d > 0:
            for i in range(m2):
                ux[:,i] = FiniteDiff(u[:,i+offset_t],dx,d)
        else: ux = np.ones((n2,m2), dtype=np.float64) 
            
        for p in range(P+1):
            Theta[:, d*(P+1)+p] = np.reshape(np.multiply(ux, np.power(u2,p)), (n2*m2), order='F')

            if p == 1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u'
            elif p>1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u^' + str(p)
            if d > 0: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+\
                                                   'u_{' + ''.join(['x' for _ in range(d)]) + '}'

    return ut, Theta, rhs_description

def print_pde(w, rhs_description, ut = 'u_t'):
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            pde = pde + "(%05f %+05fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            first = False
    print(pde)

def Variational_Bayes_Code(X, y, initz0, tol, verbosity):
    if (len(X) == 0 or len(y) == 0):
        raise Exception('X and or y is missing')

    if (len(X) != len(y)):
        raise Exception('Number of observations do not match')

    N = len(X)
    # Prior parameters of noise variance (Inverse Gamma dist)
    A = 1e-4
    B = 1e-4
    vs = 10
    tau0 = 1000

    if (len(initz0) == 0):
        raise Exception('No initial value of z found')
    else:
        p0 = expit(-0.5*(np.sqrt(N)))

        # Adding the intercept indicator variable (slightly less than 1 to prevent log(0) values)
        # initz = np.hstack((1,initz0))
        initz = initz0
        DS,LLcvg  = run_VB2(X, y, vs, A, B, tau0, p0, initz, tol, verbosity)

    out_vb   = DS
    a = DS['zmean'] > 0.5
    count = 0
    modelIdx = []
    for i in a:
        if i == True:
            modelIdx.append(count)
        count += 1

    modelIdx = np.setdiff1d(modelIdx,0)
    out_vb['modelIdx'] = modelIdx-1
    out_vb['Zmed'] = DS['zmean'][modelIdx]
    out_vb['Wsel'] = DS['wmean'][modelIdx]
    out_vb['Wcov'] = DS['wCOV'][modelIdx, modelIdx]
    out_vb['sig2'] = DS['sig2']

    return out_vb


def run_VB2(Xc, yc, vs, A, B, tau0, p0, initz, tol, verbosity):
    """This function is the implementation of VB from John T. Ormerod paper (2014)
       This implementation uses slab scaling by noise variance
       vs    : treated as a constant
       A,B   : constants of the IG prior over noise variance
       tau0  : Expected value of (sigma^{-2})
       p0    : inclusion probablility
       initz : Initial value of z
       Xc    : Centered and standardized dictionary except the first column
       yc    : Centered observations """
    
    DS = {}
    Lambda    = logit(p0)
    iter_     = 0
    max_iter  = 100
    LL        = np.zeros(max_iter)
    zm        = np.reshape(initz,(-1))
    taum      = tau0
    invVs     = 1/vs
    initz0 = initz

    X = Xc
    y = yc
    XtX = (X.T) @ X
    XtX = 0.5*(XtX + (XtX).T)
    Xty = (X.T) @ y
    yty = (y.T) @ y

    eyep = np.eye(len(XtX))
    [N,p] = X.shape
    allidx = np.arange(p)
    zm[0] = 1                                                                   # Always include the intercept
    Abar    = (A + 0.5*N + 0.5*p)
    converged = 0

    while (converged==0):
        if (iter_==100):
            break

        Zm       = np.diag(zm)
        Omg      = (np.reshape(zm,(-1,1)) @ np.reshape(zm,(1,-1))) + (Zm @ (eyep-Zm))
        # Update the mean and covariance of the coefficients given mean of z
        term1    = XtX * Omg                                                       # elementwisw multiplication
        invSigma = taum * (term1 + invVs * eyep)
        invSigma = 0.5*(invSigma + invSigma.T)                                     # symmetric
        Sigma    = la.inv(invSigma) @ eyep
        mu       = taum * (Sigma @ Zm @ Xty)                                       # @ ---> matrix multiplication

        # Update tau related to sigma
        term2    = 2 * Xty @ Zm @ mu
        term3    =  np.reshape(mu,(len(initz0),1)) @ np.reshape(mu,(1,len(initz0)))+ Sigma
        term4    = yty - term2 + np.trace((term1 + invVs * eyep) @ term3)
        s        = B + 0.5*term4

        if s<0:
            warnings.warn('s turned out be less than 0. Taking absolute value')
            s = B + 0.5*abs(term4)

        taum     = Abar / s
        zstr   = zm
        order   = np.setdiff1d(np.random.permutation(p), 0, assume_unique=True)
        for j in order:
            muj     = mu[j]
            sigmaj  = Sigma[j,j]

            remidx  = np.setdiff1d(allidx,j)
            mu_j    = mu[remidx]
            Sigma_jj= Sigma[remidx,j]
            etaj    = (Lambda - 0.5 * taum * ((muj**2 + sigmaj) * XtX[j,j])
                       + taum * np.reshape(X[:,j],(1,-1)) @ (np.reshape(y,(-1,1))*muj
                        -X[:,remidx] @ np.diag(zstr[remidx]) @ ((mu_j * muj + Sigma_jj).reshape(-1,1))))
            zstr[j] = expit(etaj)

        zm = zstr

        # Calculate marginal log-likelihood
        LL[iter_] = ( 0.5*p
                    - 0.5*N*np.log(2*np.pi)
                    + 0.5*p*np.log(invVs)
                    + A*np.log(B)
                    - loggamma(A)
                    + loggamma(Abar)
                    - Abar*np.log(s)
                    + 0.5*np.log(la.det(Sigma))
                    + np.nansum(zm*(np.log(p0) - np.log(zm)))
                    + np.nansum((1-zm)*(np.log(1-p0) - np.log(1-zm))))

        if(verbosity):
            print(f'Iteration = {iter_}  log(Likelihood) = {LL[iter_]}')

        if(iter_>1):
            cvg = LL[iter_] - LL[iter_-1]

            if (cvg < 0 and verbosity):
                print('OOPS!  log(like) decreasing!!')
            elif (cvg<tol or iter_> max_iter):
                converged = 1
                LL = LL[0:iter_]

        iter_    = iter_ + 1
    DS['zmean'] = zm
    DS['wmean'] = mu
    DS['wCOV'] = Sigma
    DS['sig2'] = 1/taum
    LLcvg    = LL[-1]
    return DS, LLcvg

def logit(C):
    logitC = np.log(C) - np.log(1-C)
    return logitC

def expit(C):
    expitC = 1./(1 + np.exp(-C))
    return expitC


"""
This code uses the Gibbs sampler for drawing samples from the posterior
distribution of the parameters of
       - Heat equation
       (!!! Takes a huge amount of Time !!!)

Paper: Discovering stochastic partial differential equations fromlimited data
       using variational Bayes inference.
       - Yogesh Chandrakant Mathpati, Tapas Tripura, Rajdip Nayek, Souvik Chakraborty
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import SPDE
import time
import fun_gibbs

# In[]:

data = scipy.io.loadmat('data/Heat_dx_64m_500t.mat')
U = data['sol'][..., :1000]

dx = 20/(U.shape[0]-1)
dt = 1/(U.shape[1]-1)

Dictt = []
for i in range(U.shape[2]):
    if i%50==0:
        print(i)
    U_t,Di,description = SPDE.build_linear_system(U[:,:-1,i], dt, dx, D = 5, P = 6 )
    Dictt.append(Di)
    
Dictt = np.array(Dictt)
Dict2 = np.mean(Dictt,axis=0)

# In[]:
description


# In[]:

Xi_act = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.]

xdt = data['xdt'].reshape(-1,order='F')
xdts = data['xdiff'].reshape(-1,order='F')

# In[]:
    
""" Identification of Drift """
np.random.seed(10)

nl = len(Xi_act)
y = xdt.reshape(-1)
MCMC, burn_in = 100, 50

start_time = time.time()
zstoredrift, Zmeandrift, thetadrift, mutdrift, sigtdrift = fun_gibbs.sparse(y, Dict2, MCMC, burn_in)

end_time = time.time()
print("Elapsed time: ",end_time-start_time)


# In[21]:

""" Identification of Diffusion """
np.random.seed(10)

y = xdts.reshape(-1)
MCMC, burn_in = 100, 50

start_time = time.time()
zstorediff, Zmeandiff, thetadiff, mutdiff, sigtdiff = fun_gibbs.sparse(y, Dict2, MCMC, burn_in)

end_time = time.time()
print("Elapsed time: ",end_time-start_time)

# In[ ]:

mutdrift[Zmeandrift < 0.5] = 0    
l2_error_Gibbs = np.linalg.norm(np.array(Xi_act)-mutdrift)
print('L2 error for sindy:   ', l2_error_Gibbs)

# In[ ]:
    
# np.savez('Gibbs_result_Heat', zstoredrift, Zmeandrift, thetadrift, mutdrift, sigtdrift,
#          zstorediff, Zmeandiff, thetadiff, mutdiff, sigtdiff)

# In[ ]:

Zmeandrift


# In[ ]:

Zmeandiff


# In[ ]:


"""
Plotting Command
"""

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

figure1=plt.figure(figsize = (14, 10))
plt.subplot(211)
xr = np.array(range(nl))
plt.stem(xr, Zmeandrift, linefmt='b', markerfmt='ob', basefmt='w', label='Drift')
plt.stem(xr+0.1, Zmeandiff, linefmt='r', markerfmt='or', basefmt='w', label='Diffusion')
plt.ylabel('PIP', fontweight='bold');
plt.xlabel('Basis functions', fontweight='bold')
plt.axhline(y= 1.0, color='grey', linestyle='-.')
plt.axhline(y= 0.5, color='brown', linestyle='-.')
plt.ylim(0,1.05)
plt.legend()
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold')


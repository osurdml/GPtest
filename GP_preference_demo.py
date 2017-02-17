# Simple 1D GP classification example
import numpy as np
import matplotlib.pyplot as plt
import GPpref
import scipy.optimize as op

log_hyp = np.log([0.1,1.0,0.1]) # length_scale, sigma_f, sigma_probit
np.random.seed(1)

n_train = 20
true_sigma = 0.05
delta_f = 1e-5

# Define polynomial function to be modelled
def true_function(x):
    y = np.sin(x*2*np.pi + np.pi/4) + 0.2
    return y

# Define noisy observation function
def obs_function(x, sigma):
    fx = true_function(x)
    noise = np.random.normal(scale=sigma, size=x.shape)
    return fx + noise
    
def noisy_preference_rank(uv, sigma):
    fuv = obs_function(uv,sigma)
    y = -1*np.ones((fuv.shape[0],1),dtype='int')
    y[fuv[:,1] > fuv[:,0]] = 1
    return y, fuv

# Main program
# Plot true function
x_plot = np.linspace(0.0,1.0,101,dtype='float')
y_plot = true_function(x_plot)

# Training data - this is a bit weird, but we sample x values, then the uv pairs
# are actually indexes into x, because it is easier computationally. You can 
# recover the actual u,v values using x[ui],x[vi]
x_train = np.random.random((2*n_train,1))
uvi_train = np.random.choice(range(2*n_train), (n_train,2), replace=False)
uv_train = x_train[uvi_train][:,:,0]

# Get noisy observations f(uv) and corresponding ranks y_train
y_train, fuv_train = noisy_preference_rank(uv_train,true_sigma)

hf,ha = plt.subplots(1,1)
ha.plot(x_plot,y_plot,'r-')
for uv,fuv,y in zip(uv_train, fuv_train, y_train):
    ha.plot(uv, fuv, 'b-')
    ha.plot(uv[(y+1)/2],fuv[(y+1)/2],'k+')

ha.set_title('Training data')
ha.set_ylabel('f(x)')
ha.set_xlabel('x')


# GP hyperparameters
# note: using log scaling to aid learning hyperparameters with varied magnitudes

prefGP = GPpref.PreferenceGaussianProcess(x_train, uvi_train, y_train, delta_f=delta_f)

# Pseudocode:
# FOr a set of hyperparameters, return log likelihood that can be used by an optimiser
theta0 = log_hyp

# log_hyp = op.fmin(prefGP.calc_nlml,theta0)
f,lml = prefGP.calc_laplace(log_hyp)

ha.plot(x_train, f, 'g^')
plt.show()

# Simple 1D GP classification example
import numpy as np
import matplotlib.pyplot as plt
import GPpref
import scipy.optimize as op

log_hyp = np.log([0.1,1.0,0.1,10.0]) # length_scale, sigma_f, sigma_probit, v_beta
np.random.seed(3)

n_rel_train = 10
n_abs_train = 5
true_sigma = 0.05
delta_f = 1e-5

# Define polynomial function to be modelled
def true_function(x):
    y = (np.sin(x*2*np.pi + np.pi/4) + 1)/2
    #y = np.sin(x*2.0*np.pi + np.pi/4.0)
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
hf,ha = plt.subplots(1,1)
ha.plot(x_plot,y_plot,'r-')


# Training data - this is a bit weird, but we sample x values, then the uv pairs
# are actually indexes into x, because it is easier computationally. You can 
# recover the actual u,v values using x[ui],x[vi]
if n_rel_train > 0:
    x_train = np.random.random((2*n_rel_train,1))
    uvi_train = np.random.choice(range(2*n_rel_train), (n_rel_train,2), replace=False)
    uv_train = x_train[uvi_train][:,:,0]

    # Get noisy observations f(uv) and corresponding ranks y_train
    y_train, fuv_train = noisy_preference_rank(uv_train, true_sigma)
    for uv,fuv,y in zip(uv_train, fuv_train, y_train):
        ha.plot(uv, fuv, 'b-')
        ha.plot(uv[(y+1)/2],fuv[(y+1)/2],'k+')

else:
    x_train = np.zeros((0,1))
    uvi_train = np.zeros((0,2))
    uv_train = np.zeros((0,2))
    y_train = np.zeros((0,1))
    fuv_train = np.zeros((0,2))

x_abs_train = np.random.random((n_abs_train,1))
#y_abs_train = obs_function(x_abs_train, true_sigma)
y_abs_train = np.clip(obs_function(x_abs_train, true_sigma), 0.01, .99)

ha.plot(x_abs_train, y_abs_train, 'r+')

ha.set_title('Training data')
ha.set_ylabel('f(x)')
ha.set_xlabel('x')

# GP hyperparameters
# note: using log scaling to aid learning hyperparameters with varied magnitudes

print "Data"
# print x_train
print x_abs_train
# print y_train
print y_abs_train
# print uvi_train

prefGP = GPpref.PreferenceGaussianProcess(x_train, uvi_train, x_abs_train,  y_train, y_abs_train, delta_f=delta_f)

# Pseudocode:
# FOr a set of hyperparameters, return log likelihood that can be used by an optimiser
theta0 = log_hyp

# log_hyp = op.fmin(prefGP.calc_nlml,theta0)
#f,lml = prefGP.calc_laplace(log_hyp)
f = prefGP.calc_laplace(log_hyp)

# New y's are expectations from Beta distribution. E(X) = alpha/(alpha+beta)
alph = prefGP.abs_likelihood.alpha(f)
bet = prefGP.abs_likelihood.beta(f)
Ey = alph/(alph+bet)

if x_train.shape[0]>0:
    ha.plot(x_train, Ey[0:x_train.shape[0]], 'g^')
if x_abs_train.shape[0]>0:
    ha.plot(x_abs_train, Ey[x_train.shape[0]:], 'r^')
plt.show()

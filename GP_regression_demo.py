# Simple 1D GP regression example
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import GPr
np.random.seed(0)

# Define polynomial function to be modelled
def true_function(x):
    c = np.array([3,5,-9,-3,2],float)
    y = np.polyval(c,x)
    return y

# Define noisy observation function
def obs_function(x):
    y = true_function(x) + np.random.normal(0,0.1,len(x))
    return y

# Main program
# Plot true function
x_plot = np.arange(0,1,0.01,float)
y_plot = true_function(x_plot)
plt.figure()
plt.plot(x_plot,y_plot,'k-')

# Training data
x_train = np.random.random(10)
y_train = obs_function(x_train)
plt.plot(x_train,y_train,'rx')

# Test data
x_test = np.arange(0,1,0.01,float)

# GP hyperparameters
# note: using log scaling to aid learning hyperparameters with varied magnitudes
log_hyp = np.log([1,1,0.1])
mean_hyp = 0
like_hyp = 0

# Initialise GP for hyperparameter training
initGP = GPr.GaussianProcess(log_hyp,mean_hyp,like_hyp,"SE","zero","zero",x_train,y_train)

# Run optimisation routine to learn hyperparameters
opt_log_hyp = op.fmin(initGP.compute_likelihood,log_hyp)

# Learnt GP with optimised hyperparameters
optGP = GPr.GaussianProcess(opt_log_hyp,mean_hyp,like_hyp,"SE","zero","zero",x_train,y_train)
y_test,cov_y = optGP.compute_prediction(x_test)

# Plot true and modelled functions
plt.plot(x_test,y_test,'b-')
plt.plot(x_test,y_test+np.sqrt(cov_y),'g--')
plt.plot(x_test,y_test-np.sqrt(cov_y),'g--')
plt.show()

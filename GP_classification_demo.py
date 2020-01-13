# Simple 1D GP classification example
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from gp_tools import GPr
from scipy.special import ndtr as std_norm_cdf
np.random.seed(0)

# Define polynomial function to be modelled
def true_function(x):
    y = np.sin(x*2*np.pi + np.pi/4) + 0.2
    return y

def class_prob(x):
    y = std_norm_cdf(true_function(x))
    return y
    
# Define noisy observation function
def obs_function(x):
    fx = class_prob(x)
    return [np.random.uniform() < f for f in fx]

# Main program
# Plot true function
x_plot = np.arange(0,1,0.01,float)
y_plot = true_function(x_plot)
y_class = class_prob(x_plot)
hf,ha = plt.subplots(1,2)
ha[0].plot(x_plot,y_plot,'k-')
ha[0].set_title('True function')
ha[0].set_ylabel('f(x)')
ha[1].plot(x_plot,y_class,'k-')
ha[1].set_title('Class probability')
ha[1].set_ylabel('\pi(x)')

# Training data
x_train = np.random.random(20)
y_train = obs_function(x_train)
ha[1].plot(x_train,y_train,'rx')
ha[1].set_ylim([-0.1,1.1])

# Test data
x_test = np.arange(0,1,0.01,float)

# GP hyperparameters
# note: using log scaling to aid learning hyperparameters with varied magnitudes
log_hyp = np.linspace(0.0, 1.0, 101)
mean_hyp = 0
like_hyp = 0
l
plt.show()

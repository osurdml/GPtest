# Simple 1D GP regression example
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import GPr
import plot_tools as ptt
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

f_sigma = 0.15
n_samples = 6
r_seed = 2
optimise_hp = False

# Define polynomial function to be modelled
def true_function(x):
    # c = np.array([3,5,-9,-3,2],float)
    # y = np.polyval(c,x)
    y = np.sin(x*2*np.pi)
    return y


# Define noisy observation function
def obs_function(x, sigma):
    y = true_function(x) + np.random.normal(0, sigma, len(x))
    return y


# Main program
# Plot true function
x_plot = np.arange(-0.5, 1.5, 0.01, dtype='float')
fx_plot = true_function(x_plot)
plt.figure()
plt.plot(x_plot, fx_plot, 'k-')

# Training data
np.random.seed(r_seed)
x_train = 0.6*np.random.random(n_samples)+0.2
y_train = obs_function(x_train, f_sigma)
plt.plot(x_train,y_train,'rx')

# Test data
x_test = x_plot

# GP hyperparameters
# note: using log scaling to aid learning hyperparameters with varied magnitudes
log_hyp = np.log([0.36, 1.3, 0.2])
mean_hyp = 0
like_hyp = 0

# Initialise GP for hyperparameter training
initGP = GPr.GaussianProcess(log_hyp, mean_hyp, like_hyp, "SE", "zero", "zero", x_train, y_train)

# Run optimisation routine to learn hyperparameters
if optimise_hp:
    opt_log_hyp = op.fmin(initGP.compute_likelihood, log_hyp)
else:
    opt_log_hyp = log_hyp

# Learnt GP with optimised hyperparameters
optGP = GPr.GaussianProcess(opt_log_hyp,mean_hyp,like_hyp,"SE","zero","zero",x_train,y_train)
fhat_test, fhat_var = optGP.compute_prediction(x_test)

h_fig,h_ax = plt.subplots()
h_fx, patch_fx = ptt.plot_with_bounds(h_ax, x_plot, fx_plot, f_sigma, c=ptt.lines[0])
h_y, = h_ax.plot(x_train, y_train, 'rx', mew=1.0, ms=8)
h_fhat, patch_fhat = ptt.plot_with_bounds(h_ax, x_plot, fhat_test, np.sqrt(fhat_var.diagonal()), c=ptt.lines[1], ls='--')
# h_pp = h_ax.plot(x_plot, y_post.T, c='grey', ls='--', lw=0.8)

h_ax.set_ylim([-3, 3])
gp_str = '$\hat{{f}}(x) \sim \mathcal{{GP}}(l={0:0.2f}, \sigma_f={1:0.2f}, \sigma_n={2:0.2f})$'
gp_l, gp_sigf, gp_sign = optGP.covFun.hyp
gp_str = gp_str.format(gp_l, gp_sigf, gp_sign)
h_ax.legend((h_fx, h_y, h_fhat),('$f(x)$', '$y \sim \mathcal{{N}}(f(x), {0:0.1f}^2)$'.format(f_sigma), gp_str), loc='best')
h_ax.set_xlabel('$x$')
#h_fig.savefig('fig/regression_example.pdf', bbox_inches='tight', transparent='true')


# Plot true and modelled functions
# plt.plot(x_test,y_test,'b-')
# plt.plot(x_test,y_test+np.sqrt(cov_y),'g--')
# plt.plot(x_test,y_test-np.sqrt(cov_y),'g--')
plt.show(block=False)

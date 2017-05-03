# Simple 1D GP classification example
import numpy as np
import matplotlib.pyplot as plt
import GPpref
import scipy.optimize as op
import plot_tools as ptt
import test_data
# from scipy.stats import beta
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)
np.random.seed(0)

train_hyper = False
use_test_data = False
verbose = True

#log_hyp = np.log([0.2, 0.5, 0.1, 1.0, 10.0]) # length_scale/s, sigma_f, sigma_n_abs, sigma_beta, v_beta
# log_hyp = np.log([0.07, 1.0, 0.25, 1.0, 28.1])
# log_hyp = np.log([0.065, 0.8, 0.8, 0.8, 20.0])
# log_hyp = np.log([0.05, 1.5, 0.09, 2.0, 50.0])
log_hyp = np.log([0.02, 0.6, 0.2, 0.8, 60.0])

n_rel_train = 100
n_abs_train = 100

rel_sigma = 0.05
delta_f = 1e-5

beta_sigma = 0.8
beta_v = 80.0

n_xplot = 101
n_mcsamples = 1000
n_ysamples = 101

# Define polynomial function to be modelled
#true_function = test_data.zero_fun

# random_wave = test_data.VariableWave(amp_range=[0.6, 1.2], f_range=[5.0, 10.0], off_range=[0.2, 0.8],
#                                      damp_range=[30.0, 100.0])
# #random_wave.set_values(a=1.2, f=6.0, o=.2, d=20.0)
# random_wave.print_values()

random_wave = test_data.MultiWave(amp_range=[0.6, 1.2], f_range=[10.0, 30.0], off_range=[0.1, 0.9],
                                     damp_range=[250.0, 350.0], n_components=3)
random_wave.print_values()
true_function = random_wave.out

# true_function = test_data.damped_wave

rel_obs_fun = GPpref.RelObservationSampler(true_function, GPpref.PrefProbit(sigma=rel_sigma))
abs_obs_fun = GPpref.AbsObservationSampler(true_function, GPpref.AbsBoundProbit(sigma=beta_sigma, v=beta_v))

# Main program
# True function
x_plot = np.linspace(0.0,1.0,n_xplot,dtype='float')
x_test = np.atleast_2d(x_plot).T
f_true = abs_obs_fun.f(x_test)
mu_true = abs_obs_fun.mean_link(x_test)
mc_samples = np.random.normal(size=n_mcsamples)
abs_y_samples = np.atleast_2d(np.linspace(0.01, 0.99, n_ysamples)).T
p_abs_y_true = abs_obs_fun.observation_likelihood_array(x_test, abs_y_samples)
p_rel_y_true = rel_obs_fun.observation_likelihood_array(x_test)

# Training data - this is a bit weird, but we sample x values, then the uv pairs
# are actually indexes into x, because it is easier computationally. You can 
# recover the actual u,v values using x[ui],x[vi]
if use_test_data:
    x_rel, uvi_rel, uv_rel, y_rel, fuv_rel, x_abs, y_abs, mu_abs = test_data.data1()
else:
    x_rel, uvi_rel, uv_rel, y_rel, fuv_rel = rel_obs_fun.generate_n_observations(n_rel_train)
    x_abs, y_abs, mu_abs = abs_obs_fun.generate_n_observations(n_abs_train)

# Construct GP object
prefGP = GPpref.PreferenceGaussianProcess(x_rel, uvi_rel, x_abs, y_rel, y_abs,
                                          delta_f=delta_f,
                                          rel_likelihood=GPpref.PrefProbit(),
                                          abs_likelihood=GPpref.AbsBoundProbit(), verbose=verbose)

# If training hyperparameters, use external optimiser
if train_hyper:
    log_hyp = op.fmin(prefGP.calc_nlml,log_hyp)

f = prefGP.calc_laplace(log_hyp)
prefGP.print_hyperparameters()

# Latent predictions
fhat, vhat = prefGP.predict_latent(x_test)

# Expected values
E_y = prefGP.abs_posterior_mean(x_test, fhat, vhat)

# Posterior likelihoods (MC sampled for absolute)
p_abs_y_post = prefGP.abs_posterior_likelihood(abs_y_samples, fhat=fhat, varhat=vhat, normal_samples=mc_samples)
p_rel_y_post = prefGP.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)


# Plot true functions
fig_t, (ax_t_l, ax_t_a, ax_t_r) = ptt.true_plots(x_test, f_true, mu_true, rel_sigma,
                                                 abs_y_samples, p_abs_y_true, p_rel_y_true,
                                                 x_abs, y_abs, uv_rel, fuv_rel, y_rel,
                                                 t_l=r'True latent function, $f(x)$')

# Posterior estimates
fig_p, (ax_p_l, ax_p_a, ax_p_r) = \
    ptt.estimate_plots(x_test, f_true, mu_true, fhat, vhat, E_y, rel_sigma,
                       abs_y_samples, p_abs_y_post, p_rel_y_post,
                       x_abs, y_abs, uv_rel, fuv_rel, y_rel,
                       t_a=r'Posterior absolute likelihood, $p(y | \mathcal{Y}, \theta)$',
                       t_r=r'Posterior relative likelihood $P(x_0 \succ x_1 | \mathcal{Y}, \theta)$')

plt.show()


## SCRAP
# p_y = np.zeros((n_ysamples, n_xplot))
# y_samples = np.linspace(0.01, 0.99, n_ysamples)
# iny = 1.0/n_ysamples
# E_y2 = np.zeros(n_xplot)
#
# normal_samples = np.random.normal(size=n_mcsamples)
# for i,(fstar,vstar) in enumerate(zip(fhat, vhat.diagonal())):
#     f_samples = normal_samples*vstar+fstar
#     aa, bb = prefGP.abs_likelihood.get_alpha_beta(f_samples)
#     p_y[:, i] = [iny*np.sum(beta.pdf(yj, aa, bb)) for yj in y_samples]
#     p_y[:, i] /= np.sum(p_y[:, i])
#     E_y2[i] = np.sum(np.dot(y_samples, p_y[:, i]))
# Simple 1D GP classification example
import numpy as np
import matplotlib.pyplot as plt
import GPpref
import scipy.optimize as op
import plot_tools as ptt
import test_data
import yaml
# from scipy.stats import beta
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

train_hyper = False
use_test_data = False # test_data.data3 #
verbose = 2

with open('./data/ordinal_test.yaml', 'rt') as fh:
    wave = yaml.safe_load(fh)

try:
    np.random.seed(wave['statrun_params']['randseed'])
except KeyError:
    np.random.seed(0)
random_wave = test_data.MultiWave(**wave['wave_params'])
log_hyp = np.log(wave['hyperparameters'])

n_rel_train = 10
n_abs_train = 20

n_xplot = 101
n_posterior_samples = 3

random_wave.print_values()
true_function = random_wave.out

rel_obs_fun = GPpref.RelObservationSampler(true_function, wave['GP_params']['rel_likelihood'], wave['rel_obs_params'])
abs_obs_fun = GPpref.AbsObservationSampler(true_function, wave['GP_params']['abs_likelihood'], wave['abs_obs_params'])

# True function
x_plot = np.linspace(0.0,1.0,n_xplot,dtype='float')
x_test = np.atleast_2d(x_plot).T
f_true = abs_obs_fun.f(x_test)
mu_true = abs_obs_fun.mean_link(x_test)
abs_y_samples = abs_obs_fun.l.y_list
p_abs_y_true = abs_obs_fun.observation_likelihood_array(x_test, abs_y_samples)
p_rel_y_true = rel_obs_fun.observation_likelihood_array(x_test)

# Training data - this is a bit weird, but we sample x values, then the uv pairs
# are actually indexes into x, because it is easier computationally. You can 
# recover the actual u,v values using x[ui],x[vi]
if use_test_data:
    x_rel, uvi_rel, uv_rel, y_rel, fuv_rel, x_abs, y_abs, mu_abs = use_test_data()
else:
    x_rel, uvi_rel, uv_rel, y_rel, fuv_rel = rel_obs_fun.generate_n_observations(n_rel_train)
    x_abs, y_abs, mu_abs = abs_obs_fun.generate_n_observations(n_abs_train)

# Construct GP object
wave['GP_params']['verbose'] = verbose
prefGP = GPpref.PreferenceGaussianProcess(x_rel, uvi_rel, x_abs, y_rel, y_abs, **wave['GP_params'])

prefGP.set_hyperparameters(log_hyp)
# If training hyperparameters, use external optimiser
if train_hyper:
    log_hyp = op.fmin(prefGP.calc_nlml,log_hyp)

f = prefGP.calc_laplace(log_hyp)
prefGP.print_hyperparameters()

# Latent predictions
fhat, vhat = prefGP.predict_latent(x_test)

# Posterior likelihoods
p_abs_y_post, E_y = prefGP.abs_posterior_likelihood(abs_y_samples, fhat=fhat, varhat=vhat)
p_rel_y_post = prefGP.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)


# Plot true functions
fig_t, (ax_t_l, ax_t_a, ax_t_r) = ptt.true_plots(x_test, f_true, mu_true, wave['rel_obs_params']['sigma'],
                                                 abs_y_samples, p_abs_y_true, p_rel_y_true,
                                                 t_l=r'True latent function, $f(x)$')

# Posterior estimates
fig_p, (ax_p_l, ax_p_a, ax_p_r) = \
    ptt.estimate_plots(x_test, f_true, mu_true, fhat, vhat, E_y, wave['rel_obs_params']['sigma'],
                       abs_y_samples, p_abs_y_post, p_rel_y_post,
                       x_abs, y_abs, uv_rel, fuv_rel, y_rel, n_posterior_samples=n_posterior_samples,
                       t_a=r'Posterior absolute likelihood, $p(u | \mathcal{Y}, \theta)$',
                       t_r=r'Posterior relative likelihood $P(x_0 \succ x_1 | \mathcal{Y}, \theta)$')

wrms = test_data.wrms(mu_true, E_y)
wrms2 = test_data.wrms_misclass(mu_true, E_y)
p_err = test_data.rel_error(mu_true, p_rel_y_true, E_y, p_rel_y_post, weight=False)
print "WRMS: {0:0.3f}, WRMS_MC: {1:0.3f}, p_err: {2:0.3f}".format(wrms, wrms2, p_err)

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
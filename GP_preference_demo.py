# Simple 1D GP classification example
import numpy as np
import matplotlib.pyplot as plt
import GPpref
import scipy.optimize as op
import plot_tools as ptt
# from scipy.stats import beta
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

train_hyper = True

log_hyp = np.log([0.2, 0.5, 0.1, 1.0, 10.0]) # length_scale/s, sigma_f, sigma_n_abs, sigma_beta, v_beta
np.random.seed(1)

n_rel_train = 30
n_abs_train = 30
rel_sigma = 0.2
delta_f = 1e-5

beta_sigma = 0.8
beta_v = 20.0

n_xplot = 101
n_mcsamples = 1000
n_ysamples = 101
marker_options = {'mec':'k', 'mew':0.5}

# Define polynomial function to be modelled
def true_function(x):
    y = np.cos(6 * np.pi * (x - 0.5)) * np.exp(-10 * (x - 0.5) ** 2)
    #y = (np.sin(x*2*np.pi + np.pi/4))/1.2
    #y = np.sin(x*2.0*np.pi + np.pi/4.0)
    return y


rel_obs_fun = GPpref.RelObservationSampler(true_function, GPpref.PrefProbit(sigma=rel_sigma))
abs_obs_fun = GPpref.AbsObservationSampler(true_function, GPpref.AbsBoundProbit(sigma=beta_sigma, v=beta_v))

# Main program
# True function
x_plot = np.linspace(0.0,1.0,n_xplot,dtype='float')
x_test = np.atleast_2d(x_plot).T

# Training data - this is a bit weird, but we sample x values, then the uv pairs
# are actually indexes into x, because it is easier computationally. You can 
# recover the actual u,v values using x[ui],x[vi]
if n_rel_train > 0:
    x_train = np.random.random((2*n_rel_train,1))
    uvi_train = np.random.choice(range(2*n_rel_train), (n_rel_train,2), replace=False)
    uv_train = x_train[uvi_train][:,:,0]

    # Get labels (y), and noisy observations f(uv) and corresponding ranks y_train
    y_train, fuv_train = rel_obs_fun.generate_observations(uv_train)

else:
    x_train = np.zeros((0,1))
    uvi_train = np.zeros((0,2))
    uv_train = np.zeros((0,2))
    y_train = np.zeros((0,1))
    fuv_train = np.zeros((0,2))

# For absolute points, get observation y and beta mean mu
x_abs_train = np.random.random((n_abs_train,1))
y_abs_train, mu_abs_train = abs_obs_fun.generate_observations(x_abs_train)

prefGP = GPpref.PreferenceGaussianProcess(x_train, uvi_train, x_abs_train,  y_train, y_abs_train,
                                          delta_f=delta_f,
                                          rel_likelihood=GPpref.PrefProbit(),
                                          abs_likelihood=GPpref.AbsBoundProbit())

# If training hyperparameters, use external optimiser
if train_hyper:
    log_hyp = op.fmin(prefGP.calc_nlml,log_hyp)

f = prefGP.calc_laplace(log_hyp)
prefGP.print_hyperparameters()

# Latent predictions
fhat, vhat = prefGP.predict_latent(x_test)

# Expected values
E_y = prefGP.abs_posterior_mean(x_test, fhat, vhat)

# Absolute posterior likelihood (MC sampled)
mc_samples = np.random.normal(size=n_mcsamples)
abs_y_samples = np.atleast_2d(np.linspace(0.01, 0.99, n_ysamples)).T
p_y = prefGP.abs_posterior_likelihood(abs_y_samples, fhat=fhat, varhat=vhat, normal_samples=mc_samples)

# PLOTTING

# Plot true function, likelihoods and observations
fig_t, (ax_t_l, ax_t_a, ax_t_r) = ptt.plot_setup_2d(t_l=r'True latent function, $f(x)$')

# True latent
f_true = abs_obs_fun.f(x_test)
ptt.plot_with_bounds(ax_t_l, x_test, f_true, rel_sigma, c=ptt.lines[0])

# True absolute likelihood
p_abs_y_true = abs_obs_fun.observation_likelihood_array(x_test, abs_y_samples)
abs_extent = [x_test[0,0], x_test[-1,0], abs_y_samples[0,0], abs_y_samples[-1,0]]
h_pat = ax_t_a.imshow(p_abs_y_true, origin='lower', extent=abs_extent)
if x_abs_train.shape[0]>0:
    ax_t_a.plot(x_abs_train, y_abs_train, 'w+')
mu_true = abs_obs_fun.mean_link(x_test)
h_yt, = ax_t_a.plot(x_test, mu_true, c=ptt.lines[0])
ax_t_a.legend([h_yt], ['$E(y|f(x))$'])
fig_t.colorbar(h_pat, ax=ax_t_a)
    
# True relative likelihood
p_rel_y_true = rel_obs_fun.observation_likelihood_array(x_test)
rel_y_extent = [x_test.min(), x_test.max(), x_test.min(), x_test.max()]
h_prt = ptt.plot_relative_likelihood(ax_t_r, p_rel_y_true, extent=rel_y_extent)
class_icons = ['ko','wo']
if x_train.shape[0] > 0:
    for uv, fuv, y in zip(uv_train, fuv_train, y_train):
        ax_t_r.plot(uv[0], uv[1], class_icons[(y[0]+1)/2], **marker_options)
        ax_t_l.plot(uv, fuv, 'b-', color=ptt.lighten(ptt.lines[0]))
        ax_t_l.plot(uv[(y+1)/2], fuv[(y+1)/2], class_icons[(y[0]+1)/2], **marker_options) # '+', color=ptt.darken(ptt.lines[0], 1.5)

# Posterior estimates
fig_p, (ax_p_l, ax_p_a, ax_p_r) = ptt.plot_setup_2d(
    t_a=r'Posterior absolute likelihood, $p(y | \mathcal{Y}, \theta)$',
    t_r=r'Posterior relative likelihood $P(x_0 \succ x_1 | \mathcal{Y}, \theta)$')

# Latent function
hf, hpf = ptt.plot_with_bounds(ax_p_l, x_test, f_true, rel_sigma, c=ptt.lines[0])

hf_hat, hpf_hat = ptt.plot_with_bounds(ax_p_l, x_test, fhat, np.sqrt(np.atleast_2d(vhat.diagonal()).T), c=ptt.lines[1])
ax_p_l.legend([hf, hf_hat], [r'True latent function, $f(x)$', r'$\mathcal{GP}$ estimate $\hat{f}(x)$'])

# Absolute posterior likelihood
h_pap = ax_p_a.imshow(p_y, origin='lower', extent=abs_extent)
h_yt, = ax_p_a.plot(x_test, mu_true, c=ptt.lines[0])
hEy, = ax_p_a.plot(x_plot, E_y, color=ptt.lines[3])
if x_abs_train.shape[0]>0:
    ax_p_a.plot(x_abs_train, y_abs_train, 'w+')
ax_p_a.legend([h_yt, hEy], [r'True mean, $E_{p(y|f(x))}[y]$', r'Posterior mean, $E_{p(y|\mathcal{Y})}\left[y\right]$'])
fig_p.colorbar(h_pap, ax=ax_p_a)

# Relative posterior likelihood
p_rel_y_post = prefGP.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)
h_prp = ptt.plot_relative_likelihood(ax_p_r, p_rel_y_post, extent=rel_y_extent)
if x_train.shape[0] > 0:
    for uv, fuv, y in zip(uv_train, fuv_train, y_train):
        ax_p_r.plot(uv[0], uv[1], class_icons[(y[0]+1)/2], **marker_options)
        ax_p_l.plot(uv, fuv, 'b-', color=ptt.lighten(ptt.lines[0]))
        ax_p_l.plot(uv[(y+1)/2], fuv[(y+1)/2], class_icons[(y[0]+1)/2], **marker_options) # '+', color=ptt.darken(ptt.lines[0], 1.5)
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
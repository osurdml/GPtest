# Simple 1D GP classification example
import numpy as np
import matplotlib.pyplot as plt
import GPpref
import scipy.optimize as op
import plot_tools as ptt
import mcmc
from scipy.stats import beta

plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)

log_hyp = np.log([0.1, 0.5, 0.1, 10.0])  # length_scale, sigma_f, sigma_probit, v_beta
np.random.seed(3)

n_rel_train = 0
n_abs_train = 0
true_sigma = 0.05
delta_f = 1e-5

n_xplot = 101
n_mcsamples = 1000
n_ysamples = 101


# Define polynomial function to be modelled
def true_function(x):
    y = (np.sin(x * 2 * np.pi + np.pi / 4) + 1.25) / 2.5
    # y = np.sin(x*2.0*np.pi + np.pi/4.0)
    return y


# Define noisy observation function
def obs_function(x, sigma):
    fx = true_function(x)
    noise = np.random.normal(scale=sigma, size=x.shape)
    return fx + noise


def noisy_preference_rank(uv, sigma):
    fuv = obs_function(uv, sigma)
    y = -1 * np.ones((fuv.shape[0], 1), dtype='int')
    y[fuv[:, 1] > fuv[:, 0]] = 1
    return y, fuv


# Main program
# True function
x_plot = np.linspace(0.0, 1.0, n_xplot, dtype='float')
y_plot = true_function(x_plot)
x_test = np.atleast_2d(x_plot).T

# Training data - this is a bit weird, but we sample x values, then the uv pairs
# are actually indexes into x, because it is easier computationally. You can
# recover the actual u,v values using x[ui],x[vi]
if n_rel_train > 0:
    x_train = np.random.random((2 * n_rel_train, 1))
    uvi_train = np.random.choice(range(2 * n_rel_train), (n_rel_train, 2), replace=False)
    uv_train = x_train[uvi_train][:, :, 0]

    # Get noisy observations f(uv) and corresponding ranks y_train
    y_train, fuv_train = noisy_preference_rank(uv_train, true_sigma)

else:
    x_train = np.zeros((0, 1))
    uvi_train = np.zeros((0, 2))
    uv_train = np.zeros((0, 2))
    y_train = np.zeros((0, 1))
    fuv_train = np.zeros((0, 2))

x_abs_train = np.random.random((n_abs_train, 1))
# y_abs_train = obs_function(x_abs_train, true_sigma)
y_abs_train = np.clip(obs_function(x_abs_train, true_sigma), 0.01, .99)

# GP hyperparameters
# note: using log scaling to aid learning hyperparameters with varied magnitudes

print "Data"
# print x_train
print x_abs_train
# print y_train
print y_abs_train
# print uvi_train

prefGP = GPpref.PreferenceGaussianProcess(x_train, uvi_train, x_abs_train, y_train, y_abs_train, delta_f=delta_f)

# Pseudocode:
# FOr a set of hyperparameters, return log likelihood that can be used by an optimiser
theta0 = log_hyp

# log_hyp = op.fmin(prefGP.calc_nlml,theta0)
# f,lml = prefGP.calc_laplace(log_hyp)
f = prefGP.calc_laplace(log_hyp)

# Latent predictions
fhat, vhat = prefGP.predict_latent(x_test)
vhat = np.atleast_2d(vhat.diagonal()).T

# Expected values
E_y = prefGP.expected_y(x_test, fhat, vhat)

# Sampling from posterior to show likelihoods
p_y = np.zeros((n_ysamples, n_xplot))
y_samples = np.linspace(0.0, 1.0, n_ysamples)
iny = 1.0 / n_ysamples
E_y2 = np.zeros(n_xplot)

normal_samples = np.random.normal(size=n_mcsamples)
for i, (fstar, vstar) in enumerate(zip(fhat, vhat)):
    f_samples = normal_samples * vstar + fstar
    aa, bb = prefGP.abs_likelihood.get_alpha_beta(f_samples)
    p_y[:, i] = [iny * np.sum(beta.pdf(yj, aa, bb)) for yj in y_samples]
    E_y2[i] = np.sum(np.dot(y_samples, p_y[:, i])) / np.sum(p_y[:, i])

# New y's are expectations from Beta distribution. E(X) = alpha/(alpha+beta)
# alph = prefGP.abs_likelihood.alpha(f)
# bet = prefGP.abs_likelihood.beta(f)
# Ey = alph/(alph+bet)

hf, (ha, hb) = plt.subplots(1, 2)
hf, hpf = ptt.plot_with_bounds(ha, x_plot, y_plot, true_sigma, c=ptt.lines[0])

if x_train.shape[0] > 0:
    for uv, fuv, y in zip(uv_train, fuv_train, y_train):
        ha.plot(uv, fuv, 'b-', color=ptt.lighten(ptt.lines[0]))
        ha.plot(uv[(y + 1) / 2], fuv[(y + 1) / 2], '+', color=ptt.darken(ptt.lines[0], 1.5))

if x_abs_train.shape[0] > 0:
    ha.plot(x_abs_train, y_abs_train, '+', color=ptt.lighten(ptt.lines[2]))

hfhat, hpfhat = ptt.plot_with_bounds(hb, x_test, fhat, np.sqrt(vhat), c=ptt.lines[1])
hEy, = ha.plot(x_plot, E_y, color=ptt.lines[3])

ha.imshow(p_y, origin='lower', extent=[x_plot[0], x_plot[-1], 0.0, 1.0])
# ha.plot(x_plot, E_y2, color=ptt.lines[4])
hmap, = ha.plot(x_plot, y_samples[np.argmax(p_y, axis=0)], color='w')
ha.set_title('Training data')
ha.set_ylabel('$y$')
ha.set_xlabel('$x$')
hb.set_xlabel('$x$')
hb.set_ylabel('$f(x)$')
ha.legend([hf, hEy, hmap], [r'$f(x)$', r'$\mathbb{E}_{p(y|\mathcal{Y})}\left[y\right]$', r'$y_{MAP} | \mathcal{Y}$'])
hb.legend([hfhat], [r'Latent function $\hat{f}(x)$'])

plt.show()

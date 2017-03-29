# Simple 1D GP classification example
import numpy as np
import matplotlib.pyplot as plt
import GPpref
import scipy.optimize as op
import plot_tools as ptt
import mcmc
from scipy.stats import beta
import active_learners
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

# log_hyp = np.log([0.1,0.5,0.1,10.0]) # length_scale, sigma_f, sigma_probit, v_beta
log_hyp = np.log([0.07, 0.6, 0.25, 1.0, 28.1])
np.random.seed(3)

n_rel_train = 2
n_abs_train = 1
rel_sigma = 0.05
delta_f = 1e-5

beta_sigma = 0.8
beta_v = 20.0

n_xplot = 101
n_mcsamples = 1000
n_ysamples = 101

n_queries = 20

# Define function to be modelled
def true_function(x):
    #y = (np.sin(x*2*np.pi + np.pi/4) + 1.25)/2.5
    #y = np.sin(x*2.0*np.pi + np.pi/4.0)
    y = np.cos(6 * np.pi * (x - 0.5)) * np.exp(-10 * (x - 0.5) ** 2)
    return y

rel_obs_fun = GPpref.RelObservationSampler(true_function, GPpref.PrefProbit(sigma=rel_sigma))
abs_obs_fun = GPpref.AbsObservationSampler(true_function, GPpref.AbsBoundProbit(sigma=beta_sigma, v=beta_v))

# Gaussian noise observation function
def normal_obs_function(x):
    fx = true_function(x)
    noise = np.random.normal(scale=rel_sigma, size=x.shape)
    return fx + noise

beta_obs = GPpref.AbsBoundProbit(sigma=beta_sigma, v=beta_v)

def beta_obs_function(x):
    fx = true_function(x)
    mu = GPpref.mean_link(f)
    a, b = beta_obs.get_alpha_beta(mu)
    z = [beta.rvs(aa, bb) for aa,bb in zip(a,b)]
    return z

def noisy_preference_rank(uv):
    fuv = normal_obs_function(uv)
    y = -1*np.ones((fuv.shape[0],1),dtype='int')
    y[fuv[:,1] > fuv[:,0]] = 1
    return y, fuv

# Main program
# True function
x_plot = np.linspace(0.0,1.0,n_xplot,dtype='float')
y_plot = true_function(x_plot)
x_test = np.atleast_2d(x_plot).T

# Training data - this is a bit weird, but we sample x values, then the uv pairs
# are actually indexes into x, because it is easier computationally. You can
# recover the actual u,v values using x[ui],x[vi]
if n_rel_train > 0:
    x_train = np.random.random((2*n_rel_train,1))
    uvi_train = np.random.choice(range(2*n_rel_train), (n_rel_train,2), replace=False)
    uv_train = x_train[uvi_train][:,:,0]

    # Get noisy observations f(uv) and corresponding ranks y_train
    y_train, fuv_train = noisy_preference_rank(uv_train)

else:
    x_train = np.zeros((0,1))
    uvi_train = np.zeros((0,2))
    uv_train = np.zeros((0,2))
    y_train = np.zeros((0,1))
    fuv_train = np.zeros((0,2))

x_abs_train = np.random.random((n_abs_train,1))
y_abs_train = beta_obs_function(x_abs_train, sigma=rel_sigma)
#y_abs_train = np.clip(normal_obs_function(x_abs_train), 0.01, .99)


learner = active_learners.PeakComparitor(x_train, uvi_train, x_abs_train,  y_train, y_abs_train, delta_f=delta_f,
                                          abs_likelihood=GPpref.AbsBoundProbit(sigma=beta_sigma, v=beta_v))

theta0 = log_hyp

# Get initial solution
learner.set_hyperparameters(log_hyp)
f = learner.calc_laplace()

for obs_num in range(n_queries):
    fuv = np.array([[0, 1]])
    next_x = np.atleast_2d(learner.select_observation())
    if next_x.shape[0] == 1:
        next_y = beta_obs_function(next_x, sigma=rel_sigma)
        learner.add_observations(next_x, next_y)
    else:
        next_y, next_f = noisy_preference_rank(next_x.T)
        fuv_train = np.concatenate((fuv_train, next_f), 0)
        learner.add_observations(next_x, next_y, fuv)
    print next_x, next_y
    f = learner.calc_laplace()

# Latent predictions
fhat, vhat = learner.predict_latent(x_test)

# Expected values
E_y = learner.GP.abs_posterior_mean(x_test, fhat, vhat)

# Sampling from posterior to show likelihoods
mc_samples = np.random.normal(size=n_mcsamples)
y_samples = np.linspace(0.01, 0.99, n_ysamples)
p_y = learner.GP.abs_posterior_likelihood(y_samples, fhat=fhat, varhat=vhat, normal_samples=mc_samples)

hf_input, ha_input =plt.subplot(1,1)


hf, (hb, ha) = plt.subplots(1,2)
hf, hpf = ptt.plot_with_bounds(ha, x_plot, y_plot, rel_sigma, c=ptt.lines[0])
ha.imshow(p_y, origin='lower', extent=[x_plot[0], x_plot[-1], 0.01, 0.99])

if learner.GP.x_train.shape[0]>0:
    for uv,fuv,y in zip(learner.GP.x_train[learner.GP.uvi_train][:,:,0], fuv_train, learner.GP.y_train):
        ha.plot(uv, fuv, 'b-', color=ptt.lighten(ptt.lines[0]))
        ha.plot(uv[(y+1)/2],fuv[(y+1)/2],'+', color=ptt.darken(ptt.lines[0], 1.5))

if learner.GP.x_abs_train.shape[0]>0:
    ha.plot(learner.GP.x_abs_train, learner.GP.y_abs_train, 'k+')

hfhat, hpfhat = ptt.plot_with_bounds(hb, x_test, fhat, np.sqrt(vhat), c=ptt.lines[1])
hEy, = ha.plot(x_plot, E_y, color=ptt.lines[3])

# ha.plot(x_plot, E_y2, color=ptt.lines[4])
hmap, = ha.plot(x_plot, y_samples[np.argmax(p_y, axis=0)], color='w')
ha.set_title('Training data')
ha.set_ylabel('$y$')
ha.set_xlabel('$x$')
hb.set_xlabel('$x$')
hb.set_ylabel('$f(x)$')

ha.legend([hf, hEy, hmap], [r'$f(x)$', r'$E_{p(y|\mathcal{Y})}\left[y\right]$', r'$y_{MAP} | \mathcal{Y}$'])
hb.legend([hfhat], [r'Latent function $\hat{f}(x)$'])

plt.show()

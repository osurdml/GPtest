# Simple 1D GP classification example
import time
import numpy as np
import matplotlib.pyplot as plt
import GPpref
import plot_tools as ptt
import active_learners
import test_data

nowstr = time.strftime("%Y_%m_%d-%H_%M")
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

save_plots = True

# log_hyp = np.log([0.1,0.5,0.1,10.0]) # length_scale, sigma_f, sigma_probit, v_beta
log_hyp = np.log([0.07, 0.75, 0.25, 1.0, 28.1])
np.random.seed(0)

n_rel_train = 1
n_abs_train = 1
rel_sigma = 0.2
delta_f = 1e-5

beta_sigma = 0.8
beta_v = 20.0

n_xplot = 101
n_mcsamples = 1000
n_ysamples = 101

n_queries = 20

# Define polynomial function to be modelled
true_function = test_data.multi_peak

if save_plots:
    nowstr = time.strftime("%Y_%m_%d-%H_%M")
    fig_dir = 'fig/' + nowstr + '/'
    ptt.ensure_dir(fig_dir)
    print "Figures will be saved to: {0}".format(fig_dir)

rel_obs_fun = GPpref.RelObservationSampler(true_function, GPpref.PrefProbit(sigma=rel_sigma))
abs_obs_fun = GPpref.AbsObservationSampler(true_function, GPpref.AbsBoundProbit(sigma=beta_sigma, v=beta_v))

# True function
x_plot = np.linspace(0.0,1.0,n_xplot,dtype='float')
x_test = np.atleast_2d(x_plot).T
f_true = abs_obs_fun.f(x_test)
mu_true = abs_obs_fun.mean_link(x_test)
mc_samples = np.random.normal(size=n_mcsamples)
abs_y_samples = np.atleast_2d(np.linspace(0.01, 0.99, n_ysamples)).T
p_abs_y_true = abs_obs_fun.observation_likelihood_array(x_test, abs_y_samples)
p_rel_y_true = rel_obs_fun.observation_likelihood_array(x_test)

# Training data
x_rel, uvi_rel, uv_rel, y_rel, fuv_rel = rel_obs_fun.generate_n_observations(n_rel_train, n_xdim=1)
x_abs, y_abs, mu_abs = abs_obs_fun.generate_n_observations(n_abs_train, n_xdim=1)


# Plot true functions
fig_t, (ax_t_l, ax_t_a, ax_t_r) = ptt.true_plots(x_test, f_true, mu_true, rel_sigma,
                                                 abs_y_samples, p_abs_y_true, p_rel_y_true,
                                                 x_abs, y_abs, uv_rel, fuv_rel, y_rel,
                                                 t_l=r'True latent function, $f(x)$')
if save_plots:
    fig_t.savefig(fig_dir+'true.pdf', bbox_inches='tight')

# Construct active learner object
learner = active_learners.LikelihoodImprovement(x_rel, uvi_rel, x_abs,  y_rel, y_abs, delta_f=delta_f,
                                         rel_likelihood=GPpref.PrefProbit(), abs_likelihood=GPpref.AbsBoundProbit())

# Get initial solution
learner.set_hyperparameters(log_hyp)
f = learner.solve_laplace()

if save_plots:
    fig_p, (ax_p_l, ax_p_a, ax_p_r) = learner.create_posterior_plot(x_test, f_true, mu_true, rel_sigma, fuv_rel,
                                                                    abs_y_samples, mc_samples)
    fig_p.savefig(fig_dir+'posterior00.pdf', bbox_inches='tight')

for obs_num in range(n_queries):
    next_x, next_uvi = learner.select_observation(req_improvement=0.55, gamma=2.0, n_comparators=4)
    if next_uvi is None:
        next_y, next_f = abs_obs_fun.generate_observations(next_x)
        learner.add_observations(next_x, next_y)
    else:
        next_y, next_f = rel_obs_fun.generate_observations(next_x[next_uvi][:,:,0])
        fuv_rel = np.concatenate((fuv_rel, next_f), 0)
        learner.add_observations(next_x, next_y, next_uvi)
    print next_x, next_y
    f = learner.solve_laplace()
    if save_plots:
        fig_p, (ax_p_l, ax_p_a, ax_p_r) = learner.create_posterior_plot(x_test, f_true, mu_true, rel_sigma, fuv_rel,
                                                                        abs_y_samples, mc_samples)
        fig_p.savefig(fig_dir+'posterior{0:02d}.pdf'.format(obs_num+1), bbox_inches='tight')
        plt.close(fig_p)

learner.print_hyperparameters()

if not save_plots:
    fig_p, (ax_p_l, ax_p_a, ax_p_r) = learner.create_posterior_plot(x_test, f_true, mu_true, rel_sigma, fuv_rel,
                                                                    abs_y_samples, mc_samples)

plt.show()

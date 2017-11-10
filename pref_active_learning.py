# Simple 1D GP classification example
import time
import numpy as np
import matplotlib.pyplot as plt
import GPpref
import plot_tools as ptt
import active_learners
import test_data
import yaml
from matplotlib.backends.backend_pdf import PdfPages

nowstr = time.strftime("%Y_%m_%d-%H_%M")
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

save_plots = False

with open('./data/statruns2_oct2017.yaml', 'rt') as fh:
    wave = yaml.safe_load(fh)

try:
    np.random.seed(wave['statrun_params']['randseed'])
except KeyError:
    np.random.seed(0)

n_rel_train = 0
n_abs_train = 1
n_queries = 20

delta_f = 1e-5

n_xplot = 101
n_mcsamples = 1000
n_ysamples = 101

keep_f = True
verbose = 1

log_hyp = np.log(wave['hyperparameters'])

# Define polynomial function to be modelled
random_wave = test_data.MultiWave(**wave['wave_params'])
true_function = random_wave.out
random_wave.print_values()

if save_plots:
    nowstr = time.strftime("%Y_%m_%d-%H_%M")
    fig_dir = 'fig/' + nowstr + '/'
    ptt.ensure_dir(fig_dir)
    print "Figures will be saved to: {0}".format(fig_dir)
    with open(fig_dir+'params.yaml', 'wt') as fh:
        yaml.safe_dump(wave, fh)
    pdf_pages = PdfPages(fig_dir+'posterior_all.pdf')

rel_obs_fun = GPpref.RelObservationSampler(true_function, GPpref.PrefProbit(**wave['rel_obs_params']))
abs_obs_fun = GPpref.AbsObservationSampler(true_function, GPpref.AbsBoundProbit(**wave['abs_obs_params']))
rel_sigma = wave['rel_obs_params']['sigma']

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
GP_kwargs = {'x_rel':x_rel, 'uvi_rel':uvi_rel, 'x_abs':x_abs,  'y_rel':y_rel, 'y_abs':y_abs, 'delta_f':delta_f,
               'rel_likelihood':GPpref.PrefProbit(), 'abs_likelihood':GPpref.AbsBoundProbit()}
learner_kwargs = wave['learners'][0]
learner = active_learners.Learner(**learner_kwargs)
learner.build_model(GP_kwargs)
learner.model.set_hyperparameters(log_hyp)

# Get initial solution
learner.model.set_hyperparameters(log_hyp)
f = learner.model.solve_laplace()
learner.model.print_hyperparameters()

fig_p, (ax_p_l, ax_p_a, ax_p_r) = learner.model.create_posterior_plot(x_test, f_true, mu_true, rel_sigma, fuv_rel,
                                                                abs_y_samples, mc_samples)
if save_plots:
    pdf_pages.savefig(fig_p, bbox_inches='tight')
    # fig_p.savefig(fig_dir+'posterior00.pdf', bbox_inches='tight')

for obs_num in range(n_queries):
    # if 'p_rel' in obs_arguments:
    #     obs_arguments['p_rel'] = max(0.0, (n_queries - obs_num) / float(n_queries))
    next_x = learner.select_observation()
    if next_x.shape[0] == 1:
        next_y, next_f = abs_obs_fun.generate_observations(next_x)
        learner.model.add_observations(next_x, next_y, keep_f=keep_f)
        print 'Abs: x:{0}, y:{1}'.format(next_x[0], next_y[0])
    else:
        next_y, next_uvi, next_fx = rel_obs_fun.gaussian_multi_pairwise_sampler(next_x)
        next_fuv = next_fx[next_uvi][:, :, 0]
        fuv_rel = np.concatenate((fuv_rel, next_fuv), 0)
        learner.model.add_observations(next_x, next_y, next_uvi, keep_f=keep_f)
        print 'Rel: x:{0}, best_index:{1}'.format(next_x.flatten(), next_uvi[0, 1])

    f = learner.model.solve_laplace()

    if save_plots:
        fig_p, (ax_p_l, ax_p_a, ax_p_r) = learner.model.create_posterior_plot(x_test, f_true, mu_true, rel_sigma, fuv_rel,
                                                                        abs_y_samples, mc_samples)
        pdf_pages.savefig(fig_p, bbox_inches='tight')
        # fig_p.savefig(fig_dir+'posterior{0:02d}.pdf'.format(obs_num+1), bbox_inches='tight')
        plt.close(fig_p)

learner.model.print_hyperparameters()

if save_plots:
    pdf_pages.close()
else:
    fig_post, (ax_p_l, ax_p_a, ax_p_r) = learner.model.create_posterior_plot(x_test, f_true, mu_true, rel_sigma, fuv_rel,
                                                                    abs_y_samples, mc_samples)

plt.show()
print "Finished!"

## SCRAP
# learner = active_learners.UCBAbsRelD(**active_args)
# # obs_arguments = {'req_improvement': 0.60, 'n_test': 50, 'gamma': 2.0, 'n_rel_samples': 5, 'p_thresh': 0.7}
# obs_arguments = {'n_test': 100, 'p_rel': 0.5, 'n_rel_samples': 5, 'gamma': 2.0}

# learner = active_learners.ExpectedImprovementRel(**GP_kwargs)
# obs_arguments = {'n_test': 100, 'zeta': 0.1, 'p_rel':1.0}

# learner = active_learners.SampledClassification(verbose=verbose, **GP_kwargs)
# obs_arguments = {'n_test':50, 'n_samples':20, 'y_threshold':0.7, 'p_pref_tol':1e-3, 'n_mc_abs':50}

# learner = active_learners.DetSelect(**GP_kwargs)
# obs_arguments = {'n_test': 100, 'n_rel_samples': 5, 'gamma': 2.0, 'tau': 0.5}

# learner = active_learners.ActiveLearner(**GP_kwargs)
# obs_arguments =  {'p_rel':0.0, 'n_rel_samples': 5}

# learner = active_learners.MaxVar(**GP_kwargs)
# obs_arguments = {'n_rel_samples': 5, 'p_rel': -10.0, 'rel_tau': 0.1, 'abs_tau': 0.1}

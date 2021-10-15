# Simple 1D GP classification example
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gp_tools import GPpref, active_learners
import utils.plot_tools as ptt
from utils import test_data
import yaml
from matplotlib.backends.backend_pdf import PdfPages
import argparse

nowstr = time.strftime("%Y_%m_%d-%H_%M")
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

parser = argparse.ArgumentParser(description='Statruns for active learning with preference GP - wine data')
parser.add_argument('-s', '--save-plots', dest='save_plots', action='store_true', help='Save output plots')
parser.add_argument('-y', '--yaml-config', default='./data/shortrun_2D.yaml', help='YAML config file')
parser.add_argument('-if', '--inter-frames', default=20, type=int, help='Intermediate plotting frames (higher for smooth animations)')
parser.add_argument('-pt', '--plot-type', default='video_frames', help='Video frame type (pdf or video_frames)')
parser.add_argument('-l', '--learner-index', default=0, type=int, help='Index of learner to use (from config file)')
args = parser.parse_args()

save_plots = args.save_plots
plot_type = args.plot_type  # 'pdf'
inter_frames = args.inter_frames

with open(args.yaml_config, 'rt') as fh:
    wave = yaml.safe_load(fh)

try:
    np.random.seed(wave['statrun_params']['randseed'])
    n_rel_train, n_abs_train = wave['statrun_params']['n_rel_train'], wave['statrun_params']['n_abs_train']
    n_queries =  wave['statrun_params']['n_queries']
except KeyError:
    np.random.seed(0)
    n_rel_train = 1
    n_abs_train = 0
    n_queries = 20

n_xplot = 31
keep_f = True
learner_index = args.learner_index

log_hyp = np.log(wave['hyperparameters'])

# Define polynomial function to be modelled
d_x = wave['GP_params']['hyper_counts'][0]-1
random_wave = test_data.MultiWave(n_dimensions=d_x, **wave['wave_params'])
true_function = random_wave.out
random_wave.print_values()

if save_plots:
    nowstr = time.strftime("%Y_%m_%d-%H_%M")
    fig_dir = 'fig/' + nowstr + '/'
    ptt.ensure_dir(fig_dir)
    print "Figures will be saved to: {0}".format(fig_dir)
    if plot_type == 'pdf':
        pdf_pages = PdfPages(fig_dir+'posterior_all.pdf')


rel_obs_fun = GPpref.RelObservationSampler(true_function, wave['GP_params']['rel_likelihood'], wave['rel_obs_params'])
abs_obs_fun = GPpref.AbsObservationSampler(true_function, wave['GP_params']['abs_likelihood'], wave['abs_obs_params'])
rel_sigma = wave['rel_obs_params']['sigma']

# True function
x_plot = np.linspace(0.0,1.0,n_xplot,dtype='float')
x_test = ptt.make_meshlist(x_plot, d_x)
f_true = abs_obs_fun.f(x_test)
mu_true = abs_obs_fun.mean_link(x_test)
abs_y_samples = abs_obs_fun.l.y_list
p_abs_y_true = abs_obs_fun.observation_likelihood_array(x_test, abs_y_samples)
if d_x is 1:
    p_rel_y_true = rel_obs_fun.observation_likelihood_array(x_test)
else:
    p_rel_y_true = None

# Training data - note the shifted domain to get the sample out of the way
far_domain = np.tile(np.array([[-3.0], [-2.0]]), d_x)
x_rel, uvi_rel, y_rel, fuv_rel = rel_obs_fun.generate_n_observations(n_rel_train, n_xdim=d_x, domain=far_domain)
x_abs, y_abs, mu_abs = abs_obs_fun.generate_n_observations(n_abs_train, n_xdim=d_x, domain=far_domain)


# Plot true functions
plot_kwargs = {'xlim':[0.0, 1.0], 'ylim':[0.0, 1.0]}
if plot_type != 'pdf':
    if p_rel_y_true is None:
        ax_t = [[], []]
        fig_t = plt.figure()
        fig_t.set_size_inches([11.44,  8.02])
        ax_t[0].append(fig_t.add_subplot(221, projection='3d', **plot_kwargs))
        ax_t[0].append(fig_t.add_subplot(222, projection='3d', **plot_kwargs))
        ax_t[1].append(fig_t.add_subplot(223, projection='3d', **plot_kwargs))
        ax_t[1].append(fig_t.add_subplot(224, projection='3d', **plot_kwargs))
    else:
        fig_t, ax_t = plt.subplots(2,3)
    true_ax = ax_t[0]
    est_ax = ax_t[1]
else:
    true_ax = None
    est_ax = None

fig_t, ax_t = ptt.true_plots(x_test, f_true, mu_true, rel_sigma,
                                                 abs_y_samples, p_abs_y_true, p_rel_y_true,
                                                 x_abs, y_abs, x_rel[uvi_rel], fuv_rel, y_rel,
                                                 t_l=r'True latent function, $f(x)$', ax=true_ax, **plot_kwargs)
if save_plots and (plot_type is 'pdf'):
    fig_t.savefig(fig_dir+'true.pdf', bbox_inches='tight')

# Construct active learner object

# Construct GP object
prefGP = GPpref.PreferenceGaussianProcess(x_rel, uvi_rel, x_abs, y_rel, y_abs, **wave['GP_params'])

model_kwargs = {'x_rel':x_rel, 'uvi_rel':uvi_rel, 'x_abs':x_abs, 'y_rel':y_rel, 'y_abs':y_abs,
                'rel_kwargs': wave['rel_obs_params'], 'abs_kwargs': wave['abs_obs_params']}
model_kwargs.update(wave['GP_params'])
learner_kwargs = wave['learners'][learner_index]
learner = active_learners.Learner(**learner_kwargs)
learner.build_model(model_kwargs)

# Get initial solution
learner.model.set_hyperparameters(log_hyp)
f = learner.model.solve_laplace()
learner.model.print_hyperparameters()

fig_p, ax_p, pre_data = learner.model.create_posterior_plot(x_test, f_true, mu_true, rel_sigma, fuv_rel, abs_y_samples, ax=est_ax, **plot_kwargs)
if save_plots:
    wave['learners'] = [learner_kwargs]
    with open(fig_dir+'params.yaml', 'wt') as fh:
        yaml.safe_dump(wave, fh)
    if plot_type is 'pdf':
        pdf_pages.savefig(fig_p, bbox_inches='tight')
    else:
        fig_p.savefig(fig_dir+'posterior000.png', bbox_inches='tight')
print learner_kwargs

for obs_num in range(n_queries):
    if learner.update_p_rel:
        linear_p_rel = max(0.0, (n_queries - obs_num) / float(n_queries))
        learner.obs_arguments['p_rel'] = linear_p_rel
    # if 'p_rel' in obs_arguments:
    #     obs_arguments['p_rel'] = max(0.0, (n_queries - obs_num) / float(n_queries))
    t0 = time.time()
    next_x = learner.select_observation()
    if next_x.shape[0] == 1:
        next_y, next_f = abs_obs_fun.generate_observations(next_x)
        learner.model.add_observations(next_x, next_y, keep_f=keep_f)
        print '{n:02} - Abs: x:{0}, y:{1}'.format(next_x[0], next_y[0], n=obs_num+1),
    else:
        next_y, next_uvi, next_fx = rel_obs_fun.gaussian_multi_pairwise_sampler(next_x)
        next_fuv = next_fx[next_uvi][:, :, 0]
        fuv_rel = np.concatenate((fuv_rel, next_fuv), 0)
        learner.model.add_observations(next_x, next_y, next_uvi, keep_f=keep_f)
        print '{n:02} - Rel: x:{0}, best_index:{1}'.format(next_x.flatten(), next_uvi[0, 1], n=obs_num+1),
    print 't = {0:0.3f}s'.format(time.time()-t0)
    f = learner.model.solve_laplace()

    if save_plots:
        ptt.reset_axes2d(est_ax)
        fig_p, ax_p, post_data = learner.model.create_posterior_plot(x_test, f_true, mu_true, rel_sigma, fuv_rel,
                                                                     abs_y_samples, ax=est_ax, **plot_kwargs)
        if plot_type is 'pdf':
            pdf_pages.savefig(fig_p, bbox_inches='tight')
        else:
            fig_p.savefig(fig_dir + 'posterior{0:03d}.png'.format((obs_num+1)*inter_frames), bbox_inches='tight')
            for iframe in range(1, inter_frames):
                scale = float(iframe)/inter_frames
                if pre_data[-1] is None:
                    pre_data = pre_data[:-1]
                inter_data = [(1-scale)*pre + scale*post for pre, post in zip(pre_data, post_data)]
                if len(inter_data) < len(post_data):
                    inter_data.append(None)
                fig_p, ax_p = learner.model.update_posterior_plot(x_test, f_true, mu_true, rel_sigma, fuv_rel,
                                                                     abs_y_samples, *inter_data, ax_p=est_ax)
                fig_p.savefig(fig_dir + 'posterior{0:03d}.png'.format(obs_num*inter_frames + iframe), bbox_inches='tight')
            pre_data = post_data
        plt.close(fig_p)
        # fig_p.savefig(fig_dir+'posterior{0:02d}.pdf'.format(obs_num+1), bbox_inches='tight')


learner.model.print_hyperparameters()

if save_plots and plot_type is 'pdf':
    pdf_pages.close()
else:
    fig_post, ax_p, h_plotobj = learner.model.create_posterior_plot(x_test, f_true, mu_true, rel_sigma, fuv_rel, abs_y_samples, **plot_kwargs)

plt.show(block=False)
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

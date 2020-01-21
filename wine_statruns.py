# Simple 1D GP classification example
import time
import numpy as np
from gp_tools import GPpref
import utils.plot_tools as ptt
from gp_tools import active_learners
from utils import test_data
from utils import plot_statruns
from utils.wine_data import WineQualityData, WineObsSampler
import yaml
import argparse

np.set_printoptions(precision=3)
wrms_fun = test_data.wrms_misclass
wrms_args = {'w_power': 1}

now_time = time.strftime("%Y_%m_%d-%H_%M")
yaml_config ='./config/red_statruns.yaml'

parser = argparse.ArgumentParser(description='Statruns for active learning with preference GP - wine data')
parser.add_argument('-np', '--no-plots', dest='make_plots', action='store_false', help='Turn off plots (default False)')
parser.add_argument('-y', '--yaml-config', default=yaml_config, help='YAML config file')
parser.add_argument('-na', '--n-abs', default=50, type=int, help='Number of prior absolute training points')
parser.add_argument('-nr', '--n-rel', default=10, type=int, help='Number of prior relative training points')
parser.add_argument('-ml', '--min-label', default=5, type=int, help='Minimum label value to count as good point')

args = parser.parse_args()

print "Using YAML config: {0}".format(args.yaml_config)
if not args.make_plots:
    print "No plot output."
with open(args.yaml_config, 'rt') as fh:
    config = yaml.safe_load(fh)

hyper = config['hyperparameters']
d_x = len(hyper['l'])
all_hyper = np.concatenate((hyper['l'], [hyper['sig_f'], hyper['sig_rel'], hyper['sig_beta'], hyper['v_beta']]))
log_hyp = np.log(all_hyper)

# Statrun parameters
statrun_params = config['statrun_params']
np.random.seed(statrun_params['randseed'])
n_trials = statrun_params['n_trials']
n_queries = statrun_params['n_queries']
calc_relative_error = False  # NOT IMPLEMENTED (might not even make sense)

n_rel_samples = config['statrun_params']['n_rel_samples']

# Get wine data:
input_data = WineQualityData(wine_type=config['wine_params']['type'], cols=config['wine_params']['variables'], norm=config['wine_params']['normalise_data'], scale_y=True)

now_time = time.strftime("%Y_%m_%d-%H_%M")
data_dir = 'data/' + now_time + '/'
ptt.ensure_dir(data_dir)
print "Data will be saved to: {0}".format(data_dir)

# Test data
n_xtest = min(statrun_params['n_xtest'], input_data.x.shape[0])

# Construct active learner objects
n_learners = len(config['learners'])
learners = []
names = []
obs_array = []
full_obs = {}
obs_samplers = []
for l in config['learners']:
    obs_samplers.append(WineObsSampler(input_data))
    if 'n_rel_samples' in l['obs_args']:
        l['obs_args']['n_rel_samples'] = n_rel_samples
    learners.append(active_learners.Learner(**l))
    names.append(l['name'])
    obs_array.append({'name': l['name'], 'obs': []})
    full_obs[l['name']] = [None] * n_trials
    # We need a sampler for each learner

wrms_results = np.zeros((n_learners, n_queries+1, n_trials))
true_pos_results = np.zeros((n_learners, n_queries+1, n_trials), dtype='int')
selected_error = np.zeros((n_learners, n_queries+1, n_trials))

with open(data_dir + 'params.yaml', 'wt') as fh:
    yaml.safe_dump(config, fh)

# Get the set of test data - note we always use all the data
x_test, y_abs_true = input_data.x, input_data.y

# Get the set of actual best points
best_points, _null = np.where(input_data.y >= args.min_label)
best_points_set = set(best_points)
n_best_points = len(best_points_set)

trial_number = 0
while trial_number < n_trials:

    # try:
    print 'Trial {0}'.format(trial_number)

    # Use first obs sampler to select first observations
    obs_samplers[0].reset()

    i_abs = obs_samplers[0].get_available_indexes(statrun_params['n_abs_train'])
    x_abs, y_abs = obs_samplers[0].pop_abs_observations(i_abs)

    i_rel = obs_samplers[0].get_available_indexes(statrun_params['n_rel_train']*2)
    i_rel = np.reshape(i_rel, (statrun_params['n_rel_train'],2))
    x_rel, uvi_rel, y_rel, fuv_rel = obs_samplers[0].pop_rel_observations(i_rel)

    # Remove these observations from available list of all observers
    for s in obs_samplers[1:]:
        s.reset()
        s.remove_indexes(i_abs)
        s.remove_indexes(i_rel.flatten())

    # Initial data
    model_kwargs = {'x_rel': x_rel, 'uvi_rel': uvi_rel, 'x_abs': x_abs, 'y_rel': y_rel, 'y_abs': y_abs,
                    'rel_kwargs': config['rel_obs_params'], 'abs_kwargs': config['abs_obs_params']}
    model_kwargs.update(config['GP_params'])

    # Get initial solution
    for nl, (learner, sampler) in enumerate(zip(learners, obs_samplers)):
        model_kwargs['domain_sampler'] = sampler._gen_x_obs
        learner.build_model(model_kwargs)
        learner.model.set_hyperparameters(log_hyp)
        f = learner.model.solve_laplace()
        fhat, vhat = learner.model.predict_latent(x_test)
        y_abs_est = learner.model.abs_posterior_mean(x_test, fhat, vhat)

        best_points_est = set(np.argpartition(y_abs_est.flatten(), -n_best_points)[-n_best_points:])
        wrms_results[nl, 0, trial_number] = wrms_fun(y_abs_true, y_abs_est, **wrms_args)
        true_pos_results[nl, 0, trial_number] = len(best_points_set.intersection(best_points_est))
        selected_error[nl, 0, trial_number] = test_data.wrms(y_abs_true[best_points], y_abs_est[best_points], weight=False)

        obs_tuple = learner.model.get_observations()
        full_obs[learner.name][trial_number] = [test_data.ObsObject(*obs_tuple)]

    for obs_num in range(n_queries):
        t0 = time.time()
        linear_p_rel = max(0.0, (n_queries-obs_num)/float(n_queries))

        for nl, (learner, sampler) in enumerate(zip(learners, obs_samplers)):
            # Update p_rel for certain methods (hacky?)
            if learner.update_p_rel:
                learner.obs_arguments['p_rel'] = linear_p_rel

            next_x = learner.model.select_observation(**learner.obs_arguments)

            next_i = [np.argwhere((input_data.x == nx).all(axis=1))[0][0] for nx in next_x]

            next_uvi = None
            if next_x.shape[0] == 1:
                # Slow and bad way to do this...

                next_x, next_y = sampler.pop_abs_observations(next_i)
                # print 'Abs: x:{0}, y:{1}'.format(next_x[0], next_y[0])
            else:
                next_x, next_uvi, next_y, next_fx = sampler.make_rel_observations(next_i)
                # next_y, next_uvi, next_fx = rel_obs_fun.gaussian_multi_pairwise_sampler(next_x)
                # next_fuv = next_fx[next_uvi][:,:,0]
                # fuv_rel = np.concatenate((fuv_rel, next_fuv), 0)
                # print 'Rel: x:{0}, best_index:{1}'.format(next_x.flatten(), next_uvi[0, 1])
            full_obs[learner.name][trial_number].append((next_x, next_y, next_uvi))
            learner.model.add_observations(next_x, next_y, next_uvi)
            f = learner.model.solve_laplace()
            fhat, vhat = learner.model.predict_latent(x_test)
            y_abs_est = learner.model.abs_posterior_mean(x_test, fhat, vhat)

            # Get selected best point set and error results
            best_points_est = set(np.argpartition(y_abs_est.flatten(), -n_best_points)[-n_best_points:])
            wrms_results[nl, obs_num+1, trial_number] = wrms_fun(y_abs_true, y_abs_est, **wrms_args)
            true_pos_results[nl, obs_num+1, trial_number] = len(best_points_set.intersection(best_points_est))
            selected_error[nl, obs_num+1, trial_number] = test_data.wrms(y_abs_true[best_points], y_abs_est[best_points], weight=False)

        print "{0}, t={t:0.2f}s, tp = {1}, wrms = {2}".format(obs_num, true_pos_results[:, obs_num+1, trial_number], wrms_results[:, obs_num+1, trial_number], t=time.time()-t0)
    for nl, learner in enumerate(learners):
        obs_tuple = learner.model.get_observations()
        obs_array[nl]['obs'].append(test_data.ObsObject(*obs_tuple))

    # except Exception as e:
    #     print 'Exception error is: %s, attempting a new sample' % e
    #     continue

    trial_number += 1
    plot_statruns.save_data(data_dir, wrms_results, true_pos_results,
                            selected_error, obs_array, full_obs=full_obs,
                            t=trial_number)

if args.make_plots:
    hfig = plot_statruns.plot_results(wrms_results, true_pos_results, selected_error, obs_array, relative_error=None, data_dir=data_dir, bars=True, norm_comparator=0)
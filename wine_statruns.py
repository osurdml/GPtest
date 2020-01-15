# Simple 1D GP classification example
import time
import numpy as np
import GPpref
import plot_tools as ptt
import active_learners
import test_data
import plot_statruns
import yaml
import argparse

np.set_printoptions(precision=3)
wrms_fun = test_data.wrms_misclass
wrms_args = {'w_power': 1}

now_time = time.strftime("%Y_%m_%d-%H_%M")
yaml_config ='./data/statruns_dec2017.yaml'

parser = argparse.ArgumentParser(description='Statruns for active learning with preference GP - wine data')
parser.add_argument('-np', '--no-plots', dest='make_plots', action='store_false', help='Turn off plots (default False)')
parser.add_argument('-y', '--yaml-config', default=yaml_config, help='YAML config file')
parser.add_argument('-w', '--wine-type', default='red', choices=['white', 'red'], help='Wine type')

args = parser.parse_args()

print "Using YAML config: {0}".format(args.yaml_config)
if not args.make_plots:
    print "No plot output."
with open(args.yaml_config, 'rt') as fh:
    run_parameters = yaml.safe_load(fh)

log_hyp = np.log(run_parameters['hyperparameters'])

# Statrun parameters
statrun_params = run_parameters['statrun_params']
np.random.seed(statrun_params['randseed'])
n_best_points = statrun_params['n_best_points']
n_trials = statrun_params['n_trials']
n_queries = statrun_params['n_queries']
if 'calc_relative_error' in run_parameters['statrun_params']:
    calc_relative_error = statrun_params['calc_relative_error']
else:
    calc_relative_error = False

n_rel_samples = run_parameters['statrun_params']['n_rel_samples']

# Define polynomial function to be modelled
d_x = run_parameters['GP_params']['hyper_counts'][0]-1

# Get wine data set
input_data = WineQualityData(wine_type=wine_type)
wine_norm = (input_data.data.values - input_data.data.values.min(axis=0)) / (input_data.data.values.max(axis=0) - input_data.data.values.min(axis=0))


random_wave  = test_data.MultiWave(n_dimensions=d_x, **run_parameters['wave_params'])

now_time = time.strftime("%Y_%m_%d-%H_%M")
data_dir = 'data/' + now_time + '/'
ptt.ensure_dir(data_dir)
print "Data will be saved to: {0}".format(data_dir)
waver = test_data.WaveSaver(n_trials, random_wave.n_components, n_dim=d_x)


# True function
x_plot = np.linspace(0.0, 1.0, statrun_params['n_xtest'], dtype='float')

try:
    if statrun_params['sample_type'] is 'uniform':
        x_test = ptt.make_meshlist(x_plot, d_x)
    else:
        x_test = np.random.uniform(0.0, 1.0,
                                   size = (statrun_params['n_xtest'], d_x))
except KeyError:
    print('Config does not contain statrun_params: sample_type value. Default to uniform sampling')
    x_test = ptt.make_meshlist(x_plot, d_x)

far_domain = np.tile(np.array([[-3.0], [-2.0]]), d_x)

# Construct active learner objects
n_learners = len(run_parameters['learners'])
learners = []
names = []
obs_array = []
full_obs = {}
for l in run_parameters['learners']:
    if 'n_rel_samples' in l['obs_args']:
        l['obs_args']['n_rel_samples'] = n_rel_samples
    learners.append(active_learners.Learner(**l))
    names.append(l['name'])
    obs_array.append({'name': l['name'], 'obs': []})
    full_obs[l['name']] = [None] * n_trials

wrms_results = np.zeros((n_learners, n_queries+1, n_trials))
true_pos_results = np.zeros((n_learners, n_queries+1, n_trials), dtype='int')
selected_error = np.zeros((n_learners, n_queries+1, n_trials))
if calc_relative_error:
    relative_error = np.zeros((n_learners, n_queries+1, n_trials))
else:
    relative_error = None

with open(data_dir + 'params.yaml', 'wt') as fh:
    yaml.safe_dump(run_parameters, fh)

trial_number = 0
while trial_number < n_trials:

    try:
        print 'Trial {0}'.format(trial_number)
        random_wave.randomize()
        random_wave.print_values()
        waver.set_vals(trial_number, *random_wave.get_values())
        rel_obs_fun = GPpref.RelObservationSampler(random_wave.out, run_parameters['GP_params']['rel_likelihood'], run_parameters['rel_obs_params'])
        abs_obs_fun = GPpref.AbsObservationSampler(random_wave.out, run_parameters['GP_params']['abs_likelihood'], run_parameters['abs_obs_params'])

        f_true = abs_obs_fun.f(x_test)
        y_abs_true = abs_obs_fun.mean_link(x_test)
        best_points = np.argpartition(y_abs_true.flatten(), -n_best_points)[-n_best_points:]
        best_points_set = set(best_points)
        if calc_relative_error:
            p_rel_y_true = rel_obs_fun.observation_likelihood_array(x_test)

        # Initial data
        x_rel, uvi_rel, y_rel, fuv_rel = rel_obs_fun.generate_n_observations(statrun_params['n_rel_train'],
                                                                                     n_xdim=d_x, domain=far_domain)
        x_abs, y_abs, mu_abs = abs_obs_fun.generate_n_observations(statrun_params['n_abs_train'], n_xdim=d_x,
                                                                                     domain=far_domain)
        model_kwargs = {'x_rel': x_rel, 'uvi_rel': uvi_rel, 'x_abs': x_abs, 'y_rel': y_rel, 'y_abs': y_abs,
                        'rel_kwargs': run_parameters['rel_obs_params'], 'abs_kwargs': run_parameters['abs_obs_params']}
        model_kwargs.update(run_parameters['GP_params'])

        # Get initial solution
        for nl, learner in enumerate(learners):
            learner.build_model(model_kwargs)
            learner.model.set_hyperparameters(log_hyp)
            f = learner.model.solve_laplace()
            fhat, vhat = learner.model.predict_latent(x_test)
            y_abs_est = learner.model.abs_posterior_mean(x_test, fhat, vhat)

            best_points_est = set(np.argpartition(y_abs_est.flatten(), -n_best_points)[-n_best_points:])
            wrms_results[nl, 0, trial_number] = wrms_fun(y_abs_true, y_abs_est, **wrms_args)
            true_pos_results[nl, 0, trial_number] = len(best_points_set.intersection(best_points_est))
            selected_error[nl, 0, trial_number] = test_data.wrms(y_abs_true[best_points], y_abs_est[best_points], weight=False)
            if calc_relative_error:
                p_rel_y_post = learner.model.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)
                relative_error[nl, 0, trial_number] = test_data.rel_error(y_abs_true, p_rel_y_true, y_abs_est, p_rel_y_post, weight=True)

            obs_tuple = learner.model.get_observations()
            full_obs[learner.name][trial_number] = [test_data.ObsObject(*obs_tuple)]

        for obs_num in range(n_queries):
            t0 = time.time()
            linear_p_rel = max(0.0, (n_queries-obs_num)/float(n_queries))

            for nl, learner in enumerate(learners):
                # Update p_rel for certain methods (hacky?)
                if learner.update_p_rel:
                    learner.obs_arguments['p_rel'] = linear_p_rel

                next_x = learner.model.select_observation(**learner.obs_arguments)
                next_uvi = None
                if next_x.shape[0] == 1:
                    next_y, next_f = abs_obs_fun.generate_observations(next_x)
                    # print 'Abs: x:{0}, y:{1}'.format(next_x[0], next_y[0])
                else:
                    next_y, next_uvi, next_fx = rel_obs_fun.gaussian_multi_pairwise_sampler(next_x)
                    next_fuv = next_fx[next_uvi][:,:,0]
                    fuv_rel = np.concatenate((fuv_rel, next_fuv), 0)
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
                if calc_relative_error:
                    p_rel_y_post = learner.model.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)
                    relative_error[nl, obs_num+1, trial_number] = test_data.rel_error(y_abs_true, p_rel_y_true, y_abs_est,
                                                                              p_rel_y_post, weight=True)
            if calc_relative_error:
                print "{0}, t={t:0.2f}s, tp = {1}, wrms = {2}, p_err={3}".format(obs_num, true_pos_results[:, obs_num+1, trial_number], wrms_results[:, obs_num+1, trial_number], relative_error[:, obs_num+1, trial_number], t=time.time()-t0)
            else:
                print "{0}, t={t:0.2f}s, tp = {1}, wrms = {2}".format(obs_num, true_pos_results[:, obs_num+1, trial_number], wrms_results[:, obs_num+1, trial_number], t=time.time()-t0)
        for nl, learner in enumerate(learners):
            obs_tuple = learner.model.get_observations()
            obs_array[nl]['obs'].append(test_data.ObsObject(*obs_tuple))


    except Exception as e:
        print 'Exception error is: %s, attempting a new wave' % e
        continue

    trial_number += 1
    if relative_error is not None:
        plot_statruns.save_data(data_dir, wrms_results, true_pos_results, selected_error, obs_array, full_obs=full_obs,
                                rel_err=relative_error, t=trial_number)
    else:
        plot_statruns.save_data(data_dir, wrms_results, true_pos_results, selected_error, obs_array, full_obs=full_obs,
                                t=trial_number)

    waver.save(data_dir+'wave_data.pkl')

if args.make_plots:
    hfig = plot_statruns.plot_results(wrms_results, true_pos_results, selected_error, obs_array, relative_error=relative_error, data_dir=data_dir, bars=True, norm_comparator=0)
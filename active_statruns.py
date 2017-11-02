# Simple 1D GP classification example
import time
import numpy as np
import GPpref
import plot_tools as ptt
import active_learners
import test_data
import pickle
import plot_statruns
import yaml

np.set_printoptions(precision=3)

class Learner(object):
    def __init__(self, model_type, obs_args, name, update_p_rel = False):
        self.model_type = getattr(active_learners, model_type)
        self.obs_arguments = obs_args
        self.name = name
        self.update_p_rel = update_p_rel

    def build_model(self, training_data):
        self.model = self.model_type(**training_data)

now_time = time.strftime("%Y_%m_%d-%H_%M")

with open('./data/statruns2_oct2017.yaml', 'rt') as fh:
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

# Learner parameters
n_rel_samples = run_parameters['learner_params']['n_rel_samples']

# Define polynomial function to be modelled
random_wave  = test_data.MultiWave(**run_parameters['wave_params'])

now_time = time.strftime("%Y_%m_%d-%H_%M")
data_dir = 'data/' + now_time + '/'
ptt.ensure_dir(data_dir)
print "Data will be saved to: {0}".format(data_dir)
waver = test_data.WaveSaver(n_trials, random_wave.n_components)


# True function
x_plot = np.linspace(0.0, 1.0, statrun_params['n_xtest'], dtype='float')
x_test = np.atleast_2d(x_plot).T

# Construct active learner objects
n_learners = len(run_parameters['learners'])
learners = []
names = []
obs_array = []
for l in run_parameters['learners']:
    if 'n_rel_samples' in l['obs_args']:
        l['obs_args']['n_rel_samples'] = n_rel_samples
    learners.append(Learner(**l))
    names.append(l['name'])
    obs_array.append({'name': l['name'], 'obs': []})

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
        rel_obs_fun = GPpref.RelObservationSampler(random_wave.out, GPpref.PrefProbit(**run_parameters['rel_obs_params']))
        abs_obs_fun = GPpref.AbsObservationSampler(random_wave.out, GPpref.AbsBoundProbit(**run_parameters['abs_obs_params']))

        f_true = abs_obs_fun.f(x_test)
        y_abs_true = abs_obs_fun.mean_link(x_test)
        best_points = np.argpartition(y_abs_true.flatten(), -n_best_points)[-n_best_points:]
        best_points_set = set(best_points)
        if calc_relative_error:
            p_rel_y_true = rel_obs_fun.observation_likelihood_array(x_test)

        # Initial data
        x_rel, uvi_rel, uv_rel, y_rel, fuv_rel = rel_obs_fun.generate_n_observations(statrun_params['n_rel_train'], n_xdim=1)
        x_abs, y_abs, mu_abs = abs_obs_fun.generate_n_observations(statrun_params['n_abs_train'], n_xdim=1)
        training_data = {'x_rel': x_rel, 'uvi_rel': uvi_rel, 'x_abs': x_abs, 'y_rel': y_rel, 'y_abs': y_abs,
                         'delta_f': run_parameters['learner_params']['delta_f'], 'rel_likelihood': GPpref.PrefProbit(),
                         'abs_likelihood': GPpref.AbsBoundProbit()}

        # Get initial solution
        for nl, learner in enumerate(learners):
            learner.build_model(training_data)
            learner.model.set_hyperparameters(log_hyp)
            f = learner.model.solve_laplace()
            fhat, vhat = learner.model.predict_latent(x_test)
            y_abs_est = learner.model.abs_posterior_mean(x_test, fhat, vhat)

            best_points_est = set(np.argpartition(y_abs_est.flatten(), -n_best_points)[-n_best_points:])
            wrms_results[nl, 0, trial_number] = test_data.wrms(y_abs_true, y_abs_est)
            true_pos_results[nl, 0, trial_number] = len(best_points_set.intersection(best_points_est))
            selected_error[nl, 0, trial_number] = test_data.wrms(y_abs_true[best_points], y_abs_est[best_points], weight=False)
            if calc_relative_error:
                p_rel_y_post = learner.model.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)
                relative_error[nl, 0, trial_number] = test_data.rel_error(y_abs_true, p_rel_y_true, y_abs_est, p_rel_y_post, weight=True)

        for obs_num in range(n_queries):
            t0 = time.time()
            linear_p_rel = max(0.0, (n_queries-obs_num)/float(n_queries))

            for nl, learner in enumerate(learners):
                # Update p_rel for certain methods (hacky?)
                if learner.update_p_rel:
                    learner.obs_arguments['p_rel'] = linear_p_rel

                next_x = learner.model.select_observation(**learner.obs_arguments)
                if next_x.shape[0] == 1:
                    next_y, next_f = abs_obs_fun.generate_observations(next_x)
                    learner.model.add_observations(next_x, next_y)
                    # print 'Abs: x:{0}, y:{1}'.format(next_x[0], next_y[0])
                else:
                    next_y, next_uvi, next_fx = rel_obs_fun.gaussian_multi_pairwise_sampler(next_x)
                    next_fuv = next_fx[next_uvi][:,:,0]
                    fuv_rel = np.concatenate((fuv_rel, next_fuv), 0)
                    learner.model.add_observations(next_x, next_y, next_uvi)
                    # print 'Rel: x:{0}, best_index:{1}'.format(next_x.flatten(), next_uvi[0, 1])
                f = learner.model.solve_laplace()
                fhat, vhat = learner.model.predict_latent(x_test)
                y_abs_est = learner.model.abs_posterior_mean(x_test, fhat, vhat)

                # Get selected best point set and error results
                best_points_est = set(np.argpartition(y_abs_est.flatten(), -n_best_points)[-n_best_points:])
                wrms_results[nl, obs_num+1, trial_number] = test_data.wrms_misclass(y_abs_true, y_abs_est)
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
    with open(data_dir+'wrms.pkl', 'wb') as fh:
        pickle.dump(wrms_results[:,:,:trial_number], fh)

    with open(data_dir+'true_pos.pkl', 'wb') as fh:
        pickle.dump(true_pos_results[:,:,:trial_number], fh)

    with open(data_dir+'selected_error.pkl', 'wb') as fh:
        pickle.dump(selected_error[:,:,:trial_number], fh)

    if calc_relative_error:
        with open(data_dir + 'relative_error.pkl', 'wb') as fh:
            pickle.dump(relative_error[:, :, :trial_number], fh)

    with open(data_dir+'obs.pkl', 'wb') as fh:
        pickle.dump(obs_array, fh)

    waver.save(data_dir+'wave_data.pkl')

hfig = plot_statruns.plot_results(wrms_results, true_pos_results, selected_error, obs_array, relative_error=relative_error, data_dir=data_dir, bars=True, norm_comparator=0)
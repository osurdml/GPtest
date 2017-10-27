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
    def __init__(self, model_type, obs_arguments, name):
        self.model_type = getattr(active_learners, self.model_type)
        self.obs_arguments = obs_arguments
        self.name = name

    def build_model(self, training_data):
        self.model = self.model_type(**training_data)

now_time = time.strftime("%Y_%m_%d-%H_%M")

with open('./data/statruns2_oct2017.yaml', 'rt') as fh:
    statrun_params = yaml.safe_load(fh)

log_hyp = np.log(statrun_params['hyperparameters'])

# Statrun parameters
np.random.seed(statrun_params['statrun_params']['randseed'])
n_rel_train = statrun_params['statrun_params']['n_rel_train']
n_abs_train = statrun_params['statrun_params']['n_abs_train']
n_xtest = statrun_params['statrun_params']['n_xtest']
n_best_points = statrun_params['statrun_params']['n_best_points']
n_trials = statrun_params['statrun_params']['n_trials']
n_queries = statrun_params['statrun_params']['n_queries']
if 'calc_relative_error' in statrun_params['statrun_params']:
    calc_relative_error = statrun_params['statrun_params']['calc_relative_error']
else:
    calc_relative_error = False

# Learner parameters
n_rel_samples = statrun_params['learner_params']['n_rel_samples']
delta_f = statrun_params['learner_params']['delta_f']


# Define polynomial function to be modelled
random_wave  = test_data.MultiWave(**statrun_params['wave_params'])

now_time = time.strftime("%Y_%m_%d-%H_%M")
data_dir = 'data/' + now_time + '/'
ptt.ensure_dir(data_dir)
print "Data will be saved to: {0}".format(data_dir)
waver = test_data.WaveSaver(n_trials, random_wave.n_components)
with open(data_dir + 'params.yaml', 'wt') as fh:
    yaml.safe_dump(statrun_params, fh)

# True function
x_plot = np.linspace(0.0, 1.0, n_xtest,dtype='float')
x_test = np.atleast_2d(x_plot).T

# Learners
learners = []
for l in statrun_params['learners']:
    learners.append(Learner(**l))

# Construct active learner object
learners = [Learner(active_learners.ActiveLearner, {'p_rel': 1.0, 'n_rel_samples': n_rel_samples},  'Random (rel)'),
            Learner(active_learners.ActiveLearner, {'p_rel': 0.0, 'n_rel_samples': n_rel_samples},  'Random (abs)'),
            Learner(active_learners.ActiveLearner, {'p_rel': 0.5, 'n_rel_samples': n_rel_samples},  'Random ($p_{rel}=0.5$)'),
            Learner(active_learners.MaxVar, {'n_test': 100},  'MaxVar (abs)'),
            Learner(active_learners.UCBLatent, {'gamma': 3.0, 'n_test': 100},  'UCB (abs)'),
            Learner(active_learners.UCBLatentSoftmax, {'gamma': 2.0, 'n_test': 100, 'tau':1.0},  'UCBSoft (abs)'),
            Learner(active_learners.UCBCovarianceSoftmax, {'gamma': 2.0, 'n_test': 100},  'UCBCovSoft (abs)'),
            Learner(active_learners.DetRelBoo, {'n_test': 100, 'n_rel_samples': n_rel_samples, 'gamma': 2.0, 'tau': 0.5}, 'DetRelBoo (rel)'),
            Learner(active_learners.DetSelect, {'n_test': 100, 'n_rel_samples': n_rel_samples, 'gamma': 2.0, 'abs_tau': -1, 'rel_tau': 1.0e-5}, 'DetSelectRel (rel)'),
            Learner(active_learners.DetSelect, {'n_test': 100, 'n_rel_samples': n_rel_samples, 'gamma': 2.0, 'abs_tau': 1.0e-5, 'rel_tau': 1.0e-5}, 'DetSelectGreedy (rel, abs)'),
            Learner(active_learners.DetSelect, {'n_test': 100, 'n_rel_samples': n_rel_samples, 'gamma': 2.0, 'abs_tau': 0.5, 'rel_tau': 0.5}, 'DetSelectSoft (rel, abs)'),
            Learner(active_learners.ExpectedImprovementAbs, { 'n_test': 100 }, 'EI (Abs)'),
            Learner(active_learners.ExpectedImprovementRel, { 'n_test': 100 }, 'EI (Rel)'),
            Learner(active_learners.UCBAbsRel, { 'n_test': 100, 'n_rel_samples': n_rel_samples, 'gamma': 2.0, 'tau': 1.0}, 'UCBCombined (rel, abs)'),
            # Learner(active_learners.UCBAbsRelD, { 'n_test': 100, 'n_rel_samples': n_rel_samples, 'gamma': 2.0, 'tau': 1.0}, 'UCBD (rel and abs)'),
            # Learner(active_learners.ABSThresh, {'n_test': 100, 'p_thresh': 0.7}, 'ABSThresh'),
            # Learner(active_learners.PeakComparitor, {'gamma': 2.0, 'n_test': 50, 'n_rel_samples': n_rel_samples}, 'PeakComparitor'),
            # Learner(active_learners.LikelihoodImprovement, {'req_improvement': 0.60, 'n_test': 50, 'gamma': 2.0, 'n_rel_samples': n_rel_samples, 'p_thresh': 0.7}, 'LikelihoodImprovement'),
            # Learner(active_learners.SampledThreshold, {'n_test':50, 'n_samples':10, 'y_threshold':0.8, 'p_pref_tol':1e-3, 'n_mc_abs':5}, 'SampledThreshold'),
            # Learner(active_learners.SampledClassification, {'n_test':50, 'n_samples':10, 'y_threshold':0.8, 'p_pref_tol':1e-3, 'n_mc_abs':5}, 'SampledClassification'),
            ]
names = [l.name for l in learners]
assert len(names) == len(learners), "Number of names does not match number of learners."
n_learners = len(learners)

obs_array = [{'name': name, 'obs': []} for name in names]

wrms_results = np.zeros((n_learners, n_queries+1, n_trials))
true_pos_results = np.zeros((n_learners, n_queries+1, n_trials), dtype='int')
selected_error = np.zeros((n_learners, n_queries+1, n_trials))
if calc_relative_error:
    relative_error = np.zeros((n_learners, n_queries+1, n_trials))
else:
    relative_error = None

trial_number = 0
# for trial_number in range(n_trials):
while trial_number < n_trials:

    try:
        print 'Trial {0}'.format(trial_number)
        random_wave.randomize()
        random_wave.print_values()
        waver.set_vals(trial_number, *random_wave.get_values())
        rel_obs_fun = GPpref.RelObservationSampler(random_wave.out, GPpref.PrefProbit(**statrun_params['rel_obs_params']))
        abs_obs_fun = GPpref.AbsObservationSampler(random_wave.out, GPpref.AbsBoundProbit(**statrun_params['abs_obs_params']))

        f_true = abs_obs_fun.f(x_test)
        y_abs_true = abs_obs_fun.mean_link(x_test)
        best_points = np.argpartition(y_abs_true.flatten(), -n_best_points)[-n_best_points:]
        best_points_set = set(best_points)
        # abs_y_samples = np.atleast_2d(np.linspace(0.01, 0.99, n_ysamples)).T
        # p_abs_y_true = abs_obs_fun.observation_likelihood_array(x_test, abs_y_samples)
        if calc_relative_error:
            p_rel_y_true = rel_obs_fun.observation_likelihood_array(x_test)

        # Initial data
        x_rel, uvi_rel, uv_rel, y_rel, fuv_rel = rel_obs_fun.generate_n_observations(n_rel_train, n_xdim=1)
        x_abs, y_abs, mu_abs = abs_obs_fun.generate_n_observations(n_abs_train, n_xdim=1)
        training_data = {'x_rel': x_rel, 'uvi_rel': uvi_rel, 'x_abs': x_abs, 'y_rel': y_rel, 'y_abs': y_abs,
                         'delta_f': delta_f, 'rel_likelihood': GPpref.PrefProbit(),
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
            learners[-1].obs_arguments['p_rel'] = max(0.0, (n_queries-obs_num)/float(n_queries))

            for nl, learner in enumerate(learners):
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
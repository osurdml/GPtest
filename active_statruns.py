# Simple 1D GP classification example
import time
import numpy as np
import GPpref
import plot_tools as ptt
import active_learners
# from active_learners import ActiveLearner, UCBLatent, UCBLatentSoftmax, LikelihoodImprovement, ABSThresh, UCBAbsRel, UCBAbsRelD
import test_data
import pickle
import plot_statruns

class Learner(object):
    def __init__(self, model_type, obs_arguments):
        self.model_type = model_type
        self.obs_arguments = obs_arguments

    def build_model(self, training_data):
        self.model = self.model_type(**training_data)


def wrms(y_true, y_est, weight=True):
    if weight:
        w = y_true
    else:
        w = 1.0
    return np.sqrt(np.mean(((y_true - y_est)*w)**2))


def wrms2(y_true, y_est):
    w = np.power(np.maximum(y_true, y_est), 2)
    return np.sqrt(np.mean(((y_true - y_est)*w)**2))


now_time = time.strftime("%Y_%m_%d-%H_%M")

# log_hyp = np.log([0.1,0.5,0.1,10.0]) # length_scale, sigma_f, sigma_probit, v_beta
# log_hyp = np.log([0.07, 0.75, 0.25, 1.0, 28.1])
# log_hyp = np.log([0.05, 1.5, 0.09, 2.0, 50.0])
log_hyp = np.log([0.018, 1.0, 0.2, 0.5, 60.0])
np.random.seed(1)

n_rel_train = 1
n_abs_train = 0
rel_sigma = 0.05
delta_f = 1e-5

beta_sigma = 0.5
beta_v = 80.0

n_xtest = 101
n_best_points = 15

# n_ysamples = 101
n_trials = 100
randomize_waves = True
n_rel_samples = 5

n_queries = 80

# Define polynomial function to be modelled
# random_wave = test_data.VariableWave([0.6, 1.0], [5.0, 10.0], [0.0, 1.0], [10.0, 20.0])
random_wave = test_data.MultiWave(amp_range=[0.6, 1.2], f_range=[10.0, 30.0], off_range=[0.1, 0.9],
                                     damp_range=[250.0, 350.0], n_components=3)

now_time = time.strftime("%Y_%m_%d-%H_%M")
data_dir = 'data/' + now_time + '/'
ptt.ensure_dir(data_dir)
print "Data will be saved to: {0}".format(data_dir)
waver = test_data.WaveSaver(n_trials, random_wave.n_components)

# True function
x_plot = np.linspace(0.0, 1.0, n_xtest,dtype='float')
x_test = np.atleast_2d(x_plot).T

# Construct active learner object
learners = [Learner(active_learners.ActiveLearner, {'p_rel': 0.5, 'n_rel_samples': n_rel_samples}),  # 'Random (rel and abs)',
            Learner(active_learners.ActiveLearner, {'p_rel': 1.0, 'n_rel_samples': n_rel_samples}),  # 'Random (rel)',
            Learner(active_learners.ActiveLearner, {'p_rel': 0.0, 'n_rel_samples': n_rel_samples}),  # 'Random (abs)',
            Learner(active_learners.UCBLatent, {'gamma': 4.0, 'n_test': 100}),  # 'UCBLatent'
            # Learner(UCBLatentSoftmax, {'gamma': 2.0, 'n_test': 100, 'tau':1.0}),  # 'UCBSoft (abs)',
            Learner(active_learners.ExpectedImprovementAbs, { 'n_test': 100 }),  # 'EI (Abs)',
            Learner(active_learners.ExpectedImprovementRel, { 'n_test': 100 }),  # 'EI (Rel)',
            Learner(active_learners.UCBAbsRel, { 'n_test': 100, 'n_rel_samples': n_rel_samples, 'gamma': 2.0, 'tau': 1.0}),  # 'UCBCombined',
            # Learner(UCBAbsRelD, { 'n_test': 100, 'n_rel_samples': n_rel_samples, 'gamma': 2.0, 'tau': 1.0}), # , 'UCBD (rel and abs)'
            # Learner(ABSThresh, {'n_test': 100, 'p_thresh': 0.7}),  # 'ABSThresh'
            # Learner(PeakComparitor, {'gamma': 2.0, 'n_test': 50, 'n_rel_samples': n_rel_samples}),  #  'PeakComparitor'
            # Learner(LikelihoodImprovement, {'req_improvement': 0.60, 'n_test': 50, 'gamma': 2.0, 'n_rel_samples': n_rel_samples, 'p_thresh': 0.7})  #  'LikelihoodImprovement'
            ]
names = ['Random (rel and abs)','Random (rel only)', 'Random (abs only)', 'UCB (abs only)', 'EI (Abs)', 'EI (Rel)', 'UCBCombined (rel and abs)']
assert len(names) == len(learners), "Number of names does not match number of learners."
n_learners = len(learners)

obs_array = [{'name': name, 'obs': []} for name in names]

wrms_results = np.zeros((n_learners, n_queries+1, n_trials))
true_pos_results = np.zeros((n_learners, n_queries+1, n_trials), dtype='int')
selected_error = np.zeros((n_learners, n_queries+1, n_trials))

trial_number = 0
# for trial_number in range(n_trials):
while trial_number < n_trials:

    try:
        print 'Trial {0}'.format(trial_number)
        if randomize_waves:
            random_wave.randomize()
        random_wave.print_values()
        waver.set_vals(trial_number, *random_wave.get_values())
        rel_obs_fun = GPpref.RelObservationSampler(random_wave.out, GPpref.PrefProbit(sigma=rel_sigma))
        abs_obs_fun = GPpref.AbsObservationSampler(random_wave.out, GPpref.AbsBoundProbit(sigma=beta_sigma, v=beta_v))
        f_true = abs_obs_fun.f(x_test)
        y_abs_true = abs_obs_fun.mean_link(x_test)
        best_points = np.argpartition(y_abs_true.flatten(), -n_best_points)[-n_best_points:]
        best_points_set = set(best_points)
        # abs_y_samples = np.atleast_2d(np.linspace(0.01, 0.99, n_ysamples)).T
        # p_abs_y_true = abs_obs_fun.observation_likelihood_array(x_test, abs_y_samples)
        # p_rel_y_true = rel_obs_fun.observation_likelihood_array(x_test)

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
            wrms_results[nl, 0, trial_number] = wrms(y_abs_true, y_abs_est)
            true_pos_results[nl, 0, trial_number] = len(best_points_set.intersection(best_points_est))
            selected_error[nl, 0, trial_number] = wrms(y_abs_true[best_points], y_abs_est[best_points], weight=False)

        for obs_num in range(n_queries):
            # learners[-2].obs_arguments['p_rel'] = max(0.0, (n_queries-obs_num)/float(n_queries))
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
                wrms_results[nl, obs_num+1, trial_number] = wrms2(y_abs_true, y_abs_est)
                true_pos_results[nl, obs_num+1, trial_number] = len(best_points_set.intersection(best_points_est))
                selected_error[nl, obs_num+1, trial_number] = wrms(y_abs_true[best_points], y_abs_est[best_points], weight=False)


            print true_pos_results[:, obs_num+1, trial_number]
            print wrms_results[:, obs_num+1, trial_number]
        for nl, learner in enumerate(learners):
            obs_tuple = learner.model.get_observations()
            obs_array[nl]['obs'].append(test_data.ObsObject(*obs_tuple))
        trial_number += 1

    except RuntimeError:
        if randomize_waves == True:
            print "Caught a bad laplace, try new trial wave"
            continue
        else:
            raise
    except np.linalg.LinAlgError:
        if randomize_waves == True:
            print "Caught a bad linalg inversion, try new trial wave"
            continue
        else:
            raise

with open(data_dir+'wrms.pkl', 'wb') as fh:
    pickle.dump(wrms_results, fh)

with open(data_dir+'true_pos.pkl', 'wb') as fh:
    pickle.dump(true_pos_results, fh)

with open(data_dir+'selected_error.pkl', 'wb') as fh:
    pickle.dump(selected_error, fh)

with open(data_dir+'obs.pkl', 'wb') as fh:
    pickle.dump(obs_array, fh)

waver.save(data_dir+'wave_data.pkl')

hfig = plot_statruns.plot_results(wrms_results, true_pos_results, selected_error, obs_array, data_dir=data_dir, bars=True, norm_comparator=0)
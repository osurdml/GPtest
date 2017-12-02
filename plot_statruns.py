import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
import test_data
import GPpref
import active_learners
from Tkinter import Tk
from tkFileDialog import askdirectory
# from test_data import ObsObject
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)


def single_plot(data, x=None, names=None, title='', xlabel='Number of samples', ylabel='', bars=False, percentile=0, precut=0, errorevery=5):
    data = data[:,precut:,:]
    n_trials = data.shape[2]

    if x is None:
        x = np.arange(data.shape[1])+precut

    hf, hax = plt.subplots()
    hl = []
    for dd in data:
        mean_err = np.nanmean(dd, axis=1)
        if percentile < 0:
            err_lo = np.nanstd(dd, axis=1, ddof=1) / np.sqrt(n_trials)
            err_hi = err_lo
        elif percentile == 0:
            err_lo = np.nanstd(dd, axis=1)
            err_hi = err_lo
        else:
            err_lo = mean_err - np.percentile(dd, percentile, axis=1)
            err_hi = np.percentile(dd, percentile + 50, axis=1) - mean_err
        err = np.array([err_lo, err_hi])

        if bars:
            hl.append(hax.errorbar(x, mean_err, yerr=err, capsize=2.0, errorevery=errorevery))
        else:
            hl.append(hax.plot(x, mean_err)[0])
    hax.legend(hl, names, loc='best')
    hax.set_title(title)
    hax.set_xlabel(xlabel)
    hax.set_ylabel(ylabel)
    return hf, hax

def plot_results(wrms_results, true_pos_results, selected_error, obs_array, relative_error=None, data_dir=None, bars=True, norm_comparator=0, exclusions=[]):
    methods_indexes = []
    for i in range(wrms_results.shape[0]):
        if i not in exclusions:
            methods_indexes.append(i)
    methods_indexes = np.array(methods_indexes)

    names = [obs_array[i]['name'] for i in methods_indexes]

    wrms_results=wrms_results[methods_indexes,:,:]
    true_pos_results=true_pos_results[methods_indexes,:,:]
    selected_error=selected_error[methods_indexes,:,:]


    f0, ax0 = single_plot(wrms_results, names=names, ylabel='Weighted RMSE', bars=bars, percentile=25)
    f1, ax1 = single_plot(true_pos_results, names=names, ylabel='True positive selections (out of 30)', bars=True, precut=1, percentile=25)
    f2, ax2 = single_plot(selected_error, names=names, ylabel='RMSE of best paths', bars=True, precut=1, percentile=25)
    f = [f0, f1, f2]
    ax = [ax0, ax1, ax2]

    try:
        norm_wrms = wrms_results/wrms_results[norm_comparator]
        f3, ax3 = single_plot(norm_wrms, names=names, title='Normalized weighted RMSE', bars=bars, percentile=0)
        f.append(f3)
        ax.append(ax3)
    except:
        pass

    if relative_error is not None:
        relative_error=relative_error[methods_indexes,:,:]
        fr, axr = single_plot(relative_error, names=names, ylabel='Mean relative prediction error', bars=True)
        f.append(fr); ax.append(axr)
        if data_dir is not None:
            fr.savefig(data_dir + 'rel_error.pdf', bbox_inches='tight')

    if data_dir is not None:
        f0.savefig(data_dir + 'wrms.pdf', bbox_inches='tight')
        f1.savefig(data_dir + 'true_pos.pdf', bbox_inches='tight')
        f2.savefig(data_dir + 'rms_best.pdf', bbox_inches='tight')
    # hl = []
    # for i in range(mean_err.shape[0]):
    #     hl.append(plt.errorbar(np.arange(mean_err.shape[1]), mean_err[i,:], yerr=std_err[i, :]))
    # plt.legend(hl, names)
    plt.show()
    return f


def _save_file(file, data):
    with open(file, 'wb') as fh:
        pickle.dump(data, fh)
    return

def save_data(data_dir, wrms, true_pos, sel_err, obs, rel_err=None, full_obs=None, t=None):
    if t is None:
        t = wrms.shape[2]
    _save_file(data_dir + 'wrms.pkl', wrms[:,:,:t])
    _save_file(data_dir + 'true_pos.pkl', true_pos[:,:,:t])
    _save_file(data_dir + 'selected_error.pkl', sel_err[:,:,:t])
    _save_file(data_dir + 'obs.pkl', obs)
    if rel_err is not None:
        _save_file(data_dir + 'relative_error.pkl', rel_err[:,:,:t])
    if full_obs is not None:
        fobs = {key:full_obs[key][:t] for key in full_obs}
        _save_file(data_dir + 'full_obs.pkl', fobs)

def _load_file(file):
    with open(file, 'rb') as fh:
        data = pickle.load(fh) # Dimensions n_learners, n_queries+1, n_trials
    return data

def _load_params(file):
    with open(file, 'rt') as fh:
        params = yaml.safe_load(fh)
    return params

def get_data_dir():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    data_dir = askdirectory(initialdir='./data/')
    if data_dir == '':  # Cancel clicked
        raise IOError('No directory selected')
    elif data_dir[-1] is not '/':
        data_dir += '/'
    return data_dir

def load_data(data_dir=None):
    if data_dir is None:
        data_dir = get_data_dir()

    wrms_results = _load_file(data_dir+'wrms.pkl')
    true_pos_results = _load_file(data_dir+'true_pos.pkl')
    selected_error = _load_file(data_dir+'selected_error.pkl')
    obs_array = _load_file(data_dir+'obs.pkl')
    try:
        relative_error = _load_file(data_dir+'relative_error.pkl')
    except IOError:
        print "No relative error data found."
        relative_error = None
    try:
        full_obs = _load_file(data_dir+'full_obs.pkl')
    except IOError:
        print "No full observation set found."
        full_obs = None
    return data_dir, wrms_results, true_pos_results, selected_error, obs_array, relative_error, full_obs


def load_and_plot(save_plots=True, *args, **kwargs):
    try:
        data_dir, wrms_results, true_pos_results, selected_error, obs_array, relative_error, full_obs = load_data()
    except IOError:
        return None

    if save_plots is False:
        data_dir = None
    hf = plot_results(wrms_results, true_pos_results, selected_error, obs_array, data_dir=data_dir, relative_error=relative_error, *args, **kwargs)
    plt.show()
    return hf

def get_selection():
    _dd, wrms, true_pos, sel_err, obs, rel_err, full_obs = load_data()
    print "The following models were found:"
    for i, obs_item in enumerate(obs):
        print "{0}: {1}".format(i, obs_item['name'])
    dex = input('Select desired models by index as an array: ')
    wrms, true_pos, sel_err = wrms[dex, :, :], true_pos[dex, :, :], sel_err[dex, :, :]
    if rel_err is not None:
        rel_err = rel_err[dex, :, :]
    obs = [obs[i] for i in dex]
    # finished = input('Finished? (0/1)')
    return wrms, true_pos, sel_err, obs, rel_err


def load_multiple(*args, **kwargs):
    wrms, true_pos, sel_err, obs, rel_err = get_selection()
    app = lambda a, b: np.append(a, b, axis=0)
    while True:
        try:
            wrms2, true_pos2, sel_err2, obs2, rel_err2 = get_selection()
        except IOError:
            break
        wrms, true_pos, sel_err = app(wrms, wrms2), app(true_pos, true_pos2), app(sel_err, sel_err2)
        if rel_err is not None: rel_err = app(rel_err, rel_err2)
        obs.extend(obs2)
    hf = plot_results(wrms, true_pos, sel_err, obs, relative_error=rel_err, *args, **kwargs)
    plt.show()
    if 'data_dir' in kwargs:
        save_data(kwargs['data_dir'], wrms, true_pos, sel_err, obs, rel_err)
        print 'Data saved to {0}'.format(kwargs['data_dir'])
    return hf, wrms, true_pos, sel_err, obs, rel_err


def build_ordinal_wrms(max_y = 0.8, *args, **kwargs):
    data_dir = get_data_dir()
    run_parameters = _load_params(data_dir+'params.yaml')
    wave_data = _load_file(data_dir + 'wave_data.pkl')
    full_obs = _load_file(data_dir + 'full_obs.pkl')
    random_wave = test_data.MultiWave(**run_parameters['wave_params'])

    if full_obs is None:
        raise IOError('full_obs.pkl not found, cannot reconstruct without full observation history')

    log_hyp = np.log(run_parameters['hyperparameters'])
    n_learners, n_queries = len(run_parameters['learners']), run_parameters['statrun_params']['n_queries']
    n_trials = len(full_obs[run_parameters['learners'][0]['name']])
    x_plot = np.linspace(0.0, 1.0, run_parameters['statrun_params']['n_xtest'], dtype='float')
    x_test = np.atleast_2d(x_plot).T

    wkld = np.zeros((n_learners, n_queries+1, n_trials), dtype='float')
    max_count = np.zeros((n_learners, n_queries+1, n_trials), dtype='float')

    learners, names = [], []
    for l in run_parameters['learners']:
        if 'n_rel_samples' in l['obs_args']:
            l['obs_args']['n_rel_samples'] = run_parameters['statrun_params']['n_rel_samples']
        learners.append(active_learners.Learner(**l))
        names.append(l['name'])

    for trial_number in range(n_trials):
        print 'Trial {0}'.format(trial_number)
        a, f, o, d = wave_data.amplitude[trial_number], wave_data.frequency[trial_number], wave_data.offset[trial_number], wave_data.damping[trial_number]
        random_wave.set_values(a, f, o, d)
        random_wave.print_values()
        abs_obs_fun = GPpref.AbsObservationSampler(random_wave.out, run_parameters['GP_params']['abs_likelihood'],
                                                   run_parameters['abs_obs_params'])

        min_max_label = np.floor(max_y * abs_obs_fun.l.y_list[-1])

        p_y_true = abs_obs_fun.observation_likelihood_array(x_test)
        true_y = abs_obs_fun.l.y_list[p_y_true.argmax(axis=0)]     # True maximum likelihood labels
        max_y_true = (true_y >= min_max_label)    # True best label indexes (bool array)
        n_max = float(max_y_true.sum())

        for obs_num in range(0, n_queries+1):
            for nl, learner in enumerate(learners):
                # Get initial observations and build models
                if obs_num == 0:
                    obs0 = full_obs[learner.name][trial_number][0]
                    x_rel, uvi_rel, x_abs, y_rel, y_abs = obs0.x_rel, obs0.uvi_rel, obs0.x_abs, obs0.y_rel, obs0.y_abs
                    model_kwargs = {'x_rel': x_rel, 'uvi_rel': uvi_rel, 'x_abs': x_abs, 'y_rel': y_rel, 'y_abs': y_abs,
                                    'rel_kwargs': run_parameters['rel_obs_params'],
                                    'abs_kwargs': run_parameters['abs_obs_params']}
                    model_kwargs.update(run_parameters['GP_params'])
                    learner.build_model(model_kwargs)
                    learner.model.set_hyperparameters(log_hyp)

                else:
                    next_x, next_y, next_uvi = full_obs[learner.name][trial_number][obs_num]
                    learner.model.add_observations(next_x, next_y, next_uvi)

                learner.model.solve_laplace()
                fhat, vhat = learner.model.predict_latent(x_test)
                p_y_est, mu_est = learner.model.abs_likelihood.posterior_likelihood(fhat, vhat)
                est_y = abs_obs_fun.l.y_list[p_y_est.argmax(axis=0)]
                max_y_est = (est_y >= min_max_label)

                wkld[nl, obs_num, trial_number] = test_data.ordinal_kld(p_y_true, p_y_est, np.maximum(true_y, est_y))
                max_count[nl, obs_num, trial_number] = np.logical_and(max_y_true, max_y_est).sum() / n_max


    _save_file(data_dir+'wkld.pkl', wkld)
    _save_file(data_dir+'max_count.pkl', max_count)
    plot_ordinal_results(wkld, max_count, run_parameters=run_parameters, data_dir=data_dir)

def plot_ordinal_results(wkld, max_count, run_parameters = None, data_dir=None, bars=True, exclusions=[]):
    if run_parameters is None:
        with open(data_dir + 'params.yaml', 'rt') as fh:
            run_parameters = yaml.safe_load(fh)

    names = []
    for l in run_parameters['learners']:
        names.append(l['name'])

    methods_indexes = []
    for i in range(len(names)):
        if i not in exclusions:
            methods_indexes.append(i)
    methods_indexes = np.array(methods_indexes)

    wkld=wkld[methods_indexes,:,:]
    max_count=max_count[methods_indexes,:,:]

    f0, ax0 = single_plot(wkld, names=names, ylabel='Weighted KLD', bars=bars, percentile=-1)
    f1, ax1 = single_plot(max_count, names=names, ylabel='Proportion of selected max points', bars=True, precut=1, percentile=-1)
    f = [f0, f1]
    ax = [ax0, ax1]

    if data_dir is not None:
        f0.savefig(data_dir + 'wkld.pdf', bbox_inches='tight')
        f1.savefig(data_dir + 'max_count.pdf', bbox_inches='tight')
    plt.show()
    return f

if __name__ == "__main__":
    # build_ordinal_wrms()
    hf = load_and_plot(save_plots=False, bars=True)
import numpy as np
import matplotlib.pyplot as plt
import pickle
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


    f0, ax0 = single_plot(wrms_results, names=names, ylabel='Weighted RMSE', bars=bars, percentile=-1)
    f1, ax1 = single_plot(true_pos_results, names=names, ylabel='True positive selections (out of 30)', bars=True, precut=1, percentile=-1)
    f2, ax2 = single_plot(selected_error, names=names, ylabel='RMSE of best paths', bars=True, precut=1, percentile=-1)
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


def save_data(data_dir, wrms, true_pos, sel_err, obs, rel_err=None):
    _save_file(data_dir + 'wrms.pkl', wrms)
    _save_file(data_dir + 'true_pos.pkl', true_pos)
    _save_file(data_dir + 'selected_error.pkl', sel_err)
    _save_file(data_dir + 'obs.pkl', obs)
    if rel_err is not None:
        _save_file(data_dir + 'relative_error.pkl', rel_err)


def _load_file(file):
    with open(file, 'rb') as fh:
        data = pickle.load(fh) # Dimensions n_learners, n_queries+1, n_trials
    return data


def load_data(data_dir=None):
    if data_dir is None:
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        data_dir = askdirectory(initialdir='./data/')
    if data_dir == '':  # Cancel clicked
        raise IOError('No directory selected')
    elif data_dir[-1] is not '/':
        data_dir += '/'

    wrms_results = _load_file(data_dir+'wrms.pkl')
    true_pos_results = _load_file(data_dir+'true_pos.pkl')
    selected_error = _load_file(data_dir+'selected_error.pkl')
    obs_array = _load_file(data_dir+'obs.pkl')
    try:
        relative_error = _load_file(data_dir+'relative_error.pkl')
    except IOError:
        print "No relative error data found."
        relative_error = None

    return data_dir, wrms_results, true_pos_results, selected_error, obs_array, relative_error


def load_and_plot(save_plots=True, *args, **kwargs):
    try:
        data_dir, wrms_results, true_pos_results, selected_error, obs_array, relative_error = load_data()
    except IOError:
        return None

    if save_plots is False:
        data_dir = None
    hf = plot_results(wrms_results, true_pos_results, selected_error, obs_array, data_dir=data_dir, relative_error=relative_error, *args, **kwargs)
    plt.show()
    return hf

def get_selection():
    _dd, wrms, true_pos, sel_err, obs, rel_err = load_data()
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

if __name__ == "__main__":
    hf = load_and_plot(save_plots=False, bars=True)
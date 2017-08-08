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
        mean_err = np.mean(dd, axis=1)
        if percentile < 0:
            err_lo = np.std(dd, axis=1, ddof=1) / np.sqrt(n_trials)
            err_hi = err_lo
        elif percentile == 0:
            err_lo = np.std(dd, axis=1)
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


def plot_results(wrms_results, true_pos_results, selected_error, obs_array, data_dir=None, bars=True, norm_comparator=0, exclusions=[4]):
    methods_indexes = []
    for i in range(wrms_results.shape[0]):
        if i not in exclusions:
            methods_indexes.append(i)
    methods_indexes = np.array(methods_indexes)

    names = [obs_array[i]['name'] for i in methods_indexes]

    wrms_results=wrms_results[methods_indexes,:,:]
    true_pos_results=true_pos_results[methods_indexes,:,:]
    selected_error=selected_error[methods_indexes,:,:]

    f0, ax0 = single_plot(wrms_results, names=names, ylabel='Weighted RMSE', bars=bars)
    f1, ax1 = single_plot(true_pos_results, names=names, ylabel='True positive selections (out of 15)', bars=True, precut=1, percentile=0)
    f2, ax2 = single_plot(selected_error, names=names, ylabel='RMSE of best paths', bars=True, precut=1)
    f = [f0, f1, f2]
    ax = [ax0, ax1, ax2]

    try:
        norm_wrms = wrms_results/wrms_results[norm_comparator]
        f3, ax3 = single_plot(norm_wrms, names=names, title='Normalized weighted RMSE', bars=bars, percentile=0)
        f.append(f3)
        ax.append(ax3)
    except:
        pass

    if data_dir is not None:
        f0.savefig(data_dir + '/wrms.pdf', bbox_inches='tight')
        f1.savefig(data_dir + '/true_pos.pdf', bbox_inches='tight')
        f2.savefig(data_dir + '/rms_best.pdf', bbox_inches='tight')
    # hl = []
    # for i in range(mean_err.shape[0]):
    #     hl.append(plt.errorbar(np.arange(mean_err.shape[1]), mean_err[i,:], yerr=std_err[i, :]))
    # plt.legend(hl, names)
    plt.show()
    return f

def load_data(data_dir):
    with open(data_dir+'/wrms.pkl', 'rb') as fh:
        wrms_results = pickle.load(fh) # Dimensions n_learners, n_queries+1, n_trials

    with open(data_dir+'/true_pos.pkl', 'rb') as fh:
        true_pos_results = pickle.load(fh)

    with open(data_dir+'/selected_error.pkl', 'rb') as fh:
        selected_error = pickle.load(fh)

    with open(data_dir+'/obs.pkl', 'rb') as fh:
        obs_array = pickle.load(fh)

    return wrms_results, true_pos_results, selected_error, obs_array

def load_and_plot(save_plots=True, *args, **kwargs):
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    data_dir = askdirectory(initialdir='./data/') # open folder GUI

    wrms_results, true_pos_results, selected_error, obs_array = load_data(data_dir)

    if not save_plots:
        data_dir = None
    hf = plot_results(wrms_results, true_pos_results, selected_error, obs_array, data_dir=data_dir, *args, **kwargs)
    plt.show()
    return hf

if __name__ == "__main__":
    hf = load_and_plot(save_plots=False, bars=True)
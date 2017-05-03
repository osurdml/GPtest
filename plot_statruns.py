import numpy as np
import matplotlib.pyplot as plt
import pickle
from Tkinter import Tk
from tkFileDialog import askdirectory
# from test_data import ObsObject


def single_plot_SEM(data, x=None, names=None, title=None, xlabel='Number of samples', bars=False):
    mean_err = np.mean(data, axis=2)
    std_err_mean = np.std(data, axis=2, ddof=1)
    if x is None:
        x = np.arange(data.shape[1])

    hf, hax = plt.subplots()
    hl = []
    for mu, sig in zip(mean_err, std_err_mean):
        if bars:
            hl.append(hax.errorbar(x, mu, yerr=sig, capsize=2.0))
        else:
            hl.append(hax.plot(x, mu)[0])
    hax.legend(hl, names, loc='best')
    hax.set_title(title)
    hax.set_xlabel(xlabel)
    return hf, hax


def plot_statrun_results(wrms_results, true_pos_results, selected_error, obs_array, data_dir=None, bars=False):
    names = [l['name'] for l in obs_array]

    f0, ax0 = single_plot_SEM(wrms_results, names=names, title='Weighted RMSE', bars=bars)
    f1, ax1 = single_plot_SEM(true_pos_results, names=names, title='True positive selections', bars=False)
    f2, ax2 = single_plot_SEM(selected_error, names=names, title='RMSE of best paths', bars=False)

    if data_dir is not None:
        f0.savefig(data_dir + '/wrms.pdf', bbox_inches='tight')
        f1.savefig(data_dir + '/true_pos.pdf', bbox_inches='tight')
        f2.savefig(data_dir + '/rms_best.pdf', bbox_inches='tight')
    # hl = []
    # for i in range(mean_err.shape[0]):
    #     hl.append(plt.errorbar(np.arange(mean_err.shape[1]), mean_err[i,:], yerr=std_err[i, :]))
    # plt.legend(hl, names)
    return f0, f1, f2

def load_and_plot(save_plots=True, *args, **kwargs):
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    data_dir = askdirectory() # show an "Open" dialog box and return the path to the selected file

    with open(data_dir+'/wrms.pkl', 'rb') as fh:
        wrms_results = pickle.load(fh) # Dimensions n_learners, n_queries+1, n_trials

    with open(data_dir+'/true_pos.pkl', 'rb') as fh:
        true_pos_results = pickle.load(fh)

    with open(data_dir+'/selected_error.pkl', 'rb') as fh:
        selected_error = pickle.load(fh)

    with open(data_dir+'/obs.pkl', 'rb') as fh:
        obs_array = pickle.load(fh)

    if not save_plots:
        data_dir = None
    plot_statrun_results(wrms_results, true_pos_results, selected_error, obs_array, data_dir=data_dir, *args, **kwargs)
    plt.show()

if __name__ == "__main__":
    load_and_plot(save_plots=True, bars=True)
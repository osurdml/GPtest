import numpy as np
import matplotlib.pyplot as plt
import pickle
from Tkinter import Tk
from tkFileDialog import askopenfilename, askdirectory
from test_data import ObsObject

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


names = [l['name'] for l in obs_array]
n_queries = wrms_results.shape[1]-1
mean_err = np.mean(wrms_results, axis=2)
std_err = np.std(wrms_results, axis=2)

f0, ax0 = plt.subplots()
hl = ax0.plot(np.arange(n_queries+1), np.mean(wrms_results, axis=2).T)
f0.legend(hl, names)
ax0.set_title('Weighted RMSE')

f1, ax1 = plt.subplots()
hl1 = ax1.plot(np.arange(n_queries+1), np.mean(true_pos_results, axis=2).T/15.0)
f1.legend(hl1, names)
ax1.set_title('True positive selections')

f2, ax2 = plt.subplots()
hl2 = ax2.plot(np.arange(n_queries+1), np.mean(selected_error, axis=2).T)
f2.legend(hl2, names)
ax2.set_title('RMSE of best paths')


# hl = []
# for i in range(mean_err.shape[0]):
#     hl.append(plt.errorbar(np.arange(mean_err.shape[1]), mean_err[i,:], yerr=std_err[i, :]))
# plt.legend(hl, names)

plt.show()
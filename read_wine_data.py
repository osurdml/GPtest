import numpy as np
import pandas
import matplotlib.pyplot as plt
import yaml
import GPpref
import scipy.optimize as op
from utils.data_downloader import MrDataGrabber


class WineQualityData(object):
    def __init__(self, data_file):
        self.file = data_file
        self.data = pandas.read_csv(self.file, delimiter=';')

optimise_hyper = True
use_normalised = True
R_limit = 0.1
wine_type = 'white'

# Get wine data:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-{0}.csv'.format(wine_type)
file_loc = 'data/wine_quality/'
download_data = MrDataGrabber(url, file_loc)
download_data.download()

input_data = WineQualityData(download_data.target_file)

fh, ah = plt.subplots(1, 1)
ah.hist(input_data.data.quality, np.arange(-0.5, 11, 1.0))
ah.set_xticks(np.arange(11))
ah.set_xlabel('Score')
ah.set_ylabel('Count')
ah.set_title('{0} wine'.format(wine_type))

with open('./data/statruns_wine.yaml', 'rt') as fh:
    config = yaml.safe_load(fh)
d_x = config['GP_params']['hyper_counts'][0]-1
log_hyp = np.log(config['hyperparameters'])

wine_norm = (input_data.data.values - input_data.data.values.min(axis=0)) / (input_data.data.values.max(axis=0) - input_data.data.values.min(axis=0))

if use_normalised:
    wine_data = wine_norm
else:
    wine_data = input_data.data.values
fh2, ah2 = plt.subplots(3, 4)
for i, aa in enumerate(ah2.flat):
   aa.cla()
   aa.scatter(wine_data[:,i], wine_data[:,-1], 2, wine_data[:,0])
   aa.set_title('{0}, {1}'.format(input_data.data.keys()[i], config['hyperparameters'][i]))

fh3, ah3 = plt.subplots()
C = np.corrcoef(wine_norm, rowvar=False)
hmat = ah3.matshow(np.abs(C))

plt.show()
for R, var_name in zip(C[0:-1,-1], input_data.data.keys()):
    print('{0}: R = {1:0.2f}'.format(var_name, R))
corr_variables = (abs(C[-1]) >= R_limit)

# Subsample wine data down to only correlated variables
# wine_data = wine_data[:, corr_variables]

if optimise_hyper:
    n_rel = 1
    n_abs = 300
    x_rel = wine_data[0:2*n_rel, 0:-1]
    uvi_rel = np.array([[0, 1]])
    fuv_rel = wine_data[0:2*n_rel, -1]
    y_rel = np.array([1])

    x_abs = wine_data[2*n_rel:(2*n_rel+n_abs), 0:-1]
    y_abs = np.expand_dims(input_data.data.values[2 * n_rel:(2 * n_rel + n_abs), -1] - 2, axis=1).astype(dtype='int')
    model_kwargs = {'x_rel': x_rel, 'uvi_rel': uvi_rel, 'x_abs': x_abs, 'y_rel': y_rel, 'y_abs': y_abs,
                    'rel_kwargs': config['rel_obs_params'], 'abs_kwargs': config['abs_obs_params']}
    model_kwargs.update(config['GP_params'])
    model_kwargs['verbose'] = 2
    prefGP = GPpref.PreferenceGaussianProcess(**model_kwargs)

    prefGP.set_hyperparameters(log_hyp)
    f = prefGP.calc_laplace(log_hyp)
    print np.exp(log_hyp)
    log_hyp_opt = op.fmin(prefGP.calc_nlml,log_hyp)
    print np.exp(log_hyp_opt)
    config['hyperparameters'] = np.exp(log_hyp_opt).tolist()
    with open('data/wine_quality/learned_{0}_params.yaml'.format(wine_type), 'w') as fh:
        yaml.safe_dump(config, fh)

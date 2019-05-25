import numpy as np
import pandas
import matplotlib.pyplot as plt
import yaml
import GPpref
import scipy.optimize as op

# Get wine data:
# wget -P data/wine_quality https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
# wget -P data/wine_quality https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
class WineQualityData(object):

    def __init__(self, data_file):
        self.file = data_file
        self.data = pandas.read_csv(self.file, delimiter=';')

red_data = WineQualityData('data/wine_quality/winequality-red.csv')
white_data = WineQualityData('data/wine_quality/winequality-white.csv')

fh, ah = plt.subplots(1, 2)
ah[0].hist(red_data.data.quality, np.arange(-0.5, 11, 1.0))
ah[1].hist(white_data.data.quality, np.arange(-0.5, 11, 1.0))
for a in ah:
    a.set_xticks(np.arange(11))
    a.set_xlabel('Score')
ah[0].set_ylabel('Count')
ah[0].set_title('Red wine')
ah[1].set_title('White wine')

plt.show(block=False)

with open('./data/statruns_wine.yaml', 'rt') as fh:
    config = yaml.safe_load(fh)
d_x = config['GP_params']['hyper_counts'][0]-1
log_hyp = np.log(config['hyperparameters'])

n_rel = 1
n_abs = 300
x_rel = red_data.data.values[0:2*n_rel, 0:-1]
uvi_rel = np.array([[0, 1]])
fuv_rel = red_data.data.values[0:2*n_rel, -1]
y_rel = np.array([1])

x_abs = red_data.data.values[2*n_rel:(2*n_rel+n_abs), 0:-1]
y_abs = np.expand_dims(red_data.data.values[2*n_rel:(2*n_rel+n_abs), -1]-2, axis=1).astype(dtype='int')
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

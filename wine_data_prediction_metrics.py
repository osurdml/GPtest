import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from gp_tools import GPpref
from utils import test_data
import scipy.optimize as op
from utils.wine_data import WineQualityData

parser = argparse.ArgumentParser(description='Wine data metrics')
parser.add_argument('-np', '--no-plots', dest='make_plots', action='store_false', help='Turn off plots (default False)')
parser.add_argument('-y', '--yaml-config', default='./config/learned_red_params.yaml', help='YAML config file')
parser.add_argument('-na', '--n-abs', default=300, type=int, help='Number of absolute training points')
parser.add_argument('-nr', '--n-rel', default=1, type=int, help='Number of relative training points')
args = parser.parse_args()

make_plots = args.make_plots

# Read config file
with open(args.yaml_config, 'rt') as fh:
    config = yaml.safe_load(fh)

wine_type = config['wine_params']['type']

# Get wine data:
input_data = WineQualityData(wine_type=wine_type, cols=config['wine_params']['variables'], norm=config['wine_params']['normalise_data'], scale_y=True)
print('Loaded {0} wine data. Contains {1} samples, {2} input dimensions loaded'.format(wine_type, input_data.x.shape[0], input_data.x.shape[1]))

# Generate quality metrics by ramping over number of samples and
hyper = config['hyperparameters']
d_x = len(hyper['l'])
all_hyper = np.concatenate((hyper['l'], [hyper['sig_f'], hyper['sig_rel'], hyper['sig_beta'], hyper['v_beta']]))
log_hyp = np.log(all_hyper)

# Reset hyper count
config['GP_params']['hyper_counts'] = [d_x+1, 1, 2]

# We just get the first entries (data is not sorted)
n_rel = args.n_rel
uv_first = np.arange(2*n_rel, dtype=int).reshape((n_rel,2))
x_rel, uvi_rel, y_rel, f_rel = input_data.get_relative_obs(uv_first)

jump_val = 5
n_abs = args.n_abs
abs_first = np.arange(n_abs)+(n_rel*2)-jump_val
x_abs, y_abs = input_data.get_data(abs_first)
p_y_true = test_data.renorm_p(input_data.p_y_true.T, axis=0)

model_kwargs = {'x_rel': x_rel, 'uvi_rel': uvi_rel, 'x_abs': x_abs, 'y_rel': y_rel, 'y_abs': y_abs,
                'rel_kwargs': config['rel_obs_params'], 'abs_kwargs': config['abs_obs_params']}
model_kwargs.update(config['GP_params'])
model_kwargs['verbose'] = 2
prefGP = GPpref.PreferenceGaussianProcess(**model_kwargs)

prefGP.set_hyperparameters(log_hyp)
f = prefGP.calc_laplace(log_hyp)

# Use remaining as test data
test_index = np.arange(abs_first[-1] + 1, input_data.x.shape[0], 1, dtype=int)
wrms, kld = [], []

for end_index in range(2*n_rel + n_abs, input_data.x.shape[0]-50, jump_val):
    x_test, y_true = input_data.get_data(test_index)
    fhat, vhat = prefGP.predict_latent(x_test)

    p_y_est, mu_est = prefGP.abs_likelihood.posterior_likelihood(fhat, vhat)
    p_y_est = test_data.renorm_p(p_y_est, axis=0)
    est_y = p_y_est.argmax(axis=0)
    y_abs_est = prefGP.abs_posterior_mean(x_test, fhat, vhat)
    wrms.append(test_data.wrms())
    kld.append(test_data.ordinal_kld(input_data.p_y_true, p_y_est, np.maximum(input_data.y[test_index], est_y))

    new_index = np.arange(new_index[-1], end_index)

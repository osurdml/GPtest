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
parser.add_argument('-j', '--jump-val', default=0, type=int, help='Step size for sweeping over samples')
parser.add_argument('--shuffle', action='store_true', help='Shuffle input data')
args = parser.parse_args()

make_plots = args.make_plots
jump_val = args.jump_val

# Read config file
with open(args.yaml_config, 'rt') as fh:
    config = yaml.safe_load(fh)

wine_type = config['wine_params']['type']

# Get wine data:
input_data = WineQualityData(wine_type=wine_type, cols=config['wine_params']['variables'], norm=config['wine_params']['normalise_data'], scale_y=True)
print('Loaded {0} wine data. Contains {1} samples, {2} input dimensions loaded'.format(wine_type, input_data.x.shape[0], input_data.x.shape[1]))
# Shuffle the data
if args.shuffle:
    input_data.shuffle()

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

n_abs = args.n_abs
abs_first = np.arange(n_abs-jump_val)+(n_rel*2)
x_abs, y_abs = input_data.get_data(abs_first)
p_y_true = test_data.renorm_p(input_data.p_y_true.T, axis=0)

model_kwargs = {'x_rel': x_rel, 'uvi_rel': uvi_rel, 'x_abs': x_abs, 'y_rel': y_rel, 'y_abs': y_abs,
                'rel_kwargs': config['rel_obs_params'], 'abs_kwargs': config['abs_obs_params']}
model_kwargs.update(config['GP_params'])
model_kwargs['verbose'] = 0
prefGP = GPpref.PreferenceGaussianProcess(**model_kwargs)

prefGP.set_hyperparameters(log_hyp)
f = prefGP.calc_laplace()

# Each step we add jump_val
new_obs_index = np.arange(jump_val) + abs_first[-1]+1
wrms, kld, data_proportion = [], [], []

while new_obs_index[-1] < input_data.x.shape[0]:
# for i in range(2*n_rel + n_abs, input_data.x.shape[0]-50, jump_val):

    next_x, next_y = input_data.get_data(new_obs_index)
    prefGP.add_observations(next_x, next_y)
    f = prefGP.calc_laplace()

    # Use remaining as test data
    test_index = np.arange(new_obs_index[-1]+1, input_data.x.shape[0], 1, dtype=int)

    x_test, y_true = input_data.get_data(test_index)
    fhat, vhat = prefGP.predict_latent(x_test)

    p_y_est, mu_est = prefGP.abs_likelihood.posterior_likelihood(fhat, vhat)
    p_y_est = test_data.renorm_p(p_y_est, axis=0)
    est_y = p_y_est.argmax(axis=0)
    y_abs_est = prefGP.abs_posterior_mean(x_test, fhat, vhat)
    wrms.append(test_data.wrms(y_true, y_abs_est))
    kld.append(test_data.ordinal_kld(p_y_true[:,test_index], p_y_est, np.maximum(input_data.y[test_index], est_y)))

    data_proportion.append(float(prefGP.x_abs.shape[0])/input_data.x.shape[0])
    print('Proportion: {0}'.format(data_proportion[-1]*100.0))

    new_obs_index = np.arange(jump_val) + new_obs_index[-1]+1

if make_plots:
    fh, ah = plt.subplots(1, 2)
    ah[0].plot(np.array(data_proportion)*100.0, wrms)
    ah[0].set_xlabel('Data proportion (%)')
    ah[0].set_ylabel('WRMS')
    ah[1].plot(np.array(data_proportion)*100.0, kld)
    ah[1].set_xlabel('Data proportion (%)')
    ah[1].set_ylabel('KL-Divergence (one-hot vs $p(y|X)$)')
    # fh.savefig('fig/red_wine_losses.pdf', bbox_inches='tight')

plt.show()

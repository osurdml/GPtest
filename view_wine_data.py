import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from gp_tools import GPpref
import scipy.optimize as op
from utils.wine_data import WineQualityData

parser = argparse.ArgumentParser(description='Viewing and learning hypers from wine data')
parser.add_argument('-np', '--no-plots', dest='make_plots', action='store_false', help='Turn off plots (default False)')
parser.add_argument('-oh', '--optimise-hyper', action='store_true', help='Optimise hyperparameters')
parser.add_argument('-R', '--R-limit', default=0.1, type=float, help='Remove dimensions with regression coffecient < R')
parser.add_argument('-y', '--yaml-config', default='./config/wine_basedata.yaml', help='YAML config file')
parser.add_argument('-o', '--yaml-out', default=None, help='YAML config file')
parser.add_argument('-na', '--n-abs', default=300, type=int, help='Number of absolute training points')
parser.add_argument('-nr', '--n-rel', default=1, type=int, help='Number of relative training points')
args = parser.parse_args()

optimise_hyper = args.optimise_hyper
R_limit = args.R_limit
make_plots = args.make_plots

# Read config file
with open(args.yaml_config, 'rt') as fh:
    config = yaml.safe_load(fh)

wine_type = config['wine_params']['type']
if optimise_hyper:
    if args.yaml_out is None:
        out_yaml = './config/learned_{0}_params.yaml'.format(wine_type)
    else:
        out_yaml = args.yaml_out

# Get wine data:
input_data = WineQualityData(wine_type=wine_type, cols=config['wine_params']['variables'], norm=config['wine_params']['normalise_data'], scale_y=False)
print('Loaded {0} wine data. Contains {1} samples, {2} input dimensions loaded'.format(wine_type, input_data.x.shape[0], input_data.x.shape[1]))

if make_plots:
    fh, ah = plt.subplots(1, 1)
    ah.hist(input_data.y, np.arange(-0.5, 11, 1.0))
    ah.set_xticks(np.arange(11))
    ah.set_xlabel('Score')
    ah.set_ylabel('Count')
    ah.set_title('{0} wine'.format(input_data.type))

    fh2, ah2 = plt.subplots(3, 4)
    for i, dc in enumerate(input_data.data_cols):
       ah2.flat[i].cla()
       ah2.flat[i].scatter(input_data.x[:,i], input_data.y, 2, input_data.x[:,0])
       ah2.flat[i].set_title('{0}, l={1}'.format(dc, config['hyperparameters']['l'][i]))

# Scale y to have minimum score 1
input_data._scale_y()
full_mat = np.concatenate((input_data.x, input_data.y), axis=1)
C = np.corrcoef(full_mat, rowvar=False)

if make_plots:
    fh3, ah3 = plt.subplots()
    hmat = ah3.matshow(np.abs(C))
    ah3.set_title('Input dimension correlation coefficient matrix')

for R, var_name in zip(C[0:-1,-1], input_data.data_cols):
    print('{0}: R = {1:0.2f}'.format(var_name, R))
corr_variables = (abs(C[-1]) >= R_limit)

# Subsample wine data down to only correlated variables
new_cols = [c for i,c in enumerate(input_data.data_cols) if corr_variables[i] == True]
input_data._reset_cols(new_cols)

hyper = config['hyperparameters']
all_hyper = np.array(hyper['l'])[corr_variables[:-1]]
d_x = len(all_hyper)
all_hyper = np.concatenate((all_hyper, [hyper['sig_f'], hyper['sig_rel'], hyper['sig_beta'], hyper['v_beta']]))
log_hyp = np.log(all_hyper)

# Reset hyper count
config['GP_params']['hyper_counts'] = [d_x+1, 1, 2]
config['abs_obs_params']['n_ordinals'] = int(input_data.y.max())

if optimise_hyper:
    n_rel = args.n_rel
    x_rel, uvi_rel, y_rel, f_rel = input_data.random_relative_obs(n_rel)

    n_abs = args.n_abs
    x_abs, y_abs = input_data.random_absolute_obs(n_abs)

    model_kwargs = {'x_rel': x_rel, 'uvi_rel': uvi_rel, 'x_abs': x_abs, 'y_rel': y_rel, 'y_abs': y_abs,
                    'rel_kwargs': config['rel_obs_params'], 'abs_kwargs': config['abs_obs_params']}
    model_kwargs.update(config['GP_params'])
    model_kwargs['verbose'] = 2
    prefGP = GPpref.PreferenceGaussianProcess(**model_kwargs)

    prefGP.set_hyperparameters(log_hyp)
    f = prefGP.calc_laplace(log_hyp)
    print ('Old hyper: {0}'.format(all_hyper))
    log_hyp_opt = op.fmin(prefGP.calc_nlml,log_hyp)

    learned_hyperparameters = np.exp(log_hyp_opt).tolist()
    print ('New hyper: {0}'.format(learned_hyperparameters))

    config['hyperparameters']['l'] = learned_hyperparameters[0:-4]
    config['hyperparameters']['sig_f'] = learned_hyperparameters[-4]
    config['hyperparameters']['sig_rel'] = learned_hyperparameters[-3]
    config['hyperparameters']['sig_beta'] = learned_hyperparameters[-2]
    config['hyperparameters']['v_beta'] = learned_hyperparameters[-1]

    config['wine_params']['variables'] = input_data.data_cols

    with open(out_yaml, 'w') as fh:
        yaml.safe_dump(config, fh)

if make_plots:
    plt.show()

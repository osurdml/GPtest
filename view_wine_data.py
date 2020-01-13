import numpy as np
import matplotlib.pyplot as plt
import yaml
from gp_tools import GPpref
import scipy.optimize as op
from utils.wine_data import WineQualityData


optimise_hyper = False
R_limit = 0.1

# Read config file
with open('./config/wine_basedata.yaml', 'rt') as fh:
    config = yaml.safe_load(fh)

wine_type = config['wine_params']['type']
out_yaml = './config/learned_{0}_params.yaml'.format(wine_type)

# Get wine data:
input_data = WineQualityData(wine_type=wine_type, cols='all', norm=config['wine_params']['normalise_data'], scale_y=False)

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

fh3, ah3 = plt.subplots()
full_mat = np.concatenate((input_data.x, input_data.y), axis=1)
C = np.corrcoef(full_mat, rowvar=False)
hmat = ah3.matshow(np.abs(C))

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

if optimise_hyper:
    n_rel = 1
    uvi_rel = np.array([[0, 1]])
    x_rel, y_rel, fuv_rel = input_data.get_relative_obs(uvi_rel)
    # x_rel = wine_data[0:2*n_rel, 0:-1]
    # fuv_rel = wine_data[0:2*n_rel, -1]
    # y_rel = np.array([1])

    n_abs = 300
    x_abs, y_abs = input_data.random_absolute_obs(n_abs)
    # x_abs = wine_data[2*n_rel:(2*n_rel+n_abs), 0:-1]
    # y_abs = np.expand_dims(input_data.data.values[2 * n_rel:(2 * n_rel + n_abs), -1] - 2, axis=1).astype(dtype='int')

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

    config['GP_params']['hyper_counts'] = [d_x, 1, 2]
    config['hyperparameters']['l'] = learned_hyperparameters[0:-4]
    config['hyperparameters']['sig_f'] = learned_hyperparameters[-4]
    config['hyperparameters']['sig_rel'] = learned_hyperparameters[-3]
    config['hyperparameters']['sig_beta'] = learned_hyperparameters[-2]
    config['hyperparameters']['v_beta'] = learned_hyperparameters[-1]

    config['wine_data']['variables'] = input_data.data_cols

    with open(out_yaml, 'w') as fh:
        yaml.safe_dump(config, fh)

plt.show()

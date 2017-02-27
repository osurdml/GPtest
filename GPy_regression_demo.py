import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#from matplotlib.collections import PatchCollection
import nice_plot_colors as npc
import GPy
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)


np.random.seed(1)
noise = 0.2
n_training = 8
opt_hyper = False

gp_l = 0.02
gp_var = 1.0**2
gp_noisevar = 0.1**2

# Define polynomial function to be modelled
def true_function(x):
    #c = np.array([3,5,-9,-3,2],float)
    #y = np.polyval(c,x)
    y = np.sin(x*np.pi*1.5)
    return y

# Define noisy observation function
def obs_function(x, sigma):
    y = true_function(x) + np.random.normal(0,sigma,(len(x),1))
    return y
    
def make_poly_array(x,y,sigma):
    nx = len(x)
    sig = np.array(sigma)
    xy = np.zeros((2*nx, 2))
    xy[:,0] = np.append(x, x[::-1])
    if sig.size > 1:
        rsig = sig[::-1]
    else:
        rsig = sig
    xy[:,1] = np.append(y-sig, y[::-1]+rsig)
    return xy

    
# Main program
# Plot true function
nxp = 100
x_plot = np.atleast_2d(np.linspace(-2.0,2.0,nxp,dtype='float')).T
fx_plot = true_function(x_plot)

# Training data
x_train = np.random.random((n_training,1))*1.5-0.75
y_train = obs_function(x_train, noise)

# GP 
kernel = GPy.kern.RBF(input_dim=1, variance=gp_var, lengthscale=gp_l)
gp = GPy.models.GPRegression(x_train,y_train,kernel)
gp.Gaussian_noise.variance = gp_noisevar
if opt_hyper:
    gp.optimize()

# Predict
fhat_test,var_test = gp.predict(x_plot)

# Plot
h_fig,h_ax = plt.subplots()
h_fx, = h_ax.plot(x_plot,fx_plot,lw=1.5,c=npc.lines[0])

patch_fx = Polygon(make_poly_array(x_plot,fx_plot,noise), ec=npc.lines[0], fc=npc.lighten(npc.lines[0], 3),alpha=0.5)
h_ax.add_patch(patch_fx)
h_y, = h_ax.plot(x_train,y_train,'rx',mew=1.0,ms=8)

patch_fhat = Polygon(make_poly_array(x_plot,fhat_test,np.sqrt(var_test)), ec=npc.lines[1], fc=npc.lighten(npc.lines[1], 3),alpha=0.5)
h_ax.add_patch(patch_fhat)
h_fhat, = h_ax.plot(x_plot, fhat_test,lw=1.5,ls='--',c=npc.lines[1])

h_ax.set_ylim([-3,3])
gp_str = '$\hat{{f}}(x) \sim \mathcal{{GP}}(l={0:0.2f}, \sigma_f={1:0.2f}, \sigma_n={2:0.2f})$'
gp_str = gp_str.format(kernel.lengthscale.values[0], np.sqrt(kernel.variance.values[0]), np.sqrt(gp.Gaussian_noise.variance.values[0]))
h_ax.legend((h_fx, h_y, h_fhat),('$f(x)$', '$y \sim \mathcal{N}(f(x), \sigma^2)$', gp_str), loc='best')
h_ax.set_xlabel('$x$')
#h_fig.savefig('fig/regression_example.pdf', bbox_inches='tight', transparent='true')
plt.show()
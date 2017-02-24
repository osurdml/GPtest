import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import nice_plot_colors as npc
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

n_training = 5
noise = 0.1

def true_function(x, a=1.0, b=0.2):
    y = a*x + b
    return y
    
def obs_function(x, sigma):
    y = true_function(x) + np.random.normal(0,sigma,(len(x),1))
    return y
    
def make_poly_array(x,y,sigma):
    nx = len(x)
    xy = np.zeros((2*nx, 2))
    xy[:,0] = np.append(x, x[::-1])
    xy[:,1] = np.append(y-sigma, y[::-1]+sigma)
    return xy  

class mse_linear_optimiser(object):
    def __init__(self, x, y, f):
        self.f = f
        self.x = x
        self.y = y
        
    def mse(self, theta):
        fx = self.f(self.x, *theta)
        mse = np.mean((fx-self.y)**2)
        return mse

nxp = 100
x_plot = np.atleast_2d(np.linspace(0.0,1.0,nxp,dtype='float')).T
fx_plot = true_function(x_plot)

# Training data
x_train = np.random.random((n_training,1))
y_train = obs_function(x_train, noise)

theta0 = [0.0, 0.0]
mso = mse_linear_optimiser(x_train, y_train, true_function)
theta_mse = op.fmin(mso.mse,theta0)
fhat_mse_test = true_function(x_plot, *theta_mse)

# Bayesian learning
# theta

# Plot
h_fig,h_ax = plt.subplots()
h_fx, = h_ax.plot(x_plot,fx_plot,lw=1.5,c=npc.lines[0])

patch_fx = Polygon(make_poly_array(x_plot,fx_plot,noise), ec=npc.lines[0], fc=npc.lighten(npc.lines[0], 3),alpha=0.5)
h_ax.add_patch(patch_fx)
h_y, = h_ax.plot(x_train,y_train,'rx',mew=1.0,ms=8)

h_fhat, = h_ax.plot(x_plot, fhat_mse_test, lw=1.5, ls='--', c=npc.lines[1])

h_ax.legend((h_fx, h_y, h_fhat),('$f(x)$', '$y \sim \mathcal{N}(f(x), \sigma^2)$','$\hat{{f}}(x)$'), loc='best')
h_ax.set_xlabel('$x$')
#h_fig.savefig('fig/mse_linear_regression.pdf', bbox_inches='tight', transparent='true')



plt.show()
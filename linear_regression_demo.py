import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import nice_plot_colors as npc
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal

plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)
cm = 'viridis'
plt.rcParams['image.cmap'] = cm
savefig = False

n_training = 5
noise = 0.1
i_var = 1.0/(noise**2)
np.random.seed(2)
n_posterior_samples = 5

# Prior over weights
w0 = np.array([0.0, 0.0])
w_var = np.diag([1.0, 1.0])


def true_function(x, a=0.5, b=0.2):
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

def likelihood(y, X, theta, sigma):
    p_y = 0.0
    for xi,yi in zip(X, y):
        p_y += (yi-np.dot(xi, theta))**2
    p_y = np.exp(-1.0/(2*sigma**2)*p_y)
    return p_y

nxp = 100
x_plot = np.atleast_2d(np.linspace(-1.0,1.0,nxp,dtype='float')).T
X_plot = np.hstack((x_plot, np.ones(x_plot.shape)))
fx_plot = true_function(x_plot)
patch_fx = Polygon(make_poly_array(x_plot,fx_plot,noise), ec=npc.lines[0], fc=npc.lighten(npc.lines[0], 3),alpha=0.5)

# Training data
x_train = np.random.random((max(n_training, 100),1))*2.0-1.0
y_train = obs_function(x_train, noise)

x_train = x_train[0:n_training]
X_train = np.hstack((x_train, np.ones(x_train.shape)))
y_train = y_train[0:n_training]

# Least squares fit
if n_training > 1:
    mso = mse_linear_optimiser(x_train, y_train, true_function)
    theta_mse = op.fmin(mso.mse,w0)
    fhat_mse_test = true_function(x_plot, *theta_mse)
    # Plot
    h_fig,h_ax = plt.subplots()
    h_fx, = h_ax.plot(x_plot,fx_plot,lw=1.5,c=npc.lines[0])
    h_ax.add_patch(patch_fx)
    h_y, = h_ax.plot(x_train,y_train,'rx',mew=1.0,ms=8)

    h_fhat, = h_ax.plot(x_plot, fhat_mse_test, lw=1.5, ls='--', c=npc.lines[1])
    h_ax.legend((h_fx, h_y, h_fhat),('$f(x)$', '$y \sim \mathcal{N}(f(x), \sigma^2)$','$\hat{{f}}(x)$ (Least squares)'), loc='best')
    h_ax.set_xlabel('$x$')
    if savefig:
        h_fig.savefig('fig/mse_linear_regression.pdf', bbox_inches='tight', transparent='true')

# Bayesian learning

h_f2, h_a2 = plt.subplots(2,2)
h_f2.set_size_inches(7,7)
delta = 0.025
x = np.arange(-1.0, 1.0, delta)
y = np.arange(-1.0, 1.0, delta)
X, Y = np.meshgrid(x, y)
ptheta = mlab.bivariate_normal(X, Y, np.sqrt(w_var[0,0]), np.sqrt(w_var[1,1]), *w0)
CS = h_a2[0,0].contour(X, Y, ptheta.T)
h_a2[0,0].set_title('Prior $p(\mathbf{w})$')

# Likelihood
like = np.ones((len(x),len(y)))
for i,m in enumerate(x):
    for j,b in enumerate(y):
        like[i,j] = likelihood(y_train, X_train, [m,b], noise)
if n_training > 0:
    CS = h_a2[0,1].contour(X, Y, like.T)
h_a2[0,1].set_title('Likelihood $p(\mathbf{y}|X,\mathbf{w})$')

# Posterior
post_w = like*ptheta
CS = h_a2[1,0].contour(X, Y, post_w.T)
h_a2[1,0].set_title('Posterior $p(\mathbf{w} | \mathbf{y},X)$')

for hh in h_a2.flat[0:3]:
    hh.set_xlabel('$\mathbf{w}_0$, slope $m$')
    hh.set_ylabel('$\mathbf{w}_1$, intercept $b$')


A = i_var*np.matmul(X_train.T, X_train) + np.diag(1.0/w_var.diagonal())
iA = np.linalg.inv(A)
w_bar = i_var*np.matmul(np.matmul(iA, X_train.T), y_train)
fhat = np.matmul(X_plot, w_bar)
xstar_var = np.matmul(np.matmul(X_plot, iA), X_plot.T)
xstar_sigma = np.atleast_2d(np.sqrt(xstar_var.diagonal())).T

# Plot true function
h_fx, = h_a2[1,1].plot(x_plot, fx_plot, lw=1.5, c=npc.lines[3])
patch_fx2 = Polygon(make_poly_array(x_plot,fx_plot,noise), ec=npc.lines[3], fc=npc.lighten(npc.lines[3], 3),alpha=0.5)
h_a2[1,1].add_patch(patch_fx2)

# Plot training samples
h_y, = h_a2[1,1].plot(x_train, y_train, 'rx', mew=1.0, ms=8)


# Draw samples from posterior
w_samples = np.random.multivariate_normal(w_bar.flatten(), iA, n_posterior_samples)
h_a2[1,0].plot(w_samples[:,0], w_samples[:,1], 'ko')
w_prior = multivariate_normal(w0, w_var)
cmap = plt.get_cmap(cm)
for wi in w_samples:
    pw = w_prior.pdf(wi)*likelihood(y_train, X_train, wi, noise)
    if pw <= 0:
        col = 'r'
    else:
        col = cmap(pw[0]/post_w.max())
    h_a2[1,1].plot(x_plot, true_function(x_plot, *wi), '-', lw=1.0, color=col)

# Plot predictive distributions
h_fhat, = h_a2[1,1].plot(x_plot, fhat, lw=1.5, ls='--', c=npc.lines[1])
patch_fhat = Polygon(make_poly_array(x_plot,fhat,xstar_sigma), ec=npc.lines[1], fc=npc.lighten(npc.lines[1], 3),alpha=0.5)
h_a2[1,1].add_patch(patch_fhat)
h_a2[1,1].set_ylim((fhat-xstar_sigma).min(), (fhat+xstar_sigma).max())
h_a2[1,1].set_xlabel('$x$')
h_a2[1,1].set_ylabel('$y$')
h_a2[1,1].set_title('Predictions $\hat{f}$')

if savefig:
    h_f2.savefig('fig/bayesion_linear_regression.pdf', bbox_inches='tight', transparent='true')
plt.tight_layout()
plt.show()
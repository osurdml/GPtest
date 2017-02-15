# Simple 1D GP classification example
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import GPr
#from GPy.utils.univariate_Gaussian import std_norm_cdf, std_norm_pdf
from scipy.special import ndtr as std_norm_cdf

#define a standard normal pdf
_sqrt_2pi = np.sqrt(2*np.pi)
def std_norm_pdf(x):
    x = np.clip(x,-1e150,1e150)
    return np.exp(-np.square(x)/2)/_sqrt_2pi
    
    
np.random.seed(0)

n_train = 20
true_sigma = 0.05
delta_f = 1e-3

# Define polynomial function to be modelled
def true_function(x):
    y = np.sin(x*2*np.pi + np.pi/4) + 0.2
    return y

#def class_prob(x):
#    y = std_norm_cdf(true_function(x))
#    return y
    
# Define noisy observation function
def obs_function(x, sigma):
    fx = true_function(x)
    noise = np.random.normal(scale=sigma, size=x.shape)
    return fx + noise
    
def noisy_preference_rank(uv, sigma):
    fuv = obs_function(uv,sigma)
    y = -1*np.ones((fuv.shape[0],1),dtype='int')
    y[fuv[:,1] > fuv[:,0]] = 1
    return y, fuv

def I_k(x, uv): # Jensen and Nielsen
    if x == uv[0]:
        return -1
    elif x == uv[1]:
        return 1
    else:
        return 0

def z_k(uvi, f, sigma, y):
    i2sig = 1.0 / (2 * np.sqrt(sigma))
    zc = i2sig*(f[uvi[:,1]]-f[uvi[:,0]])
    return y*zc, i2sig
    

def calc_W(uvi, y, f, sigma):
    nx = len(f)
    z,i2sig = z_k(uvi, f, sigma, y)
    phi_z = std_norm_cdf(z)
    N_z = std_norm_pdf(z)
    # First derivative (Jensen and Nielsen)
    dpy_df = np.zeros((nx,1), dtype='float')
    dpyuv_df = -y*i2sig*N_z/phi_z
    dpy_df[uvi[:,0]] += dpyuv_df # This implements I_k
    dpy_df[uvi[:,1]] += -dpyuv_df
    
    inner = -(i2sig**2)*(z*N_z/phi_z + (N_z/phi_z)**2)
    W = np.zeros((nx,nx), dtype='float')
    for uvik,ddpy_df in zip(uvi,inner):
        xi,yi = uvik
        W[xi,xi] -= ddpy_df      # If x_i = x_i = u_k then I(x_i)*I(y_i) = 1*1 = 1
        W[yi,yi] -= ddpy_df      # If x_i = x_i = v_k then I(x_i)*I(y_i) = -1*-1 = 1
        W[xi,yi] -= -ddpy_df     # Otherwise, I(x_i)*I(y_i) = -1*1 = -1
        W[yi,xi] -= -ddpy_df
    return W, dpy_df


def rmse(f_est,f_real):
    return np.sqrt(((f_est - f_real) ** 2).mean())


def psi_rasmussen(uvi, y, f, iK, sigma):
    z,i2sig = z_k(uvi, f, sigma, y)
    phi_z = std_norm_cdf(z)
    psi = np.sum(np.log(phi_z)) - 0.5*np.matmul(np.matmul(f.T, iK), f)
    return psi

# Main program
# Plot true function
x_plot = np.linspace(0.0,1.0,101,dtype='float')
y_plot = true_function(x_plot)

# Training data - this is a bit weird, but we sample x values, then the uv pairs
# are actually indexes into x, because it is easier computationally. You can 
# recover the actual u,v values using x[ui],x[vi]
x_train = np.random.random((2*n_train,1))
uvi_train = np.random.choice(range(2*n_train), (n_train,2), replace=False)
uv_train = x_train[uvi_train][:,:,0]

# Get noisy observations f(uv) and corresponding ranks y_train
y_train, fuv_train = noisy_preference_rank(uv_train,true_sigma)

hf,ha = plt.subplots(1,1)
ha.plot(x_plot,y_plot,'r-')
for uv,fuv,y in zip(uv_train, fuv_train, y_train):
    ha.plot(uv, fuv, 'b-')
    ha.plot(uv[(y+1)/2],fuv[(y+1)/2],'k+')

ha.set_title('Training data')
ha.set_ylabel('f(x)')
ha.set_xlabel('x')


# GP hyperparameters
# note: using log scaling to aid learning hyperparameters with varied magnitudes
log_hyp = np.log([0.1,0.1,1,0.1]) # sigma_probit, length_scale, sigma_f, sigma_n
mean_hyp = 0
like_hyp = 0

initGP = GPr.GaussianProcess(log_hyp[1:],mean_hyp,like_hyp,"SE","zero","zero",x_train,[])

# Pseudocode:
# FOr a set of hyperparameters, return log likelihood that can be used by an optimiser
theta0 = log_hyp

#while not theta_converged:
initGP.log_hyp = theta0[1:] # Note that theta[0] is sigma, the noise in the probit
f = np.zeros((x_train.shape[0],1))

# With current hyperparameters:
Ix = np.eye(x_train.shape[0])
Kxx = initGP.covFun.compute_Kxx_matrix()
L = np.linalg.cholesky(Kxx)
iKxx = np.linalg.solve(L.T,np.linalg.solve(L,Ix))
# detK = (np.product(L.diagonal()))**2
# log_det_K = np.sum(np.log(L.diagonal()))

# First, solve for \hat{f} and W (mode finding Laplace approximation)
f_error = delta_f + 1
stopped = False
f_true = true_function(x_train)
while not stopped:
    W,dpy_df = calc_W(uvi_train, y_train, f, np.exp(theta0[0]))
    g = (iKxx + W)
    f_new = np.matmul(np.linalg.inv(g), np.matmul(W,f) + dpy_df)
    df = np.abs((f_new-f))
    f_error = np.max(df)
    print f_error, psi_rasmussen(uvi_train, y_train, f_new, iKxx, np.exp(theta0[0]))
    f = f_new
    if f_error < delta_f:
        stopped = True

ha.plot(x_train, f_new, 'g^')
plt.show()

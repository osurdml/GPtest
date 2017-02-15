# Simple 1D GP classification example
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import GPr
import GPy
from scipy.special import ndtr as std_norm_cdf

#define a standard normal pdf
_sqrt_2pi = np.sqrt(2*np.pi)
def std_norm_pdf(x):
    x = np.clip(x,-1e150,1e150)
    return np.exp(-(x**2)/2)/_sqrt_2pi

# Define squared distance calculation function
def squared_distance(A,B):
    A = np.reshape(A,(len(A),1))
    B = np.reshape(B,(len(B),1))
    A2_sum = A*A
    B2_sum = B*B
    AB = 2*np.dot(A,B.T)
    A2 = np.tile(A2_sum,(1,np.size(B2_sum,axis=0)))
    B2 = np.tile(B2_sum.T,(np.size(A2_sum,axis=0),1))
    sqDist = A2 + B2 - AB
    return sqDist

# Define squared exponential CovarianceFunction function
class SquaredExponential(object):
    def __init__(self, logHyp, x):
        self.x = x
        self.hyp = np.exp(logHyp)  # hyperparameters
        xdim = self.x.shape[1]
        self.length = self.hyp[0:xdim]  # length scales
        self.logvar = self.hyp[-1] ** 2  # squared exponential variance

    def compute_Kxx_matrix(self):
        scaledX = self.x / self.M
        sqDist = squared_distance(scaledX, scaledX)
        Kxx = self.sn2 * np.eye(np.size(self.x, axis=0)) + self.sf2 * np.exp(-0.5 * sqDist)
        return Kxx

    def compute_Kxz_matrix(self, z):
        scaledX = self.x / self.M
        scaledZ = z / self.M
        sqDist = squared_distance(scaledX, scaledZ)
        Kxz = self.sf2 * np.exp(-0.5 * sqDist)
        return Kxz

class PrefProbit(object):
    def __init__(self, sigma=1.0):
        self.set_sigma(sigma)
        self.log2pi = np.log(2.0*np.pi)

    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isqrt2sig = 1.0 / (self.sigma * np.sqrt(2.0))
        self._i2var =  self._isqrt2sig**2

    def z_k(self, uvi, f, y):
        zc = self._isqrt2sig * (f[uvi[:, 1]] - f[uvi[:, 0]])
        return y * zc

    def I_k(self, x, uv):  # Jensen and Nielsen
        if x == uv[0]:
            return -1
        elif x == uv[1]:
            return 1
        else:
            return 0

    def derivatives(self, uvi, y, f):
        nx = len(f)
        z = self.z_k(uvi, f, y)
        phi_z = std_norm_cdf(z)
        N_z = std_norm_pdf(z)

        # First derivative (Jensen and Nielsen)
        dpy_df = np.zeros((nx, 1), dtype='float')
        dpyuv_df = y * self._isqrt2sig * N_z / phi_z
        dpy_df[uvi[:, 0]] += -dpyuv_df  # This implements I_k (note switch because Jensen paper has funky backwards z_k)
        dpy_df[uvi[:, 1]] += dpyuv_df

        inner = -self._i2var * (z * N_z / phi_z + (N_z / phi_z) ** 2)
        W = np.zeros((nx, nx), dtype='float')
        for uvik, ddpy_df in zip(uvi, inner):
            xi, yi = uvik
            W[xi, xi] -= ddpy_df  # If x_i = x_i = u_k then I(x_i)*I(y_i) = 1*1 = 1
            W[yi, yi] -= ddpy_df  # If x_i = x_i = v_k then I(x_i)*I(y_i) = -1*-1 = 1
            W[xi, yi] -= -ddpy_df  # Otherwise, I(x_i)*I(y_i) = -1*1 = -1
            W[yi, xi] -= -ddpy_df
        return W, dpy_df

    def log_marginal(self, uvi, y, f, iK, logdetK):
        z = self.z_k(uvi, f, y)
        phi_z = std_norm_cdf(z)
        psi = np.sum(np.log(phi_z)) - 0.5 * np.matmul(np.matmul(f.T, iK), f) - 0.5*logdetK - iK.shape[0]/2.0*self.log2pi
        return psi.flat[0]

class PreferenceGaussianProcess(object):

    def __init__(self, x_train, uvi_train, y_train, likelihood=PrefProbit):
        # log_hyp are log of hyperparameters, note that it is [length_0, ..., length_d, sigma_f, sigma_probit]
        self._xdim = x_train.shape[1]
        self._nx = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train
        self.uvi_train = uvi_train

        self.likelihood = likelihood()

        self.kern = GPy.kern.RBF(self._xdim, ARD=True)


    def calc_laplace(self, loghyp, delta_f = 1e-6):
        self.kern.lengthscale = np.exp(loghyp[0:self._xdim])
        self.kern.variance = np.exp(loghyp[self.x_dim])
        self.likelihood.set_sigma = np.exp(loghyp[-1])

        f = np.zeros((self._nx, 1))

        # With current hyperparameters:
        Ix = np.eye(self._nx)
        Kxx = self.kern.K(self.x_train)
        L = np.linalg.cholesky(Kxx)
        iKxx = np.linalg.solve(L.T, np.linalg.solve(L, Ix))
        # detK = (np.product(L.diagonal()))**2
        logdetK = np.sum(np.log(L.diagonal()))

        # First, solve for \hat{f} and W (mode finding Laplace approximation, Newton-Raphson)
        f_error = delta_f + 1

        while f_error > f:
            W, dpy_df = self.likelihood.derivatives(self.uvi_train, self.y_train, f)
            g = (iKxx + W)
            f_new = np.matmul(np.linalg.inv(g), np.matmul(W, f) + dpy_df)
            df = np.abs((f_new - f))
            f_error = np.max(df)
            lml = self.likelihood.log_marginal(self.uvi_train, self.y_train, f_new, iKxx, logdetK)
            print f_error,lml
            f = f_new

        return f, lml

# Simple 1D GP classification example
import numpy as np
import GPy
#from scipy.special import ndtr as std_norm_cdf, digamma, polygamma
from scipy.special import digamma, polygamma
from scipy.stats import norm, beta
from scipy.linalg import block_diag
import sys

class LaplaceException(Exception):
    pass

# Define a standard normal pdf
_sqrt_2pi = np.sqrt(2*np.pi)
def std_norm_pdf(x):
    x = np.clip(x,-1e150,1e150)
    return norm.pdf(x)
    # return np.exp(-(x**2)/2)/_sqrt_2pi


def std_norm_cdf(x):
    x = np.clip(x, -30, 100 )
    return norm.cdf(x)


def norm_pdf_norm_cdf_ratio(z):
    # Inverse Mills ratio for stability
    out = -z
    out[z>-30] = std_norm_pdf(z[z>-30])/std_norm_cdf(z[z>-30])
    return out


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


def print_hyperparameters(theta, log=False):
    if log is True:
        theta = np.exp(theta)
    nl = len(theta)-3
    lstr = ', '.join(['%.2f']*nl) % tuple(theta[:nl])
    print "l: {0}, sig_f: {1:0.2f}, sig: {2:0.2f}, v: {3:0.2f}".format(lstr, theta[-3], theta[-2], theta[-1])


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
    type = 'preference'
    y_type = 'discrete'
    y_list = np.array([[-1], [1]], dtype='int')

    def __init__(self, sigma=1.0):
        self.set_hyper([sigma])
        self.log2pi = np.log(2.0*np.pi)

    def set_hyper(self, hyper):
        self.set_sigma(hyper[0])

    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isqrt2sig = 1.0 / (self.sigma * np.sqrt(2.0))
        self._i2var =  self._isqrt2sig**2
        
    def print_hyperparameters(self):
        print "Probit relative, Gaussian noise on latent. ",
        print "Sigma: {0:0.2f}".format(self.sigma)

    def z_k(self, y, f, scale=None):
        if scale is None:
            scale = self._isqrt2sig
        zc = scale * (f[:, 1, None] - f[:, 0, None]) # Weird None is to preserve shape
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
        z = self.z_k(y=y, f=self.get_rel_f(f, uvi))
        N_over_phi = norm_pdf_norm_cdf_ratio(z)
        # phi_z = std_norm_cdf(z)
        # N_z = std_norm_pdf(z)

        # First derivative (Jensen and Nielsen)
        dpy_df = np.zeros((nx, 1), dtype='float')
        dpyuv_df = y * self._isqrt2sig * N_over_phi # N_z / phi_z
        for i, (uvii, uvij) in enumerate(uvi):
            dpy_df[uvii] += -dpyuv_df[i]  ## ^^FIXED BIT - THANE LOOK HERE^^
            dpy_df[uvij] += dpyuv_df[i]  # This implements I_k (note switch because Jensen paper has funky backwards z_k)
        # dpy_df[uvi[:, 0]] += -dpyuv_df  # NOTE: THESE TWO LINES ARE INCORRECT FOR REPEATED INDEXES!!!!!
        # dpy_df[uvi[:, 1]] += dpyuv_df  # This implements I_k (note switch because Jensen paper has funky backwards z_k)

        inner = -self._i2var * (z * N_over_phi + (N_over_phi) ** 2)
        W = np.zeros((nx, nx), dtype='float')
        for uvik, ddpy_df in zip(uvi, inner):
            xi, yi = uvik
            W[xi, xi] -= ddpy_df  # If x_i = x_i = u_k then I(x_i)*I(y_i) = 1*1 = 1
            W[yi, yi] -= ddpy_df  # If x_i = x_i = v_k then I(x_i)*I(y_i) = -1*-1 = 1
            W[xi, yi] -= -ddpy_df  # Otherwise, I(x_i)*I(y_i) = -1*1 = -1
            W[yi, xi] -= -ddpy_df

        py = np.log(std_norm_cdf(z))
        return W, dpy_df, py

    def get_rel_f(self, f, uvi):
        return np.hstack((f[uvi[:, 0]], f[uvi[:, 1]]))

    def likelihood(self, y, f, scale=None):
        z = self.z_k(y, f, scale=scale)
        phi_z = std_norm_cdf(z)
        return phi_z

    def log_likelihood(self, y, f):
        return np.log(self.likelihood(y, f))

    def posterior_likelihood(self, fhat, varhat, uvi, y=1): # This is the likelihood assuming a Gaussian over f
        var_star = 2*self.sigma**2 + np.atleast_2d([varhat[u, u] + varhat[v, v] - varhat[u, v] - varhat[v, u] for u,v in uvi]).T
        p_y = self.likelihood(y, self.get_rel_f(fhat, uvi), 1.0/np.sqrt(var_star))
        return p_y

    def generate_samples(self, f):
        fuv = f + np.random.normal(scale=self.sigma, size=f.shape)
        y = -1 * np.ones((fuv.shape[0], 1), dtype='int')
        y[fuv[:, 1] > fuv[:, 0]] = 1
        return y, fuv


class OrdinalProbit(object):
    type = 'categorical'
    y_type = 'discrete'

    def __init__(self, sigma=1.0, b=1.0, n_ordinals=5, eps=1.0e-10):
        self.n_ordinals=n_ordinals
        self.set_hyper([sigma, b])
        self.eps = eps
        self.y_list = np.atleast_2d(np.arange(1, self.n_ordinals+1, 1, dtype='int')).T

    def set_hyper(self, hyper):
        self.set_sigma(hyper[0])
        self.set_b(hyper[1])

    def set_b(self, b):
        if not hasattr(b, "__len__"):
            b = abs(b)
            self.b = np.hstack(([-np.Inf],np.linspace(-b, b, self.n_ordinals-1), [np.Inf]))
        elif len(b) == self.n_ordinals+1:
            self.b = b
        elif len(b) == self.n_ordinals-1:
            self.b = np.hstack(([-np.Inf], b, [np.Inf]))
        else:
            raise ValueError('Specified b should be a scalar or vector of breakpoints')

    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isigma = 1.0/self.sigma
        self._ivar = self._isigma**2

    def print_hyperparameters(self):
        print "Ordinal probit, {0} ordered categories.".format(self.n_ordinals),
        print "Sigma: {0:0.2f}, b: ".format(self.sigma),
        print self.b

    def z_k(self, y, f):
        return self._isigma*(self.b[y] - f)

    def norm_pdf(self, y, f):
        f = f*np.ones(y.shape, dtype='float')       # This ensures the number of f values matches the number of y
        out = np.zeros(y.shape, dtype='float')
        for i in range(out.shape[0]):
            if y[i] != 0 and y[i] != self.n_ordinals:  # b0 = -Inf -> N(-Inf) = 0
                z = self._isigma*(self.b[y[i]] - f[i])
                out[i] = std_norm_pdf(z)
        return out

    def norm_cdf(self, y, f, var_x=0.0):
        ivar = self._isigma + var_x
        f = f*np.ones(y.shape, dtype='float')
        out = np.zeros(y.shape, dtype='float')
        for i in range(out.shape[0]):
            if y[i][0] == self.n_ordinals:
                out[i] = 1.0
            elif y[i][0] != 0:
                z = ivar*(self.b[y[i]] - f[i])
                out[i] = std_norm_cdf(z)
        return out

    def z_pdf(self, y, f):
        f = f*np.ones(y.shape, dtype='float')
        out = np.zeros(y.shape, dtype='float')
        for i in range(out.shape[0]):
            if y[i] != 0 and y[i] != self.n_ordinals:  # b0 = -Inf -> N(-Inf) = 0
                z = self._isigma*(self.b[y[i]] - f[i])
                out[i] = z*std_norm_pdf(z)
        return out


    def likelihood(self, y, f):
        return self.norm_cdf(y, f) - self.norm_cdf(y-1, f)

    def log_likelihood(self, y, f):
        return np.log(self.likelihood(y, f))

    def derivatives(self, y, f):

        l = self.likelihood(y, f)
        py = np.log(l)

        # First derivative - Chu and Gharamani
        # Having issues with derivative (likelihood denominator drops to 0)
        dpy_df = np.zeros(l.shape, dtype='float')
        d2py_df2 = np.zeros(l.shape, dtype='float')
        for i in range(l.shape[0]):
            if l[i] < self.eps:
                # l2 = self.likelihood(y[i], f[i]+self.delta_f)
                # l0 = self.likelihood(y[i], f[i]-self.delta_f)
                # dpy_df[i] = -(l2-l[i])/self.delta_f/l[i]      # (ln(f))' = f'/f
                # d2py_df2[i] = (l2 - 2*l[i] + l0)/self.delta_f**2/dpy_df[i]/l[i]

                if y[i] == 1:
                    dpy_df[i] = self._isigma*self.z_k(y[i], f[i])
                    d2py_df2[i] = -self._ivar
                elif y[i] == self.n_ordinals:
                    dpy_df[i] = self._isigma*self.z_k(y[i]-1, f[i])
                    d2py_df2[i] = -self._ivar
                else:
                    z1 = self.z_k(y[i], f[i])
                    z2 = self.z_k(y[i]-1, f[i])
                    ep = np.exp(-0.5*(z1**2 - z2**2))
                    dpy_df[i] = self._isigma*(z1*ep-z2)/(ep - 1.0)
                    d2py_df2[i] = -(self._ivar*(1.0 - (z1**2 *ep - z2**2)/(ep - 1.0)) + dpy_df[i]**2)
            else:
                dpy_df[i] = -self._isigma*(self.norm_pdf(y[i], f[i]) - self.norm_pdf(y[i]-1, f[i])) / l[i]
                d2py_df2[i] = -(dpy_df[i]**2 + self._ivar*(self.z_pdf(y[i], f[i]) - self.z_pdf(y[i]-1, f[i])) / l[i])

        W = np.diagflat(-d2py_df2)
        return W, dpy_df, py

    def posterior_likelihood(self, fhat, var_star, y=None):
        if y is None:
            y = self.y_list
        py = np.zeros((len(y), len(fhat)), dtype='float')
        mu = np.zeros(fhat.shape, dtype='float')
        for i, (f, v) in enumerate(zip(fhat, var_star.diagonal())):
            py[:,[i]] = self.norm_cdf(y, f, v) - self.norm_cdf(y-1, f, v)
            mu[i] = (py[:,[i]]*y).sum()
        if len(y) != self.n_ordinals or any(y != self.y_list):
            print "Specified y vector is not full ordinal set."
            mu = self.posterior_mean(fhat, var_star)
        return py, mu

    def posterior_mean(self, fhat, var_star):
        py, mu = self.posterior_likelihood(fhat, var_star)
        return mu

    def mean_link(self, f):
        mu = np.zeros(f.shape, dtype='float')
        for i in range(f.shape[0]):
            py = self.likelihood(self.y_list, f[i])
            mu[i] = (py*self.y_list).sum()       # Expected value, first moment
        return mu

    def generate_samples(self, f):
        z = np.zeros(f.shape, dtype='int')
        mu = np.zeros(f.shape, dtype='float')
        for i in range(f.shape[0]):
            py = self.likelihood(self.y_list, f[i])
            mu[i] = (py*self.y_list).sum()       # Expected value, first moment
            z[i] = np.sum(np.random.uniform() > py.cumsum())+1
        return z, mu


class AbsBoundProbit(object):
    type = 'bounded continuous'
    y_type = 'bounded'
    y_list = np.atleast_2d(np.linspace(0.01, 0.99, 101)).T

    def __init__(self, sigma=1.0, v=10.0):
        # v is the precision, kind of related to inverse of noise, high v is sharp distributions
        # sigma is the slope of the probit, basically scales how far away from 
        # 0 the latent has to be to to move away from 0.5 output. Sigma should 
        # basically relate to the range of the latent function
        self.set_hyper([sigma, v])
        self.log2pi = np.log(2.0*np.pi)

    def set_hyper(self, hyper):
        self.set_sigma(hyper[0])
        self.set_v(hyper[1])

    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isqrt2sig = 1.0 / (self.sigma * np.sqrt(2.0))
        self._i2var =  self._isqrt2sig**2

    def set_v(self, v):
        self.v = v
        
    def print_hyperparameters(self):
        print "Beta distribution, probit mean link.",
        print "Sigma: {0:0.2f}, v: {1:0.2f}".format(self.sigma, self.v)

    def mean_link(self, f):
        ml = np.clip(std_norm_cdf(f*self._isqrt2sig), 1e-12, 1.0-1e-12)
        return ml

    # def alpha(self, f):
    #     return self.v * self.mean_link(f)
    #
    # def beta(self, f):
    #     return self.v * (1-self.mean_link(f))

    def get_alpha_beta(self, f):
        ml = self.mean_link(f)
        aa = self.v * ml
        bb = self.v - aa    # = self.v * (1-ml)
        return aa, bb

    def likelihood(self, y, f):
        aa, bb = self.get_alpha_beta(f)
        return beta.pdf(y, aa, bb)

    def log_likelihood(self, y, f):
        return np.log(self.likelihood(y,f))

    def cdf(self, y, f):
        aa, bb = self.get_alpha_beta(f)
        return beta.cdf(y, aa, bb)

    def derivatives(self, y, f):

        aa, bb = self.get_alpha_beta(f)

        # Trouble with derivatives...
        dpy_df = self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (np.log(y) - np.log(1-y) - digamma(aa) + digamma(bb))

        Wdiag = - self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (
            f * self._i2var * (np.log(y) - np.log(1.0-y) - digamma(aa) + digamma(bb)) +
            self.v * self._isqrt2sig * std_norm_pdf(f*self._isqrt2sig) * (polygamma(1, aa) + polygamma(1, bb)) )

        W = np.diagflat(Wdiag)

        py = np.log(beta.pdf(y, aa, bb))

        return -W, dpy_df, py

    def posterior_likelihood(self, fhat, varhat, y=None, normal_samples=None): # This is MC sampled due to no closed_form solution
        if normal_samples is None:
            normal_samples = np.random.normal(size=1000)
        iny = 1.0/len(normal_samples)

        if y is None:
            y = self.y_list

        # Sampling from posterior to show likelihoods
        p_y = np.zeros((len(y), fhat.shape[0]))
        for i, (fstar, vstar) in enumerate(zip(fhat, varhat.diagonal())):
            f_samples = normal_samples * vstar + fstar
            p_y[:, i] = [iny * np.sum(self.likelihood(yj, f_samples)) for yj in y]
        return p_y, self.posterior_mean(fhat, varhat)

    def posterior_mean(self, fhat, var_star):
        #mu_t = self.mean_link(fhat)
        vv = np.atleast_2d(var_star.diagonal()).T
        E_x = np.clip(std_norm_cdf(fhat/(np.sqrt(2*self.sigma**2 + vv))), 1e-12, 1.0-1e-12)
        return E_x

    def generate_samples(self, f):
        mu = self.mean_link(f)
        a, b = self.get_alpha_beta(f)
        z = np.zeros(f.shape, dtype='float')
        for i, (aa,bb) in enumerate(zip(a,b)):
            z[i] = beta.rvs(aa, bb)
        return z, mu


class PreferenceGaussianProcess(object):

    def __init__(self, x_rel, uvi_rel, x_abs, y_rel, y_abs, rel_likelihood='PrefProbit', delta_f=1.0e-5,
                 abs_likelihood='AbsBoundProbit', verbose=0, hyper_counts=[2, 1, 2], rel_kwargs={}, abs_kwargs={}, *args, **kwargs):
        # hyperparameters are split by hyper_counts, where hyper_counts should contain three integers > 0, the first is
        # the number of hypers in the GP covariance, second is the number in the relative likelihood, last is the number
        # in the absolute likelihood. hyper_counts.sum() should be equal to len(log_hyp)
        # log_hyp are log of hyperparameters, note that it is [length_0, ..., length_d, sigma_f, sigma_probit, v_beta]

        # Training points are split into relative and absolute for calculating f, but combined for predictions.  
        self.set_observations(x_rel, uvi_rel, x_abs, y_rel, y_abs)

        self.delta_f = delta_f
        self.rel_likelihood = getattr(sys.modules[__name__], rel_likelihood)(**rel_kwargs) # This calls the constructor from string
        self.abs_likelihood = getattr(sys.modules[__name__], abs_likelihood)(**abs_kwargs)

        self.verbose = verbose

        if self._xdim is not None:
            self.kern = GPy.kern.RBF(self._xdim, ARD=True)

        self.Ix = np.eye(self._nx)

        self.f = None

        self.hyper_counts = np.array(hyper_counts)
        self._init_extras(*args, **kwargs)

    def _init_extras(self):
        pass

    def set_observations(self, x_rel, uvi_rel, x_abs, y_rel, y_abs):
        if x_rel.shape[0] is not 0:
            self._xdim = x_rel.shape[1]
        elif x_abs.shape[0] is not 0:
            self._xdim = x_abs.shape[1]
        else:
            self._xdim = None
            # raise Exception("No Input Points")
        self._n_rel = x_rel.shape[0]
        self._n_abs = x_abs.shape[0]
        self.x_rel = x_rel
        self.y_rel = y_rel
        self.x_abs = x_abs
        self.y_abs = y_abs
        self.uvi_rel = uvi_rel


        self.x_train_all = np.concatenate((self.x_rel, self.x_abs), 0)

        self._nx = self.x_train_all.shape[0]
        self.Ix = np.eye(self._nx)

    def add_observations(self, x, y, uvi=None, keep_f=False):
        # keep_f is used to reset the Laplace solution. If it's a small update, it's sometimes better to keep the old
        # values and append some 0's for the new observations (keep_f = True), otherwise it is reset (keep_f = False)
        # Default is keep_f = False
        if self._xdim is None:
            self._xdim = x.shape[1]
            self.kern = GPy.kern.RBF(self._xdim, ARD=True)
        if uvi is None: # Absolute observation/s
            if keep_f:
                self.f = np.vstack((self.f, np.zeros((x.shape[0], 1))))
            x_abs = np.concatenate((self.x_abs, np.atleast_2d(x)), 0)
            y_abs = np.concatenate((self.y_abs, np.atleast_2d(y)), 0)
            self.set_observations(self.x_rel, self.uvi_rel, x_abs, self.y_rel, y_abs)
        # TODO: Could/should check if relative input point is already an existing observation
        else:           # Relative observation/s
            if keep_f:  # The rel observations are stored at the front of f[0:_n_rel]
                self.f = np.vstack((self.f[0:self._n_rel], np.zeros((x.shape[0], 1)), self.f[self._n_rel:]))
            x_rel = np.concatenate((self.x_rel, np.atleast_2d(x)), 0)
            y_rel = np.concatenate((self.y_rel, np.atleast_2d(y)), 0)
            uvi_rel = np.concatenate((self.uvi_rel, uvi + self.x_rel.shape[0]), 0)
            self.set_observations(x_rel, uvi_rel, self.x_abs, y_rel, self.y_abs)

    def set_hyperparameters(self, loghyp):
        assert sum(self.hyper_counts) == len(loghyp), "Sum of hyper_counts must match length of log_hyp"
        assert self.hyper_counts[0] == self._xdim+1, "Currently only supporting sq exp covariance, hyper_counts[0] must be x_dim + 1"
        self.kern.lengthscale = np.exp(loghyp[0:self._xdim])
        self.kern.variance = 1.0 # (np.exp(loghyp[self._xdim]))**2
        dex = self.hyper_counts.cumsum()
        self.rel_likelihood.set_hyper(np.exp(loghyp[dex[0]:dex[1]]))
        self.abs_likelihood.set_hyper(np.exp(loghyp[dex[1]:dex[2]]))
        # self.rel_likelihood.set_sigma(np.exp(loghyp[-3])) # Do we need different sigmas for each likelihood? Yes!
        # self.abs_likelihood.set_sigma(np.exp(loghyp[-2])) # I think this sigma relates to sigma_f in the covariance, and is actually possibly redundant
        # self.abs_likelihood.set_v(np.exp(loghyp[-1]))     # Should this relate to the rel_likelihood probit noise?

    def calc_laplace(self, loghyp=None, maxloops=10000):
        if loghyp is not None:
            self.set_hyperparameters(loghyp)

        if self.f is None or self.f.shape[0] is not self._nx or np.isnan(self.f).any():
            f = np.zeros((self._nx, 1), dtype='float')
        else:
            f = self.f

        # With current hyperparameters:
        self.Kxx = self.kern.K(self.x_train_all)

        self.iK, self.logdetK = self._safe_invert_noise(self.Kxx)

        # First, solve for \hat{f} and W (mode finding Laplace approximation, Newton-Raphson)
        f_error = self.delta_f + 1
        nloops = 0

        while f_error > self.delta_f:
            if np.isnan(f).any():
                print("NaN error in Laplace, restarting with noisy f")
                f = np.zeros((self._nx, 1), dtype='float') + np.random.normal(0, self.delta_f*1e3, f_new.size)

            # Is splitting these apart correct?  Will there be off-diagonal elements of W_abs that should be 
            # non-zero because of the relative training points?
            f_rel = f[0:self._n_rel]
            f_abs = f[self._n_rel:]

            # Get relative Hessian and Gradient
            if self._n_rel>0 and self._n_abs>0:
                W_rel, dpy_df_rel, py_rel = self.rel_likelihood.derivatives(self.uvi_rel, self.y_rel, f_rel)
                # Get Absolute Hessian and Gradient
                # Note that y_abs has to be in [0,1]
                W_abs, dpy_df_abs, py_abs = self.abs_likelihood.derivatives(self.y_abs, f_abs)

                # Combine W, gradient
                py = py_abs.sum() + py_rel.sum()
                dpy_df = np.concatenate((dpy_df_rel, dpy_df_abs), axis=0)
                W = block_diag(W_rel, W_abs)

            elif self._n_rel>0:
                W, dpy_df, py = self.rel_likelihood.derivatives(self.uvi_rel, self.y_rel, f_rel)

            elif self._n_abs>0:
                W, dpy_df, py = self.abs_likelihood.derivatives(self.y_abs, f_abs)

            # # print "Total"
            # print "Dpy, W:"
            # print dpy_df
            # print W
            lambda_eye = 0.0*np.eye(self.iK.shape[0])

            g = (self.iK + W - lambda_eye)
            f_new = np.matmul(np.linalg.inv(g), np.matmul(W-lambda_eye, f) + dpy_df)
            #lml = self.rel_likelihood.log_marginal(self.uvi_rel, self.y_rel, f_new, iK, logdetK)

            ## Jensen version (iK + W)^-1 = K - K((I + WK)^-1)WK (not sure how to get f'K^-1f though...
            # ig = K - np.matmul(np.matmul(np.matmul(K, np.linalg.inv(Ix + np.matmul(W, K))), W), K)
            # f_new = np.matmul(ig, np.matmul(W, f) + dpy_df)
            # lml = 0.0

            df = np.abs((f_new - f))
            if nloops > 0 and df.max() > f_error:
                print("Laplace error increase, adding noise")
                f_new = f_new + np.random.normal(0, df.max()/10.0, f_new.size)
            f_error = np.max(df)

            # print "F Error: " + str(f_error) #,lml
            # print "F New: " + str(f_new)
            f = f_new
            nloops += 1
            if nloops > maxloops:
                raise LaplaceException("Maximum loops exceeded in calc_laplace!!")
            if self.verbose > 1:
                lml = py - 0.5*np.matmul(f.T, np.matmul(self.iK, f)) - 0.5*np.log(np.linalg.det(np.matmul(W, self.Kxx) + self.Ix))
                print("Laplace iteration {0:02d}, log p(y|f) = {1:0.2f}".format(nloops, lml[0,0]))

        self.W = W
        self.f = f
        self.iKf = np.matmul(self.iK, self.f)
        self.KWI = np.matmul(self.W, self.Kxx) + self.Ix

        return f#, lml

    def _safe_invert_noise(self, mat, start_noise=1.0e-6):
        eps = start_noise
        inv_ok = False

        while not inv_ok:
            try:
                L = np.linalg.cholesky(mat + eps*self.Ix)
                imat = np.linalg.solve(L.T, np.linalg.solve(L, self.Ix))
                # detK = (np.product(L.diagonal()))**2
                logdet = np.sum(np.log(L.diagonal()))
                inv_ok = True
            except np.linalg.linalg.LinAlgError:
                eps = max(1e-6, eps*10.0)
                print "Inversion issue, adding noise: {0}".format(eps)
        return imat, logdet

    def calc_nlml(self, loghyp, f=None):
        if f is None:
            f = self.calc_laplace(loghyp)
        # Now calculate the log likelihoods (remember log(ax) = log a + log x)
        log_py_f_rel = self.rel_likelihood.log_likelihood(self.y_rel, self.rel_likelihood.get_rel_f(f, self.uvi_rel))
        log_py_f_abs = self.abs_likelihood.log_likelihood(self.y_abs, f[self._n_rel:]) #TODO: I would prefer to use the indexing in absolute ratings too for consistency
        fiKf = np.matmul(f.T, self.iKf)
        lml = log_py_f_rel.sum()+log_py_f_abs.sum() - 0.5*fiKf - 0.5*np.log(np.linalg.det(self.KWI))
        return -lml

    def predict_latent(self, x):
        assert hasattr(self, 'iKf')
        kt = self.kern.K(self.x_train_all, x)
        mean_latent = np.matmul(kt.T, self.iKf)
        Ktt = self.kern.K(x)
        try:
            iKW = np.linalg.inv(self.KWI)
        except np.linalg.linalg.LinAlgError:
            raise
        var_latent = Ktt - np.matmul(kt.T, np.matmul(iKW, np.matmul(self.W, kt)))
        return mean_latent, var_latent

    def sample_latent_posterior(self, mean, covariance, n_samples = 1):
        assert mean.shape[0] == covariance.shape[0]
        assert covariance.shape[1] == covariance.shape[0]
        y_post = np.random.multivariate_normal(mean.flatten(), covariance, n_samples)
        return y_post

    def _check_latent_input(self, x=None, fhat=None, varhat=None):
        if (fhat is None or varhat is None):
            if x is not None:
                fhat, varhat = self.predict_latent(x)
            else:
                raise ValueError('Must supply either x or fhat and varhat')
        return fhat, varhat

    def rel_posterior_likelihood(self, uvi, y, x=None, fhat=None, varhat=None):
        fhat, varhat = self._check_latent_input(x, fhat, varhat)
        p_y = self.rel_likelihood.posterior_likelihood(fhat, varhat, uvi, y)
        return p_y

    def rel_posterior_likelihood_array(self, x=None, fhat=None, varhat=None, y=-1):
        fhat, varhat = self._check_latent_input(x, fhat, varhat)
        p_y = np.zeros((fhat.shape[0], fhat.shape[0]), dtype='float')
        cross_uvi = np.zeros((fhat.shape[0],2), dtype='int')
        cross_uvi[:, 1] = np.arange(fhat.shape[0])
        for i in cross_uvi[:, 1]:
            cross_uvi[:, 0] = i
            p_y[:, i:i+1] = self.rel_likelihood.posterior_likelihood(fhat, varhat, cross_uvi, y)
        return p_y

    def rel_posterior_MAP(self, uvi, x=None, fhat=None, varhat=None):
        p_y = self.rel_posterior_likelihood(uvi, y=1, x=x, fhat=fhat, varhat=varhat)
        return 2*np.array(p_y < 0.5, dtype='int')-1

    def abs_posterior_likelihood(self, y=None, x=None, fhat=None, varhat=None, **kwargs): # Currently sample-based
        fhat, varhat = self._check_latent_input(x, fhat, varhat)
        return self.abs_likelihood.posterior_likelihood(fhat, varhat, y=y, **kwargs)

    def abs_posterior_mean(self, x=None, fhat=None, varhat=None):
        fhat, varhat = self._check_latent_input(x, fhat, varhat)
        return self.abs_likelihood.posterior_mean(fhat, varhat)

    def print_hyperparameters(self):
        print "COV: '{0}', l: {1}, sigma_f: {2}".format(self.kern.name, self.kern.lengthscale.values, np.sqrt(self.kern.variance.values))
        print "REL: ",
        self.rel_likelihood.print_hyperparameters()
        print "ABS: ",
        self.abs_likelihood.print_hyperparameters()


class ObservationSampler(object):
    def __init__(self, true_fun, likelihood_type, likelihood_kwargs, *extra_args, **extra_kwargs):
        self.f = true_fun
        ltype = getattr(sys.modules[__name__], likelihood_type)
        self.l = ltype(**likelihood_kwargs)
        self._extra_init(*extra_args, **extra_kwargs)

    def _extra_init(self, *extra_args, **extra_kwargs):
        # This can be overwritten by derived classes for extra init options
        # (Is this abusing this system???)
        pass

    def generate_observations(self, x):
        fx = self.f(x)
        y, ff = self.l.generate_samples(fx)
        return y, ff

    def _gen_x_obs(self, n, n_xdim=1, domain=None):
        # Domain should be 2 x n_xdim, i.e [[x0_lo, x1_lo, ... , xn_lo], [x0_hi, x1_hi, ... , xn_hi ]]
        x_test = np.random.uniform(size=(n, n_xdim))
        if domain is not None:
            x_test = x_test * np.diff(domain, axis=0) + domain[0, :]
        return x_test

    def gaussian_multi_pairwise_sampler(self, x):
        # Find the maximum from a set of n samples (x should be n by d)
        # Return pairwise relationships that one point is higher than the others
        # Sample from Gaussian distributions around function values
        fx = np.random.normal(loc=self.f(x), scale=self.l.sigma)
        max_xi = np.argmax(fx)
        other_xi = np.delete(np.arange(x.shape[0]), max_xi)
        y = np.ones((x.shape[0]-1, 1), dtype='int')
        uvi = np.hstack((np.atleast_2d(other_xi).T, max_xi*y))
        return y, uvi, fx

class AbsObservationSampler(ObservationSampler):
    def observation_likelihood_array(self, x, y=None):
        fx = self.f(x)
        if y is None:
            y = self.l.y_list
        p_y = np.zeros((y.shape[0], fx.shape[0]), dtype='float')
        for i, fxi in enumerate(fx):
            p_y[:, i:i + 1] = self.l.likelihood(y, fxi[0])
        return p_y

    def mean_link(self, x):
        fx = self.f(x)
        return self.l.mean_link(fx)

    def generate_n_observations(self, n, n_xdim=1, domain=None):
        x = self._gen_x_obs(n, n_xdim, domain)
        y, mu = self.generate_observations(x)
        return x, y, mu


class RelObservationSampler(ObservationSampler):
    def observation_likelihood_array(self, x, y=-1):
        fx = self.f(x)
        p_y = np.zeros((x.shape[0], x.shape[0]), dtype='float')
        cross_fx = np.hstack((np.zeros((x.shape[0], 1)), fx))
        for i, fxi in enumerate(fx):
            cross_fx[:, 0] = fxi
            p_y[:, i:i + 1] = self.l.likelihood(y, cross_fx)
        return p_y

    def generate_observations(self, x, uvi):
        fx = self.f(x)
        y, ff = self.l.generate_samples(fx[uvi][:, :, 0])
        return y, ff

    def generate_n_observations(self, n, n_xdim=1, domain=None):
        x = self._gen_x_obs(2*n, n_xdim, domain)
        uvi = np.arange(2*n).reshape((n, 2))
        # uv = x[uvi] # [:, :, 0]
        y, fuv = self.generate_observations(x, uvi)
        return x, uvi, y, fuv




# SCRAP:
        # Estimate dpy_df
        # delta = 0.001
        # print "Estimated dpy_df"
        # est_dpy_df = (self.log_likelihood(y, f+delta) - self.log_likelihood(y, f-delta))/(2*delta)
        # print est_dpy_df
        # print 'log likelihood'
        # print np.sum(self.log_likelihood(y, f))
        #
        # print "Estimated W"
        # est_W_diag = (self.log_likelihood(y, f+2*delta) - 2*self.log_likelihood(y,f) + self.log_likelihood(y, f-2*delta))/(2*delta)**2
        #
        # print est_W_diag



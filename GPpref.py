# Simple 1D GP classification example
import numpy as np
import GPy
#from scipy.special import ndtr as std_norm_cdf, digamma, polygamma
from scipy.special import digamma, polygamma
from scipy.stats import norm, beta
from scipy.linalg import block_diag

#define a standard normal pdf
_sqrt_2pi = np.sqrt(2*np.pi)
def std_norm_pdf(x):
    x = np.clip(x,-1e150,1e150)
    return norm.pdf(x)
    #return np.exp(-(x**2)/2)/_sqrt_2pi

def std_norm_cdf(x):
    #x = np.clip(x, -30, 100 )
    return norm.cdf(x)

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
    def __init__(self, sigma=1.0):
        self.set_sigma(sigma)
        self.log2pi = np.log(2.0*np.pi)

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
        zc = scale * (f[:, 1, None] - f[:, 0, None]) # Weird none is to preserve shape
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

    def get_rel_f(self, f, uvi):
        return np.hstack((f[uvi[:, 0]], f[uvi[:, 1]]))

    def likelihood(self, y, f, scale=None):
        z = self.z_k(y, f, scale=scale)
        phi_z = std_norm_cdf(z)
        return phi_z

    def log_likelihood(self, y, f):
        return np.log(self.likelihood(y, f))

    def posterior_likelihood(self, fhat, varhat, uvi, y=1): # This is the likelihood assuming a Gaussian over f
        var_star = 2*self.sigma**2 + np.atleast_2d([varhat[u, u] + varhat[v, v] - varhat[u, v] - varhat[v, v] for u,v in uvi]).T
        p_y = self.likelihood(y, self.get_rel_f(fhat, uvi), 1.0/np.sqrt(var_star))
        return p_y

    def generate_samples(self, f):
        fuv = f + np.random.normal(scale=self.sigma, size=f.shape)
        y = -1 * np.ones((fuv.shape[0], 1), dtype='int')
        y[fuv[:, 1] > fuv[:, 0]] = 1
        return y, fuv

class AbsBoundProbit(object):
    def __init__(self, sigma=1.0, v=10.0):
        # v is the precision, kind of related to inverse of noise, high v is sharp distributions
        # sigma is the slope of the probit, basically scales how far away from 
        # 0 the latent has to be to to move away from 0.5 output. Sigma should 
        # basically relate to the range of the latent function
        self.set_sigma(sigma)
        self.set_v(v)
        self.log2pi = np.log(2.0*np.pi)

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

    def alpha(self, f):
        return self.v * self.mean_link(f)

    def beta(self, f):
        return self.v * (1-self.mean_link(f))

    def get_alpha_beta(self, f):
        ml = self.mean_link(f)
        aa = self.v * ml
        bb = self.v * (1-ml)
        return aa, bb

    def likelihood(self, y, f):
        aa, bb = self.get_alpha_beta(f)
        return beta.pdf(y, aa, bb)

    def log_likelihood(self, y, f):
        return np.log(self.likelihood(y,f))

    def derivatives(self, y, f):

        alpha = self.alpha(f)
        beta = self.beta(f) #let's make a distribution called beta that also has beta as a parameter!

        dpy_df = self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (np.log(y)-np.log(1-y) - digamma(alpha) + digamma(beta) )

        Wdiag = - self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (
                    f*self._i2var*( np.log(y)-np.log(1.0-y)-digamma(alpha) + digamma(beta) ) +
                    self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (polygamma(1, alpha) + polygamma(1, beta)) )

        W = np.diagflat(Wdiag)

        return -W, dpy_df

    def posterior_mean(self, fhat, var_star):
        #mu_t = self.mean_link(fhat)
        E_x = np.clip(std_norm_cdf(fhat/(np.sqrt(2*self.sigma**2 + var_star))), 1e-12, 1.0-1e-12)
        return E_x

    def generate_samples(self, f):
        mu = self.mean_link(f)
        a, b = self.get_alpha_beta(f)
        z = np.zeros(f.shape, dtype='float')
        for i, (aa,bb) in enumerate(zip(a,b)):
            z[i] = beta.rvs(aa, bb)
        return z, mu


class PreferenceGaussianProcess(object):

    def __init__(self, x_train, uvi_train, x_abs_train, y_train, y_abs_train, rel_likelihood=PrefProbit(), delta_f = 1e-6, abs_likelihood=AbsBoundProbit()):
        # log_hyp are log of hyperparameters, note that it is [length_0, ..., length_d, sigma_f, sigma_probit, v_beta]
        # Training points are split into relative and absolute for calculating f, but combined for predictions.  
        self.set_observations(x_train, uvi_train, x_abs_train, y_train, y_abs_train)

        self.delta_f = delta_f
        self.rel_likelihood = rel_likelihood
        self.abs_likelihood = abs_likelihood

        self.kern = GPy.kern.RBF(self._xdim, ARD=True)

        self.Ix = np.eye(self._nx)

        self.f = None

    def set_observations(self, x_train, uvi_train, x_abs_train, y_train, y_abs_train):
        if x_train.shape[0] is not 0:
            self._xdim = x_train.shape[1]
        elif x_abs_train.shape[0] is not 0:
            self._xdim = x_abs_train.shape[1]
        else:
            raise Exception("No Input Points")
        self._n_rel = x_train.shape[0]
        self._n_abs = x_abs_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train
        self.x_abs_train = x_abs_train
        self.y_abs_train = y_abs_train
        self.uvi_train = uvi_train


        self.x_train_all = np.concatenate((self.x_train, self.x_abs_train), 0)

        self._nx = self.x_train_all.shape[0]
        self.Ix = np.eye(self._nx)

    def add_observations(self, x, y, uvi=None):
        if uvi is None:
            x_abs_train = np.concatenate((self.x_abs_train, x), 0)
            y_abs_train = np.concatenate((self.y_abs_train, y), 0)
            self.set_observations(self.x_train, self.uvi_train, x_abs_train, self.y_train, y_abs_train)
        else:
            x_train = np.concatenate((self.x_train, x), 0)
            y_train = np.concatenate((self.y_train, y), 0)
            uvi_train = np.concatenate((self.uvi_train, uvi+self.x_train.shape[0]), 0)
            self.set_observations(x_train, uvi_train, self.x_abs_train, y_train, self.y_abs_train)

    def calc_laplace(self, loghyp):
        self.kern.lengthscale = np.exp(loghyp[0:self._xdim])
        self.kern.variance = (np.exp(loghyp[self._xdim]))**2
        self.rel_likelihood.set_sigma(np.exp(loghyp[-3])) # Do we need different sigmas for each likelihood? Yes!
        self.abs_likelihood.set_sigma(np.exp(loghyp[-2])) # I think this sigma relates to sigma_f in the covariance, and is actually possibly redundant
        self.abs_likelihood.set_v(np.exp(loghyp[-1]))     # Should this relate to the rel_likelihood probit noise?

        if self.f is None or self.f.shape[0] is not self._nx:
            f = np.zeros((self._nx, 1), dtype='float')
        else:
            f = self.f

        # With current hyperparameters:
        self.Kxx = self.kern.K(self.x_train_all)

        self.iK, self.logdetK = self._safe_invert_noise(self.Kxx)

        # First, solve for \hat{f} and W (mode finding Laplace approximation, Newton-Raphson)
        f_error = self.delta_f + 1

        while f_error > self.delta_f:
            # Is splitting these apart correct?  Will there be off-diagonal elements of W_abs that should be 
            # non-zero because of the relative training points?
            f_rel = f[0:self._n_rel]
            f_abs = f[self._n_rel:]

            # Get relative Hessian and Gradient
            if self._n_rel>0 and self._n_abs>0:
                W_rel, dpy_df_rel = self.rel_likelihood.derivatives(self.uvi_train, self.y_train, f_rel)
                # Get Absolute Hessian and Gradient
                # Note that y_abs_train has to be [0,1], which could be an issue.  
                W_abs, dpy_df_abs = self.abs_likelihood.derivatives(self.y_abs_train, f_abs)

                # Combine W, gradient
                dpy_df = np.concatenate((dpy_df_rel, dpy_df_abs), axis=0)
                W = block_diag(W_rel, W_abs)

            elif self._n_rel>0:
                W, dpy_df = self.rel_likelihood.derivatives(self.uvi_train, self.y_train, f_rel)

            elif self._n_abs>0:
                W, dpy_df = self.abs_likelihood.derivatives(self.y_abs_train, f_abs)

            # # print "Total"
            # print "Dpy, W:"
            # print dpy_df
            # print W
            lambda_eye = 0.0*np.eye(self.iK.shape[0])

            g = (self.iK + W - lambda_eye)
            f_new = np.matmul(np.linalg.inv(g), np.matmul(W-lambda_eye, f) + dpy_df)
            #lml = self.rel_likelihood.log_marginal(self.uvi_train, self.y_train, f_new, iK, logdetK)

            ## Jensen version (iK + W)^-1 = K - K((I + WK)^-1)WK (not sure how to get f'K^-1f though...
            # ig = K - np.matmul(np.matmul(np.matmul(K, np.linalg.inv(Ix + np.matmul(W, K))), W), K)
            # f_new = np.matmul(ig, np.matmul(W, f) + dpy_df)
            # lml = 0.0

            df = np.abs((f_new - f))
            f_error = np.max(df)

            # print "F Error: " + str(f_error) #,lml
            # print "F New: " + str(f_new)
            f = f_new

        self.W = W
        self.f = f
        self.iKf = np.matmul(self.iK, self.f)
        self.KWI = np.matmul(self.W, self.Kxx) + self.Ix

        return f#, lml

    def _safe_invert_noise(self, mat):
        eps = 1e-6
        inv_ok = False

        while not inv_ok:
            try:
                L = np.linalg.cholesky(mat + eps*self.Ix)
                imat = np.linalg.solve(L.T, np.linalg.solve(L, self.Ix))
                # detK = (np.product(L.diagonal()))**2
                logdet = np.sum(np.log(L.diagonal()))
                inv_ok = True
            except np.linalg.linalg.LinAlgError:
                eps = eps*10
                print "Inversion issue, adding noise: {0}".format(eps)
        return imat, logdet

    def calc_nlml(self, loghyp):
        f = self.calc_laplace(loghyp)
        if self.f is None:
            self.f = f
        # Now calculate the log likelihoods (remember log(ax) = log a + log x)
        log_py_f_rel = self.rel_likelihood.log_likelihood(self.y_train, self.rel_likelihood.get_rel_f(f, self.uvi_train))
        log_py_f_abs = self.abs_likelihood.log_likelihood(self.y_abs_train, f[self._n_rel:]) #TODO: I would prefer to use the indexing in absolute ratings too for consistency
        fiKf = np.matmul(f.T, self.iKf)
        lml = log_py_f_rel.sum()+log_py_f_abs.sum() - 0.5*fiKf - 0.5*np.log(np.linalg.det(self.KWI))
        return -lml

    def predict_latent(self, x):
        assert hasattr(self, 'iKf')
        kt = self.kern.K(self.x_train_all, x)
        mean_latent = np.matmul(kt.T, self.iKf)
        Ktt = self.kern.K(x)
        iKW = np.linalg.inv(self.KWI)
        var_latent = Ktt - np.matmul(kt.T, np.matmul(iKW, np.matmul(self.W, kt)))
        return mean_latent, var_latent

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

    def abs_posterior_likelihood(self, y, x=None, fhat=None, varhat=None, normal_samples = None): # Currently sample-based
        fhat, varhat = self._check_latent_input(x, fhat, varhat)
        varhat = np.atleast_2d(varhat.diagonal()).T
        if normal_samples is None:
            normal_samples = np.random.normal(size=1000)
        iny = 1.0/len(normal_samples)

        # Sampling from posterior to show likelihoods
        p_y = np.zeros((y.shape[0], fhat.shape[0]))
        for i, (fstar, vstar) in enumerate(zip(fhat, varhat)):
            f_samples = normal_samples * vstar + fstar
            p_y[:, i] = [iny * np.sum(self.abs_likelihood.likelihood(yj, f_samples)) for yj in y]
        return p_y

    def abs_posterior_mean(self, x=None, fhat=None, varhat=None):
        fhat, varhat = self._check_latent_input(x, fhat, varhat)
        varstar = np.atleast_2d(varhat.diagonal()).T
        E_y = self.abs_likelihood.posterior_mean(fhat, varstar)
        return E_y

    def print_hyperparameters(self):
        print "COV: '{0}', l: {1}, sigma_f: {2}".format(self.kern.name, self.kern.lengthscale.values, np.sqrt(self.kern.variance.values))
        print "REL: ",
        self.rel_likelihood.print_hyperparameters()
        print "ABS: ",
        self.abs_likelihood.print_hyperparameters()


class ObservationSampler(object):
    def __init__(self, true_fun, likelihood_object):
        self.f = true_fun
        self.l = likelihood_object

    def generate_observations(self, x):
        fx = self.f(x)
        y, ff = self.l.generate_samples(fx)
        return y, ff


class AbsObservationSampler(ObservationSampler):
    def observation_likelihood_array(self, x, y):
        fx = self.f(x)
        p_y = np.zeros((y.shape[0], fx.shape[0]), dtype='float')
        for i, fxi in enumerate(fx):
            p_y[:, i:i + 1] = self.l.likelihood(y, fxi[0])
        return p_y

    def mean_link(self, x):
        fx = self.f(x)
        return self.l.mean_link(fx)


class RelObservationSampler(ObservationSampler):
    def observation_likelihood_array(self, x, y=-1):
        fx = self.f(x)
        p_y = np.zeros((x.shape[0], x.shape[0]), dtype='float')
        cross_fx = np.hstack((np.zeros((x.shape[0], 1)), fx))
        for i, fxi in enumerate(fx):
            cross_fx[:, 0] = fxi
            p_y[:, i:i + 1] = self.l.likelihood(y, cross_fx)
        return p_y






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



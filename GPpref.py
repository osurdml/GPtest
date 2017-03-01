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
    x = np.clip(x, -30, 100 )
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

class AbsBoundProbit(object):
    def __init__(self, sigma=1.0, v=10.0):
        # Making v=10 looks good on a graph, but I'm not sure what it's actually supposed to be.  
        self.set_sigma(sigma)
        self.set_v(v)
        self.log2pi = np.log(2.0*np.pi)

    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isqrt2sig = 1.0 / (self.sigma * np.sqrt(2.0))
        self._i2var =  self._isqrt2sig**2

    def set_v(self, v):
        self.v = v

    def mean_link(self, f):
        ml = np.clip(std_norm_cdf(f*self._isqrt2sig), 1e-12, 1.0-1e-12)
        return ml

    def alpha(self, f):
        return self.v * self.mean_link(f)

    def beta(self, f):
        return self.v * (1-self.mean_link(f))

    def likelihood(self, y, f ):
        ml = self.mean_link(f)
        return np.log(beta.pdf(y, self.v*ml, self.v*(1-ml)))

    def derivatives(self, y, f):

        print "Start Iter.  f, y"
        print f
        print y
        alpha = self.alpha(f)
        beta = self.beta(f) #let's make a distribution called beta that also has beta as a parameter!
        # print "Alpha: " + str(alpha)
        # print "Beta: " + str(beta) 

        # Estimate dpy_df
        delta = 0.001
        #print "Estimated dpy_df"
        est_dpy_df = (self.likelihood(y, f+delta) - self.likelihood(y, f-delta))/(2*delta)
        #print est_dpy_df
        print 'log likelihood'
        print np.sum(self.likelihood(y, f))

        #print "Estimated W"
        est_W_diag = (self.likelihood(y, f+2*delta) - 2*self.likelihood(y,f) + self.likelihood(y, f-2*delta))/(2*delta)**2

        #print est_W_diag

        # Theres a dot in Jensen that I'm hoping is just a typo. I didn't fully derive it, but it looks correct.   
        # As the iteration goes, f blows up.  This makes parts of alpha, beta go to v and 0.  Digamma(0)=-inf :(
        # So why is it blowing up?
        dpy_df = self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (np.log(y)-np.log(1-y) - digamma(alpha) + digamma(beta) )

        Wdiag =  - self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (
                    f*self._i2var*( np.log(y)-np.log(1.0-y)-digamma(alpha) + digamma(beta) ) +
                    self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (polygamma(1, alpha) + polygamma(1, beta)) )
        # print "Wdiag"
        # print Wdiag

        W = np.diagflat(Wdiag)

        return -W, dpy_df

    def log_marginal(self):
        pass


class AbsProbit(object):
    # The Probit Likelihood given in Rasmussen, p43.  Note f(x) is scaled by sqrt(2)*sigma, as in Jensen.  
    # Ok, yeah, this was totally wrong.  Don't use this.  
    def __init__(self, sigma=1.0, v=10.0):
        self.set_sigma(sigma)
        self.log2pi = np.log(2.0*np.pi)

    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isqrt2sig = 1.0 / (self.sigma * np.sqrt(2.0))
        self._i2var =  self._isqrt2sig**2

    def derivatives(self, y, f):

        print "Start Iter.  f, y"
        print f
        print y

        # dpy_df = -y*std_norm_pdf(f*self._isqrt2sig) / std_norm_cdf(y*f*self._isqrt2sig)

        # Wdiag = -(-np.power(std_norm_pdf(f*self._isqrt2sig) / std_norm_cdf(y*f*self._isqrt2sig),2)
        #             -y*f*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) / std_norm_cdf(y*f*self._isqrt2sig))

        dpy_df = -y*std_norm_pdf(f) / std_norm_cdf(y*f)

        Wdiag = -(-np.power(std_norm_pdf(f) / std_norm_cdf(y*f),2)
                    -y*f*std_norm_pdf(f) / std_norm_cdf(y*f))

        W = np.diagflat(Wdiag)

        return W, dpy_df
        
        

class PreferenceGaussianProcess(object):

    def __init__(self, x_train, uvi_train, x_abs_train, y_train, y_abs_train, likelihood=PrefProbit, delta_f = 1e-6):
        # log_hyp are log of hyperparameters, note that it is [length_0, ..., length_d, sigma_f, sigma_probit, v_beta]
        # Training points are split into relative and absolute for calculating f, but combined for predictions.  
        if x_train.shape[0] is not 0:
            self._xdim = x_train.shape[1]
        elif x_abs_train.shape[0] is not 0:
            self._xdim = x_abs_train.shape[1]
        else: 
            raise Exception("No Input Points")
        self._nx = x_train.shape[0] + x_abs_train.shape[0]
        self._n_rel = x_train.shape[0]
        self._n_abs = x_abs_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train
        self.x_abs_train = x_abs_train
        self.y_abs_train = y_abs_train
        self.uvi_train = uvi_train
        self.delta_f = delta_f

        self.x_train_all = np.concatenate((self.x_train, self.x_abs_train), 0)

        self.rel_likelihood = likelihood()
        self.abs_likelihood = AbsBoundProbit()

        self.kern = GPy.kern.RBF(self._xdim, ARD=True)


    def calc_laplace(self, loghyp, f=None):
        self.kern.lengthscale = np.exp(loghyp[0:self._xdim])
        self.kern.variance = (np.exp(loghyp[self._xdim]))**2
        self.rel_likelihood.set_sigma = np.exp(loghyp[-2]) # Do we need different sigmas for each likelihood?  Hopefully No?
        self.abs_likelihood.set_sigma = np.exp(loghyp[-2])
        self.abs_likelihood.set_v = np.exp(loghyp[-1])

        if f is None:
            f = np.ones((self._nx, 1))
            f = f*.0

        # With current hyperparameters:
        Ix = np.eye(self._nx)
        K = self.kern.K(self.x_train_all)
        eps = 1e-6
        inv_ok = False

        while not inv_ok:
            try:
                L = np.linalg.cholesky(K + eps*Ix)
                iK = np.linalg.solve(L.T, np.linalg.solve(L, Ix))
                # detK = (np.product(L.diagonal()))**2
                logdetK = np.sum(np.log(L.diagonal()))
                inv_ok = True
            except np.linalg.linalg.LinAlgError:
                eps = eps*10
                print "Inversion issue, adding noise: {0}".format(eps)

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
            lambda_eye = 0.0*np.eye(iK.shape[0])

            g = (iK + W - lambda_eye)
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

        return f#, lml

    def calc_nlml(self, loghyp):
        f,lml = self.calc_laplace(loghyp)
        return -lml
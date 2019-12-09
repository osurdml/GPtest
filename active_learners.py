import numpy as np
import GPpref
from scipy.stats import beta
import plot_tools as ptt
import time
import sys

def calc_ucb(fhat, vhat, gamma=2.0, sigma_offset=0.0):
    # return fhat + gamma * (np.sqrt(np.atleast_2d(vhat.diagonal()).T) - sigma_offset)
    return fhat.flatten() + gamma * (np.sqrt(np.diagonal(vhat)) - sigma_offset)


def softmax_selector(x, tau=1.0):
    # High tau is more random
    ex = np.exp((x - x.max())/tau)
    Px = ex/ex.sum()
    return np.random.choice(len(x), p=Px)

class Learner(object):
    def __init__(self, model_type, obs_args, name, update_p_rel = False):
        self.model_type = getattr(sys.modules[__name__], model_type)
        self.obs_arguments = obs_args
        self.name = name
        self.update_p_rel = update_p_rel

    def build_model(self, model_kwargs):
        self.model = self.model_type(**model_kwargs)

    def select_observation(self):
        return self.model.select_observation(**self.obs_arguments)

class ActiveLearner(GPpref.PreferenceGaussianProcess):
    def init_extras(self):
        self._default_uvi = np.array([[0, 1]])
        self._plus_y_obs = np.ones((1, 1), dtype='int')
        self._minus_y_obs = -1*self._plus_y_obs

    def solve_laplace(self, log_hyp=None):
        self.f = self.calc_laplace(log_hyp)
        return self.f

    def get_observations(self):
        return self.x_rel, self.uvi_rel, self.x_abs, self.y_rel, self.y_abs

    def select_observation(self, p_rel=0.5, domain=None, n_rel_samples=2):
        if np.random.uniform() > p_rel: # i.e choose an absolute sample
            n_rel_samples = 1
        return self.uniform_domain_sampler(n_rel_samples, domain)

    def uniform_domain_sampler(self, n_samples, domain=None, sortx=False):
        # Domain should be 2 x n_xdim, i.e [[x0_lo, x1_lo, ... , xn_lo], [x0_hi, x1_hi, ... , xn_hi ]]
        x_test = np.random.uniform(size=(n_samples, self._xdim))
        if domain is not None:
            x_test = x_test*np.diff(domain, axis=0) + domain[0, :]
        if sortx:
            x_test = np.sort(x_test, axis=0)
        return x_test

    def linear_domain_sampler(self, n_samples, domain=None):
        # One dimension only at the moment!
        assert self._xdim == 1
        x_test = np.atleast_2d(np.linspace(.0, 1.0, n_samples, self._xdim)).T
        if domain is not None:
            x_test = x_test*np.diff(domain, axis=0) + domain[0, :]
        return x_test


    def get_latent_ests(self, x_test, abs_y_samples ):
        # Latent predictions
        d_x = x_test.shape[1]
        fhat, vhat = self.predict_latent(x_test)
        p_abs_y_post, E_y = self.abs_posterior_likelihood(abs_y_samples, fhat=fhat, varhat=vhat)
        if d_x is 1:
            p_rel_y_post = self.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)
        else:
            p_rel_y_post = None
        return fhat, vhat, p_abs_y_post, E_y, p_rel_y_post


    def create_posterior_plot(self, x_test, f_true, mu_true, rel_sigma, fuv_train, abs_y_samples, **plot_kwargs):

        fhat, vhat, p_abs_y_post, E_y, p_rel_y_post = self.get_latent_ests(x_test, abs_y_samples)

        x_train, uvi_train, x_abs_train, y_train, y_abs_train = self.get_observations()
        uv_train = x_train[uvi_train]

        # Posterior estimates
        fig_p, ax_p = \
            ptt.estimate_plots(x_test, f_true, mu_true, fhat, vhat, E_y, rel_sigma,
                               abs_y_samples, p_abs_y_post, p_rel_y_post,
                               x_abs_train, y_abs_train, uv_train, fuv_train, y_train,
                               t_l=r'$\mathcal{GP}$ latent function estimate $\hat{f}(x)$',
                               t_a=r'Posterior absolute likelihood, $p(y | \mathcal{Y}, \theta)$',
                               t_r=r'Posterior relative likelihood $P(x_0 \succ x_1 | \mathcal{Y}, \theta)$', **plot_kwargs)
        return fig_p, ax_p, [fhat, vhat, p_abs_y_post, E_y, p_rel_y_post]

    def update_posterior_plot(self, x_test, f_true, mu_true, rel_sigma, fuv_train, abs_y_samples,
                              fhat, vhat, p_abs_y_post, E_y, p_rel_y_post, ax_p):
        x_train, uvi_train, x_abs_train, y_train, y_abs_train = self.get_observations()
        uv_train = x_train[uvi_train]

        ptt.reset_axes(ax_p)
        # Posterior estimates
        fig_p, ax_p = \
            ptt.estimate_plots(x_test, f_true, mu_true, fhat, vhat, E_y, rel_sigma,
                               abs_y_samples, p_abs_y_post, p_rel_y_post,
                               x_abs_train, y_abs_train, uv_train, fuv_train, y_train,
                               t_l=r'$\mathcal{GP}$ latent function estimate $\hat{f}(x)$',
                               t_a=r'Posterior absolute likelihood, $p(y | \mathcal{Y}, \theta)$',
                               t_r=r'Posterior relative likelihood $P(x_0 \succ x_1 | \mathcal{Y}, \theta)$', ax=ax_p)
        return fig_p, ax_p


class MaxVar(ActiveLearner):
    l2pie = 1.0 + np.log(2*np.pi)
    def logistic(self, H, n_rel):
        L = 2.0
        return 1.0 / (1.0 + np.exp(-L * (H-self.kern.variance)) )

    # Max variance
    def select_observation(self, domain=None, n_test=100, n_rel_samples = 2, p_rel = 0.0, rel_tau = 1.0, abs_tau = 1.0e-5,w_v=1.0, selector='det'):
        # p_rel is the likelihood of selecting a relative query
        # tau are softmax temperatures (low is more greedy, high is more random)
        # w_v is the UCB weighting for variance, where UCB = w_v*variance + (1-w_v)*mean
        # det_type is the type of value used to select between
        # If p_rel < 0 then we will select based on variance magnitude
        if p_rel < 0.0:
            p_select = 1.0
        else:
            p_select = p_rel
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        vv = np.sqrt(np.diagonal(vhat))

        if p_select >= 1.0 or np.random.uniform() < p_select: # i.e choose a relative sample
            available_indexes = set(range(len(x_test)))
            dK = np.zeros(len(available_indexes))
            best_n = [softmax_selector(vv, tau=rel_tau)]
            while len(best_n) < n_rel_samples:
                dK[best_n[-1]] = -1.0e10  # Bit of a scam because it is still possible to sample this value
                available_indexes.remove(best_n[-1])
                best_n.append(-1)
                for cn in available_indexes:
                    best_n[-1] = cn
                    K = vhat[np.ix_(best_n, best_n)]
                    dK[cn] = np.linalg.det(K)
                best_n[-1] = softmax_selector(dK/dK.max(), tau=rel_tau)  # np.argmax(dK) #
        else:
            best_n = [softmax_selector(w_v*vv+(1.0-w_v)*fhat.flatten(), tau=abs_tau)]   #[np.argmax(ucb)]  #

        # This chooses an absolute query based on determinant. Choose relK using beta cdf likelihood
        if p_rel < 0.0:
            best_abs = softmax_selector(w_v * vv + (1.0 - w_v) * fhat.flatten(), tau=abs_tau)
            if selector == 'det':
                best_detK = -p_rel*np.sqrt(dK[best_n[-1]])
                K_ratio = best_detK/(vv.max() + best_detK)
            elif selector == 'entropy':
                H_rel = 0.5*(n_rel_samples*self.l2pie + np.log(dK[best_n[-1]]))
                H_abs = 0.5*(self.l2pie + np.log(vhat[best_abs, best_abs]))
                K_ratio = H_rel/(H_rel + H_abs)
            p_select = beta.cdf(K_ratio, -p_rel, -p_rel)
            if np.random.uniform() > p_select:
                best_n = [best_abs]
        return x_test[best_n, :]

class UCBLatent(ActiveLearner):
    # All absolute returns
    def select_observation(self, domain=None, n_test=100, gamma=2.0):
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        ucb = calc_ucb(fhat, vhat, gamma)
        return x_test[[np.argmax(ucb)], :]


class UCBLatentSoftmax(ActiveLearner):
    # All absolute returns
    def select_observation(self, domain=None, n_test=100, gamma=2.0, tau=1.0):
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        ucb = calc_ucb(fhat, vhat, gamma)
        return x_test[[softmax_selector(ucb, tau)], :]

class UCBCovarianceSoftmax(ActiveLearner):
    # All absolute returns
    def select_observation(self, domain=None, n_test=100, gamma=2.0, tau=1.0):
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        ucb = fhat.flatten() + gamma * (vhat.sum(axis=1) ** 0.25)
        return x_test[[softmax_selector(ucb, tau)], :]

class UCBOut(ActiveLearner):
    # NOT FULLY IMPLEMENTED - BROKEN
    def select_observation(self, domain=None, n_test=100, gamma=2.0):
        # Don't know how to recover the second moment of the predictive distribution, so this isn't done
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        Ey = self.expected_y(x_test, fhat, vhat)
        return x_test[[np.argmax(Ey)], :]


class ProbabilityImprovementAbs(ActiveLearner):
    def mean_var_sampler(self, n_test, domain):
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        shat = np.atleast_2d(np.sqrt(vhat.diagonal())).T
        return x_test, fhat, shat

    @staticmethod
    def delta(f_star, f_est, sig_est, zeta):
        return (f_est - f_star - zeta) / sig_est

    def probability_improvement(self, f_star, f_est, sig_est, zeta):
        Z = self.delta(f_star, f_est, sig_est, zeta)
        PI = GPpref.std_norm_cdf(Z)
        return PI

    def select_observation(self, domain=None, n_test=100, zeta=0.0, *args, **kwargs):
        f_star = self.f.max()
        x_test, fhat, shat = self.mean_var_sampler(n_test, domain)
        PI = self.probability_improvement(f_star, fhat, shat, zeta)
        return x_test[[np.argmax(PI)], :]


class ProbabilityImprovementRel(ProbabilityImprovementAbs):
    def select_observation(self, domain=None, n_test=100, zeta=0.0, *args, **kwargs):
        i_star = np.argmax(self.f)
        x_out = np.zeros((2, self.x_train_all.shape[1]), dtype='float')
        x_out[0] = self.x_train_all[i_star]
        x_out[1] = super(ProbabilityImprovementRel, self).select_observation(domain, n_test, zeta, *args, **kwargs)[0]
        return x_out


class ExpectedImprovementAbs(ProbabilityImprovementAbs):
    def expected_improvement(self, f_star, f_est, sig_est, zeta):
        Z = self.delta(f_star, f_est, sig_est, zeta)
        EI = sig_est * (Z * GPpref.std_norm_cdf(Z) + GPpref.std_norm_pdf(Z))
        return EI

    def select_observation(self, domain=None, n_test=100, zeta=0.0, *args, **kwargs):
        f_star = self.f.max()
        x_test, fhat, shat = self.mean_var_sampler(n_test, domain)
        EI = self.expected_improvement(f_star, fhat, shat, zeta)
        return x_test[[np.argmax(EI)], :]


class ExpectedImprovementRel(ExpectedImprovementAbs):
    def select_observation(self, domain=None, n_test=100, zeta=0.0, *args, **kwargs):
        i_star = np.argmax(self.f)
        x_out = np.zeros((2, self.x_train_all.shape[1]), dtype='float')
        x_out[0] = self.x_train_all[i_star]
        x_out[1] = super(ExpectedImprovementRel, self).select_observation(domain, n_test, zeta, *args, **kwargs)[0]
        return x_out


# class PredictiveEntropy(ActiveLearner):
#     # All absolute returns
#     def select_observation(self, domain=None, n_test=100):
#
#         x_test = self.uniform_domain_sampler(n_test, domain)
#         fhat, vhat = self.predict_latent(x_test)
#         ucb = calc_ucb(fhat, vhat, gamma)
#         return x_test[[softmax_selector(ucb, tau)], :]

class ABSThresh(ActiveLearner):
    def select_observation(self, domain=None, n_test=100, p_thresh=0.7):
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        aa, bb = self.abs_likelihood.get_alpha_beta(fhat)
        p_under_thresh = beta.cdf(p_thresh, aa, bb)
        # ucb = calc_ucb(fhat, vhat, gamma)
        return x_test[[np.argmax(p_under_thresh * (1.0 - p_under_thresh))], :]


class UCBAbsRel(ActiveLearner):
    def select_observation(self, domain=None, n_test=100, p_rel=0.5, n_rel_samples=2, gamma=2.0, tau=1.0):
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        ucb = calc_ucb(fhat, vhat, gamma)

        if np.random.uniform() < p_rel: # i.e choose a relative sample
            best_n = [softmax_selector(ucb, tau=tau)]   #[np.argmax(ucb)]  #
            # sq_dist = GPpref.squared_distance(x_test, x_test)
            p_rel_y = self.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)

            while len(best_n) < n_rel_samples:
                # ucb = ucb*sq_dist[best_n[-1], :] # Discount ucb by distance
                ucb[best_n[-1]] = 0.0
                ucb *= 4*p_rel_y[best_n[-1],:]*(1.0 - p_rel_y[best_n[-1],:]) # Reduce by likelihood that each point is better than previous best
                best_n.append(softmax_selector(ucb, tau=tau))
                # best_n.append(np.argmax(ucb))
        else:
            best_n = [softmax_selector(ucb, tau=tau/2.0)]   #[np.argmax(ucb)]  #
        return x_test[best_n, :]


class UCBAbsRelD(ActiveLearner):
    def select_observation(self, domain=None, n_test=100, p_rel=0.5, n_rel_samples=2, gamma=2.0, tau=1.0):
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        ucb = calc_ucb(fhat, vhat, gamma)

        if np.random.uniform() < p_rel: # i.e choose a relative sample
            best_n = [softmax_selector(ucb, tau=tau)]   #[np.argmax(ucb)]  #
            p_rel_y = self.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)
            # sq_dist = np.sqrt(GPpref.squared_distance(x_test, x_test))
            while len(best_n) < n_rel_samples:
                # ucb *= sq_dist[best_n[-1], :] # Discount ucb by distance
                ucb[best_n[-1]] = 0.0
                ucb *= 4*p_rel_y[best_n[-1],:]*(1.0 - p_rel_y[best_n[-1],:]) # Divide by likelihood that each point is better than previous best
                best_n.append(softmax_selector(ucb, tau=tau))
                #best_n.append(np.argmax(ucb))
        else:
            best_n = [softmax_selector(ucb, tau=tau/2.0)]   #[np.argmax(ucb)]  #
        return x_test[best_n, :]

class DetRelBoo(ActiveLearner):
    def select_observation(self, domain=None, n_test=100, n_rel_samples=2, gamma=2.0, tau=1.0):
        x_test = self.uniform_domain_sampler(n_test, domain, sortx=True)
        fhat, vhat = self.predict_latent(x_test)
        # ucb = calc_ucb(fhat, vhat, gamma)

        # Select the first location using the mean, uncertainty and likelihood of improvement over everywhere (?) else
        p_rel = self.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)
        pp_rel = p_rel.mean(axis=0)
        ucb = 4*pp_rel*(1-pp_rel)*np.diagonal(vhat)**0.5 # THIS SEEMS BACKWARD!!! WHY DOES IT WORK?

        available_indexes = set(range(len(x_test)))
        dK = np.zeros(len(available_indexes))

        best_n = [softmax_selector(ucb, tau=tau)]   #[np.argmax(ucb)]  #
        uvi = self._default_uvi.copy()
        uvi[0][0] = best_n[0]
        while len(best_n) < n_rel_samples:
            dK[best_n[-1]] = -1e5                   # Bit of a scam because it is still possible to sample this value
            available_indexes.remove(best_n[-1])
            best_n.append(-1)
            for cn in available_indexes:
                best_n[-1] = cn; uvi[0][1] = cn
                K = vhat[np.ix_(best_n, best_n)]
                # p_y = self.rel_likelihood.posterior_likelihood(fhat, vhat, uvi, y=self._plus_y_obs)
                dK[cn] = (gamma*np.sqrt(np.linalg.det(K)) + fhat[cn]) # *p_y
            best_n[-1] = softmax_selector(dK, tau=tau)  # np.argmax(dK) #
        return x_test[best_n, :]

class DetSelect(ActiveLearner):
    def select_observation(self, domain=None, n_test=100, n_rel_samples=2, gamma=2.0, rel_tau=1.0, abs_tau=1.0):
        x_test = self.uniform_domain_sampler(n_test, domain, sortx=True)
        fhat, vhat = self.predict_latent(x_test)
        # ucb = calc_ucb(fhat, vhat, gamma).flatten()

        # Select the first location using the mean, uncertainty and likelihood of improvement over everywhere (?) else
        p_rel = self.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)
        pp_rel = p_rel.mean(axis=0)
        x_std = np.diagonal(vhat)**0.5
        rel_value = 4*pp_rel*(1-pp_rel)*(x_std.max() - x_std) # fhat.flatten() + gamma*(vhat.sum(axis=1)**0.25) #

        available_indexes = set(range(len(x_test)))
        dK = np.zeros(len(available_indexes))
        best_n = [softmax_selector(rel_value, tau=rel_tau)]   #[np.argmax(rel_value)]  #
        uvi = self._default_uvi.copy()
        uvi[0][0] = best_n[0]
        while len(best_n) < n_rel_samples:
            dK[best_n[-1]] = -1e5                   # Bit of a scam because it is still possible to sample this value
            available_indexes.remove(best_n[-1])
            best_n.append(-1)
            for cn in available_indexes:
                best_n[-1] = cn; uvi[0][1] = cn
                K = vhat[np.ix_(best_n, best_n)]
                p_y = self.rel_likelihood.posterior_likelihood(fhat, vhat, uvi, y=self._plus_y_obs)
                dK[cn] = p_y*(gamma*np.sqrt(np.linalg.det(K)) + fhat[cn])
            best_n[-1] = softmax_selector(dK, tau=rel_tau)  # np.argmax(dK) #

        K = vhat[np.ix_(best_n, best_n)]
        mdK = gamma*np.sqrt(np.linalg.det(K)) + fhat[cn]
        # mdK = dK.max()
        ucb = calc_ucb(fhat, vhat, gamma)
        p_rel = mdK/(mdK + ucb.max())

        if abs_tau > 0 and np.random.uniform() > p_rel:     # i.e choose an abs:
            # best_n = [softmax_selector(fhat.flatten() + gamma*(vhat.sum(axis=1)**0.25), tau)]  #[best_n[0]]   #
            best_n = [softmax_selector(ucb, tau=abs_tau)]
        return x_test[best_n, :]


class PeakComparitor(ActiveLearner):

    def test_observation(self, x, y, x_test, gamma):
        self.store_observations()
        self.add_observations(x, y, self._default_uvi)
        self.solve_laplace()
        fhat, vhat = self.predict_latent(x_test)
        ucb = calc_ucb(fhat, vhat, gamma)
        self.reset_observations()
        return ucb.max()

    def store_observations(self):
        crx, cuv, cax, cry, cay = self.get_observations()
        self.crx, self.cuv, self.cax, self.cry, self.cay = crx.copy(), cuv.copy(), cax.copy(), cry.copy(), cay.copy()
        self.crf = self.f.copy()

    def reset_observations(self):
        try:
            self.set_observations(self.crx, self.cuv, self.cax, self.cry, self.cay)
            self.f = self.crf
        except AttributeError:
            print "reset_observations failed: existing observations not found"

    def select_observation(self, domain=None, n_test=50, gamma=2.0, n_rel_samples=2):
        n_comparators = n_rel_samples-1
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        ucb = calc_ucb(fhat, vhat, gamma)
        max_xi = np.argmax(ucb)  # Old method used highest x, not ucb
        other_xi = np.delete(np.arange(n_test), max_xi)
        uvi = np.vstack((max_xi * np.ones(n_test - 1, dtype='int'), other_xi)).T

        p_pref = self.rel_likelihood.posterior_likelihood(fhat, vhat, uvi, y=-1)
        V = np.zeros(n_test - 1)
        x = np.zeros((2, 1), dtype='float')
        x[0] = x_test[max_xi]

        # Now calculate the expected value for each observation pair
        for i,uvi1 in enumerate(other_xi):
            x[1] = x_test[uvi1]
            V[i] += p_pref[i]*self.test_observation(x, self._minus_y_obs, x_test, gamma)
            if (1 - p_pref[i]) > 1e-3:
                V[i] += (1-p_pref[i])*self.test_observation(x, self._plus_y_obs, x_test, gamma)

        best_n = np.argpartition(V, -n_comparators)[-n_comparators:]
        # best = np.argmax(V)
        cVmax = np.argmax(ucb)  # This is repeated in case I want to change max_xi

        if ucb[cVmax] > V.max():
            return x_test[[cVmax], :]
        else:
            xi = np.zeros(n_comparators+1, dtype='int')
            xi[0] = max_xi
            xi[1:] = other_xi[best_n]
            return x_test[xi, :]


class LikelihoodImprovement(PeakComparitor):

    def test_observation(self, x, y, x_test, max_xi):
        self.store_observations()
        self.add_observations(x, y, self._default_uvi)
        f = self.solve_laplace()
        fhat, vhat = self.predict_latent(x_test)
        new_xi = np.argmax(fhat)
        p_new_is_better = self.rel_likelihood.posterior_likelihood(fhat, vhat, np.array([[max_xi, new_xi]]), self._plus_y_obs)
        self.reset_observations()
        return p_new_is_better

    def select_observation(self, domain=None, n_test=50, req_improvement=0.6, n_rel_samples=2, gamma=1.5, p_thresh=0.7):
        n_comparators = n_rel_samples-1
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        max_xi = np.argmax(fhat)
        other_xi = np.delete(np.arange(n_test), max_xi)
        uvi = np.vstack((max_xi * np.ones(n_test - 1, dtype='int'), other_xi)).T

        p_pref = self.rel_likelihood.posterior_likelihood(fhat, vhat, uvi, y=-1)
        V = np.zeros(n_test - 1)
        x = np.zeros((2, 1), dtype='float')
        x[0] = x_test[max_xi]


        # Now calculate the expected value for each observation pair
        for i,uvi1 in enumerate(other_xi):
            x[1] = x_test[uvi1]
            V[i] += p_pref[i]*self.test_observation(x, self._minus_y_obs, x_test, max_xi)
            if (1-p_pref[i]) > 1e-3:
                V[i] += (1-p_pref[i])*self.test_observation(x, self._plus_y_obs, x_test, max_xi)

        Vmax = V.max()
        # best_n = np.argpartition(V, -n_comparators)[-n_comparators:]
        # best = np.argmax(V)
        if self.verbose:
            print 'V_max = {0}'.format(Vmax)

        if Vmax < req_improvement:
            # aa, bb = self.abs_likelihood.get_alpha_beta(fhat)
            # p_under_thresh = beta.cdf(p_thresh, aa, bb)
            # return x_test[[np.argmax(p_under_thresh*(1.0-p_under_thresh))], :]
            ucb = calc_ucb(fhat, vhat, gamma, self.rel_likelihood.sigma)
            return x_test[[np.argmax(ucb)], :]
        else:
            best_n = []
            while len(best_n) < n_comparators:
                cbest = np.argmax(V)
                best_n.append(cbest)
                V = V * np.sqrt(GPpref.squared_distance(x_test[[other_xi[cbest]], :], x_test[other_xi])[0])
            xi = np.zeros(n_comparators+1, dtype='int')
            xi[0] = max_xi
            xi[1:] = other_xi[best_n]
            return x_test[xi, :]


class SampledThreshold(PeakComparitor):

    def calculate_threshold_utility(self, fhat, vhat, n_samples, y_threshold):
        f_sampled = self.sample_latent_posterior(fhat, vhat, n_samples = n_samples)
        threshold_utility = 0.0
        # For each one, evaluate the probability mass above the threshold in the absolute function
        for fs in f_sampled:
            threshold_utility += self.point_utility(fhat, fs, y_threshold)
        return threshold_utility/n_samples

    def point_utility(self, fhat, fs, y_threshold):
        # This is the mean probability mass above the threshold value
        pmass = 0.0
        for fi in fs:
            pmass += 1.0 - self.abs_likelihood.cdf(y_threshold, fi)
        return pmass/fs.shape[0]

    def test_observation(self, x, y, uvi, x_test, n_samples, y_threshold, f = None):
        self.add_observations(x, y, uvi, keep_f=True)
        self.solve_laplace()
        fhat, vhat = self.predict_latent(x_test)
        util = self.calculate_threshold_utility(fhat, vhat, n_samples, y_threshold)
        self.reset_observations()
        return util

    def select_observation(self, domain=None, x_test=None, n_test=50, n_samples=50, y_threshold=0.8, p_pref_tol=1e-3, n_mc_abs=5):
        # n_test is the number of test point locations on the input function
        # n_samples is the number of functions sampled from the posterior for estimating the utility
        # n_mc_abs is the number of proposed observations sampled from the current posterior for absolute estimates
        # Generate a set of test points in the domain (if not specified)
        if x_test is None:
            # x_test = self.linear_domain_sampler(n_test, domain)
            x_test = self.uniform_domain_sampler(n_test, domain)
            # x_test.sort(axis=0)
        n_test = len(x_test)

        # Sample a set of functions from the current posterior
        # We save the f value because otherwise it gets out of whack when we add observations
        flap = self.solve_laplace()
        fhat, vhat = self.predict_latent(x_test)
        # base_utility = self.calculate_threshold_utility(fhat, vhat, n_samples, y_threshold)

        # Check a random set of pairwise relatives and all absolutes (massive sampling)

        # Generate a set of random pairs (this randomly pairs all x_test points)
        uvi = np.random.choice(n_test, (n_test/2, 2), replace=False)
        p_pref = self.rel_likelihood.posterior_likelihood(fhat, vhat, uvi, y=-1)
        V_max_rel = 0.0

        self.store_observations()

        # Relative observations
        t_rel = time.time()
        # Now calculate the expected value for each observation pair
        for i, uv in enumerate(uvi):
            x = x_test[uv]
            V_rel = 0.0
            try:
                if p_pref[i] < p_pref_tol:
                    V_rel += (1-p_pref[i])*self.test_observation(x, self._plus_y_obs, self._default_uvi, x_test, n_samples, y_threshold, f=flap)
                elif p_pref[i] > 1.0-p_pref_tol:
                    V_rel += p_pref[i]*self.test_observation(x, self._minus_y_obs, self._default_uvi, x_test, n_samples, y_threshold, f=flap)
                else:
                    V_rel += (1-p_pref[i])*self.test_observation(x, self._plus_y_obs, self._default_uvi, x_test, n_samples, y_threshold, f=flap)
                    V_rel += p_pref[i] * self.test_observation(x, self._minus_y_obs, self._default_uvi, x_test, n_samples, y_threshold, f=flap)
                if V_rel >= V_max_rel:
                    V_max_rel = V_rel
                    x_best_rel = x
            except GPpref.LaplaceException as exc:
                print "Failed in relative test observation, x = [{0}, {1}]".format(x[0], x[1])
                raise exc

        # best_n = np.argpartition(V, -n_comparators)[-n_comparators:]
        # best = np.argmax(V)
        if self.verbose:
            print 'V_max_rel = {0}, x = {2}, t = {1}s'.format(V_max_rel[0], time.time()-t_rel, x_best_rel[:,0])

        # Absolute queries
        V_max = 0.0
        t_rel = time.time()
        for i, x in enumerate(x_test):
            F = fhat[i] + np.random.randn(n_mc_abs)*np.sqrt(vhat[i,i])
            Y, mu = self.abs_likelihood.generate_samples(F)
            V_abs = 0.0
            Y = np.clip(Y, 1e-2, 1-1e-2) # I had stability problems in Laplace with values approaching 0 or 1
            for y in Y:
                try:
                    V_abs += self.test_observation(x, y, None, x_test, n_samples, y_threshold, f=flap)
                except ValueError:
                    print "NaN Issue"
                except GPpref.LaplaceException as exc:
                    print "Failed in absolute test observation, x = {0}, y = {1}".format(x, y)
                    raise exc
            V_abs /= n_mc_abs
            if V_abs > V_max:
                V_max = V_abs
                x_best = x
        if self.verbose:
            print 'V_max_abs = {0}, x = {2}, t = {1}s'.format(V_max, time.time() - t_rel, x_best)

        if V_max_rel > V_max:
            x_best = x_best_rel

        return x_best


class SampledClassification(SampledThreshold):
    def calculate_threshold_utility(self, fhat, vhat, n_samples, y_threshold):
        # Class accuracy
        # Note this assumes the sampled functions are used to make the classification decision, and we calculate the
        # probability mass from the current mean estimate above the threshold
        P_below = self.abs_likelihood.cdf(y_threshold, fhat)

        f_sampled = self.sample_latent_posterior(fhat, vhat, n_samples=n_samples)
        threshold_utility = 0.0
        # For each one, evaluate the probability mass above the threshold in the absolute function
        for fs in f_sampled:
            fm = self.abs_likelihood.mean_link(fs)
            predict_below = fm < y_threshold
            predicted_accuracy = P_below[predict_below].sum() + (1.0 - P_below[~predict_below]).sum()
            threshold_utility += predicted_accuracy/len(fhat)
        return threshold_utility/n_samples


class OrdinalSampler(PeakComparitor):

    def model_value(self, fhat, vhat, y_threshold):
        # Probability of correct classification
        p_y, mu = self.abs_likelihood.posterior_likelihood(fhat, vhat)
        selection = np.argmax(p_y, axis=0)+1
        value = 0.0
        for i in range(len(fhat)):
            if selection[i] >= y_threshold:
                value += p_y[y_threshold-1:,i].sum()
            else:
                value += p_y[0:y_threshold-1,i].sum()
        return value/len(fhat)

    def test_observation(self, x, y, uvi, x_test, y_threshold):
        self.add_observations(x, y, uvi, keep_f=True)
        self.solve_laplace()
        fhat, vhat = self.predict_latent(x_test)
        util = self.model_value(fhat, vhat, y_threshold)
        self.reset_observations()
        return util

    def get_rating_set_uv(self, n_ratings):
        y = np.ones((n_ratings - 1, 1), dtype='int')
        uv = []
        for i in range(n_ratings):
            other_i = np.delete(np.arange(n_ratings), i)
            uv.append(np.hstack((np.atleast_2d(other_i).T, y*i)))
        return y, uv


    def select_observation(self, domain=None, x_test=None, n_test=50, y_threshold=5, p_pref_tol=1e-3, n_rel_samples=2, n_mc_samples=100):
        # Generate a set of test points in the domain (if not specified)
        if x_test is None:
            # x_test = self.linear_domain_sampler(n_test, domain)
            x_test = self.uniform_domain_sampler(n_test, domain)
        n_test = len(x_test)

        # We save the f value because otherwise it gets out of whack when we add observations
        flap = self.solve_laplace()
        fhat, vhat = self.predict_latent(x_test)

        # Sample a set of functions from the current posterior
        f_post = np.random.multivariate_normal(fhat.flatten(), vhat, n_mc_samples)

        # Generate a set of random groups of n_rel_samples from the input space
        uvi = np.array([np.random.choice(n_test, n_rel_samples, replace=False) for i in range(n_test)])
        V_max_rel = 0.0

        self.store_observations()

        # Relative observations
        t_rel = time.time()
        y_uv, full_uv = self.get_rating_set_uv(n_rel_samples)

        # Calculate the expected value for each observation set
        for i, uv in enumerate(uvi):
            x = x_test[uv]
            p_pref = np.zeros(n_rel_samples)
            for fp in f_post:
                p_pref[np.argmax(fp[uv])] += 1
            p_pref /= n_mc_samples
            V_rel = 0.0

            for i, p_x in enumerate(p_pref):
                if p_x > p_pref_tol:
                    V_rel += p_x*self.test_observation(x, y_uv, full_uv[i], x_test, y_threshold)
            if V_rel >= V_max_rel:
                V_max_rel = V_rel
                x_best_rel = x

        # best_n = np.argpartition(V, -n_comparators)[-n_comparators:]
        # best = np.argmax(V)
        if self.verbose >= 1:
            print 'V_max_rel = {0}, x = {2}, t = {1}s'.format(V_max_rel, time.time()-t_rel, x_best_rel[:,0])

        # Absolute queries
        V_max = 0.0
        t_rel = time.time()
        p_y, mu = self.abs_likelihood.posterior_likelihood(fhat, vhat)
        for x, ppy in zip(x_test, p_y.T):
            V_abs = 0.0
            for y_obs, p_obs in enumerate(ppy):
                if p_obs > p_pref_tol:
                    V_abs += p_obs*self.test_observation(x, y_obs+1, None, x_test, y_threshold)
            if V_abs > V_max:
                V_max = V_abs
                x_best = x
        if self.verbose >= 1:
            print 'V_max_abs = {0}, t = {1}s'.format(V_max, time.time() - t_rel)

        if V_max_rel > V_max:
            x_best = x_best_rel

        return x_best


class OrdinalOptimist(OrdinalSampler):
    def model_value(self, fhat, vhat, y_threshold):
        # Probability of values above threshold
        p_y, mu = self.abs_likelihood.posterior_likelihood(fhat, vhat)
        value = p_y[y_threshold-1:,:].sum()/len(fhat)
        return value/len(fhat)

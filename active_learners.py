import numpy as np
import GPpref
from scipy.stats import beta
import plot_tools as ptt


def calc_ucb(fhat, vhat, gamma=2.0, sigma_offset=0.0):
    return fhat + gamma * (np.sqrt(np.atleast_2d(vhat.diagonal()).T) - sigma_offset)

def softmax_selector(x, tau=1.0):
    # High tau is more random
    ex = np.exp((x - x.max())/tau)
    Px = ex/ex.sum()
    return np.random.choice(len(x), p=Px)

class ActiveLearner(GPpref.PreferenceGaussianProcess):
    def init_extras(self):
        self._default_uvi = np.array([[0, 1]])
        self._plus_y_obs = np.ones((1, 1), dtype='int')
        self._minus_y_obs = -1*self._plus_y_obs

    def set_hyperparameters(self, log_hyp):
        self.log_hyp = log_hyp

    def solve_laplace(self, log_hyp=None):
        if log_hyp is None:
            log_hyp = self.log_hyp
        self.f = self.calc_laplace(log_hyp)
        return self.f

    def get_observations(self):
        return self.x_rel, self.uvi_rel, self.x_abs, self.y_rel, self.y_abs

    def select_observation(self, p_rel=0.5, domain=None, n_rel_samples=2):
        if np.random.uniform() > p_rel: # i.e choose an absolute sample
            n_rel_samples = 1
        return self.uniform_domain_sampler(n_rel_samples, domain)

    def uniform_domain_sampler(self, n_samples, domain=None):
        # Domain should be 2 x n_xdim, i.e [[x0_lo, x1_lo, ... , xn_lo], [x0_hi, x1_hi, ... , xn_hi ]]
        x_test = np.random.uniform(size=(n_samples, self._xdim))
        if domain is not None:
            x_test = x_test*np.diff(domain, axis=0) + domain[0, :]
        return x_test

    def linear_domain_sampler(self, n_samples, domain=None):
        # One dimension only at the moment!
        assert self._xdim == 1
        x_test = np.atleast_2d(np.linspace(.0, 1.0, n_samples, self._xdim)).T
        if domain is not None:
            x_test = x_test*np.diff(domain, axis=0) + domain[0, :]
        return x_test

    def create_posterior_plot(self, x_test, f_true, mu_true, rel_sigma, fuv_train, abs_y_samples, mc_samples):
        # Latent predictions
        fhat, vhat = self.predict_latent(x_test)

        # Expected values
        E_y = self.abs_posterior_mean(x_test, fhat, vhat)

        # Absolute posterior likelihood (MC sampled)
        # Posterior likelihoods (MC sampled for absolute)
        p_abs_y_post = self.abs_posterior_likelihood(abs_y_samples, fhat=fhat, varhat=vhat, normal_samples=mc_samples)
        p_rel_y_post = self.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)
        x_train, uvi_train, x_abs_train, y_train, y_abs_train = self.get_observations()
        uv_train = x_train[uvi_train][:, :, 0]

        # Posterior estimates
        fig_p, (ax_p_l, ax_p_a, ax_p_r) = \
            ptt.estimate_plots(x_test, f_true, mu_true, fhat, vhat, E_y, rel_sigma,
                               abs_y_samples, p_abs_y_post, p_rel_y_post,
                               x_abs_train, y_abs_train, uv_train, fuv_train, y_train,
                               t_a=r'Posterior absolute likelihood, $p(y | \mathcal{Y}, \theta)$',
                               t_r=r'Posterior relative likelihood $P(x_0 \succ x_1 | \mathcal{Y}, \theta)$')
        return fig_p, (ax_p_l, ax_p_a, ax_p_r)

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
        ucb = calc_ucb(fhat, vhat, gamma).flatten()
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
#         ucb = calc_ucb(fhat, vhat, gamma).flatten()
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
        ucb = calc_ucb(fhat, vhat, gamma).flatten()

        if np.random.uniform() < p_rel: # i.e choose a relative sample
            best_n = [softmax_selector(ucb, tau=tau)]   #[np.argmax(ucb)]  #
            # sq_dist = GPpref.squared_distance(x_test, x_test)
            p_rel_y = self.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)

            while len(best_n) < n_rel_samples:
                # ucb = ucb*sq_dist[best_n[-1], :] # Discount ucb by distance
                ucb[best_n[-1]] = 0.0
                ucb *= 2*p_rel_y[best_n[-1],:]*(1.0 - p_rel_y[best_n[-1],:]) # Divide by likelihood that each point is better than previous best
                best_n.append(softmax_selector(ucb, tau=tau))
                # best_n.append(np.argmax(ucb))
        else:
            best_n = [softmax_selector(ucb, tau=tau/2.0)]   #[np.argmax(ucb)]  #
        return x_test[best_n, :]

class UCBAbsRelD(ActiveLearner):
    def select_observation(self, domain=None, n_test=100, p_rel=0.5, n_rel_samples=2, gamma=2.0, tau=1.0):
        x_test = self.uniform_domain_sampler(n_test, domain)
        fhat, vhat = self.predict_latent(x_test)
        ucb = calc_ucb(fhat, vhat, gamma).flatten()

        if np.random.uniform() < p_rel: # i.e choose a relative sample
            best_n = [softmax_selector(ucb, tau=tau)]   #[np.argmax(ucb)]  #
            p_rel_y = self.rel_posterior_likelihood_array(fhat=fhat, varhat=vhat)
            # sq_dist = np.sqrt(GPpref.squared_distance(x_test, x_test))
            while len(best_n) < n_rel_samples:
                # ucb *= sq_dist[best_n[-1], :] # Discount ucb by distance
                ucb[best_n[-1]] = 0.0
                ucb *= 2*p_rel_y[best_n[-1],:]*(1.0 - p_rel_y[best_n[-1],:]) # Divide by likelihood that each point is better than previous best
                best_n.append(softmax_selector(ucb, tau=tau))
                #best_n.append(np.argmax(ucb))
        else:
            best_n = [softmax_selector(ucb, tau=tau/2.0)]   #[np.argmax(ucb)]  #
        return x_test[best_n, :]


class PeakComparitor(ActiveLearner):

    def test_observation(self, x, y, x_test, gamma):
        self.store_observations()
        self.add_observations(x, y, self._default_uvi)
        f = self.solve_laplace()
        fhat, vhat = self.predict_latent(x_test)
        ucb = calc_ucb(fhat, vhat, gamma)
        self.reset_observations()
        return ucb.max()

    def store_observations(self):
        self.crx, self.cuv, self.cax, self.cry, self.cay = self.get_observations()

    def reset_observations(self):
        try:
            self.set_observations(self.crx, self.cuv, self.cax, self.cry, self.cay)
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
    # TODO: FINISH IMPLEMENTATION

    def calculate_threshold_utility(self, fhat, vhat, n_samples, y_threshold):
        y_post = self.sample_latent_posterior(fhat, vhat, n_samples = n_samples)
        threshold_utility = 0.0
        # For each one, evaluate the probability mass above the threshold in the absolute function
        for yi in y_post:
            threshold_utility += self.mean_P_above_threshold(yi, y_threshold)
        return threshold_utility/n_samples

    def mean_P_above_threshold(self, f, y_threshold):
        pmass = 0.0
        for fi in f:
            pmass += 1.0 - self.abs_likelihood.cdf(y_threshold, fi)
        return pmass/f.shape[0]

    def test_observation(self, x, y, x_test, n_samples, y_threshold):
        self.store_observations()
        self.add_observations(x, y, self._default_uvi)
        util, f, v = self.calculate_threshold_utility(x_test, n_samples, y_threshold)
        self.reset_observations()
        return util

    def select_observation(self, domain=None, x_test=None, n_test=100, n_samples=100, y_threshold=0.8):
        # Generate a set of test points in the domain (if not specified)
        if x_test is None:
            x_test = self.linear_domain_sampler(n_test, domain)

        # Sample a set of functions from the current posterior
        self.solve_laplace()
        fhat, vhat = self.predict_latent(x_test)
        base_utility = self.calculate_threshold_utility(fhat, vhat, n_samples, y_threshold)
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
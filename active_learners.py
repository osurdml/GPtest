import numpy as np
import GPpref
from scipy.stats import beta

class ActiveLearner(object):
    def __init__(self, x_train, uvi_train, x_abs_train,  y_train, y_abs_train, delta_f, abs_likelihood):

        self.nx_dim = x_abs_train.shape[1]
        self.GP = GPpref.PreferenceGaussianProcess(x_train, uvi_train, x_abs_train, y_train, y_abs_train,
                                                  delta_f=delta_f,
                                                  abs_likelihood=abs_likelihood)
        self.add_observations = self.GP.add_observations

    def set_hyperparameters(self, log_hyp):
        self.log_hyp = log_hyp

    def calc_laplace(self):
        self.f = self.GP.calc_laplace(self.log_hyp)
        return self.f

    def predict_latent(self, x_test):
        fhat, vhat = self.GP.predict_latent(x_test)
        vhat = np.atleast_2d(vhat.diagonal()).T
        return fhat, vhat

    def expected_y(self, x_test, fhat, vhat):
        # Expected values0
        return self.GP.expected_y(x_test, fhat, vhat)

    def posterior_likelihood(self, fhat, vhat, y_samples, mc_samples):
        # Sampling from posterior to generate output pdfs
        p_y = np.zeros((len(y_samples), len(fhat)))
        iny = 1.0/len(y_samples)
        for i,(fstar,vstar) in enumerate(zip(fhat, vhat)):
            f_samples = mc_samples*vstar+fstar
            aa, bb = self.GP.abs_likelihood.get_alpha_beta(f_samples)
            p_y[:, i] = [iny*np.sum(beta.pdf(yj, aa, bb)) for yj in y_samples]
            # p_y[:, i] /= np.sum(p_y[:, i])
        return p_y

    def get_observations(self):
        return self.GP.x_train, self.GP.uvi_train, self.GP.x_abs_train,  self.GP.y_train, self.GP.y_abs_train

    def select_observation(self, domain=[0.0, 1.0]):
        if np.random.uniform() < 0.5:
            return np.random.uniform(low=domain[0], high=domain[1], size=(1,1))
        else:
            return np.random.uniform(low=domain[0], high=domain[1], size=(2,1))

    #def init_plots(self):
        


class UCBLatent(ActiveLearner):
    def select_observation(self, domain=[0.0, 1.0], ntest=100, gamma=2.0):
        x_test = np.random.uniform(low=domain[0], high=domain[1], size=(ntest, 1))
        fhat, vhat = self.predict_latent(x_test)
        return x_test[np.argmax(fhat + gamma*np.sqrt(vhat))]

class UCBOut(ActiveLearner):
    def select_observation(self, domain=[0.0, 1.0], ntest=100, gamma=2.0):
        # Don't know how to recover the second moment of the predictive distribution, so this isn't done
        x_test = np.random.uniform(low=domain[0], high=domain[1], size=(ntest, 1))
        fhat, vhat = self.predict_latent(x_test)
        Ey = self.expected_y(x_test, fhat, vhat)
        return x_test[np.argmax(Ey)]

class PeakComparitor(ActiveLearner):

    def test_observation(self, x, y, uv, x_test, gamma):
        self.add_observations(x, y, uv)
        f = self.calc_laplace()
        fhat, vhat = self.predict_latent(x_test)
        max_ucb = (fhat + gamma*np.sqrt(np.atleast_2d(vhat.diagonal()).T)).max()
        return max_ucb


    def select_observation(self, domain=[0.0, 1.0], ntest=50, gamma=2.0):
        crx, cuv, cax, cry, cay = self.get_observations()

        x_test = np.random.uniform(low=domain[0], high=domain[1], size=(ntest, 1))
        fhat, vhat = self.GP.predict_latent(x_test)
        max_x = np.argmax(fhat)
        other_x = np.delete(np.arange(ntest), max_x)
        uv = np.vstack( (max_x*np.ones(ntest-1, dtype='int'), other_x) ).T

        p_pref = self.GP.rel_likelihood.prediction(fhat, vhat, uv)
        V = np.zeros(ntest-1)
        x = np.zeros((2,1), dtype='float')
        x[0] = x_test[max_x]
        ypos = np.ones((1, 1),dtype='int')
        ruv = np.array([[0, 1]])

        # Now calculate the expected value for each observation pair
        for i,uv1 in enumerate(other_x):
            x[1] = x_test[uv1]
            V[i] += p_pref[i]*self.test_observation(x, -1*ypos, ruv, x_test, gamma)
            self.GP.set_observations(crx, cuv, cax, cry, cay)
            V[i] += (1-p_pref[i])*self.test_observation(x, ypos, ruv, x_test, gamma)
            self.GP.set_observations(crx, cuv, cax, cry, cay)

        best = np.argmax(V)
        x[1] = x_test[uv[best,1]]
        cV = gamma*np.sqrt(np.atleast_2d(vhat.diagonal())).T + fhat
        cVmax = np.argmax(cV)
        if cV[cVmax] > V.max():
            x = np.array([x_test[cVmax]])
        return x






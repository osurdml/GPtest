import numpy as np


class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


Samplers = Enum(['Metropolis', 'MetropolisHastings', 'Gibbs'])


class MCMCSampler(object):
    def __init__(self, func, limits, q=None, seed=None, func_kwargs={}):
        self.f = func
        # self.sampler = sampler ## , sampler=Samplers.'MetropolisHastings'
        self.limits = limits
        self.lim_range = self.limits[1, :] - self.limits[0, :]
        self.n_dim = self.limits.shape[1]
        np.random.seed(seed)
        if q == None:
            q_std = self.lim_range / 10.0
            q = (lambda x: x + np.random.normal(size=self.n_dim) * q_std)
        self.q = q
        self.func_kwargs = func_kwargs

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, s):
        if s not in Samplers: raise Exception("Specified sampler must be in Sampler set")
        self._sampler = s

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, l):
        l = np.array(l)
        if l.shape[0] != 2: raise Exception("Limits must be a 2*n_dim array")
        self._limits = l

    def generate_sample(self, X_current, f_current):
        # Generate proposed sample
        X_proposed = self.q(X_current)
        while True:
            inlimits = np.logical_and(X_proposed >= self.limits[0, :], X_proposed <= self.limits[1, :])
            if not inlimits.all():
                X_proposed = self.q(X_current)
            else:
                break

        # Acceptance ratio
        f_proposed = self.f(X_proposed, **self.func_kwargs)
        alpha = f_proposed / f_current

        # Accept?
        pp = np.random.rand(1)
        if pp < alpha:  # Accept
            return X_proposed, f_proposed
        return X_current, f_current

    def sample_chain(self, n_samples, burn_in=None, X_start=None):
        if burn_in is None:
            burn_in = np.floor_divide(n_samples, 10)
        if X_start is None:
            X_start = self.limits[0, :] + np.random.rand(self.n_dim) * self.lim_range
        X_current = X_start
        f_current = self.f(X_current, **self.func_kwargs)

        # Run for burn-in and throw away samples
        for t in range(burn_in):
            X_current, f_current = self.generate_sample(X_current, f_current)
            if t % 100 == 0:
                print "Burn-in: {0}/{1}".format(t, burn_in)

        X = np.zeros((n_samples, self.n_dim))
        f_X = np.zeros(n_samples)
        X[0, :] = X_current
        f_X[0] = f_current

        for t in range(1, n_samples):
            X_current, f_current = self.generate_sample(X_current, f_current)
            X[t, :] = X_current
            f_X[t] = f_current
            if t % 100 == 0:
                print "Chain sampling: {0}/{1}".format(t, n_samples)

        return X, f_X
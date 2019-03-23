import numpy as np
import matplotlib.pyplot as plt
import GPpref
import nice_plot_colors as npc
import plot_tools
from scipy.stats import norm
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

def beta_discrete(b, f, n_cat=5):
    P = np.zeros(n_cat+1, dtype='float')
    for i in range(1, n_cat+1):
        P[i] = b.cdf(float(i)/n_cat, f) - P[0:i].sum()
    return P[1:]


# Examine how continuous variable is converted to discrete categories through beta distribution and logistic ordinal
n_ords = 5

n_f = 101
f = np.linspace(-3.0, 3.0, n_f)

# Integrated beta
sigma = 1.0
v = 15.0
beta_likelihood = GPpref.AbsBoundProbit(sigma=sigma, v=v)
Py = np.zeros((n_f, n_ords), dtype='float')
for i, fi in enumerate(f):
    Py[i] = beta_discrete(beta_likelihood, fi, n_ords)

fh, ah = plt.subplots()
beta_lines = []
for i, p in enumerate(Py.T):
    beta_lines.extend(ah.plot(f, p, color=npc.lines[i]))
ah.legend(beta_lines, ['$p(y={0}|f)$'.format(ni+1) for ni in range(n_ords)])

# Logistic ordinal
ord_sigma = 0.25
ord_b = 1.2
b = np.linspace(-ord_b, ord_b, n_ords-1)   # Log breakpoints
Py_ord = np.zeros((n_f, n_ords), dtype='float')
z = lambda ff, bb: (bb-ff) / ord_sigma
Py_ord[:, 0] = norm.cdf(z(f, b[0]))
Py_ord[:, -1] = 1.0-norm.cdf(z(f, b[-1]))
for i in range(1, n_ords-1):
    Py_ord[:,i] = norm.cdf(z(f, b[i])) - norm.cdf(z(f, b[i-1]))

ord_lines = []
for i, p in enumerate(Py_ord.T):
    ord_lines.extend(ah.plot(f, p, color=npc.lines[i], ls='--'))

ah.set_xlabel('$f(x)$')
ah.set_ylabel('$p(y|f)$')


# Testing the Ordinal Probit class in GPPref
ord_kwargs = {'sigma': ord_sigma, 'b': ord_b, 'n_ordinals': n_ords, 'eps': 1.0e-5}
orpro = GPpref.OrdinalProbit(**ord_kwargs)
Py_c = np.zeros((n_f, n_ords), dtype='float')
dPy_c = np.zeros((n_f, n_ords), dtype='float')
d2Py_c = np.zeros((n_f, n_ords), dtype='float')
fv = np.atleast_2d(f).T
y = np.ones(fv.shape, dtype='int')
for i in range(n_ords):
    d2p, dp, p = orpro.derivatives(y*(i+1), fv)
    d2Py_c[:,i], dPy_c[:,i], Py_c[:,i] = d2p.diagonal(), dp.flat, p.flat

fh2, ah2 = plt.subplots(2, 2)
ah2 = ah2.flatten()
ah2[0].plot(f, np.exp(Py_c))
ah2[0].set_title('$P(y|f(x))$')
ah2[0].set_ylabel('$P(y)$')

ah2[1].plot(f, Py_c)
ah2[1].set_title('$ln P(y|f(x))$')
ah2[1].set_ylabel('$ln P(y)$')

ah2[2].plot(f, dPy_c)
ah2[2].set_title('$\partial ln P(y|f(x) / \partial f(x)$')
ah2[2].set_ylabel('$\partial ln P(y) / \partial f(x)$')

ah2[3].plot(f, d2Py_c)
ah2[3].set_title('$\partial^2 l(y,f(x)) / \partial f(x)^2$')
ah2[3].set_ylabel('$\partial^2 ln P(y) / \partial f(x)^2$')
for a in ah2:
    a.set_xlabel('$f(x)$')


# Test the sampler
def fun(x):
    return x
    # return 2.5*np.sin(x/2.0)
obs_fun = GPpref.AbsObservationSampler(fun, 'OrdinalProbit', ord_kwargs)
x_plot = f
x_test = np.atleast_2d(x_plot).T
f_true = obs_fun.f(x_test)
y_samples = np.atleast_2d(np.arange(1, n_ords+1, 1, dtype='int')).T
p_y_array = obs_fun.observation_likelihood_array(x_test, y_samples)
z, mu = obs_fun.generate_observations(x_test)

fh3, ah3 = plt.subplots(1, 2)
plot_tools.plot_with_bounds(ah3[0], x_test, f_true, ord_sigma, c=npc.lines[0])
abs_extent = [x_test[0, 0], x_test[-1, 0], y_samples[0, 0]-0.5, y_samples[-1, 0]+0.5]
h_pat = ah3[1].imshow(p_y_array, origin='lower', extent=abs_extent, aspect='auto')
ah3[1].plot(x_test, z, 'w+')
h_yt, = ah3[1].plot(x_test, mu, c=npc.lines[1])
ah3[1].legend([h_yt], ['$E[y]$'])
fh3.colorbar(h_pat, ax=ah3[1])


# Plot some beta likelihood stuff for Thane
# beta_likelihood = GPpref.AbsBoundProbit(sigma=1.0, v=25.0)
# y = np.atleast_2d(np.linspace(0.0, 1.0, 201)).T
# f = np.linspace(-2.0, 2.0, 5)
# p_y = np.zeros((y.shape[0], len(f)), dtype='float')
# for i, fxi in enumerate(f):
#     p_y[:, i:i + 1] = beta_likelihood.likelihood(y, fxi)
# fh4, ah4 = plt.subplots()
# h_b = plt.plot(y, p_y)
# ah4.legend(h_b, ['$f = {0:0.1f}$'.format(fi) for fi in f])
# ah4.set_ylim([0, 10.0])
# ah4.set_xlabel('$y$')
# ah4.set_ylabel('$p(y|f)$')
# ah4.set_title('Symmetric beta likelihood')
plt.show(block=False)




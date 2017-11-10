import numpy as np
import matplotlib.pyplot as plt
import GPpref
import nice_plot_colors as npc
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
f = np.linspace(-9.0, 9.0, n_f)

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
ah.legend(beta_lines, ['$p(y={0}|f)$'.format(ni) for ni in range(n_ords)])

# Logistic ordinal
log_sigma = 0.5
b = np.linspace(-1.1, 1.1, n_ords-1)   # Log breakpoints
Py_ord = np.zeros((n_f, n_ords), dtype='float')
z = lambda ff, bb: (bb-ff)/log_sigma
Py_ord[:, 0] = norm.cdf(z(f, b[0]))
Py_ord[:, -1] = 1.0-norm.cdf(z(f, b[-1]))
for i in range(1, n_ords-1):
    Py_ord[:,i] = norm.cdf(z(f, b[i])) - norm.cdf(z(f, b[i-1]))

ord_lines = []
for i, p in enumerate(Py_ord.T):
    ord_lines.extend(ah.plot(f, p, color=npc.lines[i], ls='--'))

ah.set_xlabel('$f(x)$')
ah.set_ylabel('$p(y|f)$')


# Testing the Ordinal Probit class
orpro = GPpref.OrdinalProbit(log_sigma, b[-1], n_ords, eps=1.0e-5)
Py_c = np.zeros((n_f, n_ords), dtype='float')
dPy_c = np.zeros((n_f, n_ords), dtype='float')
d2Py_c = np.zeros((n_f, n_ords), dtype='float')
fv = np.atleast_2d(f).T
y = np.ones(fv.shape, dtype='int')
for i in range(n_ords):
    d2p, dp, p = orpro.derivatives(y*(i+1), fv)
    d2Py_c[:,i], dPy_c[:,i], Py_c[:,i] = d2p.diagonal(), dp.flat, p.flat

fh2, ah2 = plt.subplots(1, 3)
ah2[0].plot(f, np.exp(Py_c))
ah2[0].set_title('$P(y|f(x))$')
ah2[1].plot(f, dPy_c)
ah2[1].set_title('$\partial l(y,f(x)) / \partial f(x)$')
ah2[2].plot(f, d2Py_c)
ah2[2].set_title('$\partial^2 l(y,f(x)) / \partial f(x)^2$')

plt.show()




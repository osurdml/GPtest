import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import GPpref
import nice_plot_colors as npc

def convert_to_discrete(b, f, n_cat=5):
    P = np.zeros(n_cat+1, dtype='float')
    for i in range(1, n_cat+1):
        P[i] = b.cdf(float(i)/n_cat, f) - P[0:i].sum()
    return P[1:]

def draw_probs(ax, x, P):
    del_x = x[1]-x[0]
    ddel_x = del_x / P.shape[0]
    for xi, p in zip(x, P.T): # Loop over columns
        ip = np.argsort(p, )
        for i in ip[::-1]:
            box = Rectangle((xi+ddel_x*i, 0.0), ddel_x, p[i], color=npc.bars[i])
            ax.add_artist(box)

# Examine how continuous variable is converted to discrete categories through beta distribution
sigma = 1.0
v = 10.0
n = 5

beta_likelihood = GPpref.AbsBoundProbit(sigma=sigma, v=v)

f = np.linspace(-3.0, 3.0, 7)

y = np.linspace(0.0, 1.0, 201)
py = np.zeros((len(f), len(y)), dtype='float')
Py = np.zeros((len(f), n), dtype='float')
for i, fi in enumerate(f):
    py[i] = beta_likelihood.likelihood(y, fi)
    Py[i] = convert_to_discrete(beta_likelihood, fi, n)

fh, ah = plt.subplots()
xbars = np.arange(0.0, 1.0, step=1.0/n)
draw_probs(ah, xbars, Py)
ah2 = ah.twinx()
lines = []
for i, p in enumerate(py):
    lines.extend(ah2.plot(y, p, color=npc.lines[i]))
ah2.set_ybound(lower=0.0)
ah2.legend(lines, ['${0:0.1f}\sigma$'.format(fi) for fi in f])

# Logistic ordinal
log_sigma = 0.1
b = np.linspace(-3.0, 3.0, n)   # Log breakpoints
Py_ord = np.zeros((len(f), n), dtype='float')
for i, fi in enumerate(f):
    py[i] = beta_likelihood.likelihood(y, fi)
    Py[i] = convert_to_discrete(beta_likelihood, fi, n)



plt.show()




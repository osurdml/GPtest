import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from nice_plot_colors import *

def make_poly_array(x,y,sigma):
    nx = len(x)
    sigma = np.atleast_2d(sigma)
    xy = np.zeros((2*nx, 2))
    xy[:,0] = np.append(x, x[::-1])
    xy[:,1] = np.append(y-sigma, y[::-1]+sigma[::-1])
    return xy


def plot_with_bounds(ax, x, y, s, c=lines[0]):
    xy = make_poly_array(x, y, s)
    h_patch = Polygon(xy, ec=c, fc=lighten(c, 3), alpha=0.5)
    h_fx, = ax.plot(x, y, lw=1.5, c=c)
    ax.add_patch(h_patch)
    clim = ax.get_ylim()
    ax.set_ylim(bottom = min(clim[0],xy[:,1].min()), top = max(clim[1], xy[:,1].max()))
    return h_fx, h_patch


def plot_setup_2d(t_l = r'Latent function, $f(x)$', t_a = r'Absolute likelihood, $p(y | f(x))$',
                  t_r = r'Relative likelihood, $P(x_0 \succ x_1 | f(x_0), f(x_1))$'):

    fig, (ax_l, ax_a, ax_r) = plt.subplots(1, 3)
    fig.set_size_inches(14.7, 3.5)

    # Latent function
    ax_l.set_title(t_l)
    ax_l.set_xlabel('$x$')
    ax_l.set_ylabel('$f(x)$')

    # Absolute likelihood
    ax_a.set_title(t_a)
    ax_a.set_xlabel('$x$')
    ax_a.set_ylabel('$y$')

    # Relative likelihood
    ax_r.set_title(t_r)
    ax_r.set_xlabel('$x_0$')
    ax_r.set_ylabel('$x_1$')
    return fig, (ax_l, ax_a, ax_r)


def plot_relative_likelihood(ax, p_y, extent):
    h_p = ax.imshow(p_y, origin='lower', extent=extent, vmin=0.0, vmax=1.0)
    h_pc = ax.contour(p_y, levels=[0.5], origin='lower', linewidths=2, extent=extent)
    plt.clabel(h_pc, inline=1, fontsize=10)
    ax.get_figure().colorbar(h_p, ax=ax)
    return h_p
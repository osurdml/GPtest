import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from nice_plot_colors import *

def make_poly_array(x,y,sigma):
    nx = len(x)
    xy = np.zeros((2*nx, 2))
    xy[:,0] = np.append(x, x[::-1])
    xy[:,1] = np.append(y-sigma, y[::-1]+sigma)
    return xy


def plot_with_bounds(ax, x, y, s, c=lines[0]):
    h_patch = Polygon(make_poly_array(x, y, s), ec=c, fc=lighten(c, 3), alpha=0.5)
    h_fx, = ax.plot(x, y, lw=1.5, c=c)
    ax.add_patch(h_patch)
    return h_fx, h_patch
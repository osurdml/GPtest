import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from nice_plot_colors import *
from cycler import cycler

# plt.rc('axes', prop_cycle=(cycler('color', [greyify(c, .5, .8) for c in reversed(lines)])))
plt.rc('axes', prop_cycle=(cycler('color', lines)))

def make_poly_array(x,y,sigma):
    nx = len(x)
    sigma = np.atleast_2d(sigma)
    xy = np.zeros((2*nx, 2))
    xy[:,0] = np.append(x, x[::-1])
    xy[:,1] = np.append(y-sigma, y[::-1]+sigma[::-1])
    return xy


def plot_with_bounds(ax, x, y, s, c=lines[0]):
    isort = np.argsort(x.flat)
    xx, yy = x[isort], y[isort]
    try:
        ss = s[isort]
    except:
        ss = s
    xy = make_poly_array(xx, yy, ss)
    h_patch = Polygon(xy, ec=c, fc=lighten(c, 3), alpha=0.5)
    h_fx, = ax.plot(xx, yy, lw=1.5, c=c)
    ax.add_patch(h_patch)
    clim = ax.get_ylim()
    ax.set_ylim(bottom = min(clim[0],xy[:,1].min()), top = max(clim[1], xy[:,1].max()))
    return h_fx, h_patch

def plot_setup_rel(t_l = r'Latent function, $f(x)$', t_r = r'Relative likelihood, $P(x_0 \succ x_1 | f(x_0), f(x_1))$'):

    fig, (ax_l, ax_r) = plt.subplots(1, 2)
    fig.set_size_inches(9.8, 3.5)

    # Latent function
    ax_l.set_title(t_l)
    ax_l.set_xlabel('$x$')
    ax_l.set_ylabel('$f(x)$')

    # Relative likelihood
    ax_r.set_title(t_r)
    ax_r.set_xlabel('$x_0$')
    ax_r.set_ylabel('$x_1$')
    return fig, (ax_l, ax_r)

def plot_setup_2d(t_l = r'Latent function, $f(x)$', t_a = r'Absolute likelihood, $p(u | f(x))$',
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
    ax_a.set_ylabel('$u$')

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


def true_plots(xt, ft, mu_t, rel_sigma, y_samples, p_a_y, p_r_y, xa_train, ya_train, uvr_train, fuvr_train, yr_train,
               class_icons=['ko', 'wo'], marker_options={'mec':'k', 'mew':0.5}, *args, **kwargs):


    # Plot true function, likelihoods and observations
    fig, (ax_l, ax_a, ax_r) = plot_setup_2d(**kwargs)

    # True latent
    plot_with_bounds(ax_l, xt, ft, rel_sigma, c=lines[0])

    # True absolute likelihood
    abs_extent = [xt[0, 0], xt[-1, 0], y_samples[0, 0], y_samples[-1, 0]]
    h_pat = ax_a.imshow(p_a_y, origin='lower', extent=abs_extent)
    if xa_train.shape[0] > 0:
        ax_a.plot(xa_train, ya_train, 'w+')
    h_yt, = ax_a.plot(xt, mu_t, c=lines[0])
    ax_a.legend([h_yt], ['$E[u]$'])
    fig.colorbar(h_pat, ax=ax_a)

    # True relative likelihood
    rel_y_extent = [xt[0, 0], xt[-1, 0], xt[0, 0], xt[-1, 0]]
    h_prt = plot_relative_likelihood(ax_r, p_r_y, extent=rel_y_extent)
    if xt.shape[0] > 0:
        for uv, fuv, y in zip(uvr_train, fuvr_train, yr_train):
            ax_r.plot(uv[0], uv[1], class_icons[(y[0] + 1) / 2], **marker_options)
            ax_l.plot(uv, fuv, 'b-', color=lighten(lines[0]))
            ax_l.plot(uv[(y + 1) / 2], fuv[(y + 1) / 2], class_icons[(y[0] + 1) / 2], **marker_options)
    return fig, (ax_l, ax_a, ax_r)


def true_plots_rel(xt, ft, mu_t, rel_sigma, y_samples, p_a_y, p_r_y, xa_train, ya_train, uvr_train, fuvr_train, yr_train,
               class_icons=['ko', 'wo'], marker_options={'mec':'k', 'mew':0.5}, *args, **kwargs):

    # Plot true function, likelihoods and observations
    fig, (ax_l, ax_r) = plot_setup_rel(**kwargs)

    # True latent
    plot_with_bounds(ax_l, xt, ft, rel_sigma, c=lines[0])

    # True absolute likelihood

    # True relative likelihood
    rel_y_extent = [xt[0, 0], xt[-1, 0], xt[0, 0], xt[-1, 0]]
    h_prt = plot_relative_likelihood(ax_r, p_r_y, extent=rel_y_extent)
    if xt.shape[0] > 0:
        for uv, fuv, y in zip(uvr_train, fuvr_train, yr_train):
            ax_r.plot(uv[0], uv[1], class_icons[(y[0] + 1) / 2], **marker_options)
            ax_l.plot(uv, fuv, 'b-', color=lighten(lines[0]))
            ax_l.plot(uv[(y + 1) / 2], fuv[(y + 1) / 2], class_icons[(y[0] + 1) / 2], **marker_options)
    return fig, (ax_l, ax_r)


def estimate_plots(xt, ft, mu_t, fhat, vhat, E_y, rel_sigma,
                   y_samples, p_a_y, p_r_y, xa_train, ya_train, uvr_train, fuvr_train, yr_train,
                   class_icons = ['ko', 'wo'], marker_options = {'mec':'k', 'mew':0.5}, n_posterior_samples=0, *args, **kwargs):
    fig, (ax_l, ax_a, ax_r) = plot_setup_2d(**kwargs)

    # Posterior samples
    if n_posterior_samples > 0:
        y_post = np.random.multivariate_normal(fhat.flatten(), vhat, n_posterior_samples)
        h_pp = ax_l.plot(xt, y_post.T, lw=0.8)

    # Latent function
    hf, hpf = plot_with_bounds(ax_l, xt, ft, rel_sigma, c=lines[0])

    hf_hat, hpf_hat = plot_with_bounds(ax_l, xt, fhat, np.sqrt(np.atleast_2d(vhat.diagonal()).T), c=lines[1])

    ax_l.legend([hf, hf_hat], [r'True latent function, $f(x)$', r'$\mathcal{GP}$ estimate $\hat{f}(x)$'])

    # Absolute posterior likelihood
    abs_extent = [xt[0, 0], xt[-1, 0], y_samples[0, 0], y_samples[-1, 0]]
    h_pap = ax_a.imshow(p_a_y, origin='lower', extent=abs_extent)
    h_yt, = ax_a.plot(xt, mu_t, c=lines[0])
    hEy, = ax_a.plot(xt, E_y, color=lines[3])
    if xa_train.shape[0] > 0:
        ax_a.plot(xa_train, ya_train, 'w+')
    ax_a.legend([h_yt, hEy],
                  [r'True mean, $E[u]$', r'Posterior mean, $E_{p(u|\mathcal{Y})}\left[u\right]$'])
    fig.colorbar(h_pap, ax=ax_a)

    # Relative posterior likelihood
    rel_y_extent = [xt[0, 0], xt[-1, 0], xt[0, 0], xt[-1, 0]]
    h_prp = plot_relative_likelihood(ax_r, p_r_y, extent=rel_y_extent)
    if uvr_train.shape[0] > 0:
        for uv, fuv, y in zip(uvr_train, fuvr_train, yr_train):
            ax_r.plot(uv[0], uv[1], class_icons[(y[0] + 1) / 2], **marker_options)
            ax_l.plot(uv, fuv, 'b-', color=lighten(lines[0]))
            ax_l.plot(uv[(y + 1) / 2], fuv[(y + 1) / 2], class_icons[(y[0] + 1) / 2], **marker_options)

    return fig, (ax_l, ax_a, ax_r)

def estimate_plots_rel(xt, ft, mu_t, fhat, vhat, E_y, rel_sigma,
                   y_samples, p_a_y, p_r_y, xa_train, ya_train, uvr_train, fuvr_train, yr_train,
                   class_icons = ['ko', 'wo'], marker_options = {'mec':'k', 'mew':0.5}, *args, **kwargs):
    fig, (ax_l, ax_r) = plot_setup_rel(**kwargs)

    # Latent function
    hf, hpf = plot_with_bounds(ax_l, xt, ft, rel_sigma, c=lines[0])

    hf_hat, hpf_hat = plot_with_bounds(ax_l, xt, fhat, np.sqrt(np.atleast_2d(vhat.diagonal()).T), c=lines[1])
    ax_l.legend([hf, hf_hat], [r'True latent function, $f(x)$', r'$\mathcal{GP}$ estimate $\hat{f}(x)$'])

    # Absolute posterior likelihood
    abs_extent = [xt[0, 0], xt[-1, 0], y_samples[0, 0], y_samples[-1, 0]]

    # Relative posterior likelihood
    rel_y_extent = [xt[0, 0], xt[-1, 0], xt[0, 0], xt[-1, 0]]
    h_prp = plot_relative_likelihood(ax_r, p_r_y, extent=rel_y_extent)
    if uvr_train.shape[0] > 0:
        for uv, fuv, y in zip(uvr_train, fuvr_train, yr_train):
            ax_r.plot(uv[0], uv[1], class_icons[(y[0] + 1) / 2], **marker_options)
            ax_l.plot(uv, fuv, 'b-', color=lighten(lines[0]))
            ax_l.plot(uv[(y + 1) / 2], fuv[(y + 1) / 2], class_icons[(y[0] + 1) / 2], **marker_options)

    return fig, (ax_l, ax_r)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
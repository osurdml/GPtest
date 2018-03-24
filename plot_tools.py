import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from nice_plot_colors import *
from cycler import cycler
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# plt.rc('axes', prop_cycle=(cycler('color', [greyify(c, .5, .8) for c in reversed(lines)])))
plt.rc('axes', prop_cycle=(cycler('color', lines)))

def xgen(xp, n_dim):
    for i in range(n_dim):
        yield xp

def make_meshlist(x_plot, d_x):
    xx = xgen(x_plot, d_x)
    x_mesh = np.vstack(np.meshgrid(*xx, copy=False)).reshape(d_x, -1).T
    return x_mesh

def make_poly_array(x,y,sigma):
    nx = len(x)
    sigma = np.atleast_2d(sigma)
    xy = np.zeros((2*nx, 2))
    xy[:,0] = np.append(x, x[::-1])
    xy[:,1] = np.append(y-sigma, y[::-1]+sigma[::-1])
    return xy


def plot_with_bounds(ax, x, y, s, c=lines[0], lw=1.5, *args, **kwargs):
    isort = np.argsort(x.flat)
    xx, yy = x[isort], y[isort]
    try:
        ss = s[isort]
    except:
        ss = s
    xy = make_poly_array(xx, yy, ss)
    h_patch = Polygon(xy, ec=c, fc=lighten(c, 3), alpha=0.5)
    h_fx, = ax.plot(xx, yy, lw=lw, c=c, *args, **kwargs)
    ax.add_patch(h_patch)
    clim = ax.get_ylim()
    ax.set_ylim(bottom = min(clim[0],xy[:,1].min()), top = max(clim[1], xy[:,1].max()))
    return h_fx, h_patch

def plot_setup_1d(t_l = r'Latent function, $f(x)$', t_a = r'Absolute likelihood, $p(y | f(x))$',
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

def true_plots(xt, ft, mu_t, rel_sigma, y_samples, p_a_y, p_r_y, xa_train=None, ya_train=None, uvr_train=None, fuvr_train=None, yr_train=None,
               class_icons=['ko', 'wo'], marker_options={'mec':'k', 'mew':0.5}, *args, **kwargs):


    # Plot true function, likelihoods and observations
    fig, (ax_l, ax_a, ax_r) = plot_setup_1d(**kwargs)

    # True latent
    plot_with_bounds(ax_l, xt, ft, rel_sigma, c=lines[0])

    # True absolute likelihood
    dely = y_samples[1, 0] - y_samples[0, 0]
    abs_extent = [xt[0, 0], xt[-1, 0], y_samples[0, 0] - 0.5*dely, y_samples[-1, 0] + 0.5*dely]
    vmax = max(1.0, p_a_y.max())
    h_pat = ax_a.imshow(p_a_y, origin='lower', extent=abs_extent, aspect='auto', vmin=0.0, vmax=vmax)
    if xa_train is not None and xa_train.shape[0] > 0:
        ax_a.plot(xa_train, ya_train, 'w+')
    h_yt, = ax_a.plot(xt, mu_t, c=lines[0])
    ax_a.legend([h_yt], ['$E[y]$'])
    ax_a.set_xlim(xt[0], xt[-1])
    fig.colorbar(h_pat, ax=ax_a)

    # True relative likelihood
    rel_y_extent = [xt[0, 0], xt[-1, 0], xt[0, 0], xt[-1, 0]]
    h_prt = plot_relative_likelihood(ax_r, p_r_y, extent=rel_y_extent)
    if uvr_train is not None and xt.shape[0] > 0:
        for uv, fuv, y in zip(uvr_train, fuvr_train, yr_train):
            ax_r.plot(uv[0], uv[1], class_icons[(y[0] + 1) / 2], **marker_options)
            ax_l.plot(uv, fuv, 'b-', color=lighten(lines[0]))
            ax_l.plot(uv[(y + 1) / 2], fuv[(y + 1) / 2], class_icons[(y[0] + 1) / 2], **marker_options)
    return fig, (ax_l, ax_a, ax_r)

def estimate_plots(xt, ft, mu_t, fhat, vhat, E_y, rel_sigma,
                   y_samples, p_a_y, p_r_y, xa_train, ya_train, uvr_train, fuvr_train, yr_train,
                   class_icons = ['ko', 'wo'], marker_options = {'mec':'k', 'mew':0.5}, n_posterior_samples=0, *args, **kwargs):
    fig, (ax_l, ax_a, ax_r) = plot_setup_1d(**kwargs)

    # Posterior samples
    if n_posterior_samples > 0:
        f_post = np.random.multivariate_normal(fhat.flatten(), vhat, n_posterior_samples)
        h_pp = ax_l.plot(xt, f_post.T, lw=0.8)

    # Latent function
    hf, hpf = plot_with_bounds(ax_l, xt, ft, rel_sigma, c=lines[0])

    hf_hat, hpf_hat = plot_with_bounds(ax_l, xt, fhat, np.sqrt(np.atleast_2d(vhat.diagonal()).T), c=lines[1])

    ax_l.legend([hf, hf_hat], [r'True latent function, $f(x)$', r'$\mathcal{GP}$ estimate $\hat{f}(x)$'])

    # Absolute posterior likelihood
    dely = y_samples[1, 0]-y_samples[0, 0]
    abs_extent = [xt[0, 0], xt[-1, 0], y_samples[0, 0]-0.5*dely, y_samples[-1, 0]+0.5*dely]
    vmax = max(1.0, p_a_y.max())
    h_pap = ax_a.imshow(p_a_y, origin='lower', extent=abs_extent, aspect='auto')
    h_yt, = ax_a.plot(xt, mu_t, c=lines[0])
    hEy, = ax_a.plot(xt, E_y, color=lines[3])
    if xa_train.shape[0] > 0:
        ax_a.plot(xa_train, ya_train, 'w+')
    ax_a.set_xlim(xt[0], xt[-1])
    ax_a.legend([h_yt, hEy],
                  [r'True mean, $E[y]$', r'Posterior mean, $E_{p(y|\mathcal{Y})}\left[y\right]$'])
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

# 2D Plot tools

def plot_setup_2d(t_l = r'Latent function, $f(x)$', t_a = r'Absolute likelihood, $p(y | f(x))$'):

    fig = plt.figure()
    ax_l = fig.add_subplot(121, projection='3d')
    ax_a = fig.add_subplot(122, projection='3d')
    fig.set_size_inches(10.0, 3.5)

    # Latent function
    ax_l.set_title(t_l)
    ax_l.set_xlabel('$x_0$')
    ax_l.set_ylabel('$x_1$')
    ax_l.set_zlabel('$f(x)$')

    # Absolute likelihood
    ax_a.set_title(t_a)
    ax_a.set_xlabel('$x_0$')
    ax_a.set_ylabel('$x_1$')
    ax_a.set_zlabel('$y$')
    return fig, (ax_l, ax_a)

def true_plots2D(xt, ft, mu_t, rel_sigma, y_samples, p_a_y, xa_train=None, ya_train=None, uvr_train=None, fuvr_train=None, yr_train=None,
               class_icons=['ko', 'wo'], marker_options={'mec':'k', 'mew':0.5}, *args, **kwargs):

    # Plot true function, likelihoods and observations
    nx = int(np.sqrt(xt.shape[0]))
    x0 = xt[0:nx, 0]  # Assuming generated using make_meshlist
    x1 = xt[0:-1:nx, 1]
    xx, yy = np.meshgrid(x0, x1)

    fig, (ax_l, ax_a) = plot_setup_2d(**kwargs)

    # True latent
    ax_l.plot_surface(xx, yy, np.reshape(ft, (nx, nx)), color=lines[0])
    # ax_l.imshow(np.reshape(ft, (nx, nx)), extent=[x0[0], x1[-1], x1[0], x1[1]], origin='lower', aspect='auto')

    # True absolute likelihood
    h_yt = ax_a.plot_wireframe(xx, yy, np.reshape(mu_t, (nx, nx)), color=lines[1])

    norm_py = p_a_y/p_a_y.max()
    cc = cm.get_cmap('Blues')
    h_py = []
    for y, py in zip(y_samples, norm_py):
        h_py.append(ax_a.scatter(xt[:, 0], xt[:, 1], y, s=py*15.0, marker='o', c=cc(py)))

    if xa_train is not None and xa_train.shape[0] > 0:
        ax_a.scatter(xa_train[:, 0], xa_train[:, 1], ya_train, c='w', marker='+')
    ax_a.legend([h_yt], [r'True mean $E[y]$'])
    # ax_a.set_xlim(xt[0], xt[-1])

    return fig, (ax_l, ax_a)


def estimate_plots2D(xt, ft, mu_t, fhat, vhat, E_y, rel_sigma,
                   y_samples, p_a_y, xa_train, ya_train, uvr_train, fuvr_train, yr_train,
                   class_icons = ['ko', 'wo'], marker_options = {'mec':'k', 'mew':0.5}, *args, **kwargs):

    # Plot estimated function, likelihoods and observations
    nx = int(np.sqrt(xt.shape[0]))
    x0 = xt[0:nx, 0]  # Assuming generated using make_meshlist
    x1 = xt[0:-1:nx, 1]
    xx, yy = np.meshgrid(x0, x1)

    fig, (ax_l, ax_a) = plot_setup_2d(**kwargs)
    cc = cm.get_cmap('inferno')

    # Latent function estimate
    # hf =ax_l.plot_wireframe(xx, yy, np.reshape(ft, (nx, nx)), color=lines[0])
    tfc = np.reshape(vhat.diagonal(), (nx, nx))
    hf_hat = ax_l.plot_surface(xx, yy, np.reshape(fhat, (nx, nx)), facecolors=cc(tfc/tfc.max()))
    # ax_l.imshow(np.reshape(fhat, (nx, nx)), extent=[x0[0], x1[-1], x1[0], x1[1]], origin='lower', aspect='auto')
    # ax_l.legend([hf], [r'True latent function, $f(x)$']) #, r'$\mathcal{GP}$ estimate $\hat{f}(x)$'])

    if uvr_train.shape[0] > 0:
        rel_segments = np.zeros((uvr_train.shape[0], 2, 3))
        rel_segments[:, :, 0:2] = uvr_train
        rel_segments[:, :, 2] = fuvr_train
        rel_lines = Line3DCollection(rel_segments, color=lines[3])
        ax_l.add_collection(rel_lines)

        rel_hipoints = np.array([uv[(i+1)/2] for uv, i in zip(rel_segments, yr_train.flat)])
        ax_l.plot(rel_hipoints[:, 0], rel_hipoints[:, 1], rel_hipoints[:, 2], 'ko', **marker_options)

    # Absolute posterior likelihood
    # h_yt = ax_a.plot_wireframe(xx, yy, np.reshape(mu_t, (nx, nx)), color=lines[1])
    hEy = ax_a.plot_wireframe(xx, yy, np.reshape(E_y, (nx, nx)), color=lines[3])

    cc = cm.get_cmap('Blues')
    norm_py = p_a_y/p_a_y.max()
    for y, py in zip(y_samples, norm_py):
        ax_a.scatter(xt[:, 0], xt[:, 1], y, s=py*15.0, marker='o', c=cc(py))

    if xa_train.shape[0] > 0:
        ax_a.plot(xa_train[:,0], xa_train[:,1], ya_train.flat, 'r^', color=lines[1])
    ax_a.legend([hEy], [r'Posterior mean, $E_{p(y|\mathcal{Y})}\left[y\right]$'])
    # fig.colorbar(h_pap, ax=ax_a)


    return fig, (ax_l, ax_a)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
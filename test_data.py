import numpy as np


def damped_wave(x):
    y = np.cos(6 * np.pi * (x - 0.5)) * np.exp(-10 * (x - 0.5) ** 2)
    return y

def multi_peak(x):
    y = np.cos(6 * np.pi * (x - 0.5))
    return y

def basic_sine(x):
    y = (np.sin(x*2*np.pi + np.pi/4))/1.2
    return y


def zero_fun(x):
    return 0*x

def data1():
    x_rel = np.array([[0.6], [0.7]])
    uvi_rel = np.array([[0, 1], [1, 0]], dtype='int')
    uv_rel = x_rel[uvi_rel][:,:,0]
    y_rel = np.array([[1], [1]], dtype='int')
    fuv_rel = np.array([[-0.1, 0.1], [-0.1, 0.1]])

    x_abs = np.array([[0.2]])
    y_abs = np.array([[0.5]])
    mu_abs = np.array([[0.0]])

    return x_rel, uvi_rel, uv_rel, y_rel, fuv_rel, x_abs, y_abs, mu_abs

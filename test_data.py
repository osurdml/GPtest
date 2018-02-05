import numpy as np
import pickle


def wrms(y_true, y_est, weight=True):
    # RMS weighted by the true value of y (high value high importance)
    if weight:
        w = y_true
    else:
        w = 1.0
    return np.sqrt(np.mean(((y_true - y_est)*w)**2))


def wrms_misclass(y_true, y_est, w_power=2):
    # This is the misclassification error, where the weight is the max of the true or predicted value (penalise
    # predicting high values if the true value is low)
    w = np.power(np.maximum(y_true, y_est), w_power)
    return np.sqrt(np.mean(((y_true - y_est)*w)**2))

def rel_error(y_true, prel_true, y_est, prel_est, weight=False):
    if weight:
        y_max = np.maximum(y_true.flatten(), y_est.flatten())
    else:
        y_max = np.ones(y_true.shape[0], dtype='float')
    nx = prel_true.shape[0]
    mean_p_err = 0.0
    w_sum = 0.0
    for i in range(nx):
        for j in range(i, nx):
            w = max(y_max[i], y_max[j])
            w_sum += w
            mean_p_err += w*np.abs(prel_true[i,j] - prel_est[i,j])
    return mean_p_err/w_sum

def ordinal_kld(p_y_true, p_y_est, w = np.array([1.0])):
    kld = -(p_y_true * np.log(p_y_est/p_y_true)).sum(axis=0)
    wkld = w*kld/w.mean()
    return wkld.mean()

class ObsObject(object):
    def __init__(self, x_rel, uvi_rel, x_abs, y_rel, y_abs):
        self.x_rel, self.uvi_rel, self.x_abs, self.y_rel, self.y_abs = x_rel, uvi_rel, x_abs, y_rel, y_abs

    def get_obs(self):
        return self.x_rel, self.uvi_rel, self.x_abs, self.y_rel, self.y_abs

def obs_stats(obs_array, n_rel_samples):
    for method in obs_array:
        n_rel = 0.0
        n_abs = 0.0
        for ob in method['obs']:
            n_rel += ob.x_rel.shape[0]/n_rel_samples
            n_abs += ob.x_abs.shape[0]
        print "{0}: p_rel = {1}, p_abs = {2}".format(method['name'], n_rel/(n_rel+n_abs), n_abs/(n_rel+n_abs))


class VariableWave(object):
    def __init__(self, amp_range, f_range, off_range, damp_range, n_components=1, n_dimensions=1):
        self.amp_range = amp_range
        self.f_range = f_range
        self.off_range = off_range
        self.damp_range = damp_range
        self.n_components = n_components
        self.n_dimensions = n_dimensions
        self.randomize()

    def out(self, x):
        y = self.amplitude*np.cos(self.frequency * np.pi * (x-self.offset)) * np.exp(-self.damping*(x-self.offset)**2)
        return y

    def set_values(self, a, f, o, d):
        self.amplitude = a
        self.frequency = f
        self.offset = o
        self.damping = d

    def randomize(self, print_vals=False):
        self.amplitude = np.random.uniform(low=self.amp_range[0], high=self.amp_range[1]/self.n_components)
        self.frequency = np.random.uniform(low=self.f_range[0], high=self.f_range[1])
        self.offset = np.random.uniform(low=self.off_range[0], high=self.off_range[1])
        self.damping = np.random.uniform(low=self.damp_range[0], high=self.damp_range[1])
        if print_vals:
            self.print_values()

    def print_values(self):
        print "a: {0:.2f}, f: {1:.2f}, o: {2:.2f}, d: {3:.2f}".format(self.amplitude, self.frequency, self.offset, self.damping)

    def export_state(self):
        return dict(amplitude=self.amplitude, frequency=self.frequency, offset=self.offset, damping=self.damping)


class MultiWave(VariableWave):
    def out(self, x):
        y = np.zeros(np.array(x).shape)
        for a, f, o, d in zip(self.amplitude, self.frequency, self.offset, self.damping):
            y += a*np.cos(f*np.pi*(x-o)) * np.exp(-d*(x-o)**2)
        return y.sum(axis=1, keepdims=True)

    def randomize(self, print_vals=False):
        self.amplitude = np.random.uniform(low=self.amp_range[0], high=self.amp_range[1], size=(self.n_components, self.n_dimensions))
        self.frequency = np.random.uniform(low=self.f_range[0], high=self.f_range[1], size=(self.n_components, self.n_dimensions))
        self.offset = np.random.uniform(low=self.off_range[0], high=self.off_range[1], size=(self.n_components, self.n_dimensions))
        self.damping = np.random.uniform(low=self.damp_range[0], high=self.damp_range[1], size=(self.n_components, self.n_dimensions))
        if print_vals:
            self.print_values()

    def print_values(self):
        print "a: {0}, f: {1}, o: {2}, d: {3}".format(self.amplitude, self.frequency, self.offset, self.damping)

    def get_values(self):
        return self.amplitude, self.frequency, self.offset, self.damping


class DoubleMultiWave(object):
    def __init__(self, amp_range, f_range, off_range, damp_range, n_components=2):
        self.f_low = MultiWave(amp_range[0:2], f_range[0:2], off_range[0:2], damp_range[0:2], n_components=1)
        self.f_high = MultiWave(amp_range[2:4], f_range[2:4], off_range[2:4], damp_range[2:4], n_components=n_components-1)
        self.randomize()

    def out(self, x):
        return self.f_low.out(x)+self.f_high.out(x)

    def randomize(self, print_vals=False):
        self.f_low.randomize()
        self.f_high.randomize()
        if print_vals:
            self.print_values()

    def print_values(self):
        self.f_low.print_values()
        self.f_high.print_values()


class WaveSaver(object):
    def __init__(self, n_trials, n_components):
        self.amplitude = np.zeros((n_trials, n_components), dtype='float')
        self.frequency = np.zeros((n_trials, n_components), dtype='float')
        self.offset = np.zeros((n_trials, n_components), dtype='float')
        self.damping = np.zeros((n_trials, n_components), dtype='float')
        self.n = 0

    def set_vals(self, n, a, f, o, d):
        self.n = n
        self.amplitude[n] = a
        self.frequency[n] = f
        self.offset[n] = o
        self.damping[n] = d

    def save(self, filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def get_vals(self, n):
        return self.amplitude[n], self.frequency[n], self.offset[n], self.damping[n]

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


def data2():
    x_rel = np.array([[ 0.8517731 ], [ 0.66358258],
                      [ 0.06054717], [ 0.45331369],
                      [ 0.8461625 ], [ 0.58854979]])
    uvi_rel = np.array([[0, 1], [2, 3], [4, 5]], dtype='int')
    uv_rel = x_rel[uvi_rel][:,:,0]
    y_rel = np.array([[-1], [1], [1]], dtype='int')
    fuv_rel = np.array([[0.0043639, -0.10653237], [0.01463141, 0.05046293],
                        [0.01773679, 0.45730181]])

    x_abs = np.array([[0.43432351]])
    y_abs = np.array([[0.38966307]])
    mu_abs = np.array([[0.0]])

    return x_rel, uvi_rel, uv_rel, y_rel, fuv_rel, x_abs, y_abs, mu_abs


def data3():
    x_rel = np.array([[0.8517731 ], [0.66358258],
                      [0.06054717], [0.45331369],
                      [0.8461625 ], [0.58854979]])
    uvi_rel = np.array([[0, 1], [2, 3], [4, 5]], dtype='int')
    uv_rel = x_rel[uvi_rel] # [:,:,0]
    y_rel = np.array([[-1], [1], [1]], dtype='int')
    fuv_rel = np.array([[0.0043639, -0.10653237], [0.01463141, 0.05046293],
                        [0.01773679, 0.45730181]])

    x_abs = np.array([[0.43432351], [0.03362113]])
    y_abs = np.array([[0.38966307], [0.999]])
    mu_abs = np.array([[0.0], [0.0]])

    return x_rel, uvi_rel, uv_rel, y_rel, fuv_rel, x_abs, y_abs, mu_abs

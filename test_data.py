import numpy as np
import pickle


class ObsObject(object):
    def __init__(self, x_rel, uvi_rel, x_abs, y_rel, y_abs):
        self.x_rel, self.uvi_rel, self.x_abs, self.y_rel, self.y_abs = x_rel, uvi_rel, x_abs, y_rel, y_abs


class VariableWave(object):
    def __init__(self, amp_range, f_range, off_range, damp_range, n_components=1):
        self.amp_range = amp_range
        self.f_range = f_range
        self.off_range = off_range
        self.damp_range = damp_range
        self.n_components = n_components
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


class MultiWave(VariableWave):
    def out(self, x):
        y = np.zeros(np.array(x).shape)
        for a, f, o, d in zip(self.amplitude, self.frequency, self.offset, self.damping):
            y += a*np.cos(f*np.pi*(x-o)) * np.exp(-d*(x-o)**2)
        return y

    def randomize(self, print_vals=False):
        self.amplitude = np.random.uniform(low=self.amp_range[0], high=self.amp_range[1], size=self.n_components)
        self.frequency = np.random.uniform(low=self.f_range[0], high=self.f_range[1], size=self.n_components)
        self.offset = np.random.uniform(low=self.off_range[0], high=self.off_range[1], size=self.n_components)
        self.damping = np.random.uniform(low=self.damp_range[0], high=self.damp_range[1], size=self.n_components)
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

    def set_vals(self, n, a, f, o, d):
        self.amplitude[n] = a
        self.frequency[n] = f
        self.offset[n] = o
        self.damping[n] = d

    def save(self, filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

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

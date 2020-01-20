import pandas
import os
import numpy as np
from utils.data_downloader import MrDataGrabber
from gp_tools.GPpref import AbsObservationSampler, RelObservationSampler

class WineQualityData(object):

    def __init__(self, wine_type='red', file_loc='data/wine_quality/',
                 cols='all', y_index='quality', norm=False, scale_y=True):
        self.type = wine_type
        self.y_index = y_index
        self.norm = norm
        self.scale_y = scale_y

        # Create target directory
        try:
            os.mkdir(file_loc)
            print("Directory ", file_loc, " Created ")
        except OSError:
            print("Directory ", file_loc, " already exists")

        # Download data
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-{0}.csv'.format(
            wine_type)
        self.download_data = MrDataGrabber(url, file_loc)
        self.download_data.download()
        self.file = self.download_data.target_file

        # Read data file
        self.data = pandas.read_csv(self.file, delimiter=';')

        # Setup data handlers
        self._reset_cols(cols)

        print('Loaded {0} wine data. Contains {1} samples, {2} ' +
              'input dimensions.'.format(wine_type, self.x.shape[0],
                                               self.x.shape[1]))

    def _reset_cols(self, cols='all'):
        if cols == 'all':
            self.data_cols = list(self.data.keys())
            self.data_cols.remove(self.y_index)
        else:
            self.data_cols = cols

        self._reset()

    def _reset(self):
        # Data views
        self.x = self.data[self.data_cols].values
        if self.norm:
            self._norm_x()

        self.y = np.expand_dims(self.data[self.y_index].values, -1)
        # self.y = self.data[self.y_index].values
        if self.scale_y:
            self._scale_y()

        # Probability of labels (for us, one hot)
        self.p_y_true = np.zeros((self.x.shape[0], self.y.max()),
                                 dtype=float)
        for py, y in zip(self.p_y_true, self.y):
            py[y - 1] = 1.0

        self.available_indexes = range(self.x.shape[0])

    def _norm_x(self):
        self.norm = True
        self.x = (self.x - self.x.min(axis=0)) / (self.x.max(axis=0) - self.x.min(axis=0))

    def _scale_y(self):
        # Rescale y to integer ratings, with 1 minimum
        self.scale_y = True
        self.y = self.y + 1 - self.y.min()

    def shuffle(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self._reset()

    def get_data(self, entries=None):
        # Get specified (X, y) pair from data
        if entries is not None:
            return self.x[entries, :], self.y[entries, :]
        else:
            return self.x, self.y

    def random_absolute_obs(self, n=1):
        # Sample random entries
        indexes = np.random.choice(self.data.shape[0], n, replace=False)
        return self.get_data(indexes)

    def get_relative_obs(self, data_uv):
        assert data_uv.shape[1] == 2, 'UV index must be n x 2'
        # No noise observation from data
        x_rel, f_rel = self.get_data(data_uv.flat)

        # Basically just does the indices in order (since we sampled from x)
        # [[0,1],[2,3],[3,4], ...]
        uvi_rel = np.arange(data_uv.shape[0]*2).reshape((data_uv.shape[0], 2))

        y_rel = -1 * np.ones((data_uv.shape[0], 1), dtype='int')
        y_rel[f_rel[uvi_rel[:, 1], 0] > f_rel[uvi_rel[:, 0], 0]] = 1
        return x_rel, uvi_rel, y_rel, f_rel
        # x_rel = wine_data[0:2 * n_rel, 0:-1]
        # uvi_rel = np.array([[0, 1]])
        # fuv_rel = wine_data[0:2 * n_rel, -1]

    def random_relative_obs(self, n=1):
        # Returns uvi_rel, x_rel, y_rel, fuv_rel
        indexes = np.random.choice(self.data.shape[0], (n,2), replace=False)
        x_rel, uvi_rel, y_rel, fuv_rel = self.get_relative_obs(indexes)
        return x_rel, uvi_rel, y_rel, fuv_rel

    def pop_observations(self, indexes):
        # This will return an observation and remove it from the list of
        # available observations
        for i in indexes:
            self.available_indexes.remove(i)
        return self.get_data(indexes)


class WineAbsObsSampler(AbsObservationSampler):

    def _extra_init(self, wine_data_object):
        # This gives us access to the list of indexes of data that has
        # not yet been used, and is therefore available for selecting new
        # samples from.
        self.wine_data_object = wine_data_object

    def _gen_x_obs(self, n, n_xdim=1, domain=None):
        assert n <= len(self.wine_data_object.available_indexes)

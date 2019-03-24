import numpy as np
import pandas
import matplotlib.pyplot as plt

# Get wine data:
# wget -P data/wine_quality https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
# wget -P data/wine_quality https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
class WineQualityData(object):

    def __init__(self, data_file):
        self.file = data_file
        self.data = pandas.read_csv(self.file, delimiter=';')

red_data = WineQualityData('data/wine_quality/winequality-red.csv')
white_data = WineQualityData('data/wine_quality/winequality-white.csv')

fh, ah = plt.subplots(1, 2)
ah[0].hist(red_data.data.quality, np.arange(-0.5, 11, 1.0))
ah[1].hist(white_data.data.quality, np.arange(-0.5, 11, 1.0))
for a in ah:
    a.set_xticks(np.arange(11))
    a.set_xlabel('Score')
ah[0].set_ylabel('Count')
ah[0].set_title('Red wine')
ah[1].set_title('White wine')

plt.show(block=False)
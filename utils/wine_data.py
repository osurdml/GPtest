import pandas
import os
from utils.data_downloader import MrDataGrabber


class WineQualityData(object):
    def __init__(self, wine_type='red', file_loc='data/wine_quality/'):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-{0}.csv'.format(
            wine_type)

        try:
            # Create target Directory
            os.mkdir(file_loc)
            print("Directory ", file_loc, " Created ")
        except OSError:
            print("Directory ", file_loc, " already exists")

        self.download_data = MrDataGrabber(url, file_loc)
        self.download_data.download()
        self.file = self.download_data.target_file
        self.data = pandas.read_csv(self.file, delimiter=';')

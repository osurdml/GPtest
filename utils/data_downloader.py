from __future__ import print_function
import os
import urllib3
import zipfile

class MrDataGrabber(object):  # Does not work with bananas

    def __init__(self, url, target_path):
        self.url = url
        self.path = target_path
        self.target_basename = os.path.basename(self.url)
        self.target_file = os.path.join(self.path, self.target_basename)

    def download(self):
        if os.path.exists(self.target_file):
            print('Target file already downloaded, please remove if you wish to update.')
        else:
            http = urllib3.PoolManager()
            r = http.request('GET', self.url, preload_content=False)

            with open(self.target_file, 'wb') as out:
                print('Downloading {0}...'.format(self.url), end='')
                while True:
                    data = r.read(2**16)
                    if not data:
                        break
                    out.write(data)
            r.release_conn()
            print(' done.')

            if os.path.splitext(self.target_basename)[1].lower() == '.zip':
                self.unzip_data()
        return self.target_file

    def unzip_data(self, target_dir=None, force=False):
        if target_dir is None:
            target_dir = os.path.join(self.path, os.path.splitext(self.target_basename)[0])
        if (not force) and os.path.isdir(target_dir):
            print('Target directory already exists, not unzipping.')
        else:
            zip_ref = zipfile.ZipFile(self.target_file, 'r')
            zip_ref.extractall(target_dir)

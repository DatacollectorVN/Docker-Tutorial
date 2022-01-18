import urllib
import requests
import argparse
import os

class Cfg(object):
    def __init__(self):
        super(Cfg, self).__init__()
        self.data = ["https://github.com/DatacollectorVN/Docker-Tutorial/releases/download/data/train.csv"]
    
    def down_model(self, destination):
        data = self.data[0]
        print ('Start to download, this process take a few minutes')
        urllib.request.urlretrieve(data, destination)
        print("Downloaded dataset - {} to-'{}'".format(data, destination))


def main(directory):
    cfg = Cfg()
    os.makedirs(directory, exist_ok = True)
    cfg.down_model(destination = os.path.join(directory, "train.csv"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', help = 'Destination to save data', type = str,
                        default = './')

    args = parser.parse_args()
    main(directory = args.data_directory)
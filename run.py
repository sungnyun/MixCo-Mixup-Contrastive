import argparse
import warnings
# warnings.filterwarnings('ignore')

parser = argparse.ArgumentParset(description='Process Configuration Dict')

parser.add_argument('--config_path', type=str, help='file path of config file')
parser.add_argument('--device', default='cuda:0', type=str, help='gpu device')
parser.add_argument('--seed', default=100, type=int, help='random seed')
args = parser.parse_args()

class objectview(object):
    def __init__(self, test):
        for k, v in test.items():
            if isinstance(v, dict):
                self.__dict__[k] = objectview(v)
            else:
                self.__dict__[k] = v
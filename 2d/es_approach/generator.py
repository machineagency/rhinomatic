import numpy as np
import imageio
from cmaes import CMAES

class Generator:
    def __init__(self, train_featurizer=False):
        self.cmaes = CMAES(train_featurizer)

if __name__ == '__main__':
    # g = Generator(train_featurizer=True)
    g = Generator()
    g.cmaes.test()


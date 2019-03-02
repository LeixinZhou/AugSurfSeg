import numpy as np


class AddNoise(object):
    """
    The base class for additive noises.
    """

    def noise_gen(self, nparray, *args):
        raise NotImplementedError()

    def __call__(self, nparray):
        return nparray + self.noise_gen(nparray)


class AddNoiseGaussian(AddNoise):
    """
    Add random Gaussian noise. 
    Args:
        loc: Gaussian mean (default 0)
        scale: Gaussian std (default 0.2)
    """

    def __init__(self, loc=0, scale=0.2):
        self.loc = loc
        self.scale = scale

    def noise_gen(self, nparray):
        return np.random.normal(self.loc, self.scale, size=nparray.shape)


class AddNoiseLaplace(AddNoise):
    """
    Add random Laplace noise. 
    Args:
        loc: Laplace distribution peak (default 0)
        scale: Laplace exponential decay (default 0.2)
    """

    def __init__(self, loc=0., scale=0.2):
        self.loc = loc
        self.scale = scale

    def noise_gen(self, nparray):
        return np.random.laplace(self.loc, self.scale, size=nparray.shape)

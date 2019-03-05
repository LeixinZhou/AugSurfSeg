import numpy as np
from .augmentor import AugNoGTChange

class AddNoise(AugNoGTChange):
    """
    The base class for additive noises.
    """

    def noise_gen(self,  *args):
        raise NotImplementedError()

    def img_aug(self, input_img_gt):
        nparray = input_img_gt['img']
        return nparray + self.noise_gen(input_img_gt)


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

    def noise_gen(self, input_img_gt):
        nparray = input_img_gt['img']
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

    def noise_gen(self, input_img_gt):
        nparray = input_img_gt['img']
        return np.random.laplace(self.loc, self.scale, size=nparray.shape)

import numpy as np
from augmentor import AugNoGTChange

class NormalizeSTD(AugNoGTChange):
    """
    Just normalize the input nparray to have zero mean and unit std.
    """
    def img_aug(self, input_img_gt):
        nparray = input_img_gt['img']
        if not type(nparray) is np.ndarray:
            raise TypeError('input is not a numpy array.')
        mean, std = np.mean(nparray), np.std(nparray)
        nparray -= mean
        nparray /= std
        return nparray



class NormalizeLinear(AugNoGTChange):
    """
    Just normalize the input array to have fixed range by linear transformation.object
    Args:
        t_min: the target minimal 
        t_max: the target maximal
    Returns:
        normalized nparray
    """

    def __init__(self, t_min, t_max):
        assert t_max > t_min
        self.t_min = t_min
        self.t_max = t_max

    def img_aug(self, input_img_gt):
        nparray = input_img_gt['img']
        c_min, c_max = np.min(nparray), np.max(nparray)
        return (self.t_max - self.t_min) * (nparray - c_min) / (c_max - c_min) + self.t_min


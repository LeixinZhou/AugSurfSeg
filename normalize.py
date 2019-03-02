import numpy as np


class NormalizeSTD(object):
    """
    Just normalize the input nparray to have zero mean and unit std.
    """
    @staticmethod
    def normalize(nparray):
        if not type(nparray) is np.ndarray:
            raise TypeError('input is not a numpy array.')
        mean, std = np.mean(nparray), np.std(nparray)
        nparray -= mean
        nparray /= std
        return nparray

    def __call__(self, nparray):
        """
        Args:
            nparray: nparray of any size to be normalized.
        Returns:
            nparray: normalized nparray
        """
        return self.normalize(nparray)


class NormalizeLinear(object):
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

    def normalize(self, nparray):
        c_min, c_max = np.min(nparray), np.max(nparray)
        return (self.t_max - self.t_min) * (nparray - c_min) / (c_max - c_min) + t_min

    def __call__(self, nparray):
        return self.normalize(nparray)

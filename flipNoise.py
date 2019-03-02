import numpy as np


class SaltPepperNoise(object):
    """
    Just add salt pepper noise.
    Args:
        sp_ratio: ratio of positions to change to salt (max value) or pepper (min value), default to be 0.1.
        salt_ratio: ratio of salt, default to be 0.5.
    """

    def __init__(self, sp_ratio=0.1, salt_ratio=0.5):
        self.sp_ratio = sp_ratio
        self.salt_ratio = salt_ratio

    def __call__(self, nparray):
        """
        Args:
            nparray: nparray of any size to be added noise.
        Returns:
            nparray: noised nparray
        """
        flipped = np.random.choice([True, False], size=nparray.shape, p=[
                                   self.sp_ratio, 1-self.sp_ratio])
        salted = np.random.choice([True, False], size=nparray.shape, p=[
                                  self.salt_ratio, 1-self.salt_ratio])
        peppered = ~salted
        min_val, max_val = np.min(nparray), np.max(nparray)
        nparray[flipped & salted] = max_val
        nparray[flipped & peppered] = min_val
        return nparray

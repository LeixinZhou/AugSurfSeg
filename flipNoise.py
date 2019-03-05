import numpy as np
from .augmentor import AugNoGTChange


class SaltPepperNoise(AugNoGTChange):
    """
    Just add salt pepper noise.
    Args:
        sp_ratio: ratio of positions to change to salt (max value) or pepper (min value), default to be 0.1.
        salt_ratio: ratio of salt, default to be 0.5.
    """

    def __init__(self, sp_ratio=0.1, salt_ratio=0.5):
        self.sp_ratio = sp_ratio
        self.salt_ratio = salt_ratio

    def img_aug(self, input_img_gt):
        nparray = input_img_gt['img'].copy()
        flipped = np.random.choice([True, False], size=nparray.shape, p=[
                                   self.sp_ratio, 1-self.sp_ratio])
        salted = np.random.choice([True, False], size=nparray.shape, p=[
                                  self.salt_ratio, 1-self.salt_ratio])
        peppered = ~salted
        min_val, max_val = np.min(nparray), np.max(nparray)
        nparray[flipped & salted] = max_val
        nparray[flipped & peppered] = min_val
        return nparray

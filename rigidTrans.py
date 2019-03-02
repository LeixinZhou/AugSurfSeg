from augmentor import AugWithGTChange
import numpy as np


class MirrorLR(AugWithGTChange):
    def img_aug(self, input_img_gt):
        nparray = input_img_gt['img']
        return np.fliplr(nparray)

    def gt_aug(self, input_img_gt):
        nparray = input_img_gt['gt']
        return np.flipud(nparray)


class MirrorUD(AugWithGTChange):
    def img_aug(self, input_img_gt):
        nparray = input_img_gt['img']
        return np.flipud(nparray)

    def gt_aug(self, input_img_gt):
        """
        Note this function assumes the img_nparray has the shape: row x column and the surface position starts from 0. 
        The gt needs to do the operation: row_nb - surface position -1.
        """
        gt_nparray = input_img_gt['gt']
        img_nparray = input_img_gt['img']
        return img_nparray.shape[0] - gt_nparray - 1

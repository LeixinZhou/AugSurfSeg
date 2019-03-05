from .augmentor import AugWithGTChange
import numpy as np
import random


class MirrorLR(AugWithGTChange):
    def img_aug(self, input_img_gt):
        nparray = input_img_gt['img']
        return np.fliplr(nparray).copy()

    def gt_aug(self, input_img_gt):
        nparray = input_img_gt['gt']
        return np.flipud(nparray).copy()


class MirrorUD(AugWithGTChange):
    def img_aug(self, input_img_gt):
        nparray = input_img_gt['img']
        return np.flipud(nparray).copy()

    def gt_aug(self, input_img_gt):
        """
        Note this function assumes the img_nparray has the shape: row x column and the surface position starts from 0. 
        The gt needs to do the operation: row_nb - surface position -1.
        """
        gt_nparray = input_img_gt['gt']
        img_nparray = input_img_gt['img']
        return img_nparray.shape[0] - gt_nparray - 1


class CirculateLR(AugWithGTChange):
    def img_aug(self, input_img_gt, shift_step):
        img_nparray = input_img_gt['img']
        aug_img_nparray = np.empty_like(img_nparray)
        aug_img_nparray[:, :shift_step] = img_nparray[:, -shift_step:]
        aug_img_nparray[:, shift_step:] = img_nparray[:, :-shift_step]
        return aug_img_nparray

    def gt_aug(self, input_img_gt, shift_step):
        gt_nparray = input_img_gt['gt']
        aug_gt_nparray = np.empty_like(gt_nparray)
        aug_gt_nparray[:shift_step] = gt_nparray[-shift_step:]
        aug_gt_nparray[shift_step:] = gt_nparray[:-shift_step]
        return aug_gt_nparray

    def __call__(self, input_img_gt):
        """
        This function circulates image and gt toward right with a random step number.
        """
        shift_step = random.randint(1, input_img_gt['gt'].shape[0]-1)
        return {'img': self.img_aug(input_img_gt, shift_step),
                'gt': self.gt_aug(input_img_gt, shift_step)}


class CirculateUD(AugWithGTChange):
    def img_aug(self, img_nparray, shift_step):
        aug_img_nparray = np.empty_like(img_nparray)
        aug_img_nparray[:shift_step, ] = img_nparray[-shift_step:, ]
        aug_img_nparray[shift_step:, ] = img_nparray[:-shift_step, ]
        return aug_img_nparray

    def gt_aug(self, gt_nparray, shift_step, row_len):
        aug_gt_nparray = (gt_nparray + shift_step) % row_len
        return aug_gt_nparray

    def __call__(self, input_img_gt):
        """
        This function circulates image and gt down with a random step number.
        """
        gt_nparray = input_img_gt['gt']
        img_nparray = input_img_gt['img']
        gt_max, gt_min = np.max(gt_nparray), np.min(gt_nparray)
        row_len = img_nparray.shape[0]
        if row_len - gt_max > gt_min:
            shift_step = random.randint(1, (row_len-gt_max)//2)
        else:
            shift_step = random.randint(-gt_min//2, -1) + row_len
        return {'img': self.img_aug(img_nparray, shift_step),
                'gt': self.gt_aug(gt_nparray, shift_step, row_len)}

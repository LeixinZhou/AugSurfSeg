from augmentor import AugWithGTChange
import numpy as np
import random
from scipy import interpolate
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class RandomCropResize(AugWithGTChange):
    """
    Random crop a continuous subset of columns and then crop a subset of rows containing all surface pixels, and 
    finally resize the cropped image and its gt truth to the original size.
    Args:
        crop ratio: 0-1 (default 0.75)
        img interpolate method: 'linear' or 'cubic' (default 'linear')
        gt interpolated method: 'linear' or 'cubic' (default 'linear')
    Returns:
        Cropped and Resized image and gt
    """

    def __init__(self, crop_ratio=0.75, img_interp='linear', gt_interp='linear'):
        assert crop_ratio > 0 and crop_ratio <= 1
        self.crop_ratio = crop_ratio
        self.img_interp = img_interp
        self.gt_interp = gt_interp

    @staticmethod
    def _get_para(input_img_gt, crop_ratio):
        input_img = input_img_gt['img']
        row_crop_size = int(input_img.shape[0]*crop_ratio)
        col_crop_size = int(input_img.shape[1]*crop_ratio)
        # compute col range
        col_start = random.randint(0, input_img.shape[1] - col_crop_size)
        col_end = col_start + col_crop_size

        input_gt = input_img_gt['gt'][col_start:col_end]
        gt_min, gt_max = int(np.min(input_gt)), int(np.max(input_gt))
        # compute row range
        assert gt_max - gt_min < row_crop_size
        if gt_max - row_crop_size < 0:
            row_start_min = 0
        else:
            row_start_min = gt_max - row_crop_size
        row_start = random.randint(row_start_min, gt_min)
        row_end = row_start + row_crop_size
        
        return row_start, row_end, col_start, col_end

    @staticmethod
    def _crop_resize(input_img_gt, crop_ratio, img_interp, gt_interp, row_start, row_end, col_start, col_end):
        # crop
        cropped_img = input_img_gt['img'][row_start:row_end, col_start:col_end]
        # adjust gt: subtract row start, magnify in the same time as reverse of crop ratio
        cropped_gt = (input_img_gt['gt'][col_start:col_end] - row_start) / crop_ratio
        row_size = row_end - row_start
        col_size = col_end - col_start
        # resize img
        f_img = interpolate.interp2d(np.arange(
            col_size), np.arange(row_size), cropped_img, kind=img_interp)
        resized_img = f_img(np.arange(start=0, stop=col_size-1, step=crop_ratio), np.arange(
            start=0, stop=row_size-1+crop_ratio, step=crop_ratio))
        # resize gt
        f_gt = interpolate.interp1d(
            np.arange(col_size), cropped_gt, kind=gt_interp)
        
        resized_gt = f_gt(np.arange(start=0, stop=col_size-1, step=crop_ratio))
        assert resized_img.shape[1] == resized_gt.shape[0]
        return resized_img, resized_gt

    def __call__(self, input_img_gt):
        row_start, row_end, col_start, col_end = self._get_para(
            input_img_gt, self.crop_ratio)
        img, gt = self._crop_resize(input_img_gt, self.crop_ratio, self.img_interp,
                                    self.gt_interp, row_start, row_end, col_start, col_end)
        return {'img': img, 'gt': gt}

class ElasticTrans(AugWithGTChange):
    def __init__(self, alpha, sigma, img_interp_spline_order=1):
        self.alpha = alpha
        self.sigma = sigma
        self.img_interp = img_interp_spline_order
    def __call__(self, input_img_gt):
        input_img = input_img_gt['img']
        input_gt = input_img_gt['gt']
        img_shape = input_img.shape
        d_row = gaussian_filter(np.random.rand(*input_gt.shape) * 2 - 1, self.sigma) * self.alpha
        row, col = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
        print(d_row.shape, row.shape, col.shape)
        indices =  np.reshape(col, (-1, 1)), np.reshape(row+d_row, (-1, 1))
        new_img = map_coordinates(input_img, indices, order=self.img_interp, mode='reflect').reshape(img_shape)
        new_gt = input_gt - d_row
        return {'img': new_img, 'gt': new_gt}

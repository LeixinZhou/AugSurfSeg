class Augmentor(object):
    """
    The base class of all augmentations.
    """

    def img_aug(self, input_img_gt, *args):
        raise NotImplementedError

    def gt_aug(self, input_img_gt, *args):
        raise NotImplementedError

    def __call__(self, input_img_gt, *args):
        """
        This augments image and its corresponding surface ground truth.
        Args:
            input_img_gt: a dictionary of the shape {'img': img_nparray, 'gt': gt_nparray}. img_nparray shape: row x column, gt_array shape: column.
        Returns:
            Augmented image and ground truth surface.
        """
        return {'img': self.img_aug(input_img_gt), 'gt': self.gt_aug(input_img_gt)}


class AugNoGTChange(Augmentor):
    """
    The base class for all augmentations without need to change the ground truth.
    """

    def img_aug(self, input_img_gt, *args):
        raise NotImplementedError

    def gt_aug(self, input_img_gt, *args):
        return input_img_gt['gt']


class AugWithGTChange(Augmentor):
    """
    The base class for all augmentations with corresponding change of ground truth.
    """

    def img_aug(self, input_img_gt, *args):
        raise NotImplementedError

    def gt_aug(self, input_img_gt, *args):
        raise NotImplementedError

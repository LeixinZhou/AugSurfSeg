class Augmentor(object):
    """
    The base class of all augmentations.
    """

    def img_aug(self, img_nparray, *args):
        raise NotImplementedError

    def gt_aug(self, gt_nparray, *args):
        raise NotImplementedError

    def __call__(self, input):
        """
        This augments image and its corresponding surface ground truth.
        Args:
            input: a dictionary of the shape {'img': img_nparray, 'gt': gt_nparray}.
        Returns:
            Augmented image and ground truth surface.
        """
        return {'img': img_aug(input['img']), 'gt': gt_aug(input['gt'])}


class AugNoGTChange(Augmentor):
    """
    The base class for all augmentations without need to change the ground truth.
    """

    def img_aug(self, img_nparray, *args):
        raise NotImplementedError

    def gt_aug(self, gt_nparray, *args):
        return gt_nparray

class AugWithGTChange(Augmentor):
    """
    The base class for all augmentations with corresponding change of ground truth.
    """

    def img_aug(self, img_nparray, *args):
        raise NotImplementedError

    def gt_aug(self, gt_nparray, *args):
        raise NotImplementedError

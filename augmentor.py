import random


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


class RandomApplyTrans(object):
    """
    Randomly apply transforms from the input list and then apply the necessary transform list.len
    Args:
        trans_seq: randomly applied list
        trans_seq_must: the necessary transform list
    """
    def __init__(self, trans_seq, trans_seq_pre=[], trans_seq_post=[]):
        assert isinstance(trans_seq, (list, tuple))
        self.trans_list = trans_seq
        assert isinstance(trans_seq_post, (list, tuple))
        self.trans_seq_post = trans_seq_post
        assert isinstance(trans_seq_pre, (list, tuple))
        self.trans_seq_pre = trans_seq_pre

    def __call__(self, input_img_gt):
        trans_count = len(self.trans_list)
        rand_trans_nb = random.randint(0, trans_count)
        appllied_trans = random.sample(self.trans_list, rand_trans_nb)
        random.shuffle(appllied_trans)
        if len(self.trans_seq_pre) != 0:
            for i in self.trans_seq_pre:
                input_img_gt = i(input_img_gt)
        if len(appllied_trans) != 0:
            for i in appllied_trans:
                input_img_gt = i(input_img_gt)
        if len(self.trans_seq_post) != 0:
            for i in self.trans_seq_post:
                input_img_gt = i(input_img_gt)
        return input_img_gt

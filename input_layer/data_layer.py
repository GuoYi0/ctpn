"""
InputLayer是整个网络的输入层，在每iter中，需要获取下一个batch的数据，其核心函数是forward
"""
import numpy as np
import cv2
import os


class InputLayer(object):
    def __init__(self, roidb,  config=None):
        if not config:
            raise RuntimeError('Input layer lack config')
        self._cfg = config
        """
        self_roidb是一个列表，列表的每个元素是一个字典，字典的键有
        'image_name'：字符串，表示图片名字
        'gt_boxes'：N行4列矩阵，int32 每一行表示一个GT
        ”height“:高
        "width"：宽
        ”image_path“：图片路径
        """
        self._roidb = roidb.roidb
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = -self._cfg.TRAIN.IMS_BATCH_SIZE

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        cfg = self._cfg
        if self._cur + cfg.TRAIN.IMS_BATCH_SIZE >= len(self._roidb):
            self._shuffle_roidb_inds()
        # 先将指示剂后移一位，再返回去
        self._cur += cfg.TRAIN.IMS_BATCH_SIZE
        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_BATCH_SIZE]
        return db_inds

    def put_hard(self, hard_pos, hard_neg):
        """
        :param hard_pos: 一个多行四列的数组，
        :param hard_neg: 一个多行四列的数组，
        """
        assert hard_neg.ndim == 1, "int {},the dimmension of hard_neg must be one!".format(__file__)
        assert hard_pos.ndim == 1, "int {},the dimmension of hard_pos must be one!".format(__file__)
        self._roidb[self._cur]['hard_pos'].clear()
        self._roidb[self._cur]['hard_pos'].extend(list(hard_pos))

        self._roidb[self._cur]['hard_neg'].clear()
        self._roidb[self._cur]['hard_neg'].extend(list(hard_neg))

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        im_blob = cv2.imread(minibatch_db[0]['image_path'])
        # 去均值化
        im_blob = im_blob.astype(np.float32)
        im_blob -= self._cfg.TRAIN.PIXEL_MEANS
        im_blob = im_blob[np.newaxis, :]
        single_blob = {'data': im_blob,
                       # ”gt_boxes"须是一个N行4列的矩阵，每一行代表一个GT
                       'gt_boxes': minibatch_db[0]['gt_boxes'],
                       # im_info须是一个包含三个元素的向量，分别代表图片的高，宽，缩放比
                       'im_info': np.array([minibatch_db[0]['height'],
                                            minibatch_db[0]['width'], minibatch_db[0]["image_scale"]]),
                       'im_name': os.path.basename(minibatch_db[0]['image_name']),
                       'hard_neg': np.array(minibatch_db[0]['hard_neg'], dtype=np.int32),
                       'hard_pos': np.array(minibatch_db[0]['hard_pos'], dtype=np.int32)
                       }
        return single_blob

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        return self._get_next_minibatch()


def get_data_layer(roidb, config):
    return InputLayer(roidb, config=config)

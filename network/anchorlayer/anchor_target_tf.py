# -*- coding:utf-8 -*-
import numpy as np
import numpy.random as npr
from .generate_anchors import generate_anchors
from .iou import bbox_overlaps
from lib import load_config
from exceptions import NoPositiveError

cfg = load_config()


def anchor_target_layer_py(rpn_cls_score, gt_boxes, im_info, hard_pos, hard_neg, _feat_stride):
    # hard_pos: shape=[None]，一维数组，第0号元素表示该特征图的有效anchor总数，后续的对应hard的ID
    # hard_neg: shape=[None]
    # 生成基本的anchor,一共10个,返回一个10行4列矩阵，每行为一个anchor，返回的只是基于中心的相对坐标
    # 这里返回的4个值是对应的某个anchor的xmin, xmax, ymin, ymax
    _anchors = generate_anchors(cfg)
    _num_anchors = _anchors.shape[0]  # 10个anchor

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    assert rpn_cls_score.shape[0] == 1, 'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]  # feature-map的高宽

    # ==================================================================================
    # generate_anchors生成的是相对坐标，由于划窗之后的特征图上的一个像素点对应原图片中的16 * 16的区域，
    # 因此需要计算出每个anchor在图像中的真实位置，shift_x shift_y 对应于每个像素点的偏移量，
    shift_x = np.arange(0, width) * _feat_stride  # 返回一个列表，[0, 16, 32, 48, ...]
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    A = _num_anchors  # 10个anchor
    K = shifts.shape[0]  # feature-map的像素个数

    # 前者的shape为(1, 10, 4), 后者的shape为(像素数, 1, 4)两者相加
    # 结果为(像素数, 10, 4) python数组广播相加。。。。。。。有待理解
    # all_anchors的维度 k * 10 * 4
    # 用每个像素点所对应的anchor四条边的的相对坐标加偏移量，得到anchor box具体的值
    all_anchors = (_anchors.reshape((1, A, cfg.TRAIN.COORDINATE_NUM)) +
                   shifts.reshape((1, K, cfg.TRAIN.COORDINATE_NUM)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, cfg.TRAIN.COORDINATE_NUM))
    total_anchors = int(K * A)
    # 为鲁棒性考虑，hard_neg[0]与hard_pos[0]都是用来存储总的anchor个数的
    assert len(hard_neg) > 0, "fatal error! in file {}".format(__file__)
    if hard_neg[0] > 0:
        assert total_anchors == hard_neg[0], "the pic generate diff num of anhors during two training stage"
        assert total_anchors == hard_pos[0], "the pic generate diff num of anhors during two training stage"

    # 仅保留那些还在图像内部的anchor
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]
    total_valid_anchors = len(inds_inside)  # 在图片里面的anchors的数目
    assert total_valid_anchors > 0, "The number of total valid anchor must be lager than zero"
    # hard_neg[0]在初始化的时候是 -1，后续用来保存有效anchor的个数，
    # 这里确保同一张图片在每轮训练中，所产生的有效anchor数是一样的

    anchors = all_anchors[inds_inside, :]  # 保留那些在图像内的anchor

    # 至此，anchor准备好了
    # ===============================================================================
    # label: 1 is positive, 0 is negative, -1 is dont care
    # (A)

    # 将有用的label 筛选出来
    labels = np.empty((total_valid_anchors,), dtype=np.int32)
    labels.fill(-1)  # 初始化label，均为-1

    # 计算anchor和gt-box的overlap，用来给anchor上标签
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))  # 假设anchors有x个，gt_boxes有y个，返回的是一个（x,y）的数组

    assert overlaps.shape[0] == total_valid_anchors, "Fatal Error: in the file {}".format(__file__)
    assert overlaps.shape[1] == gt_boxes.shape[0], "Fatal Error: in the file {}".format(__file__)
    # argmax_overlaps[0]表示第0号anchor与所有GT的IOU最大值的脚标
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(total_valid_anchors), argmax_overlaps]
    # # 返回一个一维数组，第i号元素的值表示第i个anchor与最可能的GT之间的IOU

    # =================================================================================================================
    tag1 = max_overlaps > 0
    tag2 = max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP
    negative = []
    for k in range(len(tag1)):
        if tag2[k] and tag1[k]:
            negative.append(k)
    # # 最大iou < 0.3 的设置为负例,这里要写明大于0，是因为对于计算iou有问题的anchor，其iou设置为-5了，他的标签为-1
    labels[negative] = 0
    # cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.8
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1  # overlap大于0.8的认为是前景

    # 最多的前景个数
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)  # 0.5*300

    fg_inds = np.where(labels == 1)[0]  # 返回一个数组, 正例的索引

    num_fg_inds = len(fg_inds)  # 正例的长度
    if num_fg_inds > num_fg:
        npr.shuffle(fg_inds)
        labels[fg_inds[num_fg:]] = -1
        num_fg_inds = num_fg
    elif num_fg_inds == 0:
        raise NoPositiveError("Tne number of positive anchor cannot be zero")

    # 若困难正例存在，则将对应的标签置为 1
    num_hard_pos = len(hard_pos) - 1
    if num_hard_pos > 0:
        labels[hard_pos[1:]] = 1
        num_fg_inds += num_hard_pos

    ratio = (1-cfg.TRAIN.RPN_FG_FRACTION)/cfg.TRAIN.RPN_FG_FRACTION
    num_bg = int(min(cfg.TRAIN.RPN_BATCHSIZE - num_fg, ratio*num_fg_inds))

    bg_inds = np.where(labels == 0)[0]

    num_bg_inds = len(bg_inds)  # 正例的长度
    if num_bg_inds > num_bg:
        npr.shuffle(bg_inds)
        labels[bg_inds[num_bg:]] = -1
    elif num_bg_inds == 0:
        raise NoPositiveError("Tne number of negative anchor cannot be zero")

    # 若困难正例存在，则将对应的标签置为 1
    num_hard_neg = len(hard_neg) - 1
    if num_hard_neg > 0:
        labels[hard_neg[1:]] = 0

    """返回值里面，只有正例的回归是有效值"""
    # 现在 每个有效的anchor都有了自己需要回归的真值
    bbox_targets = _compute_targets(anchors, labels, gt_boxes[argmax_overlaps, :])

    # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来， 加回来的的标签全部置为-1
    # labels是内部anchor的分类， total_anchors是总的anchor数目， inds_inside是内部anchor的索引
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)  # 这些anchor的label是-1，也即dontcare

    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)  # 这些anchor的真值是0，也即没有值

    # feature map上对应的每个像素点都有自己label, 文本为1 不是文本0 不关心区域为 -1
    labels = labels.reshape((1, height, width, A))  # reshape一下label
    rpn_labels = labels

    # bbox_targets
    # 对于label是1的标签，我们关心其回归的delta_y delta_h,label是0 和 -1的，我们不关心
    bbox_targets = bbox_targets.reshape((1, height, width, A * 2))  # reshape

    rpn_bbox_targets = bbox_targets

    # 返回的类型，依次是int32，float32， int32
    return rpn_labels, rpn_bbox_targets


def get_y(x1, y1, x2, y2, x, min_val=True):
    if x1 == x2:
        if min_val:
            return min(y1, y2)
        else:
            return max(y1, y2)
    return (y2 - y1) * (x - x1) / (x2 - x1) + y1


def _next_ind(ind):
    assert 0 <= ind <= 3, "ind must be a valid index!"
    if ind <= 2:
        return ind + 1
    return 0


def _last_ind(ind):
    assert 0 <= ind <= 3, "ind must be a valid index!"
    if ind >= 1:
        return ind - 1
    return 3


def _get_h_y(anchors, inds_positive, gt):
    """
    根据anchor的中心坐标，返回该中心坐标处，gt的高度,以及中心坐标
    :param anchors: N*4 数组，每行为一个anchor
    :param inds_positive: 一维数组， 指示正例的索引
    :param gt: N*4 数组，每行为一个GT
    :return: 一个一维数组，长度为N，返回正例的索引所指向的GT的高度
    """
    length = anchors.shape[0]
    gt_heights = np.empty(shape=(length,), dtype=np.float32)
    gt_y = np.empty(shape=(length,), dtype=np.float32)
    for i in inds_positive:
        ctr_x = (anchors[i, 0] + anchors[i, 2]) / 2
        X = np.array([gt[i, 0], gt[i, 2], gt[i, 4], gt[i, 6]])
        Y = np.array([gt[i, 1], gt[i, 3], gt[i, 5], gt[i, 7]])
        ints_sort = np.argsort(X, kind='mergesort')
        if X[ints_sort[0]] <= ctr_x <= X[ints_sort[1]]:
            cur_ind = ints_sort[0]
            last_ind = _last_ind(cur_ind)
            next_ind = _next_ind(cur_ind)
            x = X[cur_ind]
            y = Y[cur_ind]
            x_last = X[last_ind]
            y_last = Y[last_ind]
            x_next = X[next_ind]
            y_next = Y[next_ind]
            ymin = get_y(x, y, x_last, y_last, ctr_x, False)
            ymax = get_y(x, y, x_next, y_next, ctr_x, True)
            gt_heights[i] = abs(ymin - ymax) + 1
            gt_y[i] = (ymin + ymax) / 2

        elif X[ints_sort[2]] <= ctr_x <= X[ints_sort[3]]:
            cur_ind = ints_sort[3]
            last_ind = _last_ind(cur_ind)
            next_ind = _next_ind(cur_ind)
            x = X[cur_ind]
            y = Y[cur_ind]
            x_last = X[last_ind]
            y_last = Y[last_ind]
            x_next = X[next_ind]
            y_next = Y[next_ind]
            ymin = get_y(x, y, x_last, y_last, ctr_x, True)
            ymax = get_y(x, y, x_next, y_next, ctr_x, False)
            gt_heights[i] = abs(ymin - ymax) + 1
            gt_y[i] = (ymin + ymax) / 2

        else:
            ymin = get_y(gt[i, 0], gt[i, 1], gt[i, 2], gt[i, 3], ctr_x)
            ymax = get_y(gt[i, 4], gt[i, 5], gt[i, 6], gt[i, 7], ctr_x)
            gt_heights[i] = abs(ymax - ymin) + 1
            gt_y[i] = (ymin + ymax) / 2
    return gt_heights, gt_y


# data是内部anchor的分类标签， count是总的anchor数目， inds是内部anchor的索引
def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.int32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def bbox_transform(ex_rois, label, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes, anchors boxes
    :param label: 一维向量，是anchors的标签
    :param gt_rois: n * 8 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, y和高度的回归
    """

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1
    for mywidth in ex_widths:
        assert mywidth == 16

    length = ex_rois.shape[0]
    # 取出所有正例
    inds_positive = np.where(label == 1)[0]

    # 计算正例的高度
    ex_heights = np.empty(shape=(length,), dtype=np.float32)

    ex_heights[inds_positive] = ex_rois[inds_positive, 3] - ex_rois[inds_positive, 1] + 1.0

    # 计算正例的中心坐标
    ex_ctr_y = np.empty(shape=(length,), dtype=np.float32)

    ex_ctr_y[inds_positive] = ex_rois[inds_positive, 1] + 0.5 * ex_heights[inds_positive]
    # 根据anchor所在的横坐标，获取其对应GT的高度
    gt_heights, gt_ctr_y = _get_h_y(ex_rois, inds_positive, gt_rois)

    """
    对于ctpn文本检测，只需要回归y和高度坐标即可
    """
    targets_dy = np.empty(shape=(length,), dtype=np.float32)
    # 这里的传进来的为是正例和负例一起的anchors，将是正例的部分赋值
    targets_dy[inds_positive] = (gt_ctr_y[inds_positive] - ex_ctr_y[inds_positive]) / ex_heights[inds_positive]

    targets_dh = np.empty(shape=(length,), dtype=np.float32)
    targets_dh[inds_positive] = np.log(gt_heights[inds_positive] / ex_heights[inds_positive])

    # 对于每个正例来说需要回归两个delta
    # 返回的target为一个 正例个 * 2 的矩阵 两列分别为正例的delta_y delta_h
    targets = np.vstack(
        (targets_dy, targets_dh)).transpose()
    return targets


def _compute_targets(ex_rois, labels, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 8
    assert len(labels) == ex_rois.shape[0]

    # bbox_transform函数的输入是anchors， 和GT的坐标部分
    # 输出是一个N×2的矩阵，每行表示一个anchor与对应的IOU最大的GT的y,h回归,
    t = bbox_transform(ex_rois, labels, gt_rois)
    return t.astype(np.float32, copy=False)

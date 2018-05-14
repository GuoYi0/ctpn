#coding:utf-8
import tensorflow as tf
import numpy as np
import shutil
import os
import cv2
from lib.timer import Timer
from lib.text_connector.detectors import TextDetector
from exceptions import NoPositiveError
from shapely.geometry import Polygon


class TestClass(object):
    def __init__(self, cfg, network):
        self._cfg = cfg
        self._net = network

    # 画方框,被ctpn()调用
    def draw_boxes(self, img, image_name, boxes, color):
        """
        :param img: 最原始的图片矩阵
        :param image_name: 图片地址
        :param boxes: N×9的矩阵，表示N个拼接以后的完整的文本框。
        每一行，前八个元素一次是左上，右上，左下，右下的坐标，最后一个元素是文本框的分数
        :param color: 颜色
        :return:
        """
        # base_name = image_name.split('/')[-1]
        base_name = os.path.basename(image_name)
        b_name, ext = os.path.splitext(base_name)
        if self._cfg.TEST.CONNECT:
            with open(os.path.join(self._cfg.TEST.RESULT_DIR_TXT, '{}.txt'.format(b_name)), 'w') as f:
                for box in boxes:
                    # TODO 下面注释掉的这两行不知是啥意思
                    # if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    #     continue
                    # 默认用红色线条绘制，可能性最低
                    # color = (0, 0, 255)  # 颜色为BGR
                    cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=2)
                    cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, thickness=2)
                    cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, thickness=2)
                    cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, thickness=2)
                    # cv2.putText(img, 'score:{}'.format(box[8]), (int(box[0]), int(box[1])),
                    #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    x1 = box[0]
                    y1 = box[1]
                    x2 = box[2]
                    y2 = box[3]
                    x4 = box[4]
                    y4 = box[5]
                    x3 = box[6]
                    y3 = box[7]

                    line = ','.join([str(x1), str(y1), str(x2), str(y2), str(x3), str(y3), str(x4), str(y4)]) + '\n'
                    f.write(line)
        else:
            color = (255, 0, 255)  # 颜色为BGR
            for box in boxes:
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[1])), color, thickness=2)
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), color, thickness=2)
                cv2.line(img, (int(box[2]), int(box[3])), (int(box[2]), int(box[1])), color, thickness=2)
                cv2.line(img, (int(box[2]), int(box[3])), (int(box[0]), int(box[3])), color, thickness=2)
        # cv2.imwrite(os.path.join(self._cfg.TEST.RESULT_DIR_PIC, base_name), img)

    # 改变图片的尺寸，被ctpn()调用
    @ staticmethod
    def resize_im(im, scale, max_scale=None):
        # 缩放比定义为 修改后的图/原图
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

    def merge_y_anchor(self, boxes):
        """
        该函数的作用是在box层面，对竖直文本进行连接
        :param boxes: 一个N×9的矩阵，表示N个拼接以后的完整的文本框。
        每一行，前八个元素一次是左上，右上，左下，右下的坐标，最后一个元素是文本框的分数
        :return: 合并以后的anchor和分数
        """
        height = (abs(boxes[:, 5] - boxes[:, 1])+ abs(boxes[:, 7]  - boxes[:, 3])) / 2 + 1
        width = (abs(boxes[:, 2] - boxes[:, 0])+ abs(boxes[:, 6]  - boxes[:, 4])) / 2 + 1

        square_ratio = height / width
        square_inds = np.where(self._cfg.TEST.SQUARE_THRESH < square_ratio)[0]

        for ind, i in enumerate(square_inds[:-1]):
            if boxes[i, 8] > 0:
                # i号框框的x坐标，取为四个x的中心坐标
                x_i = (boxes[i, 0] + boxes[i, 2] + boxes[i, 4] + boxes[i, 6])/4
                # i号框框的上边y坐标和下边y坐标
                yi_up = (boxes[i, 1] + boxes[i, 3])/2
                yi_down = (boxes[i, 5] + boxes[i, 7])/2
                for j in square_inds[ind+1:]:
                    if boxes[j, 8] > 0 and abs((boxes[j, 0] + boxes[j, 2] + boxes[j, 4] + boxes[j, 6])/4 - x_i) \
                            < (width[i] + width[j])/2 and min(width[i], width[j])/max(width[i], width[j]) \
                            > self._cfg.TEST.MIN_SIZE_SIM:
                    # if boxes[j, 8] > 0:
                        # j号框框的上边y坐标和下边y坐标
                        yj_up = (boxes[j, 1] + boxes[j, 3]) / 2
                        yj_down = (boxes[j, 5] + boxes[j, 7]) / 2
                        # y方向的差值
                        y_gap = min(abs(yi_down - yj_up), abs(yj_down - yi_up))
                        if y_gap < (width[i]+width[j])*2:

                            boxes[i, 0] = boxes[i, 0]*0.7 + boxes[j, 0]*0.3
                            boxes[i, 1] = min(boxes[i, 1], boxes[j, 1])

                            boxes[i, 2] = boxes[i, 2]*0.7 + boxes[j, 2]*0.3
                            boxes[i, 3] = min(boxes[i, 3], boxes[j, 3])

                            boxes[i, 4] = boxes[i, 4]*0.7 + boxes[j, 4]*0.3
                            boxes[i, 5] = max(boxes[i, 5], boxes[j, 5])

                            boxes[i, 6] = boxes[i, 6]*0.7 + boxes[j, 6]*0.3
                            boxes[i, 7] = max(boxes[i, 7], boxes[j, 7])

                            boxes[i, 8] = boxes[i, 8]*0.7 + boxes[j, 8]*0.3
                            boxes[j, 8] = -1.0

                            # 更新i号框框的参数值
                            yi_up = (boxes[i, 1] + boxes[i, 3]) / 2
                            yi_down = (boxes[i, 5] + boxes[i, 7]) / 2
                            width[i] = width[i]*0.7 + width[j]*0.3
                            # height[i] = height[i]*0.7 + height[j]*0.3

        valid_inds = np.where(boxes[:, 8] > 0)[0]
        return boxes[valid_inds]

    def box_nms(self, boxes):
        length = boxes.shape[0]

        areas = np.empty(shape=(length,), dtype=np.float32)
        polygen = list()
        # 获取每个文本框的面积
        for i in range(length):
            x0, y0, x1, y1, x2, y2, x3, y3 = boxes[i, 0:8]

            coord = ((x0, y0), (x1, y1), (x3, y3), (x2, y2))
            quad = Polygon(coord)
            areas[i] = quad.area
            polygen.append(quad)

        for i in range(length):
            if boxes[i, 8] > 0:
                for j in range(i+1, length):
                    if boxes[j, 8] > 0 and polygen[i].intersects(polygen[j]):
                        na = polygen[i].intersection(polygen[j]).area
                        if areas[i] > areas[j]:
                            small_ind = j
                        else:
                            small_ind = i
                        if na/areas[small_ind] > self._cfg.TEST.TEXT_NMS:
                            boxes[small_ind, 8] = -1.0
                            if small_ind == i:
                                break
        valid_inds = np.where(boxes[:, 8] > 0)[0]

        return boxes[valid_inds, :]
        # return boxes

    # 被test_net()调用
    def ctpn(self, sess, net, image_name):
        """
        :param sess: 会话
        :param net: 创建的测试网络
        :param image_name: 所要测试的单张图片的目录
        :return:
        """
        # 读取图片
        image = cv2.imread(image_name)
        # 获取文本框
        # 将一张图片缩放成两个尺寸，其中大尺寸检查小的文本，小尺寸检查大的文本
        # boxes是缩放回原图的坐标
        boxes_big = self.get_boxes(sess, net, image, mode="big")
        # len_big = boxes_big.shape[0]
        boxes_small = self.get_boxes(sess, net, image, mode="small")
        boxes = np.concatenate((boxes_big, boxes_small), axis=0)

        boxes_nms = self.box_nms(boxes)  # 对文本框进行非极大值抑制

        if self._cfg.TEST.BIG_CONNECT:
            boxes_nms = self.merge_y_anchor(boxes_nms)
            boxes_nms = self.box_nms(boxes_nms)

        self.draw_boxes(image, image_name, boxes_nms, (0, 0, 255))
        #
        # boxes_big = boxes_nms[:len_big]
        # boxes_small = boxes_nms[len_big:]
        #
        # valid_inds = np.where(boxes_big[:, 8] > 0)[0]
        # boxes_big = boxes_big[valid_inds]
        #
        # valid_inds = np.where(boxes_small[:, 8] > 0)[0]
        # boxes_small = boxes_small[valid_inds]
        #
        # self.draw_boxes(image, image_name, boxes_big, (0, 0, 255))
        # self.draw_boxes(image, image_name, boxes_small, (0, 255, 0))

        base_name = os.path.basename(image_name)
        cv2.imwrite(os.path.join(self._cfg.TEST.RESULT_DIR_PIC, base_name), image)

        # 在原始图片上画图

    def get_boxes(self, sess, net, image, mode):
        assert mode in ["big", "small"], "model must be big or small"
        # resize_im，返回缩放后的图片和相应的缩放比。缩放比定义为 修改后的图/原图
        # 根据模式的不同，缩放成不同大小的图片
        if mode == "big":
            img, scale = TestClass.resize_im(image, scale=self._cfg.TEST.SCALE_BIG,
                                             max_scale=self._cfg.TEST.MAX_SCALE_BIG)
        else:
            img, scale = TestClass.resize_im(image, scale=self._cfg.TEST.SCALE_SMALL,
                                             max_scale=self._cfg.TEST.MAX_SCALE_SMALL)

        shape = img.shape[:2]  # 获取缩放后的高，宽
        # 将图片去均值化
        im_orig = img.astype(np.float32, copy=True)
        im_orig -= self._cfg.TRAIN.PIXEL_MEANS

        # 将缩放和去均值化以后的图片，放入网络进行前向计算，获取分数和对应的文本片段，
        # 该片段为映射回缩放以后的图片坐标，并且已经进行了非极大值抑制
        scores, boxes = TestClass.test_ctpn(sess, net, im_orig)
        # 获得anchor的高
        high = abs(boxes[:, 3]-boxes[:, 1]) + 1
        # 缩放回原图
        high = high/scale
        # print(high)
        # 根据检测模式筛选anchor
        if mode == "big":
            # model=='big'时，检测大文本，只保留大的anchor
            valid_ind = np.where(high >= self._cfg.TEST.TEXT_THRESH_BIG)[0]
        else:
            # 检查小文本模式，使用小anchor
            valid_ind = np.where(high <= self._cfg.TEST.TEXT_THRESH_SMALL)[0]

        # print(valid_ind)

        scores = scores[valid_ind]
        boxes = boxes[valid_ind, :]

        # valid_len=len(scores)
        # assert valid_len>0,"no valid anchor"


        if self._cfg.TEST.CONNECT:
            # 此处调用了一个文本检测器
            textdetector = TextDetector(self._cfg)
            """
            输入参数分别为：
            N×4矩阵，每行为一个已经映射回最初的图片的文字片段坐标
            N维向量，对应的分数
            两维向量，分别为最原始图片的高宽
            返回：
            一个N×9的矩阵，表示N个拼接以后的完整的文本框。
            每一行，前八个元素一次是左上，右上，左下，右下的坐标，最后一个元素是文本框的分数
            """
            # 返回缩放前的boxes
            boxes = textdetector.detect(boxes, scores, shape)
            boxes[:, 0:8] = boxes[:, 0:8] / scale
        else:
            boxes = boxes / scale
        return boxes

    def test_net(self, graph):

        timer = Timer()
        if os.path.exists(self._cfg.TEST.RESULT_DIR_TXT):
            shutil.rmtree(self._cfg.TEST.RESULT_DIR_TXT)
        os.makedirs(self._cfg.TEST.RESULT_DIR_TXT)

        if os.path.exists(self._cfg.TEST.RESULT_DIR_PIC):
            shutil.rmtree(self._cfg.TEST.RESULT_DIR_PIC)
        os.makedirs(self._cfg.TEST.RESULT_DIR_PIC)

        saver = tf.train.Saver()
        # 创建一个Session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 不能太大，否则报错

        sess = tf.Session(config=config, graph=graph)

        # 获取一个Saver()实例

        # 恢复模型参数
        ckpt = tf.train.get_checkpoint_state(self._cfg.COMMON.CKPT)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
            print('done')
        else:
            raise 'Check your pretrained {}'.format(self._cfg.COMMON.CKPT)

        im_names = os.listdir(self._cfg.TEST.DATA_DIR)

        assert len(im_names) > 0, "Nothing to test"
        i = 0
        timer.tic()
        for im in im_names:

            im_name = os.path.join(self._cfg.TEST.DATA_DIR, im)
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # print(('Testing for image {:s}'.format(im_name)))
            try:
                self.ctpn(sess, self._net, im_name)
            except NoPositiveError:
                print("Warning!!, get no region of interest in picture {}".format(im))
                continue
            except:
                print("the pic {} may has problems".format(im))
                continue
            # self.ctpn(sess, self._net, im_name)

            i += 1
            if i % 10 == 0:
                _diff_time = timer.toc(average=False)
                print('Detection took {:.3f}s for {} pic'.format(_diff_time, i))
                timer.tic()

        # 最后关闭session
        sess.close()

    @ staticmethod
    def test_ctpn(sess, net, im):
        # im_info = np.array([im.shape[0], im.shape[1]])
        im_info = im.shape[0:2]
        im = im[np.newaxis, :]
        feed_dict = {net.data: im, net.im_info: im_info, net.keep_prob: 1.0}
        fetches = [net.get_output('rois'), ]

        # (1 x H x W x A, 5) 第一列为正例的概率,已经排序了，后四列为映射回输入图片的，经过回归修正以后的，预测的文本片段坐标
        # 已经经过了非极大值抑制！！！！！！！！！！！！！
        rois = sess.run(fetches=fetches, feed_dict=feed_dict)
        if len(rois) == 0:
            raise NoPositiveError("Found no region of interest")
        # sess.run是以列表形式返回结果，所以这里需要[0]，以取出数组
        rois = rois[0]
        scores = rois[:, 0]
        # 这里是缩放后的坐标
        boxes = rois[:, 1:5]
        return scores, boxes
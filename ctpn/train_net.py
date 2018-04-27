import tensorflow as tf
import os
from lib import Timer
from input_layer import get_data_layer
from exceptions import NoPositiveError
from shapely import *

class SolverWrapper(object):
    def __init__(self, cfg, network, roidb, checkpoints_dir, max_iter, pretrain_model, restore):
        self._cfg = cfg
        self.net = network
        # 所有图片的imdb列表
        # self.roidb = roidb  # 所有图片的列表，每个元素是一个字典，字典里面包含列所有的box
        # self.output_dir = output_dir
        self.pretrained_model = pretrain_model
        self.checkpoints_dir = checkpoints_dir
        self._restore = restore
        self.max_iter = max_iter

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=10, write_version=tf.train.SaverDef.V2)
        self.data_layer = get_data_layer(roidb, self._cfg)

    def snapshot(self, sess, iter):

        filename = ('ctpn_iter_{:d}'.format(iter + 1) + '.ckpt')
        filename = os.path.join(self.checkpoints_dir, filename)

        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    def train_model(self, sess):
        # 根据全部的roidb，获得一个data_layer对象
        # data_layer对象是一批一批地传递处理好了的数据

        total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = self.net.build_loss()
        # 返回两个tensor数组，tensor数组的每行都是一个hard box
        hard_neg, hard_pos = self.net.get_hard()
        # cfg.TRAIN.LEARNING_RATE = 0.00001
        lr = tf.Variable(self._cfg.TRAIN.LEARNING_RATE, trainable=False)
        # TRAIN.SOLVER = 'Momentum'
        if self._cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(self._cfg.TRAIN.LEARNING_RATE)
        elif self._cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(self._cfg.TRAIN.LEARNING_RATE)
        else:
            # lr = tf.Variable(0.0, trainable=False)
            momentum = self._cfg.TRAIN.MOMENTUM  # 0.9
            opt = tf.train.MomentumOptimizer(lr, momentum)

        global_step = tf.Variable(0, trainable=False)
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()  # 获取所有的可训练参数
            # 下面这句话会产生UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape.
            # This may consume a large amount of memory
            grads, norm = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 10.0)

            train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = opt.minimize(total_loss, global_step=global_step)

        # initialize variables
        sess.run(tf.global_variables_initializer())
        restore_iter = 0

        # load vgg16
        if self.pretrained_model is not None and not self._restore:
            try:
                print(('Loading pretrained model '
                       'weights from {:s}').format(self.pretrained_model))

                # 从预训练模型中导入
                self.net.load(self.pretrained_model, sess, True)
            except:
                raise 'Check your pretrained model {:s}'.format(self.pretrained_model)

        # resuming a trainer
        if self._restore:  # restore为True表示训练过程中可能死机了， 现在重新启动训练
            try:
                print(self.checkpoints_dir)
                ckpt = tf.train.get_checkpoint_state(self.checkpoints_dir)
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_iter))
                print("The starting iter is {:d}".format(restore_iter))
                print('done')
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
        timer = Timer()

        loss_list = [total_loss, model_loss, rpn_cross_entropy, rpn_loss_box]
        train_list = [hard_neg, hard_pos, train_op]
        num_neg = 0
        num_pos = 0
        timer.tic()
        for iter in range(restore_iter, self.max_iter):
            
            # learning rate
            if iter != 0 and iter % self._cfg.TRAIN.STEPSIZE == 0:  # 每STEPSIZE轮，学习率变为原来的0.1
                sess.run(tf.assign(lr, lr.eval() * self._cfg.TRAIN.GAMMA))
                print("learning rate at step {} is {}".format(iter, lr))

            blobs = self.data_layer.forward()
            gt_boxes = blobs['gt_boxes']

            if not gt_boxes.shape[0] > 0:
                print("warning: abandon a picture named {}, because it has "
                      "no gt_boxes".format(blobs['im_name']))
                continue

            feed_dict = {
                self.net.data: blobs['data'],  # 一个形状为[批数，宽，高，通道数]的源图片，命名为“data”
                self.net.im_info: blobs['im_info'],  # 一个两维向量，包含高，宽，缩放比例
                self.net.keep_prob: 0.5,
                self.net.gt_boxes: gt_boxes,  # GT_boxes信息，N×8矩阵，每一行为一个gt_box
                self.net.hard_neg: blobs['hard_neg'],
                self.net.hard_pos: blobs['hard_pos']
            }
            try:
                hard_neg2, hard_pos2, _ = sess.run(fetches=train_list, feed_dict=feed_dict)
                # 把难以区分的正负例添加进去
                self.data_layer.put_hard(hard_pos=hard_pos2, hard_neg=hard_neg2)
                num_pos += len(hard_pos2) - 2
                num_neg += len(hard_neg2) - 2

            except NoPositiveError:
                print("warning: abandon a picture named {}".format(blobs['im_name']))
            # except:
            #     print("pic {} may has problem".format(blobs['im_name']))
                continue

            if iter % self._cfg.TRAIN.DISPLAY == 0:
                total_loss_val, model_loss_val, rpn_loss_cls_val, rpn_loss_box_val = \
                    sess.run(fetches=loss_list, feed_dict=feed_dict)
                print('iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, '
                      'rpn_loss_box: %.4f, lr: %f' % (iter, self.max_iter, total_loss_val, model_loss_val,
                                                      rpn_loss_cls_val, rpn_loss_box_val, lr.eval()))
                print("hard_pos: {},  hard_neg: {}".format(
                    num_pos/self._cfg.TRAIN.DISPLAY, num_neg/self._cfg.TRAIN.DISPLAY))
                num_neg = 0
                num_pos = 0
                _diff_time = timer.toc(average=False)
                print('speed: {:.3f}s for {} pictures'.format(_diff_time, self._cfg.TRAIN.DISPLAY))
                timer.tic()
            # 每1000次保存一次模型
            if (iter + 1) % self._cfg.TRAIN.SNAPSHOT_ITERS == 0:  # 每一千次保存一下ckeckpoints
                self.data_layer.write2cache()
                self.snapshot(sess, iter)
        # for循環結束以後，記錄下最後一次
        # self.snapshot(sess, self.max_iter - 1)


def train_net(cfg, network, roidb, checkpoints_dir,  max_iter, pretrain_model, restore):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True  # 程序按需申请内存

    with tf.Session(config=config) as sess:
        '''sw = solver wrapper'''
        sw = SolverWrapper(cfg, network, roidb, checkpoints_dir, max_iter, pretrain_model, restore)
        print('Solving...')

        sw.train_model(sess=sess)
        print('done solving')

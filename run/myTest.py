# import cv2
# img = cv2.imread("E:\\ctpn_yi\\dataset\\for_train\\Imageset\\T1.AK_XX8hXXbnu_Z1_042512.jpg.jpg")
# color = (255, 255, 255)
# cv2.line(img, (0, 400), (300, 400), color, thickness=11)
# cv2.imshow("yi", img)
# cv2.waitKey(0)

# import numpy as np
# a = np.array([[1, 2],[3, 4], [5, 6]])
# b = tuple(a)
# print(b)
#
# c = [(row[0], row[1]) for row in a]
# print(tuple(c))
# print("in file {}, x2 must be larger than x1 in anchor".format(__file__))

# import numpy as np
# a = np.logspace(start=0, stop=16, num=16, base=1.25, endpoint=False)*8
# print(a)
# import numpy as np
# a = np.array([[1,8],[2,6], [0,9]])
# print(a)
# b = a[np.argsort(a[:, 0]), :]
# print(b)
# import os
# print(os.path.splitext("rad.txt"))
# print(round(3.64))
# import tensorflow as tf
# graph = tf.Graph()
# with graph.as_default():
#     w = tf.Variable(dtype=tf.float32, initial_value=1.0)
#     ema = tf.train.ExponentialMovingAverage(0.9)
#     update = tf.assign_add(w, 1.0)
#
#     with tf.control_dependencies([update]):
#         ema_op = ema.apply([w])  # 返回一个op,这个op用来更新moving_average #这句和下面那句不能调换顺序
#
#     ema_val = ema.average(w)  # 此op用来返回当前的moving_average,这个参数不能是list
#
# with tf.Session(graph=graph) as sess:
#     sess.run(tf.initialize_all_variables())
#     for i in range(3):
#         print(i)
#         print('w_old=', sess.run(w))
#         print(sess.run(ema_op))
#         print('w_new=', sess.run(w))
#         print(sess.run(ema_val))
#         print('**************')



# import tensorflow as tf
# w = tf.Variable(dtype=tf.float32, initial_value=1.0, trainable=False)
# update = tf.assign_add(w, 1.0)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# sess.run(update)
# print(sess.run(w))
# sess.close()

for i in range(4, 8):
    print(i)

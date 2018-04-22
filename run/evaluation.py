import os
import numpy as np
from shapely.geometry import LineString, Polygon


def height_normalization():
    return


def get_box_coordinate(data):
    r = np.full((4, 2), 0.0, dtype='float32')
    data = data.strip().split(",")
    for j in range(4):
        r[j][0] = float(data[j * 2])
        r[j][1] = float(data[j * 2 + 1])
    return r


def get_iou(gt, dr):
    # 对于p是针对预测结果而言的，对每一个预测结果在gt中寻找是否命中
    p = np.zeros(len(dr))
    p_a = np.zeros(len(dr))
    w_p_a = np.empty(len(dr), dtype=float)

    for i, d1 in enumerate(dr):
        # 将预测结果数组信息进行一层float32格式的转换，并封装成多边形对象
        coor = get_box_coordinate(d1)
        dr_bbox = Polygon(coor)

        l1 = LineString([coor[0], coor[1]]).length
        l2 = LineString([coor[0], coor[3]]).length
        w_p_a[i] = l1 / l2 if l1 > l2 else l2 / l1


        # 对GT遍历
        for j, d2 in enumerate(gt):
            # 获得gt多边形对象
            gt_bbox = Polygon(get_box_coordinate(d2))
            # 判断是否重合
            if dr_bbox.intersects(gt_bbox):

                # 计算交集
                intersection = dr_bbox.intersection(gt_bbox).area
                # 计算并集
                union = dr_bbox.union(gt_bbox).area

                # 计算iou
                iou = intersection / union
                # 精度的iou阈值为0.5
                if iou > 0.5:
                    p[i] = 1
                    p_a[i] = w_p_a[i]

    # 对于是针对gt而言的，对每一个gt在dr中寻找是否命中
    r = np.zeros(len(gt))

    r_a = np.zeros(len(gt))
    w_r_a = np.empty(len(gt), dtype=float)
    for i, d1 in enumerate(gt):
        coor = get_box_coordinate(d1)
        gt_bbox = Polygon(coor)

        l1 = LineString([coor[0], coor[1]]).length
        l2 = LineString([coor[0], coor[3]]).length
        w_r_a[i] = l1 / l2 if l1 > l2 else l2 / l1

        for j, d2 in enumerate(dr):
            dr_bbox = Polygon(get_box_coordinate(d2))
            if gt_bbox.intersects(dr_bbox):
                intersection = gt_bbox.intersection(dr_bbox).area
                union = gt_bbox.union(dr_bbox).area
                iou = intersection / union

                # 召回的iou阈值为0.7
                if iou > 0.7:
                    r[i] = 1
                    r_a[i] = w_r_a[i]
    return p, r, p_a, r_a, w_p_a, w_r_a


if __name__ == '__main__':
    # GroundTruth文件目录
    gt_path = "E:\\alidata\\[update] ICPR_text_train_part1_20180316\\train_1000\\txt_1000"
    # 输出的预测结果DetectResult文件目录
    dr_path = "E:\\ctpn_yi\\data\\results\\gt_txt"
    # dr_path = "E:\\alidata\\[update] ICPR_text_train_part1_20180316\\train_1000\\txt_1000"

    # 所有图片总精度的分子
    total_p = 0
    # 所有图片总召回的分子
    total_r = 0
    # 所有图片总精度的分母
    total_p_n = 0
    # 所有图片总召回的分母
    total_r_n = 0

    # 阿里测试
    # 所有图片总精度的分子
    total_p_a = 0
    # 所有图片总召回的分子
    total_r_a = 0
    # 所有图片总精度的分母
    total_p_n_a = 0
    # 所有图片总召回的分母
    total_r_n_a = 0

    # 读取数据
    dirs = os.listdir(gt_path)
    for i in dirs:
        if os.path.splitext(i)[1] == ".txt":
            # 打开同名标注文件与预测结果文件，进行比对
            # gt_data = pd.read_csv(os.path.join(gt_path, i), header=None,
            #                       quoting=csv.QUOTE_NONE, encoding='utf8').iloc[:, :8].values
            # dr_data = pd.read_csv(os.path.join(dr_path, i), header=None,
            #                       quoting=csv.QUOTE_NONE, encoding='utf8').iloc[:, :8].values
            gt = open(os.path.join(gt_path, i), "r", encoding="utf_8")
            try:
                dr = open(os.path.join(dr_path, i), "r", encoding="utf_8")
            except:
                print("the pic {} doesn't exists".format(os.path.join(dr_path, i)))
                gt.close()
                continue
            gt_data = gt.readlines()
            dr_data = dr.readlines()

            # 通过计算iou的方式获得p，r
            # p，r即计算出的单张图片的精度与召回
            # p，r数组中元素均为二值化的0或1，表示是否命中
            p, r, p_a, r_a, w_p_a, w_r_a = get_iou(gt_data, dr_data)

            # 将单张的结果保存
            total_p += np.sum(p)
            total_p_n += len(p)
            total_r += np.sum(r)
            total_r_n += len(r)

            total_p_a += np.sum(p_a)
            total_p_n_a += np.sum(w_p_a)
            total_r_a += np.sum(r_a)
            total_r_n_a += np.sum(w_r_a)

            gt.close()
            dr.close()

    # 计算总精度
    precision = total_p / total_p_n
    # 计算总召回
    recall = total_r / total_r_n
    # 计算f_score
    f_score = 2 * precision * recall / (precision + recall)

    print("业界标准===================================================")
    print("precision = ", precision)
    print("recall = ", recall)
    print("f1_score = ", f_score)
    print("\n")
    # 计算总精度
    precision = total_p_a / total_p_n_a
    # 计算总召回
    recall = total_r_a / total_r_n_a
    # 计算f_score
    f_score = 2 * precision * recall / (precision + recall)

    print("阿里标准===================================================")
    print("precision = ", precision)
    print("recall = ", recall)
    print("f1_score = ", f_score)

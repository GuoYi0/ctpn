import os
import math
import shutil
import numpy as np


def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a, b):
        part_up += a1 * b1
        a_sq += a1 * a1
        b_sq += b1 * b1
    part_down = math.sqrt(a_sq * b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down


if __name__ == '__main__':
    # 原数据集的txt文件目录
    in_path = "E:/alidata/ICPR_text_train_part2_20180313/txt_9000"
    # 生成新txt文件目录
    out_path = "E:/alidata/ICPR_text_train_part2_20180313/txt_refresh2"
    # pic_path = "E:/alidata/ICPR_text_train_part2_20180313/image_9000"
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    dirs = os.listdir(in_path)
    for i in dirs:
        # try:
        file_name, ext = os.path.splitext(i)
        if ext == ".txt":
            in_file = open(os.path.join(in_path, i), "r", encoding="utf_8")
            data = in_file.readlines()
            in_file.close()
            with open(os.path.join(out_path, i), 'w', encoding="utf_8") as f:
                for index, d1 in enumerate(data):
                    gt = d1.strip().split(",")
                    if gt is None:
                        continue
                    r = np.full((4, 2), 0.0, dtype='float32')
                    for j in range(4):
                        r[j][0] = float(gt[j * 2])
                        r[j][1] = float(gt[j * 2 + 1])

                    xSorted = r[np.argsort(r[:, 0], kind="mergesort"), :]
                    leftMost = xSorted[:2, :]
                    rightMost = xSorted[2:, :]

                    leftMost = leftMost[np.argsort(leftMost[:, 1], kind="mergesort"), :]
                    (tl, bl) = leftMost

                    vector_0 = np.array(bl - tl)
                    vector_1 = np.array(rightMost[0] - tl)
                    vector_2 = np.array(rightMost[1] - tl)
                    cosd1 = cos_dist(vector_0, vector_1)
                    cosd2 = cos_dist(vector_0, vector_2)
                    if cosd1 is None or cosd2 is None:
                        continue
                    angle = [np.arccos(cosd1), np.arccos(cosd2)]
                    (br, tr) = rightMost[np.argsort(angle), :]

                    f.write(','.join(list(map(str, (tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]))))
                            + '\n')

    print("success")

import numpy as np


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """
    def __init__(self, cfg):
        self._cfg = cfg
        # 用于存放文本片段，[N，4]
        self.text_proposals = None
        # 用于存放分数 N维向量
        self.scores = None
        # 二维向量，存放图像高宽，
        self.im_size = None
        # N维向量，依次存放文本片段的高度
        self.heights = None
        # 这是一个列表，其长度与图片宽度相等。每个元素都是一个列表，第i号元素用来存放x1=i的所有文本片段的索引
        self.boxes_table = None

    def get_successions(self, index):
        """
        根据index找其右相邻的文本片段
        :param index:
        :return: 以列表形式返回右相邻的文本片段
        """
        # 取出第index号文本片段
        box = self.text_proposals[index]
        x0 = (box[0]+box[2])/2
        y0 = (box[1]+box[3])/2
        results = []
        height = abs(box[3]-box[1])+1
        gap = height + 16
        gap2 = gap**2
        # MAX_HORIZONTAL_GAP = 50, 水平距离不超过50个像素的文本片段有可能对应同一个文本
        for left in range(int(box[0])+1, min(int(box[0])+gap, self.im_size[1])):
            # 取出x1=left的所有文本片段
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                x1 = (self.text_proposals[adj_box_index][0] + self.text_proposals[adj_box_index][2]) / 2
                y1 = (self.text_proposals[adj_box_index][1] + self.text_proposals[adj_box_index][3]) / 2
                dist1 = (x0 - x1)**2 + abs(y0 - y1)**2
                # 若这两个文本片段属于同一文本，则添加进results
                if self.meet_v_iou(adj_box_index, index) and dist1 < gap2:
                    results.append(adj_box_index)
            if len(results) > 0:
                return results
        return results

    def get_precursors(self, index):
        """
        根据index找其左相邻的文本片段
        :param index:
        :return: 以列表形式返回左相邻的文本片段
        """
        box = self.text_proposals[index]
        x0 = (box[0]+box[2])/2
        y0 = (box[1]+box[3])/2
        results = []
        height = abs(box[3]-box[1])+1
        gap = height + 16
        gap2 = gap**2
        for left in range(int(box[0])-1, max(int(box[0]-gap), 0)-1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                x1 = (self.text_proposals[adj_box_index][0] + self.text_proposals[adj_box_index][2]) / 2
                y1 = (self.text_proposals[adj_box_index][1] + self.text_proposals[adj_box_index][3]) / 2
                dist1 = (x0 - x1)**2 + abs(y0 - y1)**2
                if self.meet_v_iou(adj_box_index, index) and dist1 < gap2:
                    results.append(adj_box_index)

            # if len(results) != 0:
            if len(results) > 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        # 以列表形式返回succession_index左边的所有最邻近
        precursors = self.get_precursors(succession_index)
        # 如果index在succession_index左边的最邻近列表中，表示他们是邻近对
        if index in precursors:
            return True
        # if self.scores[index] >= np.max(self.scores[precursors]):
        #     return True
        return False

    def meet_v_iou(self, ind1, ind2):
        """
        :param ind1:
        :param ind2:
        :return: 两个文本片段是否属于同一文本
        """
        h1 = self.heights[ind1]
        h2 = self.heights[ind2]
        y0 = max(self.text_proposals[ind2][1], self.text_proposals[ind1][1])
        y1 = min(self.text_proposals[ind2][3], self.text_proposals[ind1][3])
        # y方向的IOU
        y_iou = (y1-y0+1)/(h1+h2-(y1-y0))
        # 高度的相似度
        h_similarity = min(h1, h2)/max(h1, h2)
        # y方向的iou和高度相似度是否都大于阈值
        return y_iou >= self._cfg.TEST.MIN_V_OVERLAPS and h_similarity >= self._cfg.TEST.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        """
        该函数用于创建一个N阶方阵，其中N为文本片段的个数。方阵里面，True表示对应的两个文本片段是相邻的。
        :param text_proposals:
        :param scores:
        :param im_size:
        :return: 返回一个图
        """
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3]-text_proposals[:, 1]+1

        # 创建一个列表，其长度与图片宽度相等，每个元素为一个空列表
        boxes_table = [[] for _ in range(self.im_size[1])]

        for index, box in enumerate(text_proposals):
            # 把x1坐标相同的文本片段放在一起
            boxes_table[int(box[0])].append(index)

        self.boxes_table = boxes_table
        length = text_proposals.shape[0]

        # 创建一个行列均为文本片个数的方阵， 初始化全部是False
        graph = np.zeros((length, length), np.bool)

        # 对所有的文本片段进行遍历，取出与其右相邻的文本片段
        for index, box in enumerate(text_proposals):
            # index是文本片段编号；以列表形式返回index右边的最邻近
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            # 把众多相邻片段中，分数最高的一个作为其相邻片段
            # TODO 这句话或许需要修正，不仅要以分数优先，还要以y_iou优先
            # 这里改为了以分数和y_iou的和作为指标

            criteria = scores[successions]+self.y_iou(index, successions)*self._cfg.TEST.RATIO

            ind = int(np.argmax(criteria))
            succession_index = successions[ind]
            # succession_index在index的右边
            if self.is_succession_node(index, succession_index):
                graph[index, succession_index] = True

        # 下面的代码，用于确保每个anchor只有一个左相邻
        for k in range(length):
            left = np.where(graph[:, k])[0]
            if len(left) > 1:
                graph[left, k] = False
                criteria = scores[left] + self.y_iou(k, left)*self._cfg.TEST.RATIO
                ind = int(np.argmax(criteria))
                graph[ind, k] = True
        # 该图是一个方阵，(i, j)为True表示i对j右边相邻
        return graph

    def y_iou(self, index, succession):
        """
        :param index: 一个数
        :param succession: 一个列表
        :return: 数index与列表succession的y_iou
        """
        v_iou = np.empty(shape=(len(succession),), dtype=np.float64)
        h1 = self.heights[index]

        for i, k in enumerate(succession):

            h2 = self.heights[k]
            y0 = max(self.text_proposals[k][1], self.text_proposals[index][1])
            y1 = min(self.text_proposals[k][3], self.text_proposals[index][3])
            # y方向的IOU
            v_iou[i] = (y1 - y0 + 1) / (h1 + h2 - (y1 - y0))

        return v_iou


def graphs_connected(graph):
    sub_graphs = []
    length = graph.shape[0]
    for index in range(length):
        # 这个if语句只寻找文本片段的起点
        if not graph[:, index].any() and graph[index, :].any():
            v = index
            sub_graphs.append([v])
            while graph[v, :].any():
                # 取出第一个True所在的角标
                v = np.where(graph[v, :])[0][0]
                sub_graphs[-1].append(v)
    return sub_graphs




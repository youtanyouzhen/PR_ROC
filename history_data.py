import numpy as np
from confusion_matrix import ConfusionMatrix


class HistoryData:

    def __init__(self, x_true, x_false, start_thr=0, end_thr=1, num_thr=10):
        self.x_true = x_true
        self.x_false = x_false
        self.thresholds = np.linspace(start_thr, end_thr, num_thr)
        self._init_data()

    def _init_data(self):
        self.Ps = []
        self.Rs = []
        self.TPRs = []
        self.FPRs = []

    def compute_all_data(self):

        # 生成数据
        for thr in self.thresholds:
            cm = ConfusionMatrix(self.x_true, self.x_false, thr)
            self.Ps.append(cm.P)
            self.Rs.append(cm.R)
            self.TPRs.append(cm.TPR)
            self.FPRs.append(cm.FPR)


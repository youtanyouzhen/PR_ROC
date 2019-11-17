import numpy as np
from confusion_matrix import ConfusionMatrix


class HistoryData:

    def __init__(self, mu_true, sigma_true, mu_false, sigma_false, start_thr=0, end_thr=1, num_thr=10):
        self.mu_true = mu_true
        self.sigma_true = sigma_true
        self.mu_false = mu_false
        self.sigma_false = sigma_false
        self.thresholds = np.linspace(start_thr, end_thr, num_thr)
        self._init_data()

    def _init_data(self):
        self.Ps = []
        self.Rs = []
        self.TPRs = []
        self.FPRs = []
        self.F1s = []

    def compute_all_data(self):

        # 生成数据
        for thr in self.thresholds:
            cm = ConfusionMatrix(self.mu_true, self.sigma_true, self.mu_false, self.sigma_false, thr)
            self.Ps.append(cm.P)
            self.Rs.append(cm.R)
            self.TPRs.append(cm.TPR)
            self.FPRs.append(cm.FPR)
            self.F1s.append(cm.F1)


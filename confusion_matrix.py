

class ConfusionMatrix:

    def __init__(self, x_true, x_false, threshold):
        self.x_true = x_true
        self.x_false = x_false
        self.threshold = threshold

    @property
    def TP(self):
        return self.x_true[self.x_true >= self.threshold].size

    @property
    def TN(self):
        return self.x_true[self.x_true < self.threshold].size

    @property
    def FP(self):
        return self.x_false[self.x_false < self.threshold].size

    @property
    def FN(self):
        return self.x_false[self.x_false >= self.threshold].size

    @property
    def P(self):
        if self.TP == 0:
            return 0
        else:
            return self.TP / (self.TP + self.FP)

    @property
    def R(self):
        if self.TP == 0:
            return 0
        else:
            return self.TP / (self.TP + self.FN)

    @property
    def F1(self):
        m = 2 * self.P * self.R
        if m == 0:
            return 0
        else:
            return m / (self.P + self.R)

    @property
    def TPR(self):
        return self.R

    @property
    def FPR(self):
        if self.FP == 0:
            return 0
        else:
            return self.FP / (self.TN + self.FP)
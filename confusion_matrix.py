from scipy.stats import norm


class ConfusionMatrix:

    def __init__(self, mu_true, sigma_true, mu_false, sigma_false, threshold, scale_true=0.2, scale_false=1.0):
        self.mu_true = mu_true
        self.sigma_true = sigma_true
        self.mu_false = mu_false
        self.sigma_false = sigma_false
        self.threshold = threshold
        self.scale_true = scale_true
        self.scale_false = scale_false

        self.norm_true = norm(self.mu_true, self.sigma_true)
        self.norm_false = norm(self.mu_false, self.sigma_false)

    @property
    def TP(self):
        return (1 - self.norm_true.cdf(self.threshold)) * self.scale_true

    @property
    def TN(self):
        return (self.norm_false.cdf(self.threshold)) * self.scale_false

    @property
    def FP(self):
        return (1 - self.norm_false.cdf(self.threshold)) * self.scale_false

    @property
    def FN(self):
        return (self.norm_true.cdf(self.threshold)) * self.scale_true

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
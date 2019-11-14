import seaborn as sns
import pandas as pd
from history_data import HistoryData
from utilities import get_norm_data, plot_decision
import matplotlib.pyplot as plt


sns.set(color_codes=True)


#
mu_true, sigma_true = 0.65, 0.1
mu_false, sigma_false = 0.45, 0.1
pic_num = 10


# 生成数据
x_true, y_true = get_norm_data(mu=mu_true, sigma=sigma_true)
x_false, y_false = get_norm_data(mu=mu_false, sigma=sigma_false)

# 生成历史数据
data = HistoryData(mu_true, sigma_true, mu_false, sigma_false)   # todo: 用面积计算HistoryData
data.compute_all_data()


for i, threshold in enumerate(data.thresholds):
    R, P, FPR, TPR = data.Rs[i], data.Ps[i], data.FPRs[i], data.TPRs[i]

    fig, axes = plt.subplots(3, 1)

    # 做正态分布
    plot_decision(x_true, y_true, x_false, y_false, threshold, axes[0])

    # 做P-R曲线
    line = pd.Series(data.Ps, data.Rs) # y, x
    ax_pr = sns.lineplot(data=line, label='P-R curve', color='g', ax=axes[1])
    ax_pr.set_xlabel('recall')
    ax_pr.set_ylabel('precision')
    ax_pr.plot(R, P, 'ko') # x, y

    # 做ROC曲线
    line = pd.Series(data.TPRs, data.FPRs)
    ax_roc = sns.lineplot(data=line, label='ROC', color='g', ax=axes[2])
    ax_roc.set_xlabel('FPR')
    ax_roc.set_ylabel('TPR')
    ax_roc.plot(FPR, TPR, 'ko')

    plt.show()
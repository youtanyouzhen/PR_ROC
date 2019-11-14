import seaborn as sns
import pandas as pd
from history_data import HistoryData
from utilities import get_norm_data, plot_decision
import matplotlib.pyplot as plt


sns.set(color_codes=True)

 # todo: 用kdeplot作图
# 生成数据
x_true, y_true = get_norm_data(mu=0.55)
x_false, y_false = get_norm_data(mu=0.45)

# 生成历史数据
data = HistoryData(x_true, x_false)
data.compute_all_data()

pic_num = 10
for i in range(0, pic_num):
    threshold = i/pic_num

    fig, axes = plt.subplots(3,1)

    # 做正态分布
    plot_decision(x_true, y_true, x_false, y_false, threshold, axes[0])

    # 做P-R曲线
    line = pd.Series(data.Ps, data.Rs)
    ax = sns.lineplot(data=line, label='P-R curve', color='g', ax=axes[1])

    plt.show()
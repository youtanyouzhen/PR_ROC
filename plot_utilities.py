import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio

from history_data import HistoryData

sns.set(color_codes=True)

def get_norm_data(mu=0.25, sigma=0.1, size=100):
    x = np.linspace(0, 1, size)
    y = stats.norm.pdf(x, mu, sigma)
    return x, y



def plot_norm(ax, x, y, scale=1.0, threshold=0.45, label='diseased', color='b'):
    y = scale * y
    data = pd.Series(y, x)
    ax = sns.lineplot(data=data, label=label, color=color, ax=ax)
    ax.fill_between(x[x < threshold], y[x < threshold], alpha=0.1, color=color)
    ax.fill_between(x[x > threshold], y[x > threshold], alpha=0.5, color=color)
    ax.get_yaxis().set_visible(False)
    return max(y)

def plot_decision(x_true, y_true, x_false, y_false, threshold, ax):
    max_y1 = plot_norm(ax, x_true, y_true, threshold=threshold, label='diseased', color="coral", scale=0.2)  # 正例
    max_y2 = plot_norm(ax, x_false, y_false, threshold=threshold, label='healthy', color='b')  #
    ax.plot([threshold, threshold], [0, max([max_y1, max_y2]) * 1.1], 'k', label='threshold')


def plot_PR(Ps, Rs, i, ax):
    line = pd.Series(Ps, Rs)  # y, x
    ax_pr = sns.lineplot(data=line, label='P-R curve', color='g', ax=ax)
    ax_pr.set_xlabel('R')
    ax_pr.set_ylabel('P')
    ax_pr.plot(Rs[i], Ps[i], 'ko')  # x, y
    ax_pr.set_ylim(0, 1.1)

def plot_roc(TPRs, FPRs, i, ax):
    line = pd.Series(TPRs, FPRs)
    ax_roc = sns.lineplot(data=line, label='ROC', color='g', ax=ax)
    ax_roc.set_xlabel('FPR')
    ax_roc.set_ylabel('TPR')
    ax_roc.plot(FPRs[i], TPRs[i], 'ko')
    ax_roc.set_ylim(0, 1.1)

def plot_f1(thrs, F1, i, ax):
    line = pd.Series(thrs, F1) # y, x
    ax_roc = sns.lineplot(data=line, label='F1', color='g', ax=ax)
    ax_roc.set_xlabel('threshold')
    ax_roc.set_ylabel('F1')
    ax_roc.plot(F1[i], thrs[i], 'ko')


def save_norm(mu_true=0.65, sigma_true=0.1, mu_false=0.45, sigma_false=0.1, thr=0.5):
    fig, ax = plt.subplots(1, 1)
    # 生成数据
    x_true, y_true = get_norm_data(mu=mu_true, sigma=sigma_true)
    x_false, y_false = get_norm_data(mu=mu_false, sigma=sigma_false)
    # 作正态分布
    plot_decision(x_true, y_true, x_false, y_false, thr, ax)

    fig.savefig(f"images/norm{thr}.jpg")


def plot_pr_roc_gif(mu_true=0.65, sigma_true=0.1, mu_false=0.45, sigma_false=0.1, pic_num=10):

    # 生成数据
    x_true, y_true = get_norm_data(mu=mu_true, sigma=sigma_true)
    x_false, y_false = get_norm_data(mu=mu_false, sigma=sigma_false)

    # 生成历史数据
    data = HistoryData(mu_true, sigma_true, mu_false, sigma_false, num_thr=pic_num)
    data.compute_all_data()

    images = []
    for i, threshold in enumerate(data.thresholds):
        fig, axes = plt.subplots(3, 1)

        # 作正态分布
        plot_decision(x_true, y_true, x_false, y_false, threshold, axes[0])

        # 作P-R曲线
        plot_PR(data.Ps, data.Rs, i, axes[1])

        # 作ROC曲线
        plot_roc(data.TPRs, data.FPRs, i, axes[2])

        # plt.show()
        fig.tight_layout()
        filename_dir = f"images/pic_{i}.jpg"
        fig.savefig(filename_dir)
        images.append(imageio.imread(filename_dir))

    imageio.mimsave(f"images/norm.gif", images, duration=0.5)


def plot_f1_gif(mu_true=0.65, sigma_true=0.1, mu_false=0.45, sigma_false=0.1, pic_num=10):

    # 生成数据
    x_true, y_true = get_norm_data(mu=mu_true, sigma=sigma_true)
    x_false, y_false = get_norm_data(mu=mu_false, sigma=sigma_false)

    # 生成历史数据
    data = HistoryData(mu_true, sigma_true, mu_false, sigma_false, num_thr=pic_num)
    data.compute_all_data()

    images = []
    for i, threshold in enumerate(data.thresholds):
        fig, axes = plt.subplots(2, 1)

        # 作正态分布
        plot_decision(x_true, y_true, x_false, y_false, threshold, axes[0])

        # 作F1
        plot_f1(data.F1s, data.thresholds, i,  axes[1])

        # plt.show()
        fig.tight_layout()
        filename_dir = f"images/pic_F1_{i}.jpg"
        fig.savefig(filename_dir)
        images.append(imageio.imread(filename_dir))

    imageio.mimsave(f"images/F1.gif", images, duration=0.5)
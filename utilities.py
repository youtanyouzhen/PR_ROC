import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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



import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def dict_combos(d: dict):
    """Calculates all combinations given several list

    Args:
        d: dictionary to process, where each key points to a list

    Returns:
        list of tuples consisting of all possible combinations
    
    """
    return [dict(zip(d.keys(), vals)) for vals in itertools.product(*d.values())]
        
def plot_metric_vs_time(df, metric, dataset_name, fig, ax,
                    cax=None,
                    cax_orientation="vertical",
                    cax_ticks=['Earliest', 'Latest'],
                     # figsize=(8, 5.5), 
                     title="",
                      title_font=20, label_font=15, cm_name="YlGnBu", num_seed=5, xlabel=None, ylabel=None,
                      legend=False, lw=2, 
                      # dpi=200, 
                      ylim=None, 
                      # save_path=None, 
                      org_x=None, cus_x=None):
    """Visualizes metric of interest versus time.
    
    Args:
        df: dataframe of results from Eot
        metric: performance metric that is plotted (e.g. "auc", "accuracy")
        dataset_name: name of the dataset
        figsize: size of the figure
        title: title of the plot
        title_font: font of the title
        label_font: font of the labels of X and Y axis
        cm_name: name of the colormap
        xlabel: content of X-axis label
        ylabel: content of Y-axis label
        legend: boolean variable indicates to display legend or not
        lw: line width used in figure
        dpi: dpi of the figure
        ylim: range to plot for Y-axis
        save_path: path to save the plot
        org_x: original X-axis tick
        cus_x: customized X-axis tick
    
    """
    range_len = len(df["train_end"].unique())
    cm_list = cm.get_cmap(cm_name, range_len)
    
    for (train_end, df_tmp), color in zip(df.groupby("train_end"), 
                                          cm_list(np.linspace(0, 1, range_len))[::-1]):
        
        mean_list = []
        std_list = []
        test_end_list = []
        for _, df_temp in df_tmp.groupby("staleness"):
            mean_list.append(df_temp[metric].mean())
            std_list.append(df_temp[metric].std())
            test_end_list.append(df_temp["test_end"].unique().item())
        std_list = std_list / np.sqrt(num_seed)
        # ax.plot(test_end_list, mean_auc, label=train_end, lw=lw, color=color)
        # ax.fill_between(test_end_list, [i - 1.96 * j for i, j in zip(mean_auc, std_auc)],
        #                     [i + 1.96 * j for i, j in zip(mean_auc, std_auc)], alpha=0.1, color=color)
        ax.errorbar(test_end_list, mean_list, yerr=std_list, label=train_end, lw=1, color=color)
        ax.plot(test_end_list[0], mean_list[0], marker='o', markersize=5, markeredgecolor=color, markerfacecolor=color)
        
    if title:
        ax.set_title(title, size=title_font)
    else:
        ax.set_title(f"{metric.upper()} vs. Time\n{dataset_name}", size=title_font)
    
    if xlabel:
        ax.set_xlabel(xlabel, size=label_font)
    else:
        ax.set_xlabel("Test year", size=label_font)
    if ylabel:
        ax.set_ylabel(ylabel, size=label_font)
    else:
        ax.set_ylabel(metric.upper(), size=label_font)
    ax.grid(alpha=0.3)

    if cax is not None:
        fig.colorbar(cm.ScalarMappable(cmap=cm_name), cax=cax, orientation=cax_orientation, ticks=[0,1])
        cax.set_xticklabels(cax_ticks)
    
    if ylim:
        ax.set_ylim(ylim)
        ax.set_ylim(ylim)
        
    if org_x:
        ax.set_xticks(org_x)
        ax.set_xticklabels(cus_x)

    if legend:
        ax.legend()
        
    # if save_path:
        # fig.savefig(save_path)

    # return ax
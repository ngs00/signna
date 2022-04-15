import numpy
import matplotlib.pyplot as plt


def plot_pred_result(fig_name, targets, preds, font_size=16):
    min_val = 2.0
    max_val = 3.5

    plt.tight_layout()
    plt.rcParams.update({'font.size': font_size})
    plt.grid(linestyle='--')
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.xticks([2.0, 2.5, 3.0, 3.5])
    plt.yticks([2.0, 2.5, 3.0, 3.5])
    plt.plot([min_val, max_val], [min_val, max_val], 'k', zorder=2)
    plt.scatter(targets, preds, edgecolor='k', zorder=3)
    plt.savefig(fig_name, bbox_inches='tight', dpi=500)
    plt.close()


def plot_error_dist(fig_name, errors, y_min=None, y_max=None, font_size=16):
    if y_min is not None and y_max is not None:
        plt.ylim([y_min, y_max])

    plt.rcParams.update({'font.size': font_size})
    plt.gcf().set_size_inches(8, 4)
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.bar(numpy.arange(len(errors)), errors, align='edge', zorder=3, edgecolor='k', width=numpy.full(len(errors), 1))
    plt.savefig(fig_name, bbox_inches='tight', dpi=500)
    plt.close()

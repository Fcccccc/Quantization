import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib import pyplot

fmt = "%.2f%%"
yticks = mtick.FormatStrFormatter(fmt)


n_groups = 4

origin = np.array((1, 1, 1, 1)) * 100

huffman_9 = np.array((0.32, 0.29, 0.33, 0.30)) * 100

final_9 = np.array((0.33 * 0.75, 0.29 * 0.56, 0.33 * 0.96, 0.30 * 0.63)) * 100

huffman_8 = np.array((0.44, 0.32, 0.42, 0.40)) * 100

final_8 = np.array((0.44 * 0.64, 0.32 * 0.88, 0.42 * 0.96, 0.40 * 0.97)) * 100

# fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.4
error_config = {'ecolor': '0.3'}

def plot(pos, x, y1, y2, y3, title):
    ax = plt.subplot(pos)
    

    rects1 = ax.bar(index, y1, bar_width,
                    alpha=opacity, color='b',
                    label='origin')

    rects2 = ax.bar(index + bar_width, y2, bar_width,
                    alpha=opacity, color='r',
                    label='huffman')
    rects3 = ax.bar(index + 2 * bar_width, y3, bar_width,
                    alpha=opacity, color='g',
                    label='final')

    ax.set_xlabel('Model Name')
    ax.set_ylabel('Compression Ratio')
    ax.yaxis.set_major_formatter(yticks)
    ax.set_title(title)
    ax.set_xticks(index + bar_width )
    ax.set_xticklabels(('Alexnet', 'Vgg16', 'Resnet18', 'Resnet34'))
    for x, y1, y2, y3 in zip(x, y1, y2, y3):
        ax.text(x, y1 + 4, "%.2f" % y1, ha = "center", va = "top")
        ax.text(x + bar_width, y2 + 4, "%.2f" % y2, ha = "center", va = "top")
        ax.text(x + 2 * bar_width, y3 + 4, "%.2f" % y3, ha = "center", va = "top")
    ax.legend()

plot(121, index, origin, huffman_8, final_8, "80% Sparsity")
plot(122, index, origin, huffman_9, final_9, "90% Sparsity")


# fig.tight_layout()
plt.show()




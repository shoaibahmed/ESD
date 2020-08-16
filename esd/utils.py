import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_log(stats_dict, output_file=None, title=None):
    """
    Plots the detailed logs returned from the ESD library.
    :param stats_dict: dictionary containing the detailed stats returned by the library.
    :param output_file: (optional) string specifiying the output file to be used for writing the figure. If not
                        defined, displays the figure on the screen.
    :param title: (optional) title to be used for the plot.
    """
    assert stats_dict is not None
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)

    marker_list = ['o', '*', 'X', 'P', 'D']
    line_styles = ['dashed', 'dashdot', 'dotted']  # 'solid'
    marker_colors = sns.color_palette('husl', n_colors=1)

    num_example_list = sorted(list(stats_dict.keys()))
    accuracy_list = [stats_dict[num_examples] for num_examples in num_example_list]
    assert all([0.0 <= acc <= 1.0 for acc in accuracy_list])

    idx = 0
    line = plt.plot(num_example_list, accuracy_list, linewidth=2., marker=marker_list[idx % len(marker_list)],
                    color=marker_colors[idx], alpha=0.75, markeredgecolor='k')
    line[0].set_color(marker_colors[idx])
    line[0].set_linestyle(line_styles[idx % len(line_styles)])

    plt.xlabel('Number of examples')
    plt.ylabel('Accuracy')
    if title is not None:
        plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()
    plt.close('all')

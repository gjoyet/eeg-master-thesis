import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('macOSX')
plt.close("all")


def plot_accuracies(data: np.ndarray = None, title: str = "", savefile: str = None,
                    downsample_factor: int = 5, washout: int = 0) -> None:
    """
    Plots the mean accuracy over time with confidence band over subjects.
    :param data: 2D numpy array, where each row is the decoding accuracy for one subject over all timesteps.
    :param title: title of the plot.
    :param savefile: file name to save the plot under. If None, no plot is saved.
    :param downsample_factor:
    :param washout:
    :return: None
    """

    df = pd.DataFrame(data=data.T)
    df = df.reset_index().rename(columns={'index': 'Time'})
    df = df.melt(id_vars=['Time'], value_name='Mean_Accuracy', var_name='Subject')

    sns.set_context("paper", font_scale=1.75)

    # Create a seaborn lineplot, passing the matrix directly to seaborn
    plt.figure(figsize=(10, 5))  # Optional: Set the figure size

    sns.set_palette(sns.color_palette("deep"))

    # Create the lineplot, seaborn will automatically calculate confidence intervals
    sns.lineplot(data=df, x=(df['Time'] + washout) * downsample_factor - 1000, y='Mean_Accuracy',
                 errorbar='ci', label='Accuracy')  # BUT confidence band gets much larger with 'sd'
    # Also, it is important to note that MVPA computes CIs over subjects, while the
    # neural nets compute CIs over trials.Higher n makes for narrower CIs, i.e. neural
    # nets will have much narrower CIs without this implying higher certainty.
    sns.despine()

    plt.axhline(y=0.5, color='orange', linestyle='dashdot', linewidth=1.5, label='Random Chance')
    plt.axvline(x=0, ymin=0, ymax=0.05, color='black', linewidth=1.5, label='Stimulus Onset')

    # Set plot labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    if savefile is not None:
        plt.savefig('results/{}.png'.format(savefile))

    # Show the plot
    # plt.show()
    plt.close()


if __name__ == '__main__':
    for fn in ['mvpa_accuracy_200Hz_av-1_aug-1_C-1000_win-1_KEEP_FOR_COMPARISON',
               'mvpa_accuracy_200Hz_av-1_aug-1_C-1000_win-1_HC',
               'mvpa_accuracy_200Hz_av-1_aug-1_C-1000_win-1_SCZ']:
        data = np.load(f'results/data/{fn}.npy')
        plot_accuracies(data=data, savefile=fn)

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('macOSX')
plt.close('all')

data_root = 'results/data/'


def plot_loss(prefix, suffix):
    out_file_name = 'results/{}_loss{}.png'.format(prefix, suffix[:-4])

    if os.path.isdir(out_file_name) and not overwrite:
        return

    train_loss = np.load(os.path.join(data_root, prefix, f'{prefix}_train_loss{suffix}'))
    validation_loss = np.load(os.path.join(data_root, prefix, f'{prefix}_val_loss{suffix}'))

    num_epochs = len(train_loss)
    # num_epochs = 10 if prefix == 'lstm' else 15

    sns.set_context("paper", font_scale=1.75)
    plt.figure(figsize=(7, 5))

    sns.set_palette(sns.color_palette("deep")[9::-9])

    sns.lineplot(y=train_loss, x=range(1, num_epochs + 1), label='Training Loss')
    sns.lineplot(y=validation_loss, x=range(1, num_epochs + 1), label='Validation Loss')

    sns.despine()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title("Loss over epochs")

    plt.legend()
    plt.tight_layout()

    plt.savefig(out_file_name)
    plt.close()


def plot_accuracies(prefix, suffix, title: str = "", savefile: str = None,
                    downsample_factor: int = 5, washout: int = 200) -> None:

    for t in ['acc', 'sco']:
        out_file_name = 'results/{}_{}{}.png'.format(prefix, t, suffix[:-4])

        if os.path.isfile(out_file_name) and not overwrite:
            continue

        train_data = np.load(os.path.join(data_root, prefix, f'{prefix}_train_{t}{suffix}'))
        val_data = np.load(os.path.join(data_root, prefix, f'{prefix}_val_{t}{suffix}'))

        dfs = []
        for data in [train_data, val_data]:
            df = pd.DataFrame(data=data.T)
            df = df.reset_index().rename(columns={'index': 'Time'})
            df = df.melt(id_vars=['Time'], value_name='Mean_Accuracy', var_name='Subject')
            dfs.append(df)

        sns.set_context("paper", font_scale=1.5)
        ylabel = 'Accuracy' if t == 'acc' else 'Score'

        # Create a seaborn lineplot, passing the matrix directly to seaborn
        plt.figure(figsize=(10, 5))  # Optional: Set the figure size

        sns.set_palette(sns.color_palette("deep"))

        # Create the lineplot, seaborn will automatically calculate confidence intervals
        # sns.lineplot(data=dfs[0], x=(dfs[0]['Time'] + washout) * downsample_factor - 1000, y='Mean_Accuracy',
        #              errorbar='ci', label='Training')
        sns.lineplot(data=dfs[1], x=(dfs[1]['Time'] + washout) * downsample_factor - 1000, y='Mean_Accuracy',
                     errorbar='ci', label=ylabel)
        sns.despine()

        plt.axhline(y=0.5, color='orange', linestyle='dashdot', linewidth=1, label='Random Chance')
        plt.axvline(x=0, ymin=0, ymax=0.05, color='black', linewidth=1, label='Stimulus Onset')

        # Set plot labels and title
        plt.xlabel('Time (ms)')
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(title)
        plt.tight_layout()

        plt.savefig(out_file_name)
        plt.close()


if __name__ == '__main__':
    overwrite = False

    prefix = 'lstm'
    for suffix in ['_8ep_1lyr_64hdim.npy',
                   '_25ep_1lyr_64hdim.npy',
                   '_25ep_1lyr_256hdim.npy',
                   '_25ep_3lyr_64hdim.npy',
                   '_25ep_3lyr_256hdim.npy',
                   '_15ep_3lyr_64hdim_0.0001wd.npy',
                   '_15ep_3lyr_64hdim_0.001wd.npy',
                   '_15ep_3lyr_64hdim_0.01wd.npy',
                   '_15ep_3lyr_64hdim_0.1wd.npy']:
        plot_accuracies(prefix=prefix, suffix=suffix, washout=200)

        plot_loss(prefix=prefix, suffix=suffix)

    prefix = 'chrononet'
    for suffix in ['_1ep_0wd_0idp_0gdp_0.0001lr.npy',
                   '_15ep_0.0001wd_0.5idp_0.5gdp_0.0001lr.npy',
                   '_15ep_0.0001wd_0.25idp_0.25gdp_0.0001lr.npy',
                   '_15ep_0.0001wd_0idp_0gdp_0.0001lr.npy',
                   '_25ep_0wd_0idp_0gdp_0.0001lr.npy',
                   '_5ep_0.1wd_0.5idp_0.5gdp.npy',
                   '_5ep_0.1wd_0.25idp_0.25gdp.npy',
                   '_15ep_0.1wd_0.5idp_0.5gdp.npy',
                   '_15ep_0.01wd_0.5idp_0.5gdp.npy',
                   '_15ep_0.1wd_0.25idp_0.25gdp.npy',
                   '_15ep_0.01wd_0.25idp_0.25gdp.npy',
                   '_15ep_0.01wd_0idp_0gdp.npy',
                   '_15ep_0.001wd_0idp_0gdp.npy']:

        plot_accuracies(prefix=prefix, suffix=suffix, washout=180)

        plot_loss(prefix=prefix, suffix=suffix)

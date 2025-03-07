import os.path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import seaborn as sns

import matplotlib

matplotlib.use('macOSX')
import matplotlib.pyplot as plt

data_path = '/Volumes/Guillaume EEG Project/Berlin_Data/EEG/raw'

'''
All methods at the moment ignore confidence level. They could be adapted to weight results by confidence. 
'''


def calculate_per_subject_metrics() -> Dict[str, Dict[str, list]]:
    cols = ['correct', 'response_too_early', 'choice_rt', 'contrast_2']

    lapse_rate = {'hc': [], 'scz': []}
    mean_accuracy = {'hc': [], 'scz': []}
    reaction_time = {'hc': [], 'scz': []}
    early_reaction_rate = {'hc': [], 'scz': []}
    contrast_threshold = {'hc': [], 'scz': []}

    n = {'hc': 0, 'scz': 0}

    pctgs = []
    for folder, subject, df in file_to_df_iterator(data_path, cols):
        n[subject] += 1

        # mean accuracy
        mean_accuracy[subject].append(np.nanmean(df['correct']))

        # lapse rate
        lapse_rate[subject].append(((df['correct'].isna()) & (df['response_too_early'].isna())).sum() / len(df))

        # reaction time (only checking stuff, delete later)
        val = 1.0
        percentage_below = (df['choice_rt'].dropna() < val).mean() * 100
        pctgs.append(percentage_below)
        print(f"Subject {int(folder):3d}: {percentage_below:.2f}% of the values are below {val:.2f}.")
        # print('Reaction time 95th quantile: {}'.format(df['choice_rt'].quantile(q=0.95)))

        # reaction time
        reaction_time[subject].append(np.nanmean(df['choice_rt']))

        # early choice rate
        early_reaction_rate[subject].append(df['response_too_early'].abs().sum() / len(df['response_too_early']))

        # contrast threshold
        df.loc[:, 'contrast_2_abs'] = df['contrast_2'].abs()
        df = df.dropna(subset=['choice_rt'])
        df_sorted = df.sort_values(by='contrast_2_abs')
        idx = len(df_sorted)
        for i in range(len(df_sorted)):
            mean_value = df_sorted['correct'].iloc[i:].mean()  # Calculate mean from index i to the end
            if mean_value >= 0.75:
                idx = df_sorted.index[i]  # Store the index if condition is met
                break
        # TODO: maybe rather take the mean of all contrast values starting at idx instead of just the value at idx?
        contrast_threshold[subject].append(df_sorted['contrast_2_abs'].loc[idx])
        pass

    print(f"\nOn average, {np.mean(pctgs)} are below {val:.2f}\n")  # delete later

    print('Total subjects: {}, whereof {} schizophrenia patients and {} healthy controls.'.format(n['scz'] + n['hc'],
                                                                                                  n['scz'], n['hc']))

    return {'Mean Accuracy': mean_accuracy, 'Reaction Time': reaction_time, 'Lapse Rate': lapse_rate,
            'Early Choice Rate': early_reaction_rate, 'Contrast Threshold': contrast_threshold}


def psychophysical_kernel_auroc() -> pd.DataFrame:
    ppk_df = pd.DataFrame(columns=['subject_id', 'subject_type', 'stimulus_category', 'sample_n', 'auroc'])

    cols = ['response', 'side'] + ['contrast_left_{}'.format(i) for i in range(1, 11)] + ['contrast_right_{}'.format(i)
                                                                                          for i in range(1, 11)]

    for folder, subject, df in file_to_df_iterator(data_path, cols):
        df = df.dropna(subset=['response'])

        for sample in range(1, 11):
            for stimulus_category in [-1, 1]:
                df_oi = df[df['side'] == stimulus_category]
                df_oi = df_oi[
                    ['response', 'side', 'contrast_left_{}'.format(sample), 'contrast_right_{}'.format(sample)]]
                df_oi['contrast_difference'] = df_oi['contrast_right_{}'.format(sample)] - df_oi[
                    'contrast_left_{}'.format(sample)]
                auroc = roc_auc_score(df_oi['response'], df_oi['contrast_difference'])
                ppk_df.loc[len(ppk_df)] = [int(folder), subject, stimulus_category, sample, auroc]

    return ppk_df


def psychophysical_kernel_glm() -> pd.DataFrame:
    ppk_df = pd.DataFrame(columns=['subject_id', 'subject_type', 'stimulus_category', 'sample_n', 'weight'])

    cols = ['response', 'side'] + ['contrast_left_{}'.format(i) for i in range(1, 11)] + ['contrast_right_{}'.format(i)
                                                                                          for i in range(1, 11)]

    for folder, subject, df in file_to_df_iterator(data_path, cols):
        df = df.dropna(subset=['response'])

        for sample in range(1, 11):
            df['cd_{}'.format(sample)] = df['contrast_right_{}'.format(sample)] - df['contrast_left_{}'.format(sample)]

        for stimulus_category in [-1, 1]:
            df_oi = df[df['side'] == stimulus_category]

            X = df_oi[['cd_{}'.format(i) for i in range(1, 11)]]
            y = df_oi['response']
            # TODO: try regularization?
            clf = LogisticRegression(random_state=0).fit(X, y)

            for sample in range(1, 11):
                ppk_df.loc[len(ppk_df)] = [int(folder), subject, stimulus_category, sample, clf.coef_[0][sample - 1]]

    return ppk_df


def calculate_metric_correlation():
    """
    Calculates correlation between early choice and contrast difference over samples PPK style.
    :return:
    """

    corr_df = pd.DataFrame(columns=['subject_type', 'sample_n', 'weight'])

    cols = ['response_too_early'] + ['contrast_left_{}'.format(i) for i in range(1, 11)] + \
           ['contrast_right_{}'.format(i) for i in range(1, 11)]

    df = load_all_data(data_path, cols)

    df['early'] = df['response_too_early'].apply(lambda k: 0 if np.isnan(k) else 1)

    for sample in range(1, 11):
        df['cd_{}'.format(sample)] = np.abs(df['contrast_right_{}'.format(sample)] -
                                            df['contrast_left_{}'.format(sample)])

    for subject in ['hc', 'scz']:
        df_oi = df[df['subject_type'] == subject]

        X = df_oi[['cd_{}'.format(i) for i in range(1, 11)]]
        y = df_oi['early']
        # TODO: try regularization?
        clf = LogisticRegression(random_state=0).fit(X, y)

        for sample in range(1, 11):
            corr_df.loc[len(corr_df)] = [subject, sample, clf.coef_[0][sample - 1]]

    return corr_df


def file_to_df_iterator(path: str, cols: List[str]) -> Tuple[str, str, pd.DataFrame]:
    directory_content = os.listdir(path)

    for folder in directory_content:

        if folder == '9':
            # TODO: ask what to do for this subject, where one stimulus category has only one response in the whole data.
            continue

        if folder.isdigit():
            subdirectory_content = os.listdir(os.path.join(data_path, folder))

            # TODO: correct condition for subject label
            subject = 'scz' if int(folder) < 100 else 'hc'

            dfs = []

            # TODO: correct criteria for .csv selection
            for filename in filter(lambda k: ('results.csv' in k and 'assr' not in k and 'wrong' not in k),
                                   subdirectory_content):
                data = pd.read_csv(os.path.join(data_path, folder, filename), usecols=cols)
                dfs.append(data)

            combined_df = pd.concat(dfs, ignore_index=True)

            yield folder, subject, combined_df


def load_all_data(path: str, cols: List[str]) -> pd.DataFrame:
    directory_content = os.listdir(path)

    dfs = []

    for folder in directory_content:
        if folder.isdigit():
            subdirectory_content = os.listdir(os.path.join(data_path, folder))

            # TODO: correct condition for subject label
            subject = 'scz' if int(folder) < 100 else 'hc'

            # TODO: correct criteria for .csv selection
            for filename in filter(lambda k: ('results.csv' in k and 'assr' not in k and 'wrong' not in k),
                                   subdirectory_content):
                data = pd.read_csv(os.path.join(data_path, folder, filename), usecols=cols)
                data['subject_type'] = subject
                dfs.append(data)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def plot_metrics():
    # TODO: if change return of calculate_metrics() – adapt here!
    metrics = calculate_per_subject_metrics()
    for variable_name, metric in metrics.items():
        df = pd.DataFrame({
            variable_name: metric["hc"] + metric["scz"],  # Combine both lists
            "Participant Type": ["HC"] * len(metric["hc"]) + ["SCZ"] * len(metric["scz"])
        })

        plt.figure(figsize=(4, 6))
        sns.violinplot(df, y=variable_name, hue='Participant Type', split=True, gap=.1, inner='quart')
        sns.despine()
        plt.tight_layout()
        plt.savefig('results/{}.png'.format(variable_name.lower().replace(' ', '_')))


def plot_ppk(ppk_df: pd.DataFrame, method: str):
    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot the lines for each subject_type
    sns.lineplot(data=ppk_df, x='sample_n', y=method, hue='subject_type')

    # Add title and labels
    if method == 'auroc':
        plt.title("Mean AUROC over subjects")
        plt.hlines(y=0.5, xmin=1, xmax=10, color='black', linestyle='--', linewidth=1, alpha=0.5, label='random')
        plt.ylim((0.45, 0.75))
    if method == 'weight':
        plt.title("Mean weight of logistic regression model over subjects")

    # plt.ylabel("Mean")
    # plt.xlabel("Sample Number")
    plt.legend()

    # Display the plot
    plt.savefig('results/ppk_{}.png'.format(method))
    # plt.show()


def plot_corr(corr_df: pd.DataFrame):
    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot the lines for each subject_type
    sns.lineplot(data=corr_df, x='sample_n', y='weight', hue='subject_type')

    plt.title("Weights of logistic regression model predicting early choice from contrast difference")

    # plt.ylabel("Mean")
    # plt.xlabel("Sample Number")
    plt.legend()

    # Display the plot
    plt.savefig('results/ec_cd_corr.png')
    # plt.show()


if __name__ == '__main__':
    plot_metrics()

    # ppk = psychophysical_kernel_auroc()
    #
    # plot_ppk(ppk, method='auroc')
    #
    # ppk = psychophysical_kernel_glm()
    #
    # plot_ppk(ppk, method='weight')
    #
    # corr = calculate_metric_correlation()
    #
    # plot_corr(corr)

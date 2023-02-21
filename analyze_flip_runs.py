import pandas as pd
import numpy as np
import wandb
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter


def import_wandb_runs(project):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("luth/flips-study-imagenet")

    summary_list, state_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        name_list.append(run.name)
        state_list.append(run.state)

    runs_df = pd.DataFrame.from_records(summary_list)
    runs_df['name'] = name_list
    runs_df['state'] = state_list
    return runs_df


def plot_wordcloud(words, title=None):
    words = [str(w) for w in words]
    #plt.figure(figsize=(15, 10))
    wordcloud = WordCloud(width=4000, height=2000, colormap='ocean').generate(' '.join(words))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


def get_stats(df, name, tot):
    print(f'Summary for {name}')
    print(f'Total number of runs: {len(df)} ({round(100*len(df)/tot,2)}%)')
    print(f'Summary statistics:')
    print(df[['neg_flips_rel', 'pos_flips_rel', 'modeltop1_a', 'modeltop1_b']].describe())
    models = list(df['modelname_a'])+list(df['modelname_b'])
    print(f'Total number of unique models in {name}: {len(np.unique(models))}')
    print(f'Model counter: {Counter(models)}')
    plot_wordcloud(models)


def infer_model_names(df):
    try:
        run_names = df['Name'].values
    except KeyError:
        run_names = df['name'].values
    df['modelname_a'] = [name.split('>')[0] for name in run_names]
    df['modelname_b'] = [name.split('>')[1] for name in run_names]

    return df

def main():
    runs = import_wandb_runs("flips-study-imagenet")

    # print summary statistics
    print(f'Stats for all runs')
    runs.describe()

    # get failed runs
    completed_runs = runs.dropna(subset=['neg_flips_abs'])
    get_stats(completed_runs, 'Completed Runs', len(runs))

    error_runs = runs.loc[runs['neg_flips_abs'].isna()]
    error_runs = infer_model_names(error_runs)
    get_stats(error_runs, 'Error Runs', len(runs))

    crashed_runs = runs.loc[runs['state'] == 'failed']
    crashed_runs = infer_model_names(crashed_runs)
    get_stats(crashed_runs, 'Crashed Runs', len(error_runs))

    test


if __name__=="__main__":
    main()
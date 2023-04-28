import wandb
import json
import pandas as pd
import numpy as np
from mycolorpy import colorlist as mcp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib

from aggregate_results import load_wandb_runs

plt.rc('font', family='Times New Roman', size=14)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

arch_strings = {'transformer': 'Transformer', 'cnn': 'CNN', 'mlp': 'MLP'}
appr_strings = {'KL_Dist': 'KL Distillation', 'XEKL_Dist': 'XE-KL Distillation',
                'CD_Dist': 'CD Distillation', 'CRD_Dist': 'CRD Distillation',
                'XEKL+MCL_Dist': 'XE-KL+MCL Distillation', 'KL+MT_Dist': 'KL+MT Distillation', 'KL+MT_u_Dist': 'KL+MT-U Distillation'}
teachers = [211, 268, 234, 302, 209, 10, 152, 80, 36, 182, 310, 77, 12, 239, 151, 145, 232, 101, 291, 124]
all_students = {'transformer': [41, 7, 5, 46, 26, 171],
                'cnn': [33, 131, 235, 132, 42, 130, 48],
                'mlp': [214, 2, 9, 77, 258, 160, 72]}


def mscatter(x,y,z, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x,y,z,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def prediction_flips_plot():
    data = load_wandb_runs('1-1_class_prediction_flips_study')
    data = data.dropna(subset=['ent_pos_class_flips', 'ent_pos_class_flips'])
    data = data.loc[data['tag'] != 'Random Baseline']
    pairs = [f"{data['teacher_name'][i]}-{data['student_name'][i]}" for i in data.index]
    models = np.concatenate((data['teacher_name'].values, data['student_name'].values))
    print(f'Number of evaluated pairs: {len(np.unique(pairs))}')
    print(f'Number of evaluated models: {len(np.unique(models))}')

    plt.rcParams['font.size'] = 15
    labels_size = 18
    fig = plt.figure(figsize=(8, 5))

    #plt.subplot(121)
    plt.axvline(0, color='black', alpha=0.3, linestyle='--', label='Equal Performance')
    #plt.axhline(np.min(data['pos_rel'].values), color='red', alpha=0.9, linestyle='dotted', lw=3, label='Min. Complementarity')
    plt.plot(-23.8, np.min(data['pos_rel'].values), marker='_', color='red', alpha=0.9, label='Min. Positive Flips')
    plt.scatter(data['ts_diff'], data['pos_rel'], color='C0', alpha=0.5)
    plt.xlabel('Performance Difference', fontsize=labels_size)
    plt.ylabel('Share of Positive Prediction Flips', fontsize=labels_size)
    plt.xlim((-24, 24))
    plt.ylim((0, 26))
    plt.minorticks_on()
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left')

    """
    plt.subplot(122)
    plt.axvline(0, color='black', alpha=0.3, linestyle='--', label='Equal Performance')
    plt.axhline(np.min(data['neg_rel'].values), color='red', alpha=0.9, linestyle='dotted', lw=3, label='Min. Neg. Flips')
    plt.scatter(data['ts_diff'], data['neg_rel'], color='C1', alpha=0.5)
    plt.xlabel('Performance Difference', fontsize=labels_size)
    plt.ylabel('Share of Negative Prediction Flips', fontsize=labels_size)
    plt.xlim((-24, 24))
    plt.ylim((0, 26))
    plt.minorticks_on()
    plt.grid(alpha=0.3)
    """

    fig.tight_layout()
    plt.savefig(f'images/prediction_flips_plot.pdf', bbox_inches='tight')
    plt.show()


def prediction_flips_entropy_plot():
    data = load_wandb_runs('1-1_class_prediction_flips_study')
    data = data.dropna(subset=['ent_pos_class_flips', 'ent_pos_class_flips'])
    pairs = [f"{data['teacher_name'][i]}-{data['student_name'][i]}" for i in data.index]
    models = np.concatenate((data['teacher_name'].values, data['student_name'].values))
    print(f'Number of evaluated pairs: {len(np.unique(pairs))}')
    print(f'Number of evaluated models: {len(np.unique(models))}')

    plt.rcParams['font.size'] = 15
    labels_size = 18
    fig = plt.figure(figsize=(8, 5))

    #plt.subplot(121)
    #plt.axvline(0, color='black', alpha=0.3, linestyle='--', label='Equal Performance')
    plt.axhline(6.9048, color='red', alpha=0.9, linestyle='dotted', lw=3, label='Uniform Distribution')
    plt.scatter(data['pos_rel'], data['ent_pos_class_flips'], color='C0', alpha=0.5)
    plt.xlabel('Positive Prediction Flips [%]', fontsize=labels_size)
    plt.ylabel('Entropy', fontsize=labels_size)
    plt.title('Entropy of Positive Class Prediction Flips', fontsize=labels_size+2)
    plt.xlim((0, 25))
    plt.ylim((5.8, 6.95))
    plt.minorticks_on()
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right')

    """
    plt.subplot(122)
    plt.axvline(0, color='red', alpha=0.3, linestyle='--')
    plt.axhline(6.9048, color='black', alpha=0.9, linestyle='dotted')
    plt.scatter(data['ts_diff'], data['ent_neg_class_flips'], color='C1', alpha=0.5)
    plt.xlabel('Performance Difference', fontsize=labels_size)
    plt.ylabel('Entropy of Neg. Class Prediction Flips', fontsize=labels_size)
    plt.xlim((-24, 24))
    plt.ylim((5.6, 6.95))
    plt.minorticks_on()
    plt.grid(alpha=0.3)
    """

    fig.tight_layout()
    plt.savefig(f'images/prediction_flips_entropy_plot.pdf', bbox_inches='tight')
    plt.show()


def prediction_flips_sim_plot():
    data = load_wandb_runs('1-1_class_prediction_flips_study')
    data = data.dropna(subset=['ent_pos_class_flips', 'ent_pos_class_flips'])
    pairs = [f"{data['teacher_name'][i]}-{data['student_name'][i]}" for i in data.index]
    models = np.concatenate((data['teacher_name'].values, data['student_name'].values))
    print(f'Number of evaluated pairs: {len(np.unique(pairs))}')
    print(f'Number of evaluated models: {len(np.unique(models))}')

    plt.rcParams['font.size'] = 15
    labels_size = 18
    fig = plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.axvline(0, color='black', alpha=0.3, linestyle='--', label='Equal Performance')
    plt.axhline(0.61, color='red', alpha=0.9, linestyle='dotted', lw=3, label='Average Sim.')
    plt.scatter(data['ts_diff'], data['top2%_avg_sim'], color='C0', alpha=0.5)
    plt.xlabel('Performance Difference', fontsize=labels_size)
    plt.ylabel('Average Class Similarity', fontsize=labels_size)
    plt.title('Class Similarity for the Top-2% Pos. Prediction Flips', fontsize=labels_size+2)
    plt.xlim((-24, 24))
    plt.ylim((0.49, 0.83))
    plt.minorticks_on()
    plt.grid(alpha=0.3)

    plt.subplot(122)
    plt.axvline(0, color='black', alpha=0.3, linestyle='--', label='Equal Performance')
    plt.axhline(0.61, color='red', alpha=0.9, linestyle='dotted', lw=3, label='Average Sim.')
    plt.scatter(data['ts_diff'], data['top50%_avg_sim'], color='C0', alpha=0.5)
    plt.xlabel('Performance Difference', fontsize=labels_size)
    plt.ylabel('Average Class Similarity', fontsize=labels_size)
    plt.title('Class Similarity for the Top-50% Pos. Prediction Flips', fontsize=labels_size+2)
    plt.xlim((-24, 24))
    plt.ylim((0.49, 0.83))
    plt.minorticks_on()
    plt.grid(alpha=0.3)
    plt.legend()

    fig.tight_layout()
    #plt.savefig(f'images/prediction_flips_sim_plot.pdf', bbox_inches='tight')
    plt.show()


def prediction_flips_sim_plot_v2():
    data = load_wandb_runs('1-1_class_prediction_flips_study')
    data = data.dropna(subset=['ent_pos_class_flips', 'ent_pos_class_flips'])
    data = data.loc[data['tag'] != 'Random Baseline']
    pairs = [f"{data['teacher_name'][i]}-{data['student_name'][i]}" for i in data.index]
    models = np.concatenate((data['teacher_name'].values, data['student_name'].values))
    print(f'Number of evaluated pairs: {len(np.unique(pairs))}')
    print(f'Number of evaluated models: {len(np.unique(models))}')

    plt.rcParams['font.size'] = 15
    labels_size = 18
    fig = plt.figure(figsize=(8, 5))
    top_p = [2, 5, 20, 50]
    mean_sims = np.array([np.mean(data[f'top{p}%_avg_sim'].values) for p in top_p])
    lower_sims = np.array([np.quantile(data[f'top{p}%_avg_sim'].values, 0.25) for p in top_p])
    upper_sims = np.array([np.quantile(data[f'top{p}%_avg_sim'].values, 0.75) for p in top_p])

    #plt.axhline(0.0, color='red', alpha=0.9, linestyle='dotted', lw=3, label='Average Sim.')
    plt.plot(top_p, (mean_sims-0.61)/0.61, color='C0', alpha=1, marker='o')
    plt.fill_between(top_p, (lower_sims-0.61)/0.61, (upper_sims-0.61)/0.61, alpha=0.3)
    plt.xlabel('Top-X% of the Total Positive Flips', fontsize=labels_size)
    plt.ylabel('Difference to Average Class Sim. [%]', fontsize=labels_size)
    plt.title('Similarity of the Classes Containing \n the Most Positive Prediction Flips', fontsize=labels_size+2)
    plt.ylim((0.04, 0.18))
    #plt.minorticks_on()
    plt.grid(alpha=0.3)
    plt.xticks(top_p, ['2%', '5%', '20%', '50%'])
    #plt.legend()

    fig.tight_layout()
    plt.savefig(f'images/prediction_flips_sim_plot.pdf', bbox_inches='tight')
    plt.show()


def class_pred_flips_histogramms():
    examples = ['xcit_small_24_p16_224>dla46_c', 'resmlp_24_224>regnetx_080', 'pit_ti_224>vit_base_patch8_224']
    seeds = [124, 189, 165]
    subplots = [131, 132, 133]
    titles = ['Strong Teacher - Weak Student', 'Equal Teacher - Student Performances', 'Weak Teacher - Strong Student']

    fig = plt.figure(figsize=(16, 4))
    for s, seed in enumerate(seeds):
        with open(f'prediction_flips/{seed}_prediction_flips.json') as f:
            data = json.load(f)
        plt.subplot(subplots[s])
        hist = np.array(data['results']['pos_class_flips'])/50*100
        hist = -1 * np.sort(-1*hist)
        plt.bar(range(1000), hist, align='edge', width=1, alpha=0.6)
        plt.plot(range(1000), hist)
        if s == 0:
            plt.ylabel('Positive Flips per Class \n [in % of the total class samples]', fontsize=16)
        plt.xlabel('ImageNet Classes', fontsize=16)
        plt.xticks([])
        plt.title(f'{titles[s]} (Diff: {round(data["config"]["ts_diff"])})', fontsize=18)

    fig.tight_layout()
    plt.savefig(f'images/pos_flip_histograms.pdf', bbox_inches='tight')
    plt.show()

    test


if __name__ == "__main__":
    prediction_flips_plot()
    #prediction_flips_entropy_plot()
    #prediction_flips_sim_plot_v2()
    #class_pred_flips_histogramms()




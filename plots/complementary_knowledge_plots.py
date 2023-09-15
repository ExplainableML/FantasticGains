import json
import textwrap
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from aggregate_results import load_wandb_runs

plt.rc('font', family='Times New Roman', size=14)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def prediction_flips_plot():
    """Plot the complementary knowledge of the student model compared to the teacher model.

    :Returns: None
    """
    data = load_wandb_runs('1-1_class_prediction_flips_study')
    data = data.dropna(subset=['ent_pos_class_flips', 'ent_pos_class_flips'])
    data = data.loc[data['tag'] != 'Random Baseline']
    pairs = [f"{data['teacher_name'][i]}-{data['student_name'][i]}" for i in data.index]
    models = np.concatenate((data['teacher_name'].values, data['student_name'].values))
    print(f'Number of evaluated pairs: {len(np.unique(pairs))}')
    print(f'Number of evaluated models: {len(np.unique(models))}')

    plt.rcParams['font.size'] = 19
    labels_size = 20
    fig = plt.figure(figsize=(8, 5))

    #plt.subplot(121)
    plt.axvline(0, color='black', alpha=0.3, linestyle='--', label='Equal Performance')
    #plt.axhline(np.min(data['pos_rel'].values), color='red', alpha=0.9, linestyle='dotted', lw=3, label='Min. Complementarity')
    plt.plot(-23.8, np.min(data['pos_rel'].values), marker='_', color='red', alpha=0.9, label='Min. Compl. Knowledge', markersize=20)
    plt.scatter(data['ts_diff'], data['pos_rel'], color='C0', alpha=0.5)
    plt.xlabel('Performance Difference', fontsize=labels_size)
    plt.ylabel('Complementary Knowledge [%]', fontsize=labels_size)
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
    """Plot the entropy of the positive class prediction flips of the student model compared to the teacher model.

    :Returns: None
    """
    data = load_wandb_runs('1-1_class_prediction_flips_study')
    data = data.dropna(subset=['ent_pos_class_flips', 'ent_pos_class_flips'])
    pairs = [f"{data['teacher_name'][i]}-{data['student_name'][i]}" for i in data.index]
    models = np.concatenate((data['teacher_name'].values, data['student_name'].values))
    print(f'Number of evaluated pairs: {len(np.unique(pairs))}')
    print(f'Number of evaluated models: {len(np.unique(models))}')

    plt.rcParams['font.size'] = 19
    labels_size = 20
    fig = plt.figure(figsize=(8, 5))

    #plt.subplot(121)
    #plt.axvline(0, color='black', alpha=0.3, linestyle='--', label='Equal Performance')
    plt.axhline(6.9048, color='red', alpha=0.9, linestyle='dotted', lw=3, label='Uniform Distribution')
    plt.scatter(data['pos_rel'], data['ent_pos_class_flips'], color='C0', alpha=0.5)
    plt.xlabel('Complementary Knowledge [%]', fontsize=labels_size)
    plt.ylabel('Entropy', fontsize=labels_size)
    #plt.title('Entropy of Positive Class Prediction Flips', fontsize=labels_size+2)
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
    """Plot the similarity of the classes with the largest shares of complementary knowledge.

    :Returns: None
    """
    data = load_wandb_runs('1-1_class_prediction_flips_study')
    data = data.dropna(subset=['ent_pos_class_flips', 'ent_pos_class_flips'])
    data = data.loc[data['tag'] != 'Random Baseline']
    pairs = [f"{data['teacher_name'][i]}-{data['student_name'][i]}" for i in data.index]
    models = np.concatenate((data['teacher_name'].values, data['student_name'].values))
    print(f'Number of evaluated pairs: {len(np.unique(pairs))}')
    print(f'Number of evaluated models: {len(np.unique(models))}')

    plt.rcParams['font.size'] = 19
    labels_size = 20
    fig = plt.figure(figsize=(7, 4))
    top_p = [2, 5, 20, 50]
    mean_sims = np.array([np.mean(data[f'top{p}%_avg_sim'].values) for p in top_p])
    lower_sims = np.array([np.quantile(data[f'top{p}%_avg_sim'].values, 0.25) for p in top_p])
    upper_sims = np.array([np.quantile(data[f'top{p}%_avg_sim'].values, 0.75) for p in top_p])

    #plt.axhline(0.0, color='red', alpha=0.9, linestyle='dotted', lw=3, label='Average Sim.')
    plt.plot(['Top-2%', 'Top-5%', 'Top-20%', 'Top-50%'], (mean_sims-0.61)/0.61, color='C0', alpha=1, marker='o')
    plt.fill_between(['Top-2%', 'Top-5%', 'Top-20%', 'Top-50%'], (lower_sims-0.61)/0.61, (upper_sims-0.61)/0.61, alpha=0.3)
    plt.xlabel('Classes with Top-X% of the Compl. Knowledge', fontsize=labels_size)
    plt.ylabel('Difference to Avg. Sim. [%]', fontsize=labels_size)
    #plt.title('Similarity of the Classes Containing \n the Most Positive Prediction Flips', fontsize=labels_size+2)
    plt.ylim((0.04, 0.20))
    #plt.minorticks_on()
    #plt.xscale('log')
    plt.minorticks_off()
    plt.grid(alpha=0.3)
    #plt.xticks(top_p, ['Top-2%', 'Top-5%', 'Top-20%', 'Top-50%'])
    plt.yticks([0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18], ['+6%', '+8%', '+10%', '+12%', '+14%', '+16%', '+18%'])
    #plt.legend()

    fig.tight_layout()
    plt.savefig(f'images/prediction_flips_sim_plot.pdf', bbox_inches='tight')
    plt.show()


def class_pred_flips_histogramms():
    """Plot the histogramms of the classes with the largest shares of complementary knowledge.

    :Returns: None
    """
    examples = ['xcit_small_24_p16_224>dla46_c', 'resmlp_24_224>regnetx_080', 'pit_ti_224>vit_base_patch8_224']
    seeds = [124, 189, 165]
    subplots = [133, 132, 131]
    titles = ['c) Strong Teacher - Weak Student', 'b) Equal Teacher - Student Performances', 'a) Weak Teacher - Strong Student']
    titles = [r'iii) Strong Teacher (Diff: $+$18)', r'ii) Equal Accs. (Diff: $\pm$0)', r'i) Weak Teacher (Diff: -13)']

    plt.rcParams['font.size'] = 19
    fig = plt.figure(figsize=(9, 4))
    for s, seed in enumerate(seeds):
        with open(f'prediction_flips/{seed}_prediction_flips.json') as f:
            data = json.load(f)
        plt.subplot(subplots[s])
        hist = np.array(data['results']['pos_class_flips'])/50*100
        hist = -1 * np.sort(-1*hist)
        plt.bar(range(1000), hist, align='edge', width=1, alpha=0.3)
        plt.plot(range(1000), hist, linewidth=2)
        plt.axvline(-5, ymax=hist[0]/plt.gca().get_ylim()[1], linewidth=2)
        if s == 2:
            plt.ylabel('Compl. Knowledge \n per Class [%]', fontsize=20)
        plt.xlabel('ImageNet Classes', fontsize=20)
        plt.xticks([])
        title =f'{titles[s]} (Diff: {round(data["config"]["ts_diff"])})'
        title = titles[s]
        plt.title('\n'.join(textwrap.wrap(title, 20)), fontsize=20, loc='center')

    fig.tight_layout()
    plt.savefig(f'images/pos_flip_histograms.pdf', bbox_inches='tight')
    plt.show()





if __name__ == "__main__":
    #prediction_flips_plot()
    #prediction_flips_entropy_plot()
    prediction_flips_sim_plot()
    #class_pred_flips_histogramms()
    #sim_heatmap()




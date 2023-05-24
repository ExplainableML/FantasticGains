import textwrap
import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from aggregate_results import load_wandb_runs

plt.rc('font', family='Times New Roman', size=14)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ARCHITECTURES = {'transformer': 'Trafo', 'cnn': 'CNN', 'mlp': 'MLP'}
APPROACHES = {'KL_Dist': 'KL Distillation', 'XEKL_Dist': 'XE-KL Distillation',
                'CD_Dist': 'CD Distillation', 'CRD_Dist': 'CRD Distillation',
                'XEKL+MCL_Dist': 'XE-KL+MCL Distillation', 'KL+MT_Dist': 'KL+MT Distillation', 'KL+MT_u_Dist': 'KL+MT-U Distillation'}
TEACHERS = [211, 268, 234, 302, 209, 10, 152, 80, 36, 182, 310, 77, 12, 239, 151, 145, 232, 101, 291, 124]
STUDENTS = {'transformer': [41, 7, 5, 46, 26, 171],
                'cnn': [33, 131, 235, 132, 42, 130, 48],
                'mlp': [214, 2, 9, 77, 258, 160, 72]}


def mscatter(x,y,z, ax=None, m=None, **kw):
    """Scatter plot with different marker styles.
    https://stackoverflow.com/questions/52339341/how-to-plot-a-scatter-plot-with-a-custom-shape-in-matplotlib

    :param x: x-axis values
    :param y: y-axis values
    :param z: z-axis values
    :param ax: matplotlib axis
    :param m: marker styles
    :param kw: additional keyword arguments

    :Returns: matplotlib scatter plot
    """
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


def pos_dist_delta_2():
    """Boxplot of the share of teachers improving the student's performance for different distillation approaches.

    :Returns: None
    """
    plt.rc('font', family='Times New Roman', size=18)
    data = pd.read_csv('students_leaderboard.csv', index_col=[0])
    data = data.loc[data['mean_dist_delta'] > -10]
    data = data.loc[[data['student_id'][i] not in [144, 261, 63, 139, 237, 302, 40, 299, 232, 285] for i in data.index]]

    appr_dict = {'KL Dist.': 'C0', 'XE-KL Dist.': 'C0', 'XE-KL+MCL Dist.': 'C2', 'KL+DP Dist.': 'C2'}
    appr_strings = {'KL_Dist': 'KL Distillation', 'XEKL_Dist': 'XE-KL Distillation', 'XEKL+MCL_Dist': 'XE-KL+MCL Distillation', 'KL+MT_Dist': 'KL+DP Distillation'}
    wrapped_labels = ['\n'.join(textwrap.wrap(label, 12)) for label in appr_strings.values()]

    fig = plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-colorblind')
    data = data.loc[[data['approach'][i] in appr_strings.keys() for i in data.index]]
    pdd = data[['pos_dist_delta', 'approach']]
    pdd['Share of Teachers'] = pdd['pos_dist_delta']
    pdd['Distillation Approach'] = [appr_strings[app] for app in pdd['approach'].values]
    pdd['Category'] = ['KD' if app in ['KL Dist.', 'XE-KL Dist.'] else 'KD+CL' for app in pdd['Distillation Approach'].values]
    #sns.violinplot(data=pdd, x='Distillation Approach', y='Share of Teachers', cut=0, palette=appr_dict, saturation=0.5, inner=None, linewidth=1)
    bp_vals = [pdd.loc[pdd["Distillation Approach"] == appr_strings[app]]['Share of Teachers'].values for app in appr_strings.keys()]
    plt.boxplot(bp_vals[:2], positions=range(0,2), widths=0.5, patch_artist=True,
                boxprops=dict(linewidth=3.0, facecolor='C0', color='black', alpha=0.8),
                medianprops=dict(linewidth=2.0, color='black'), whiskerprops=dict(linewidth=2.0),
                capprops=dict(linewidth=2.0), flierprops=dict(linewidth=2.0))
    plt.boxplot(bp_vals[2:], positions=range(2,4), widths=0.5, patch_artist=True,
                boxprops=dict(linewidth=3.0, facecolor='C2', color='black', alpha=0.8),
                medianprops=dict(linewidth=2.0, color='black'), whiskerprops=dict(linewidth=2.0),
                capprops=dict(linewidth=2.0), flierprops=dict(linewidth=2.0))
    plt.axvline(1.5, color='black', alpha=0.3, linestyle='--', linewidth=2)

    plt.xticks(range(4), wrapped_labels)#, rotation=45, ha='right')
    plt.xlabel('')
    plt.ylabel('Knowledge Transfer Success Rate [%]     ', fontsize=20)
    #plt.title('Share of Teachers Improving the Student', fontsize=22)

    fig.tight_layout()
    plt.savefig(f'images/dist_approaches_plot.pdf', bbox_inches='tight')
    plt.show()


def teacher_influence_plot():
    """Barplot of the influence of the teacher's model properties on the distillation success.

    :Returns: None
    """
    plt.rc('font', family='Times New Roman', size=19)
    data = load_wandb_runs('2_distill_between_experts')
    data = data.dropna(subset=['knowledge_gain', 'knowledge_loss', 'teacher_acc', 'student_params'])
    data = data.loc[data['dist_delta'] > -20]

    type_to_num = {'transformer': 0, 'cnn': 1, 'mlp': 2}
    data['teacher_type'] = [type_to_num[data['teacher_type'][i]] for i in data.index]
    data['student_type'] = [type_to_num[data['student_type'][i]] for i in data.index]

    dist_delta_corr = {'# Param.': {'Teacher':[], 'Student':[]}, 'Acc.': {'Teacher':[], 'Student':[]}, 'Type': {'Teacher':[], 'Student':[]}}
    appr_dict = {'KL_Dist': 1, 'XEKL_Dist': 2, 'XEKL+MCL_Dist': 3, 'KL+MT_Dist': 4}
    appr_strings = {'KL_Dist': 'KL Dist.', 'XEKL_Dist': 'XE-KL Dist.', 'XEKL+MCL_Dist': 'XE-KL+MCL Dist.', 'KL+MT_Dist': 'KL+MT Dist.'}
    appr_strings = {'KL_Dist': 'KL Distillation', 'XEKL_Dist': 'XE-KL Distillation',
                    'XEKL+MCL_Dist': 'XE-KL+MCL Distillation', 'KL+MT_Dist': 'KL+DP Distillation'}
    wrapped_labels = ['\n'.join(textwrap.wrap(label, 12)) for label in appr_strings.values()]
    for approach in appr_dict.keys():
        tmp = data.loc[data['tag'] == approach]
        correlation = tmp.corr(method='pearson')
        dist_delta_corr['# Param.']['Teacher'].append(correlation.loc['dist_delta', 'teacher_params'])
        dist_delta_corr['Acc.']['Teacher'].append(correlation.loc['dist_delta', 'teacher_acc'])
        dist_delta_corr['Type']['Teacher'].append(correlation.loc['dist_delta', 'teacher_type'])
        dist_delta_corr['# Param.']['Student'].append(correlation.loc['dist_delta', 'student_params'])
        dist_delta_corr['Acc.']['Student'].append(correlation.loc['dist_delta', 'student_acc'])
        dist_delta_corr['Type']['Student'].append(correlation.loc['dist_delta', 'student_type'])

    fig = plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-colorblind')
    ls = ['solid', 'dashed', 'dotted']
    marker = ['..', '//', 'xx']
    cols = [0, 1, 2]
    for f, feature in enumerate(dist_delta_corr.keys()):
        x = [val*2 - 0.5*(f-1) for val in appr_dict.values()]
        plt.bar(x, dist_delta_corr[feature]['Teacher'], label=f'Teacher {feature}', color='lightgray', hatch=marker[f], width=0.5, edgecolor='black')
    plt.axhline(0, color='black', alpha=0.5, linestyle='-')
    plt.axvline(5, color='black', alpha=0.3, linestyle='--')
    #plt.xlabel('Distillation Approaches', fontsize=18)
    plt.ylabel('Correlation with the Dist. Delta', fontsize=20)
    #plt.title('Influence of the Teacher Model on the Distillation Delta', fontsize=22)
    plt.legend()
    plt.xticks([val*2 for val in appr_dict.values()], wrapped_labels)

    fig.tight_layout()
    #fig.subplots_adjust(bottom=0.3)
    plt.savefig(f'images/teacher_influence.pdf', bbox_inches='tight')
    plt.show()


def transfer_rate_plot_a(appr='KL+MT_Dist'):
    """Plot of the transfer rate for the classes with the highest shares of complementary knowledge.

    :param appr: The distillation approach to plot.

    :Returns: None
    """
    plt.rc('font', family='Times New Roman', size=19)
    data = load_wandb_runs('2_distill_between_experts')
    data = data.dropna(subset=['knowledge_gain', 'knowledge_loss', 'pos_flips_delta'])
    data = data.loc[data['ts_diff'] != 0]
    data = data.loc[data['tag'] == appr]
    data = data.loc[[data['student_id'][i] not in [144, 261, 63, 139, 237, 302, 40, 299, 232, 285] for i in data.index]]
    for i in data.index:
        if data['student_id'][i] in [26, 171]:
            data['student_params'][i] = 11

    shapes = ['s', 'v', 'o', 'v', 's', 'P', 'v', 's', '<', 'p']
    marker_dict = {key: shapes[k] for k, key in enumerate(['mlp', 'cnn', 'transformer'])}
    #colors = {'transformer': '#5387DD', 'cnn': '#DA4C4C', 'mlp': '#EDB732'}
    colors = {'transformer': '#8172B3', 'cnn': '#C44E52', 'mlp': '#CCB974'}

    fig = plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-bright')
    #plt.subplot(121)
    p_values = [2, 5, 20, 50, 100]
    for arch in marker_dict.keys():
        tmp = data.loc[data['student_type'] == arch]
        transfer_rates = [100-(p*tmp['positive_flips'].values - tmp[f'top{p}%_improve'].values*tmp['knowledge_gain'].values)/tmp['positive_flips'].values  for p in p_values]
        means = [np.mean(t) for t in transfer_rates]
        upper = [np.quantile(t, 0.75) for t in transfer_rates]
        lower = [np.quantile(t, 0.25) for t in transfer_rates]

        plt.plot(range(len(p_values)), means, marker=marker_dict[arch], color=colors[arch], label=ARCHITECTURES[arch], linewidth=3, markersize=8, alpha=0.8, markeredgecolor='black')
        plt.fill_between(range(len(p_values)), lower, upper, alpha=0.1, color=colors[arch])
    plt.ylabel('Transfer Rate [%]', fontsize=20)
    plt.xticks(range(len(p_values)), [f'Top-{p}%' for p in p_values])
    plt.xlabel('Classes with Top-X% of the Compl. Knowledge', fontsize=20)
    #plt.title('a) Transferred Flips for Top-X% Pos. Flip Classes', fontsize=18)
    plt.ylim(18, 102)

    fig.tight_layout()
    plt.savefig(f'images/transferred_flips_a.pdf', bbox_inches='tight')
    plt.show()


def transfer_rate_plot_b(appr='KL+MT_Dist'):
    """Plot of the transfer rate by the student model size.

    :param appr: The distillation approach to plot.

    :Returns: None
    """
    plt.rc('font', family='Times New Roman', size=19)
    data = load_wandb_runs('2_distill_between_experts')
    data = data.dropna(subset=['knowledge_gain', 'knowledge_loss', 'pos_flips_delta'])
    data = data.loc[data['ts_diff'] != 0]
    data = data.loc[data['tag'] == appr]
    data = data.loc[[data['student_id'][i] not in [144, 261, 63, 139, 237, 302, 40, 299, 232, 285] for i in data.index]]
    for i in data.index:
        if data['student_id'][i] in [26, 171]:
            data['student_params'][i] = 11

    shapes = ['o', 'v', 's', 'P', 'v', 's', '<', 'p']
    marker_dict = {key: shapes[k] for k, key in enumerate(['transformer', 'cnn', 'mlp'])}
    #colors = {'transformer': '#5387DD', 'cnn': '#DA4C4C', 'mlp': '#EDB732'}
    colors = {'transformer': '#8172B3', 'cnn': '#C44E52', 'mlp': '#CCB974'}

    fig = plt.figure(figsize=(8, 5))
    transfer_rates = {}
    for arch in marker_dict.keys():
        transfer_rates[arch] = {'mean': [], 'upper': [], 'lower': [], 'params': []}
        tmp = data.loc[data['student_type'] == arch]
        student_params = np.sort(np.unique(tmp['student_params'].values))
        for params in student_params:
            tmp2 = tmp.loc[tmp['student_params'] == params]
            tr = tmp2['knowledge_gain'].values*(tmp2['top100%_improve'].values/100)/tmp2[f'positive_flips'].values*100
            transfer_rates[arch]['mean'].append(np.mean(tr))
            transfer_rates[arch]['upper'].append(np.quantile(tr, 0.75))
            transfer_rates[arch]['lower'].append(np.quantile(tr, 0.25))
            transfer_rates[arch]['params'].append(params)
    for arch in marker_dict.keys():
        plt.plot(transfer_rates[arch]['params'], transfer_rates[arch]['mean'], marker=marker_dict[arch], color=colors[arch], label=ARCHITECTURES[arch], linewidth=3, markersize=8, alpha=0.8, markeredgecolor='black')
        plt.fill_between(transfer_rates[arch]['params'], transfer_rates[arch]['lower'], transfer_rates[arch]['upper'], color=colors[arch], alpha=0.1)
    #plt.ylabel('Share of Transferred Positive Flips', fontsize=16)
    plt.xlabel('Number of Student Parameters [M]', fontsize=20)
    plt.ylabel('Transfer Rate [%]', fontsize=20)
    #plt.title('b) Total Transferred Flips by Student Size', fontsize=18)
    plt.xscale('log')
    plt.yscale('log')
    plt.minorticks_off()
    plt.xticks([10, 15, 20, 40, 60, 80, 120, 200], ['10', '15', '20', '40', '60', '80', '120', '200'])
    plt.ylim(18,85)
    plt.yticks([20, 40, 60, 80], ['20', '40', '60', '80'])
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'images/transferred_flips_b.pdf', bbox_inches='tight')
    plt.show()


def full_imagenet_table():
    """Prints the full table of the imagenet experiments.

    :Returns: None
    """
    students = [41, 5, 26, 131, 40, 130, 214, 2, 160]
    teachers = [234, 302, 77]
    data = load_wandb_runs('2_distill_between_experts')
    data = data.loc[[data['teacher_id'][i] in teachers for i in data.index]]
    data = data.loc[[data['student_id'][i] in students for i in data.index]]
    data = data.loc[data['dist_delta'] > -40]
    imagenet = data.loc[[data['data'][i]['dataset'] == 'imagenet' for i in data.index]]
    subset = data.loc[[data['data'][i]['dataset'] != 'imagenet' for i in data.index]]

    students = ['xcit_large_24_p16_224', 'pit_b_224', 'pit_xs_224', 'gluon_senet154', 'convnext_small_in22ft1k', 'resnetv2_50x1_bit_distilled', 'mixer_l16_224', 'mixer_b16_224_miil', 'resmlp_24_distilled_224']
    s_accs = [82.89, 82.44, 78.19, 81.23, 84.57, 82.80, 72.07, 82.30, 80.76]
    rows = []
    header = '| Students | Type | Acc. | Params. | KL-Dist. | KL+MT Dist. | KL+MT-u Dist.|'
    lines = '|-------|-------|-------|-------|------|-------|-------|'
    print(header)
    #print(lines)
    for s, student in enumerate(students):
        tmp = imagenet.loc[imagenet['student_name'] == student]
        kl = [np.mean(tmp.loc[tmp['tag'] == 'KL_Dist']['dist_delta'].values), np.std(tmp.loc[tmp['tag'] == 'KL_Dist']['dist_delta'].values)]
        mt = [np.mean(tmp.loc[tmp['tag'] == 'KL+MT_Dist']['dist_delta'].values), np.std(tmp.loc[tmp['tag'] == 'KL+MT_Dist']['dist_delta'].values)]
        mtu = [np.mean(tmp.loc[tmp['tag'] == 'KL+MT_u_Dist']['dist_delta'].values), np.std(tmp.loc[tmp['tag'] == 'KL+MT_u_Dist']['dist_delta'].values)]
        #row = f'|{student} | {arch_strings[tmp["student_type"].values[0]]} |  {s_accs[s]} | {tmp["student_params"].values[0]} | {round(kl[0], 2)} ({round(kl[1], 2)}) |  {round(mt[0], 2)} ({round(mt[1], 2)}) |  {round(mtu[0], 2)} ({round(mtu[1], 2)}) |'
        row = f'{student.replace("_", "-")} & {ARCHITECTURES[tmp["student_type"].values[0]]} &  {s_accs[s]} & {tmp["student_params"].values[0]} & {round(kl[0], 2)} ($\pm${round(kl[1], 2)}) &  {round(mt[0], 2)} ($\pm${round(mt[1], 2)}) &  {round(mtu[0], 2)} ($\pm${round(mtu[1], 2)}) \\\\'
        if s % 3 == 0:
            print('\hline')
        print(row)
        rows.append(row)


if __name__ == "__main__":
    #pos_dist_delta()
    #teacher_influence_plot()

    transfer_rate_plot_a()
    transfer_rate_plot_b()

    #full_imagenet_table()

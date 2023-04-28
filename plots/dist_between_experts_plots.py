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


def plot_pos_dist_delta():
    data = pd.read_csv('students_leaderboard.csv', index_col=[0])
    data = data.loc[data['mean_dist_delta'] > -10]
    data = data.loc[[data['student_id'][i] not in [144, 261, 63, 139, 237, 302, 40, 299, 232, 285] for i in data.index]]

    appr_dict = {'KL_Dist': 1, 'XEKL_Dist': 2, 'XEKL+MCL_Dist': 3, 'KL+MT_Dist': 4}

    shapes = ['o', 'v', 'X', 'P', 'v', 's', '<', 'p']
    marker_dict = {key: shapes[k] for k, key in enumerate(['transformer', 'cnn', 'mlp'])}
    colors = {'transformer': '#5387DD', 'cnn': '#DA4C4C', 'mlp': '#EDB732'}

    fig = plt.figure(figsize=(13, 5))
    plt.subplot(121)
    for arch in colors.keys():
        mean = np.array([np.mean(data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['pos_dist_delta'].values) for appr in appr_dict.keys()])
        lower = np.array([np.quantile(data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['pos_dist_delta'].values, 0.25)
                for appr in appr_dict.keys()])
        upper = np.array([np.quantile(data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['pos_dist_delta'].values, 0.75)
                for appr in appr_dict.keys()])
        plt.plot(appr_dict.values(), mean, label=arch_strings[arch], marker=marker_dict[arch], color=colors[arch])
        plt.fill_between(appr_dict.values(), upper, lower, color=colors[arch], alpha=0.1)
    #plt.xlabel('Distillation Approaches', fontsize=16)
    plt.xticks(list(appr_dict.values()), ['KL Dist.', 'XE-KL Dist.', 'XE-KL+MCL Dist.', 'KL+MT Dist.'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Share of Teachers with Positive Dist. Delta', fontsize=18)
    plt.legend(loc='upper left')

    plt.subplot(122)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    for arch in colors.keys():
        mean = np.array([np.median(data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['mean_dist_delta'].values) for appr in appr_dict.keys()])
        lower = np.array([np.quantile(data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['mean_dist_delta'].values, 0.25)
                for appr in appr_dict.keys()])
        upper = np.array([np.quantile(data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['mean_dist_delta'].values, 0.75)
                for appr in appr_dict.keys()])
        plt.plot(appr_dict.values(), mean, label=arch_strings[arch], marker=marker_dict[arch], color=colors[arch])
        plt.fill_between(appr_dict.values(), lower, upper, color=colors[arch], alpha=0.1)
    #plt.xlabel('Distillation Approaches', fontsize=16)
    plt.xticks(list(appr_dict.values()), ['KL Dist.', 'XE-KL Dist.', 'XE-KL+MCL Dist.', 'KL+MT Dist.'])
    plt.xticks(rotation=45, ha='right')
    plt.ylim([-1.5, 1.5])
    plt.title('Median Distillation Delta', fontsize=18)

    fig.tight_layout()
    plt.savefig(f'images/dist_approaches_plot.pdf', bbox_inches='tight')
    plt.show()


def pos_dist_delta_2():
    data = pd.read_csv('students_leaderboard.csv', index_col=[0])
    data = data.loc[data['mean_dist_delta'] > -10]
    data = data.loc[[data['student_id'][i] not in [144, 261, 63, 139, 237, 302, 40, 299, 232, 285] for i in data.index]]

    appr_dict = {'KL Dist.': 'C0', 'XE-KL Dist.': 'C0', 'XE-KL+MCL Dist.': 'C1', 'KL+MT Dist.': 'C1'}
    appr_strings = {'KL_Dist': 'KL Dist.', 'XEKL_Dist': 'XE-KL Dist.', 'XEKL+MCL_Dist': 'XE-KL+MCL Dist.', 'KL+MT_Dist': 'KL+MT Dist.'}

    fig = plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-colorblind')
    data = data.loc[[data['approach'][i] in appr_strings.keys() for i in data.index]]
    pdd = data[['pos_dist_delta', 'approach']]
    pdd['Share of Teachers'] = pdd['pos_dist_delta']
    pdd['Distillation Approach'] = [appr_strings[app] for app in pdd['approach'].values]
    pdd['Category'] = ['KD' if app in ['KL Dist.', 'XE-KL Dist.'] else 'KD+CL' for app in pdd['Distillation Approach'].values]
    sns.violinplot(data=pdd, x='Distillation Approach', y='Share of Teachers', cut=0, palette=appr_dict, saturation=1)
    plt.axvline(1.5, color='black', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('')
    plt.ylabel('Share of Teachers [%]', fontsize=16)
    plt.title('Share of Teachers Improving the Student', fontsize=18)

    fig.tight_layout()
    plt.savefig(f'images/dist_approaches_plot.pdf', bbox_inches='tight')
    plt.show()


def dist_deltas_plot():
    data = load_wandb_runs('2_distill_between_experts')
    data = data.loc[data['dist_delta'] > -20]
    data = data.loc[[data['student_id'][i] not in [144, 261, 63, 139, 237, 302, 40, 299, 232, 285] for i in data.index]]
    data = data.loc[[data['data'][i]['dataset'] == 'imagenet_subset' for i in data.index]]

    fig = plt.figure(figsize=(10, 5))
    appr_strings = {'KL_Dist': 'KL Dist.', 'XEKL_Dist': 'XE-KL Dist.', 'XEKL+MCL_Dist': 'XE-KL+MCL Dist.',
                    'KL+MT_Dist': 'KL+MT Dist.'}
    colors = {'Transformer': '#5387DD', 'CNN': '#DA4C4C', 'MLP': '#EDB732'}
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    data = data.loc[[data['tag'][i] in appr_strings.keys() for i in data.index]]
    dist_deltas = data[['student_type', 'tag', 'dist_delta']]
    dist_deltas['Student Arch.'] = [arch_strings[arch] for arch in dist_deltas['student_type'].values]
    dist_deltas['approach'] = [appr_strings[appr] for appr in dist_deltas['tag'].values]
    sns.violinplot(data=dist_deltas, x='approach', y='dist_delta', hue='Student Arch.', width=1, palette=colors, order=list(appr_strings.values()))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('')
    plt.ylabel('Distillation Delta', fontsize=16)
    plt.title('Distillation Delta by Distillation Approach', fontsize=18)
    plt.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(f'images/dist_deltas_plot.pdf', bbox_inches='tight')
    plt.show()


def correlation_heatmap(runs, appr=None, mode='student'):
    correlation = runs.corr(method='pearson')
    if mode == 'student':
        col_subset = ['student_params', 'student_acc', 'student_cnn', 'student_mlp', 'student_transformer']
        col_rename = {'student_params': 'Student Params.', 'student_acc': 'Student Acc.',
                      'student_cnn': 'CNN Student', 'student_mlp':'MLP Student', 'student_transformer': 'Transformer Student'}
        fig = plt.figure(figsize=(7, 5))
        plt.subplots_adjust(bottom=0.22, left=0.30)
    elif mode == 'teacher':
        col_subset = ['teacher_params', 'teacher_acc', 'teacher_cnn', 'teacher_mlp', 'teacher_transformer']
        col_rename = {'teacher_params': 'Teacher Params.', 'teacher_acc': 'Teacher Acc.',
                      'teacher_cnn': 'CNN Teacher', 'teacher_mlp':'MLP Teacher', 'teacher_transformer': 'Transformer Teacher'}
        fig = plt.figure(figsize=(7, 5))
        plt.subplots_adjust(bottom=0.22, left=0.30)
    else:
        col_subset = ['performance_diff', 'params_diff']
        col_subset += [f'{t_arch}-{s_arch}' for s_arch in arch_strings.keys() for t_arch in arch_strings.keys()]
        col_rename = {'performance_diff': 'Performance Diff.', 'params_diff': 'Params Diff.', 'cnn-cnn': 'CNN to CNN',
                      'cnn-mlp': 'CNN to MLP', 'cnn-transformer': 'CNN to Transf.', 'mlp-cnn': 'MLP to CNN', 'mlp-mlp':
                      'MLP to MLP', 'mlp-transformer': 'MLP to Transf.', 'transformer-cnn': 'Transf. to CNN', 'transformer-mlp': 'Transf. to MLP',
                      'transformer-transformer': 'Transf. to Transf.'}
        fig = plt.figure(figsize=(10, 5))
        plt.subplots_adjust(bottom=0.22, left=0.05)

    row_rename = {'dist_delta': 'Dist. Delta', 'knowledge_gain': 'Knowledge Gain', 'knowledge_loss': 'Knowledge Loss'}
    row_subset = ['dist_delta', 'knowledge_gain', 'knowledge_loss']
    correlation = correlation.loc[row_subset, col_subset]
    correlation = correlation.rename(columns=col_rename, index=row_rename)
    sns.heatmap(correlation, annot=True, fmt='.2f', annot_kws={"size": 16}, cmap='coolwarm', vmin=-0.8, vmax=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')
    if appr is not None:
        plt.title(appr_strings[appr]+ '\n', fontsize=18)
    fig.tight_layout()
    plt.savefig(f'images/{appr}_{mode}_corr_heatmap.pdf', bbox_inches='tight')
    plt.show()


def get_correlation_heatmaps():
    data = load_wandb_runs('2_distill_between_experts')
    data = data.loc[data['dist_delta'] > -40]
    for approach in appr_strings.keys():
        tmp = data.loc[(data['tag'] == approach)]
        for arch in ['transformer', 'cnn', 'mlp']:
            tmp[f'student_{arch}'] = [tmp['student_type'][i]==arch for i in tmp.index]
        correlation_heatmap(tmp, approach, 'student')
        print('next')


def get_correlation_heatmaps_teacher():
    data = load_wandb_runs('2_distill_between_experts')
    #for approach in appr_strings.keys():
    for approach in ['KL+MT_Dist']:
        tmp = data.loc[(data['tag'] == approach)]
        tmp['performance_diff'] = abs(tmp['teacher_acc'] - tmp['student_acc'])
        tmp['params_diff'] = abs(tmp['teacher_params'] - tmp['student_params'])
        for arch in ['transformer', 'cnn', 'mlp']:
            tmp[f'student_{arch}'] = [tmp['student_type'][i]==arch for i in tmp.index]
            tmp[f'teacher_{arch}'] = [tmp['teacher_type'][i]==arch for i in tmp.index]
        for t_arch in ['transformer', 'cnn', 'mlp']:
            for s_arch in ['transformer', 'cnn', 'mlp']:
                tmp[f'{t_arch}-{s_arch}'] = [tmp['student_type'][i]==s_arch and tmp['teacher_type'][i]==t_arch for i in tmp.index]
        correlation_heatmap(tmp, approach, 'ts')


def teacher_influence_plot():
    data = load_wandb_runs('2_distill_between_experts')
    data = data.dropna(subset=['knowledge_gain', 'knowledge_loss', 'teacher_acc', 'student_params'])
    data = data.loc[data['dist_delta'] > -20]

    type_to_num = {'transformer': 0, 'cnn': 1, 'mlp': 2}
    data['teacher_type'] = [type_to_num[data['teacher_type'][i]] for i in data.index]
    data['student_type'] = [type_to_num[data['student_type'][i]] for i in data.index]

    dist_delta_corr = {'Params.': {'Teacher':[], 'Student':[]}, 'Acc.': {'Teacher':[], 'Student':[]}, 'Type': {'Teacher':[], 'Student':[]}}
    appr_dict = {'KL_Dist': 1, 'XEKL_Dist': 2, 'XEKL+MCL_Dist': 3, 'KL+MT_Dist': 4}
    appr_strings = {'KL_Dist': 'KL Dist.', 'XEKL_Dist': 'XE-KL Dist.', 'XEKL+MCL_Dist': 'XE-KL+MCL Dist.', 'KL+MT_Dist': 'KL+MT Dist.'}
    for approach in appr_dict.keys():
        tmp = data.loc[data['tag'] == approach]
        correlation = tmp.corr(method='pearson')
        dist_delta_corr['Params.']['Teacher'].append(correlation.loc['dist_delta', 'teacher_params'])
        dist_delta_corr['Acc.']['Teacher'].append(correlation.loc['dist_delta', 'teacher_acc'])
        dist_delta_corr['Type']['Teacher'].append(correlation.loc['dist_delta', 'teacher_type'])
        dist_delta_corr['Params.']['Student'].append(correlation.loc['dist_delta', 'student_params'])
        dist_delta_corr['Acc.']['Student'].append(correlation.loc['dist_delta', 'student_acc'])
        dist_delta_corr['Type']['Student'].append(correlation.loc['dist_delta', 'student_type'])

    fig = plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-colorblind')
    ls = ['solid', 'dashed', 'dotted']
    marker = ['o', 'x', '^']
    for f, feature in enumerate(dist_delta_corr.keys()):
        x = [val*2 - 0.5*(f-1) for val in appr_dict.values()]
        plt.bar(x, dist_delta_corr[feature]['Teacher'], label=f'Teacher {feature}', color=f'C{f}', width=0.5)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    #plt.xlabel('Distillation Approaches', fontsize=18)
    plt.ylabel('Correlation with the Dist. Delta', fontsize=18)
    plt.title('Influence of the Teacher Model on the Distillation Delta', fontsize=20)
    plt.legend()
    plt.xticks([val*2 for val in appr_dict.values()], [appr_strings[appr] for appr in appr_dict.keys()])
    plt.xticks(rotation=45, ha='right')
    fig.subplots_adjust(bottom=0.3)
    plt.savefig(f'images/teacher_influence.pdf', bbox_inches='tight')
    plt.show()


def plot_arch_influence(appr='KL+MT_Dist'):
    data = pd.read_csv('students_leaderboard.csv', index_col=[0])
    data = data.loc[data['approach'] == appr]

    arch_students = {'Transformer': ["vit_small_patch32_224", "xcit_small_24_p16_224", "pit_b_224", "xcit_large_24_p16_224"],
                     'CNN': ["gluon_resnet34_v1b", "gluon_resnet101_v1c", "wide_resnet50_2", "ig_resnext101_32x16d"],
                     'MLP': ["mixer_b16_224_miil", "resmlp_36_224", "resmlp_12_224"]}

    shapes = ['o', 'v', 'X', 'P', 'v', 's', '<', 'p']
    marker_dict = {key: shapes[k] for k, key in enumerate(['Transformer', 'CNN', 'MLP'])}
    colors = {'Transformer': '#5387DD', 'CNN': '#DA4C4C', 'MLP': '#EDB732'}

    fig = plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    for arch in arch_students.keys():
        mean = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['mean_dist_delta'].values
        lower = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['q25_dist_delta'].values
        upper = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['q75_dist_delta'].values
        plt.plot(range(len(arch_students[arch])), mean, label=arch, marker=marker_dict[arch], color=colors[arch])
        plt.fill_between(range(len(arch_students[arch])), lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Model Size', fontsize=16)
    plt.xticks(range(4), ['Small', 'Medium', 'Large', 'XLarge'])
    plt.ylim()
    plt.title('Distillation Delta', fontsize=16)

    plt.subplot(132)
    for arch in arch_students.keys():
        mean = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['mean_k_gain'].values
        lower = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['q25_k_gain'].values
        upper = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['q75_k_gain'].values
        plt.plot(range(len(arch_students[arch])), mean, label=arch, marker=marker_dict[arch], color=colors[arch])
        plt.fill_between(range(len(arch_students[arch])), lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Model Size', fontsize=16)
    plt.xticks(range(4), ['Small', 'Medium', 'Large', 'XLarge'])
    plt.ylim([0.1, 3.5])
    plt.title('Knowledge Gain', fontsize=16)

    plt.subplot(133)
    for arch in arch_students.keys():
        mean = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['mean_k_loss'].values
        lower = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['q25_k_loss'].values
        upper = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['q75_k_loss'].values
        plt.plot(range(len(arch_students[arch])), mean, label=arch, marker=marker_dict[arch], color=colors[arch])
        plt.fill_between(range(len(arch_students[arch])), lower, upper, color=colors[arch], alpha=0.2)
    plt.legend(loc='lower right')
    plt.xlabel('Student Model Size', fontsize=16)
    plt.xticks(range(4), ['Small', 'Medium', 'Large', 'XLarge'])
    plt.ylim([0.1, 3.5])
    plt.title('Knowledge Loss', fontsize=16)

    plt.suptitle(appr_strings[appr], fontsize=20)
    fig.tight_layout()
    plt.savefig(f'images/student_architecture_influence.pdf', bbox_inches='tight')
    plt.show()


def plot_performance_influence(appr='KL+MT_Dist'):
    data = pd.read_csv('students_leaderboard.csv', index_col=[0])
    data = data.loc[data['approach'] == appr]

    perf_students = {'Transformer': [['vit_base_patch16_224','vit_base_patch16_224_sam'], ['pit_xs_distilled_224', 'pit_xs_224']],
                     'CNN':[['wide_resnet50_2', 'legacy_seresnet152'], ['resnetv2_50x1_bit_distilled', 'gluon_resnet34_v1b']],
                     'MLP': [['mixer_b16_224_miil', 'mixer_b16_224'], ['resmlp_24_224', 'resmlp_24_distilled_224']]}
    performances = {'Transformer': [[84.53, 80.24],[79.31, 78.19]],
                     'CNN':[[81.46, 78.65], [82.80, 74.59]],
                     'MLP': [[82.30, 76.61], [79.39, 80.76]]}

    shapes = ['o', 'v', 'X', 'P', 'v', 's', '<', 'p']
    marker_dict = {key: shapes[k] for k, key in enumerate(['Transformer', 'CNN', 'MLP'])}
    colors = {'Transformer': '#5387DD', 'CNN': '#DA4C4C', 'MLP': '#EDB732'}

    fig = plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    for arch in perf_students.keys():
        for p, pair in enumerate(perf_students[arch]):
            mean = np.array([data.loc[data['student_name'] == pair[i]]['mean_dist_delta'].values[0] for i in range(len(pair))])
            lower = np.array([data.loc[data['student_name'] == pair[i]]['q25_dist_delta'].values[0] for i in range(len(pair))])
            upper = np.array([data.loc[data['student_name'] == pair[i]]['q75_dist_delta'].values[0] for i in range(len(pair))])
            plt.plot(performances[arch][p], mean, label=arch, marker=marker_dict[arch], color=colors[arch])
            plt.fill_between(performances[arch][p], lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Accuracy', fontsize=16)
    plt.title('Distillation Delta', fontsize=16)

    plt.subplot(132)
    for arch in perf_students.keys():
        for p, pair in enumerate(perf_students[arch]):
            mean = np.array([data.loc[data['student_name'] == pair[i]]['mean_k_gain'].values[0] for i in range(len(pair))])
            lower = np.array([data.loc[data['student_name'] == pair[i]]['q25_k_gain'].values[0] for i in range(len(pair))])
            upper = np.array([data.loc[data['student_name'] == pair[i]]['q75_k_gain'].values[0] for i in range(len(pair))])
            plt.plot(performances[arch][p], mean, label=arch, marker=marker_dict[arch], color=colors[arch])
            plt.fill_between(performances[arch][p], lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Accuracy', fontsize=16)
    plt.ylim([0.1, 3.5])
    plt.title('Knowledge Gain', fontsize=16)

    plt.subplot(133)
    for arch in perf_students.keys():
        for p, pair in enumerate(perf_students[arch]):
            mean = np.array([data.loc[data['student_name'] == pair[i]]['mean_k_loss'].values[0] for i in range(len(pair))])
            lower = np.array([data.loc[data['student_name'] == pair[i]]['q25_k_loss'].values[0] for i in range(len(pair))])
            upper = np.array([data.loc[data['student_name'] == pair[i]]['q75_k_loss'].values[0] for i in range(len(pair))])
            plt.plot(performances[arch][p], mean, label=arch, marker=marker_dict[arch], color=colors[arch])
            plt.fill_between(performances[arch][p], lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Accuracy', fontsize=16)
    plt.title('Knowledge Loss', fontsize=16)
    plt.ylim([0.1, 3.5])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.suptitle(appr_strings[appr], fontsize=20)
    fig.tight_layout()
    plt.savefig(f'images/student_performance_influence.pdf', bbox_inches='tight')
    plt.show()


def plot_size_influence(appr='KL+MT_Dist'):
    data = pd.read_csv('students_leaderboard.csv', index_col=[0])
    data = data.loc[data['approach'] == appr]

    size_students = {'Transformer': [['xcit_large_24_p16_224', 'pit_b_224', 'vit_relpos_medium_patch16_224'],
                                     ['vit_base_patch16_224_sam', 'deit_small_patch16_224', 'pit_xs_distilled_224']],
                     'CNN':[['gluon_senet154', 'wide_resnet50_2', 'resnetv2_50'],
                            ['ig_resnext101_32x16d', 'swsl_resnext101_32x8d', 'convnext_small_in22ft1k']],
                     'MLP': [['mixer_b16_224_miil', 'resmlp_24_224', 'mixnet_xl'],
                             ['mixer_b16_224', 'resmlp_12_224', 'mixnet_s']]}

    shapes = ['o', 'v', 'X', 'P', 'v', 's', '<', 'p']
    marker_dict = {key: shapes[k] for k, key in enumerate(['Transformer', 'CNN', 'MLP'])}
    colors = {'Transformer': '#5387DD', 'CNN': '#DA4C4C', 'MLP': '#EDB732'}

    fig = plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    for arch in size_students.keys():
        for triplet in size_students[arch]:
            mean = np.array([data.loc[data['student_name'] == student]['mean_dist_delta'].values[0] for student in triplet])
            lower = np.array([data.loc[data['student_name'] == student]['q25_dist_delta'].values[0] for student in triplet])
            upper = np.array([data.loc[data['student_name'] == student]['q75_dist_delta'].values[0] for student in triplet])
            size = np.array([data.loc[data['student_name'] == student]['student_params'].values[0] for student in triplet])
            plt.plot(size, mean, label=arch, marker=marker_dict[arch], color=colors[arch])
            plt.fill_between(size, lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Parameters', fontsize=16)
    plt.xscale('log')
    plt.xticks([5, 10, 15, 20, 40, 60, 80, 120, 200], ['5', '10', '15', '20', '40', '60', '80', '120', '200'])
    plt.title('Distillation Delta', fontsize=16)

    plt.subplot(132)
    for arch in size_students.keys():
        for triplet in size_students[arch]:
            mean = np.array([data.loc[data['student_name'] == student]['mean_k_gain'].values[0] for student in triplet])
            lower = np.array([data.loc[data['student_name'] == student]['q25_k_gain'].values[0] for student in triplet])
            upper = np.array([data.loc[data['student_name'] == student]['q75_k_gain'].values[0] for student in triplet])
            size = np.array([data.loc[data['student_name'] == student]['student_params'].values[0] for student in triplet])
            plt.plot(size, mean, label=arch, marker=marker_dict[arch], color=colors[arch])
            plt.fill_between(size, lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Parameters', fontsize=16)
    plt.xscale('log')
    plt.xticks([5, 10, 15, 20, 40, 60, 80, 120, 200], ['5', '10', '15', '20', '40', '60', '80', '120', '200'])
    plt.ylim([0.1, 4.1])
    plt.title('Knowledge Gain', fontsize=16)

    plt.subplot(133)
    for arch in size_students.keys():
        for triplet in size_students[arch]:
            mean = np.array([data.loc[data['student_name'] == student]['mean_k_loss'].values[0] for student in triplet])
            lower = np.array([data.loc[data['student_name'] == student]['q25_k_loss'].values[0] for student in triplet])
            upper = np.array([data.loc[data['student_name'] == student]['q75_k_loss'].values[0] for student in triplet])
            size = np.array([data.loc[data['student_name'] == student]['student_params'].values[0] for student in triplet])
            plt.plot(size, mean, label=arch, marker=marker_dict[arch], color=colors[arch])
            plt.fill_between(size, lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Parameters', fontsize=16)
    plt.xscale('log')
    plt.xticks([5, 10, 15, 20, 40, 60, 80, 120, 200], ['5', '10', '15', '20', '40', '60', '80', '120', '200'])
    plt.ylim([0.1, 4.1])
    plt.title('Knowledge Loss', fontsize=16)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.suptitle(appr_strings[appr], fontsize=20)
    fig.tight_layout()
    plt.savefig(f'images/student_size_influence.pdf', bbox_inches='tight')
    plt.show()


def student_feature_importenace(appr='KL+MT_Dist'):
    data = pd.read_csv('students_leaderboard.csv', index_col=[0])
    data = data.loc[data['approach'] == appr]

    shapes = ['o', 'v', 'X', 'P', 'v', 's', '<', 'p']
    marker_dict = {key: shapes[k] for k, key in enumerate(['Transformer', 'CNN', 'MLP'])}
    colors = {'Transformer': '#5387DD', 'CNN': '#DA4C4C', 'MLP': '#EDB732'}

    fig = plt.figure(figsize=(15, 4))
    plt.subplot(131)
    perf_students = {'Transformer': [['vit_base_patch16_224','vit_base_patch16_224_sam'], ['pit_xs_distilled_224', 'pit_xs_224']],
                     'CNN':[['wide_resnet50_2', 'legacy_seresnet152'], ['resnetv2_50x1_bit_distilled', 'gluon_resnet34_v1b']],
                     'MLP': [['mixer_b16_224_miil', 'mixer_b16_224'], ['resmlp_24_224', 'resmlp_24_distilled_224']]}
    performances = {'Transformer': [[84.53, 80.24],[79.31, 78.19]],
                     'CNN':[[81.46, 78.65], [82.80, 74.59]],
                     'MLP': [[82.30, 76.61], [79.39, 80.76]]}

    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    for arch in perf_students.keys():
        for p, pair in enumerate(perf_students[arch]):
            mean = np.array([data.loc[data['student_name'] == pair[i]]['mean_dist_delta'].values[0] for i in range(len(pair))])
            lower = np.array([data.loc[data['student_name'] == pair[i]]['q25_dist_delta'].values[0] for i in range(len(pair))])
            upper = np.array([data.loc[data['student_name'] == pair[i]]['q75_dist_delta'].values[0] for i in range(len(pair))])
            plt.plot(performances[arch][p], mean, label=arch, marker=marker_dict[arch], color=colors[arch])
            plt.fill_between(performances[arch][p], lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Accuracy', fontsize=16)
    plt.ylabel('Distillation Delta', fontsize=16)
    plt.title('Performance', fontsize=18)

    plt.subplot(132)
    size_students = {'Transformer': [['xcit_large_24_p16_224', 'pit_b_224', 'vit_relpos_medium_patch16_224'],
                                     ['vit_base_patch16_224_sam', 'deit_small_patch16_224', 'pit_xs_distilled_224']],
                     'CNN': [['gluon_senet154', 'wide_resnet50_2', 'resnetv2_50'],
                             ['ig_resnext101_32x16d', 'swsl_resnext101_32x8d', 'convnext_small_in22ft1k']],
                     'MLP': [['mixer_b16_224_miil', 'resmlp_24_224', 'mixnet_xl'],
                             ['mixer_b16_224', 'resmlp_12_224', 'mixnet_s']]}

    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    for arch in size_students.keys():
        for triplet in size_students[arch]:
            mean = np.array(
                [data.loc[data['student_name'] == student]['mean_dist_delta'].values[0] for student in triplet])
            lower = np.array(
                [data.loc[data['student_name'] == student]['q25_dist_delta'].values[0] for student in triplet])
            upper = np.array(
                [data.loc[data['student_name'] == student]['q75_dist_delta'].values[0] for student in triplet])
            size = np.array(
                [data.loc[data['student_name'] == student]['student_params'].values[0] for student in triplet])
            plt.plot(size, mean, label=arch, marker=marker_dict[arch], color=colors[arch])
            plt.fill_between(size, lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Parameters', fontsize=16)
    plt.xscale('log')
    plt.xticks([5, 10, 15, 20, 40, 60, 80, 120, 200], ['5', '10', '15', '20', '40', '60', '80', '120', '200'])
    plt.title('Size', fontsize=18)

    plt.subplot(133)
    arch_students = {
        'Transformer': ["vit_small_patch32_224", "xcit_small_24_p16_224", "pit_b_224", "xcit_large_24_p16_224"],
        'CNN': ["gluon_resnet34_v1b", "gluon_resnet101_v1c", "wide_resnet50_2", "ig_resnext101_32x16d"],
        'MLP': ["mixer_b16_224_miil", "resmlp_36_224", "resmlp_12_224"]}

    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    for arch in arch_students.keys():
        mean = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['mean_dist_delta'].values
        lower = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['q25_dist_delta'].values
        upper = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['q75_dist_delta'].values
        plt.plot(range(len(arch_students[arch])), mean, label=arch, marker=marker_dict[arch], color=colors[arch])
        plt.fill_between(range(len(arch_students[arch])), lower, upper, color=colors[arch], alpha=0.2)
    plt.xlabel('Student Model Size', fontsize=16)
    plt.xticks(range(4), ['Small', 'Medium', 'Large', 'XLarge'])
    plt.ylim()
    plt.title('Architecture', fontsize=18)
    plt.legend(loc='lower right')

    plt.suptitle('Impact of Different Student Features on th Distillation Delta', fontsize=20)
    fig.tight_layout()
    plt.savefig(f'images/student_feature_influence.pdf', bbox_inches='tight')
    plt.show()


def improvement_dist_plot(appr='KL+MT_Dist'):
    data = load_wandb_runs('2_distill_between_experts')
    data = data.dropna(subset=['knowledge_gain', 'knowledge_loss', 'pos_flips_delta'])
    data = data.loc[data['ts_diff'] != 0]
    data = data.loc[data['tag'] == appr]

    p_values = [2, 5, 20, 50, 100]
    transfer_rates = [data['knowledge_gain'].values*(data[f'top{p}%_improve'].values)/(data['knowledge_gain'].values*p) for p in p_values]
    transfer_rates = [100-(p*data['positive_flips'].values - data[f'top{p}%_improve'].values*data['knowledge_gain'].values)/data['positive_flips'].values  for p in p_values]
    means = [np.mean(t) for t in transfer_rates]
    upper = [np.quantile(t, 0.75) for t in transfer_rates]
    lower = [np.quantile(t, 0.25) for t in transfer_rates]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(len(p_values)), means)
    plt.fill_between(range(len(p_values)), lower, upper, alpha=0.3)
    plt.ylabel('Share of Transferred Positive Flips', fontsize=16)
    plt.xticks(range(len(p_values)), [f'Top-{p}%' for p in p_values])
    plt.xlabel('Classes Containing the Top-X% of the Positive Flips', fontsize=16)
    plt.show()


def transfer_rate_plot(appr='KL+MT_Dist'):
    data = load_wandb_runs('2_distill_between_experts')
    data = data.dropna(subset=['knowledge_gain', 'knowledge_loss', 'pos_flips_delta'])
    data = data.loc[data['ts_diff'] != 0]
    data = data.loc[data['tag'] == appr]

    shapes = ['o', 'v', 'X', 'P', 'v', 's', '<', 'p']
    marker_dict = {key: shapes[k] for k, key in enumerate(['transformer', 'cnn', 'mlp'])}
    colors = {'transformer': '#5387DD', 'cnn': '#DA4C4C', 'mlp': '#EDB732'}

    fig = plt.figure(figsize=(13, 4))
    plt.subplot(121)
    p_values = [2, 5, 20, 50, 100]
    for arch in marker_dict.keys():
        tmp = data.loc[data['student_type'] == arch]
        transfer_rates = [100-(p*tmp['positive_flips'].values - tmp[f'top{p}%_improve'].values*tmp['knowledge_gain'].values)/tmp['positive_flips'].values  for p in p_values]
        means = [np.mean(t) for t in transfer_rates]
        upper = [np.quantile(t, 0.75) for t in transfer_rates]
        lower = [np.quantile(t, 0.25) for t in transfer_rates]

        plt.plot(range(len(p_values)), means, marker=marker_dict[arch], color=colors[arch], label=arch_strings[arch])
        plt.fill_between(range(len(p_values)), lower, upper, alpha=0.1, color=colors[arch])
    plt.ylabel('Share of Transferred Pos. Flips', fontsize=16)
    plt.xticks(range(len(p_values)), [f'Top-{p}%' for p in p_values])
    plt.xlabel('Classes Containing the Top-X% of the Positive Flips', fontsize=16)
    plt.title('Transferred Flips for Top-X% Pos. Flip Classes', fontsize=18)
    plt.ylim(0, 101)

    plt.subplot(122)
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
        plt.plot(transfer_rates[arch]['params'], transfer_rates[arch]['mean'], marker=marker_dict[arch], color=colors[arch], label=arch_strings[arch])
        plt.fill_between(transfer_rates[arch]['params'], transfer_rates[arch]['lower'], transfer_rates[arch]['upper'], color=colors[arch], alpha=0.1)
    #plt.ylabel('Share of Transferred Positive Flips', fontsize=16)
    plt.xlabel('Student Parameters', fontsize=16)
    plt.title('Total Transferred Flips', fontsize=18)
    plt.xscale('log')
    plt.xticks([10, 15, 20, 40, 60, 80, 120, 200], ['10', '15', '20', '40', '60', '80', '120', '200'])
    plt.ylim(0,101)
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'images/transferred_flips.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    #get_correlation_heatmaps_teacher()
    #get_correlation_heatmaps()

    #teacher_influence_plot()

    #plot_arch_influence('KL+MT_Dist')
    #plot_performance_influence('KL+MT_Dist')
    #plot_size_influence('KL+MT_Dist')
    #student_feature_importenace()

    pos_dist_delta_2()
    #dist_deltas_plot()
    #improvement_dist_plot()
    #transfer_rate_plot()

import wandb
import json
import textwrap
import pandas as pd
import numpy as np
from mycolorpy import colorlist as mcp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib

from aggregate_results import load_wandb_runs
from dist_between_experts_plots import mscatter

ARCHITECTURES = {'transformer': 'Trafo', 'cnn': 'CNN', 'mlp': 'MLP'}
APPROACHES = {'KL_Dist': 'KL Distillation', 'XEKL_Dist': 'XE-KL Distillation',
                'CD_Dist': 'CD Distillation', 'CRD_Dist': 'CRD Distillation',
                'XEKL+MCL_Dist': 'XE-KL+MCL Distillation', 'KL+MT_Dist': 'KL+DP Distillation', 'KL+MT_u_Dist': 'KL+DP-U Distillation'}
TEACHERS = [211, 268, 234, 302, 209, 10, 152, 80, 36, 182, 310, 77, 12, 239, 151, 145, 232, 101, 291, 124]
STUDENTS = {'transformer': [41, 7, 5, 46, 26, 171],
                'cnn': [33, 131, 235, 132, 42, 130, 48],
                'mlp': [214, 2, 9, 77, 258, 160, 72]}


def pos_dist_delta_appendix():
    """Boxplot of the share of teachers improving the student for each distillation approach.

    :Returns: None
    """
    plt.rc('font', family='Times New Roman', size=19)
    data = pd.read_csv('students_leaderboard.csv', index_col=[0])
    data = data.loc[data['mean_dist_delta'] > -10]
    data = data.loc[[data['student_id'][i] not in [144, 261, 63, 139, 237, 302, 40, 299, 232, 285] for i in data.index]]

    appr_strings = {'XEKL_Dist': 'XE-KL Distillation', 'CD_Dist': 'CD Distillation', 'CRD_Dist': 'CRD Distillation', 'KL+MT_Dist': 'KL+DP Distillation'}
    wrapped_labels = ['\n'.join(textwrap.wrap(label, 12)) for label in appr_strings.values()]

    fig = plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-colorblind')
    data = data.loc[[data['approach'][i] in appr_strings.keys() for i in data.index]]
    pdd = data[['pos_dist_delta', 'approach']]
    pdd['Share of Teachers'] = pdd['pos_dist_delta']
    pdd['Distillation Approach'] = [appr_strings[app] for app in pdd['approach'].values]
    pdd['Category'] = ['KD' if app in ['KL Dist.', 'XE-KL Dist.'] else 'KD+CL' for app in pdd['Distillation Approach'].values]
    bp_vals = [pdd.loc[pdd["Distillation Approach"] == appr_strings[app]]['Share of Teachers'].values for app in appr_strings.keys()]
    plt.boxplot(bp_vals[:1], positions=range(1), widths=0.5, patch_artist=True,
                boxprops=dict(linewidth=3.0, facecolor='C0', color='black', alpha=0.8),
                medianprops=dict(linewidth=2.0, color='black'), whiskerprops=dict(linewidth=2.0),
                capprops=dict(linewidth=2.0), flierprops=dict(linewidth=2.0))
    plt.boxplot(bp_vals[1:3], positions=range(1, 3), widths=0.5, patch_artist=True,
                boxprops=dict(linewidth=3.0, facecolor='C1', color='black', alpha=0.8),
                medianprops=dict(linewidth=2.0, color='black'), whiskerprops=dict(linewidth=2.0),
                capprops=dict(linewidth=2.0), flierprops=dict(linewidth=2.0))
    plt.boxplot(bp_vals[3:], positions=range(3, 4), widths=0.5, patch_artist=True,
                boxprops=dict(linewidth=3.0, facecolor='C2', color='black', alpha=0.8),
                medianprops=dict(linewidth=2.0, color='black'), whiskerprops=dict(linewidth=2.0),
                capprops=dict(linewidth=2.0), flierprops=dict(linewidth=2.0))
    plt.axvline(0.5, color='black', alpha=0.3, linestyle='--', linewidth=2)
    plt.axvline(2.5, color='black', alpha=0.3, linestyle='--', linewidth=2)

    plt.xticks(range(4), wrapped_labels)
    plt.xlabel('')
    plt.ylabel('Knowledge Transfer Success Rate [%]     ', fontsize=20)

    fig.tight_layout()
    plt.savefig(f'images/dist_approaches_plot_appendix.pdf', bbox_inches='tight')
    plt.show()


def correlation_heatmap(runs, appr=None, mode='student'):
    """Correlation heatmap of model parameters and distillation metrics.

    :param runs: DataFrame of runs
    :param appr: Distillation approach
    :param mode: 'student', 'teacher' or 'ts' for teacher-student correlation

    :Returns: None
    """
    correlation = runs.corr(method='pearson')
    if mode == 'student':
        col_subset = ['student_params', 'student_acc', 'student_cnn', 'student_mlp', 'student_transformer']
        col_rename = {'student_params': 'Student Params.', 'student_acc': 'Student Acc.',
                      'student_cnn': 'CNN Student', 'student_mlp':'MLP Student', 'student_transformer': 'Trafo Student'}
        fig = plt.figure(figsize=(7, 5))
        plt.subplots_adjust(bottom=0.22, left=0.30)
    elif mode == 'teacher':
        col_subset = ['teacher_params', 'teacher_acc', 'teacher_cnn', 'teacher_mlp', 'teacher_transformer']
        col_rename = {'teacher_params': 'Teacher Params.', 'teacher_acc': 'Teacher Acc.',
                      'teacher_cnn': 'CNN Teacher', 'teacher_mlp':'MLP Teacher', 'teacher_transformer': 'Trafo Teacher'}
        fig = plt.figure(figsize=(7, 5))
        plt.subplots_adjust(bottom=0.22, left=0.30)
    else:
        col_subset = ['performance_diff', 'params_diff']
        col_subset += [f'{t_arch}-{s_arch}' for s_arch in ARCHITECTURES.keys() for t_arch in ARCHITECTURES.keys()]
        col_rename = {'performance_diff': 'Performance Diff.', 'params_diff': 'Params Diff.', 'cnn-cnn': 'CNN to CNN',
                      'cnn-mlp': 'CNN to MLP', 'cnn-transformer': 'CNN to Trafo', 'mlp-cnn': 'MLP to CNN', 'mlp-mlp':
                      'MLP to MLP', 'mlp-transformer': 'MLP to Trafo', 'transformer-cnn': 'Trafo to CNN', 'transformer-mlp': 'Trafo to MLP',
                      'transformer-transformer': 'Trafo to Trafo'}
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
        plt.title(APPROACHES[appr] + '\n', fontsize=18)
    fig.tight_layout()
    plt.savefig(f'images/{appr}_{mode}_corr_heatmap.pdf', bbox_inches='tight')
    plt.show()


def get_correlation_heatmaps_student():
    """Correlation heatmap of student parameters and distillation metrics."""
    data = load_wandb_runs('2_distill_between_experts')
    data = data.loc[data['dist_delta'] > -40]
    for approach in APPROACHES.keys():
        tmp = data.loc[(data['tag'] == approach)]
        for arch in ['transformer', 'cnn', 'mlp']:
            tmp[f'student_{arch}'] = [tmp['student_type'][i]==arch for i in tmp.index]
        correlation_heatmap(tmp, approach, 'student')
        print('next')


def get_correlation_heatmaps_teacher():
    """Correlation heatmap of teacher parameters and distillation metrics."""
    data = load_wandb_runs('2_distill_between_experts')
    data = data.loc[data['dist_delta'] > -40]
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


def student_feature_importance(appr='KL+MT_Dist'):
    """Plot feature importance of student models.

    :param appr: approach to plot feature importance for

    :Returns: None
    """
    plt.rc('font', family='Times New Roman', size=18)
    data = pd.read_csv('students_leaderboard.csv', index_col=[0])
    data = data.loc[data['approach'] == appr]

    shapes = ['o', 'v', 's', 'P', 'v', 's', '<', 'p']
    marker_dict = {key: shapes[k] for k, key in enumerate(['Trafo', 'CNN', 'MLP'])}
    colors = {'Trafo': '#8172B3', 'CNN': '#C44E52', 'MLP': '#CCB974'}

    fig = plt.figure(figsize=(15, 4))
    plt.style.use('seaborn-bright')
    plt.subplot(131)
    perf_students = {'Trafo': [['vit_base_patch16_224','vit_base_patch16_224_sam'], ['pit_xs_distilled_224', 'pit_xs_224']],
                     'CNN':[['wide_resnet50_2', 'legacy_seresnet152'], ['resnetv2_50x1_bit_distilled', 'gluon_resnet34_v1b']],
                     'MLP': [['mixer_b16_224_miil', 'mixer_b16_224'], ['resmlp_24_224', 'resmlp_24_distilled_224']]}
    performances = {'Trafo': [[84.53, 80.24],[79.31, 78.19]],
                     'CNN':[[81.46, 78.65], [82.80, 74.59]],
                     'MLP': [[82.30, 76.61], [79.39, 80.76]]}

    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    for arch in perf_students.keys():
        for p, pair in enumerate(perf_students[arch]):
            mean = np.array([data.loc[data['student_name'] == pair[i]]['mean_dist_delta'].values[0] for i in range(len(pair))])
            lower = np.array([data.loc[data['student_name'] == pair[i]]['q25_dist_delta'].values[0] for i in range(len(pair))])
            upper = np.array([data.loc[data['student_name'] == pair[i]]['q75_dist_delta'].values[0] for i in range(len(pair))])
            plt.plot(performances[arch][p], mean, label=arch, marker=marker_dict[arch], color=colors[arch], alpha=0.8)
            plt.fill_between(performances[arch][p], lower, upper, color=colors[arch], alpha=0.1)
    plt.xlabel('Student Accuracy', fontsize=20)
    plt.ylabel('Distillation Delta', fontsize=20)
    plt.title('a) Student Performance', fontsize=22)

    plt.subplot(132)
    size_students = {'Trafo': [['xcit_large_24_p16_224', 'pit_b_224', 'vit_relpos_medium_patch16_224'],
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
            plt.plot(size, mean, label=arch, marker=marker_dict[arch], color=colors[arch], alpha=0.8)
            plt.fill_between(size, lower, upper, color=colors[arch], alpha=0.1)
    plt.xlabel('Student Parameters', fontsize=20)
    plt.xscale('log')
    plt.minorticks_off()
    plt.xticks([5, 10, 20, 40, 80, 120, 200], ['5', '10', '20', '40', '80', '120', '200'])
    plt.title('b) Student Size', fontsize=22)

    plt.subplot(133)
    arch_students = {
        'Trafo': ["vit_small_patch32_224", "xcit_small_24_p16_224", "pit_b_224", "xcit_large_24_p16_224"],
        'CNN': ["gluon_resnet34_v1b", "gluon_resnet101_v1c", "wide_resnet50_2", "ig_resnext101_32x16d"],
        'MLP': ["mixer_b16_224_miil", "resmlp_36_224", "resmlp_12_224"]}

    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    for arch in arch_students.keys():
        mean = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['mean_dist_delta'].values
        lower = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['q25_dist_delta'].values
        upper = data.loc[[data['student_name'][i] in arch_students[arch] for i in data.index]]['q75_dist_delta'].values
        plt.plot(range(len(arch_students[arch])), mean, label=arch, marker=marker_dict[arch], color=colors[arch], alpha=0.8)
        plt.fill_between(range(len(arch_students[arch])), lower, upper, color=colors[arch], alpha=0.1)
    plt.xlabel('Student Model Size', fontsize=20)
    plt.xticks(range(4), ['Small', 'Medium', 'Large', 'XLarge'])
    plt.ylim()
    plt.title('c) Student Architecture', fontsize=22)
    plt.legend(loc='lower right')

    #plt.suptitle('Impact of Different Student Features on th Distillation Delta', fontsize=20)
    fig.tight_layout()
    plt.savefig(f'images/student_feature_influence.pdf', bbox_inches='tight')
    plt.show()


def gain_loss_plot(app1, app2=None):
    """Scatter plot of knowledge gain vs. knowledge loss for all students.

    :param app1: distillation approach 1
    :param app2: distillation approach 2 (optional)

    :Returns: None
    """
    plt.rc('font', family='Times New Roman', size=19)
    data = load_wandb_runs('2_distill_between_experts')
    data = data.loc[data['dist_delta'] > -20]
    data = data.loc[[data['student_id'][i] not in [144, 261, 63, 139, 237, 302, 40, 299, 232, 285] for i in data.index]]
    data = data.dropna(subset=['knowledge_gain', 'knowledge_loss', 'teacher_acc', 'student_params'])
    df1 = data.loc[data['tag'] == app1]
    df2 = data.loc[data['tag'] == app2] if app2 is not None else df1

    shapes = ['o', 'v', 'X', 'P', 'v', 's', '<', 'p']
    #shapes = ['$C$', '$M$', '$T$']

    keys = np.unique(df1['teacher_type'])
    marker_dict = {key:shapes[k] for k, key in enumerate(keys)}
    marker_handles = []
    for key in marker_dict.keys():
        marker_handles.append(mlines.Line2D([], [], color='grey', marker=marker_dict[key], linestyle='None',
                                            markersize=10, label=ARCHITECTURES[key]))

    cvals = [11, 22, 30, 44, 60]
    cmap = plt.get_cmap('plasma')

    x_max = np.max(np.concatenate((df1['knowledge_loss'].values, df2['knowledge_loss'].values))) + 0.2
    y_max = np.max(np.concatenate((df1['knowledge_gain'].values, df2['knowledge_gain'].values))) + 0.2
    x_max_1 = np.max(df1['knowledge_loss'].values) + 0.2
    y_max_1 = np.max(df1['knowledge_gain'].values) + 0.2
    x_max_2 = np.max(df2['knowledge_loss'].values) + 0.2
    y_max_2 = np.max(df2['knowledge_gain'].values) + 0.2

    plt.rcParams['font.size'] = 15
    labels_size = 18

    if app2 is not None:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7))
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        axes = [axes]

    for a, arch in enumerate(ARCHITECTURES.keys()):
        tmp = df1.loc[df1['student_type']==arch]
        gain, loss, z, color, markers = tmp['knowledge_gain'].values, tmp['knowledge_loss'].values, tmp['teacher_acc'].values, tmp['student_params'].values, tmp['teacher_type'].values
        marks = [marker_dict[m] for m in markers]
        z = (((z - np.min(z)) / (np.max(z) - np.min(z))) + 1.8) ** 5

        axes[0][a].plot([0, 50], [0, 50], color='red', alpha=0.5)
        scat = mscatter(loss, gain, z, ax=axes[0][a], c=color, cmap=cmap, alpha=0.8, m=marks, edgecolor='black', norm=matplotlib.colors.LogNorm())
        if a == 0:
            axes[0][a].annotate(APPROACHES[app1], xy=(0, 0.5), xytext=(-axes[0][a].yaxis.labelpad - 6, 0),
                                xycoords=axes[0][a].yaxis.label, textcoords='offset points',
                                fontsize=20, ha='center', va='center', rotation='vertical')
            axes[0][a].set_ylabel(f'Knowledge Gain', fontsize=19)
        if app2 is None:
            axes[0][a].set_xlabel('Knowledge Loss', fontsize=19)
        if a == 0:
            axes[0][a].legend(handles=marker_handles, loc='lower right')
            axes[0][a].set_yticks([1.0, 2.0, 3.0, 4.0], ['1.0', '2.0', '3.0', '4.0'])
        if a == 1:
            axes[0][a].set_yticks([1, 2, 3])
        axes[0][a].set_xlim((np.min(loss)-0.2, np.max(loss)+0.2))
        axes[0][a].set_ylim((np.min(gain)-0.5, np.max(gain)+0.5))
        axes[0][a].set_title(f'{ARCHITECTURES[arch]} Student Models', fontsize=20)
        #axes[0][a].minorticks_on()
        #axes[0][a].grid()

        if app2 is not None:
            tmp = df2.loc[df2['student_type'] == arch]
            gain, loss, z, color, markers = tmp['knowledge_gain'].values, tmp['knowledge_loss'].values, tmp['teacher_acc'].values, tmp['student_params'].values, tmp['teacher_type'].values
            marks = [marker_dict[m] for m in markers]
            z = (((z - np.min(z)) / (np.max(z) - np.min(z))) + 1.8) ** 5
            axes[1, a].plot([0, 50], [0, 50], color='red', alpha=0.5)
            scat = mscatter(loss, gain, z, ax=axes[1, a], c=color, cmap=cmap, alpha=0.8, m=marks, edgecolor='black', norm=matplotlib.colors.LogNorm())
            axes[1, a].set_xlim((np.min(loss)-0.2, np.max(loss)+0.2))
            axes[1, a].set_ylim((np.min(gain)-0.5, np.max(gain)+0.5))
            axes[1, a].set_xlabel('Knowledge Loss', fontsize=19)
            if a == 0:
                axes[1, a].annotate(APPROACHES[app2], xy=(0, 0.5), xytext=(-axes[1][a].yaxis.labelpad - 5, 0),
                                    xycoords=axes[1][a].yaxis.label, textcoords='offset points',
                                    fontsize=20, ha='right', va='center', rotation='vertical')
                axes[1, a].set_ylabel(f'Knowledge Gain', fontsize=19)
            #axes[1, a].minorticks_on()
            #axes[1, a].grid()

    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.76])
    cbar = plt.colorbar(scat, cax=cbar_ax, aspect=10, ticks=[10, 25, 50, 75, 100, 150, 200])
    cbar.ax.set_yticklabels(['10', '25', '50', '75', '100', '150', '200'])
    cbar.ax.minorticks_off()
    cbar.ax.get_yaxis().labelpad = 18
    cbar.set_label('Number of Student Parameters [M]', rotation=270, fontsize=20)

    plot_name = f'{app1}_{app2}_scatterplot.pdf' if app2 is not None else f'{app1}_scatterplot.pdf'
    plt.savefig(f'images/{plot_name}', bbox_inches='tight')
    plt.show()


def cont_distillation_gain_loss(student='pit_b_224', appr='Cont_MT'):
    """Plot the knowledge gain/loss and distillation delta for sequential continual distillation.

    :param student: Name of the student model
    :param appr: Distillation approach

    :Returns: None
    """
    plt.rc('font', family='Times New Roman', size=19)
    data = load_wandb_runs('3_continual_distillation', history=True)
    data = data.loc[(data['student_name']==student) & (data['tag']==appr)]
    data = data.loc[[data['contdist'][i]['curriculum'] == 'asc' for i in data.index]]
    k_gain = data['knowledge_gain_hist'].values[0]
    k_loss = data['knowledge_loss_hist'].values[0]
    dist_delta = data['dist_delta_hist'].values[0]
    x = range(len(k_gain))
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, k_gain, label='Knowledge Gain', color='C0', linewidth=3)
    plt.plot(x, k_loss, label='Knowledge Loss', color='C2', linewidth=3)
    plt.plot(x, dist_delta, label='Distillation Delta', color='C1', linewidth=3)
    plt.text(10, 0.65, 'ResMLP-36', horizontalalignment='center', fontsize=19)
    plt.text(30, 0.65, 'SWSL-ResNext', horizontalalignment='center', fontsize=19)
    plt.text(50, 0.65, 'VOLO-D2', horizontalalignment='center', fontsize=19)
    plt.axvline(20, color='black', alpha=0.3)
    plt.axvline(40, color='black', alpha=0.3)
    plt.xlabel('Distillation Epoch', fontsize=20)
    plt.ylabel('Knowledge Gain/Loss [%]', fontsize=20)
    plt.ylim(0.3, 3.2)
    plt.xlim(0, 60)
    plt.legend(loc='upper left', handlelength=0.5)
    fig.tight_layout()
    plt.savefig(f'images/cont_dist_gain_loss.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    #gain_loss_plot('KL_Dist', 'KL+MT_Dist')
    cont_distillation_gain_loss()

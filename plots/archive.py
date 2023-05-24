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

ARCHITECTURES = {'transformer': 'Transformer', 'cnn': 'CNN', 'mlp': 'MLP'}
APPROACHES = {'KL_Dist': 'KL Distillation', 'XEKL_Dist': 'XE-KL Distillation',
                'CD_Dist': 'CD Distillation', 'CRD_Dist': 'CRD Distillation',
                'XEKL+MCL_Dist': 'XE-KL+MCL Distillation', 'KL+MT_Dist': 'KL+DP Distillation', 'KL+MT_u_Dist': 'KL+DP-U Distillation'}
TEACHERS = [211, 268, 234, 302, 209, 10, 152, 80, 36, 182, 310, 77, 12, 239, 151, 145, 232, 101, 291, 124]
STUDENTS = {'transformer': [41, 7, 5, 46, 26, 171],
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


def gain_loss_plot_old(df1, df2, name1, name2):
    shapes = ['o', 'v', 'X', 'P', 'v', 's', '<', 'p']
    #shapes = ['$C$', '$M$', '$T$']

    keys = np.unique(df1['teacher_type'])
    marker_dict = {key:shapes[k] for k, key in enumerate(keys)}
    marker_handles = []
    for key in marker_dict.keys():
        marker_handles.append(mlines.Line2D([], [], color='grey', marker=marker_dict[key], linestyle='None',
                                            markersize=10, label=ARCHITECTURES[key]))

    cvals = [11, 22, 30, 44, 60] #[11, 22, 30, 44, 60, 86]
    #colors = ["#DA4C4C", "#EDB732", "#5BC5DB", "#5387DD", "#479A5F"] #["#DA4C4C", "#EDB732", "#5BC5DB", "#5387DD", "#A0C75C", "#479A5F"]
    #norm = plt.Normalize(min(cvals), max(cvals))
    #tuples = list(zip(map(norm, cvals), colors))
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    cmap = plt.get_cmap('plasma')

    x_max = np.max(np.concatenate((df1['knowledge_loss'].values, df2['knowledge_loss'].values))) + 0.2
    y_max = np.max(np.concatenate((df1['knowledge_gain'].values, df2['knowledge_gain'].values))) + 0.2

    plt.rcParams['font.size'] = 15
    labels_size = 18
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)

    gain, loss, z, color, markers = df1['knowledge_gain'].values, df1['knowledge_loss'].values, df1['teacher_acc'].values, df1['student_params'].values, df1['teacher_type'].values
    marks = [marker_dict[m] for m in markers]
    z = (((z - np.min(z)) / (np.max(z) - np.min(z))) + 1.8) ** 5

    plt.axline([0.6, 0.6], [2, 2], color='red', alpha=0.5)
    scat = mscatter(loss, gain, z, c=color, cmap=cmap, alpha=0.8, m=marks, edgecolor='black')
    plt.xlabel('Knowledge Loss', fontsize=16)
    plt.ylabel('Knowledge Gain', fontsize=16)
    plt.xlim((0, x_max))
    plt.ylim((0, y_max))
    plt.title(f'a) {name1}', fontsize=18, fontweight='bold')
    plt.minorticks_on()
    plt.grid()

    plt.subplot(122)
    gain, loss, z, color, markers = df2['knowledge_gain'].values, df2['knowledge_loss'].values, df2['teacher_acc'].values, df2['student_params'].values, df2['teacher_type'].values
    marks = [marker_dict[m] for m in markers]
    z = (((z - np.min(z)) / (np.max(z) - np.min(z))) + 1.8) ** 5
    plt.axline([0.6, 0.6], [2, 2], color='red', alpha=0.5)
    scat = mscatter(loss, gain, z, c=color, cmap=cmap, alpha=0.8, m=marks, edgecolor='black')
    plt.xlim((0, x_max))
    plt.ylim((0, y_max))
    plt.xlabel('Knowledge Loss', fontsize=16)
    plt.title(f'b) {name2}', fontsize=18, fontweight='bold')
    plt.legend(handles=marker_handles, loc='lower right')
    plt.minorticks_on()
    plt.grid()

    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.76])
    cbar = plt.colorbar(scat, cax=cbar_ax, aspect=10)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('Student Parameters', rotation=270, fontsize=16)

    plt.savefig(f'images/{name1}_{name2}_plot.pdf', bbox_inches='tight')
    plt.show()


def gain_loss_plot(app1, app2=None):
    data = load_wandb_runs('2_distill_between_experts')
    data = data.loc[data['dist_delta'] > -20]
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

    plt.rcParams['font.size'] = 15
    labels_size = 18

    if app2 is not None:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        axes = [axes]

    for a, arch in enumerate(ARCHITECTURES.keys()):
        tmp = df1.loc[df1['student_type']==arch]
        gain, loss, z, color, markers = tmp['knowledge_gain'].values, tmp['knowledge_loss'].values, tmp['teacher_acc'].values, tmp['student_params'].values, tmp['teacher_type'].values
        marks = [marker_dict[m] for m in markers]
        z = (((z - np.min(z)) / (np.max(z) - np.min(z))) + 1.8) ** 5

        axes[0][a].plot([0, 50], [0, 50], color='red', alpha=0.5)
        scat = mscatter(loss, gain, z, ax=axes[0][a], c=color, cmap=cmap, alpha=0.8, m=marks, edgecolor='black', norm=matplotlib.colors.LogNorm())
        if a == 0:
            axes[0][a].annotate(APPROACHES[app1], xy=(0, 0.5), xytext=(-axes[0][a].yaxis.labelpad - 5, 0),
                                xycoords=axes[0][a].yaxis.label, textcoords='offset points',
                                fontsize=18, ha='center', va='center', rotation='vertical', fontweight='bold')
            axes[0][a].set_ylabel(f'Knowledge Gain', fontsize=16)
        if app2 is None:
            axes[0][a].set_xlabel('Knowledge Loss', fontsize=16)
        if a == 0:
            axes[0][a].legend(handles=marker_handles, loc='upper left')
        axes[0][a].set_xlim((0, x_max))
        axes[0][a].set_ylim((0, y_max))
        axes[0][a].set_title(f'{ARCHITECTURES[arch]} Student Models', fontsize=18, fontweight='bold')
        axes[0][a].minorticks_on()
        axes[0][a].grid()

        if app2 is not None:
            tmp = df2.loc[df2['student_type'] == arch]
            gain, loss, z, color, markers = tmp['knowledge_gain'].values, tmp['knowledge_loss'].values, tmp['teacher_acc'].values, tmp['student_params'].values, tmp['teacher_type'].values
            marks = [marker_dict[m] for m in markers]
            z = (((z - np.min(z)) / (np.max(z) - np.min(z))) + 1.8) ** 5
            axes[1, a].plot([0, 50], [0, 50], color='red', alpha=0.5)
            scat = mscatter(loss, gain, z, ax=axes[1, a], c=color, cmap=cmap, alpha=0.8, m=marks, edgecolor='black', norm=matplotlib.colors.LogNorm())
            axes[1, a].set_xlim((0, x_max))
            axes[1, a].set_ylim((0, y_max))
            axes[1, a].set_xlabel('Knowledge Loss', fontsize=16)
            if a == 0:
                axes[1, a].annotate(APPROACHES[app2], xy=(0, 0.5), xytext=(-axes[1, a].yaxis.labelpad - 5, 0),
                                    xycoords=axes[1, a].yaxis.label, textcoords='offset points',
                                    fontsize=18, ha='right', va='center', rotation='vertical', fontweight='bold')
                axes[1, a].set_ylabel(f'Knowledge Gain', fontsize=16)
            axes[1, a].minorticks_on()
            axes[1, a].grid()

    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.76])
    cbar = plt.colorbar(scat, cax=cbar_ax, aspect=10, ticks=[10, 25, 50, 75, 100, 150, 200])
    cbar.ax.set_yticklabels(['10', '25', '50', '75', '100', '150', '200'])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('Student Parameters', rotation=270, fontsize=16)

    plot_name = f'{app1}_{app2}_scatterplot.pdf' if app2 is not None else f'{app1}_scatterplot.pdf'
    plt.savefig(f'images/{plot_name}', bbox_inches='tight')
    plt.show()


def student_heatmap(appr='KL+MT_Dist'):
    data = load_wandb_runs('2_distill_between_experts')
    data = data.loc[data['tag'] == appr]

    cmap = [sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
            sns.color_palette("ch:start=-.2,rot=.6", as_cmap=True),
            sns.color_palette("ch:start=2,rot=0", as_cmap=True)
            ]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

    axes[0].annotate(APPROACHES[appr], xy=(0, 0.5), xytext=(-axes[0].yaxis.labelpad - 5, 0),
                     xycoords=axes[0].yaxis.label, textcoords='offset points',
                     fontsize=18, ha='right', va='center', rotation='vertical', fontweight='bold')
    for a, arch in enumerate(['transformer', 'cnn', 'mlp']):
        tmp = data.loc[data['student_type'] == arch]
        students = STUDENTS[arch]

        dist_deltas = np.zeros((len(TEACHERS), len(students)))
        for t, teacher in enumerate(TEACHERS):
            for s, student in enumerate(students):
                dist_deltas[t][s] = np.mean(tmp.loc[(tmp['teacher_id'] == teacher) & (tmp['student_id'] == student)]['dist_delta'].values)

        lower = np.nanquantile(dist_deltas, .1)
        upper = np.nanquantile(dist_deltas, .9)

        cbar_kwargs = {'label': 'Distillation Delta'} if a == 2 else None
        ax = sns.heatmap(dist_deltas, cmap=cmap[a], yticklabels=TEACHERS, xticklabels=students, linewidth=0.5, cbar_kws=cbar_kwargs, vmin=lower, vmax=upper, ax=axes[a])
        axes[a].set_title(f'{ARCHITECTURES[arch]} Student Models', fontsize=18, fontweight='bold')
        if a == 0:
            axes[a].set_ylabel('Teacher Models', fontsize=16)
        axes[a].set_xlabel('Student Models', fontsize=16)

    fig.tight_layout()
    plt.savefig(f'images/{appr}_st_heatmap.pdf', bbox_inches='tight')
    plt.show()


def prediction_flips_sim_plot():
    """Plot the similarity of the classes with the largest shares of complementary knowledge.

    :Returns: None
    """
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
    plt.title('Class Similarity for the Top-2% Pos. Prediction Flips', fontsize=labels_size + 2)
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
    plt.title('Class Similarity for the Top-50% Pos. Prediction Flips', fontsize=labels_size + 2)
    plt.xlim((-24, 24))
    plt.ylim((0.49, 0.83))
    plt.minorticks_on()
    plt.grid(alpha=0.3)
    plt.legend()

    fig.tight_layout()
    # plt.savefig(f'images/prediction_flips_sim_plot.pdf', bbox_inches='tight')
    plt.show()


def plot_pos_dist_delta():
    """Boxplot of the share of teachers with positive distillation delta for each distillation approach and architecture.

    :Returns: None
    """
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
        mean = np.array(
            [np.mean(data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['pos_dist_delta'].values) for
             appr in appr_dict.keys()])
        lower = np.array([np.quantile(
            data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['pos_dist_delta'].values, 0.25)
                          for appr in appr_dict.keys()])
        upper = np.array([np.quantile(
            data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['pos_dist_delta'].values, 0.75)
                          for appr in appr_dict.keys()])
        plt.plot(appr_dict.values(), mean, label=ARCHITECTURES[arch], marker=marker_dict[arch], color=colors[arch])
        plt.fill_between(appr_dict.values(), upper, lower, color=colors[arch], alpha=0.1)
    # plt.xlabel('Distillation Approaches', fontsize=16)
    plt.xticks(list(appr_dict.values()), ['KL Dist.', 'XE-KL Dist.', 'XE-KL+MCL Dist.', 'KL+MT Dist.'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Share of Teachers with Positive Dist. Delta', fontsize=18)
    plt.legend(loc='upper left')

    plt.subplot(122)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    for arch in colors.keys():
        mean = np.array(
            [np.median(data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['mean_dist_delta'].values)
             for appr in appr_dict.keys()])
        lower = np.array([np.quantile(
            data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['mean_dist_delta'].values, 0.25)
                          for appr in appr_dict.keys()])
        upper = np.array([np.quantile(
            data.loc[(data['approach'] == appr) & (data['student_type'] == arch)]['mean_dist_delta'].values, 0.75)
                          for appr in appr_dict.keys()])
        plt.plot(appr_dict.values(), mean, label=ARCHITECTURES[arch], marker=marker_dict[arch], color=colors[arch])
        plt.fill_between(appr_dict.values(), lower, upper, color=colors[arch], alpha=0.1)
    # plt.xlabel('Distillation Approaches', fontsize=16)
    plt.xticks(list(appr_dict.values()), ['KL Dist.', 'XE-KL Dist.', 'XE-KL+MCL Dist.', 'KL+MT Dist.'])
    plt.xticks(rotation=45, ha='right')
    plt.ylim([-1.5, 1.5])
    plt.title('Median Distillation Delta', fontsize=18)

    fig.tight_layout()
    plt.savefig(f'images/dist_approaches_plot.pdf', bbox_inches='tight')
    plt.show()


def dist_deltas_plot():
    """Plot of the distillation deltas for different distillation approaches.

    :Returns: None
    """
    data = load_wandb_runs('2_distill_between_experts')
    data = data.loc[data['dist_delta'] > -20]
    data = data.loc[[data['student_id'][i] not in [144, 261, 63, 139, 237, 302, 40, 299, 232, 285] for i in data.index]]
    data = data.loc[[data['data'][i]['dataset'] == 'imagenet_subset' for i in data.index]]

    plt.style.use('seaborn-bright')
    fig = plt.figure(figsize=(10, 5))
    appr_strings = {'KL_Dist': 'KL Dist.', 'XEKL_Dist': 'XE-KL Dist.', 'XEKL+MCL_Dist': 'XE-KL+MCL Dist.',
                    'KL+MT_Dist': 'KL+MT Dist.'}
    colors = {'Trafo': '#8172B3', 'CNN': '#C44E52', 'MLP': '#CCB974'}
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    data = data.loc[[data['tag'][i] in appr_strings.keys() for i in data.index]]
    dist_deltas = data[['student_type', 'tag', 'dist_delta']]
    dist_deltas['Student Arch.'] = [ARCHITECTURES[arch] for arch in dist_deltas['student_type'].values]
    dist_deltas['approach'] = [appr_strings[appr] for appr in dist_deltas['tag'].values]
    sns.violinplot(data=dist_deltas, x='approach', y='dist_delta', hue='Student Arch.', width=1, palette=colors,
                   order=list(appr_strings.values()))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('')
    plt.ylabel('Distillation Delta', fontsize=16)
    plt.title('Distillation Delta by Distillation Approach', fontsize=18)
    plt.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(f'images/dist_deltas_plot.pdf', bbox_inches='tight')
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

    plt.suptitle(APPROACHES[appr], fontsize=20)
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

    plt.suptitle(APPROACHES[appr], fontsize=20)
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

    plt.suptitle(APPROACHES[appr], fontsize=20)
    fig.tight_layout()
    plt.savefig(f'images/student_size_influence.pdf', bbox_inches='tight')
    plt.show()


def improvement_dist_plot(appr='KL+MT_Dist'):
    """Plot of the transfer rate for the classes with the highest shares of positive flips.

    :param appr: The distillation approach to plot.

    :Returns: None
    """
    data = load_wandb_runs('2_distill_between_experts')
    data = data.dropna(subset=['knowledge_gain', 'knowledge_loss', 'pos_flips_delta'])
    data = data.loc[data['ts_diff'] != 0]
    data = data.loc[data['tag'] == appr]

    p_values = [2, 5, 20, 50, 100]
    transfer_rates = [
        data['knowledge_gain'].values * (data[f'top{p}%_improve'].values) / (data['knowledge_gain'].values * p) for p in
        p_values]
    transfer_rates = [
        100 - (p * data['positive_flips'].values - data[f'top{p}%_improve'].values * data['knowledge_gain'].values) /
        data['positive_flips'].values for p in p_values]
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


if __name__ == "__main__":
    student_heatmap()
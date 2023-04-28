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


def gain_loss_plot_old(df1, df2, name1, name2):
    shapes = ['o', 'v', 'X', 'P', 'v', 's', '<', 'p']
    #shapes = ['$C$', '$M$', '$T$']

    keys = np.unique(df1['teacher_type'])
    marker_dict = {key:shapes[k] for k, key in enumerate(keys)}
    marker_handles = []
    for key in marker_dict.keys():
        marker_handles.append(mlines.Line2D([], [], color='grey', marker=marker_dict[key], linestyle='None',
                              markersize=10, label=arch_strings[key]))

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
                              markersize=10, label=arch_strings[key]))

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

    for a, arch in enumerate(arch_strings.keys()):
        tmp = df1.loc[df1['student_type']==arch]
        gain, loss, z, color, markers = tmp['knowledge_gain'].values, tmp['knowledge_loss'].values, tmp['teacher_acc'].values, tmp['student_params'].values, tmp['teacher_type'].values
        marks = [marker_dict[m] for m in markers]
        z = (((z - np.min(z)) / (np.max(z) - np.min(z))) + 1.8) ** 5

        axes[0][a].plot([0, 50], [0, 50], color='red', alpha=0.5)
        scat = mscatter(loss, gain, z, ax=axes[0][a], c=color, cmap=cmap, alpha=0.8, m=marks, edgecolor='black', norm=matplotlib.colors.LogNorm())
        if a == 0:
            axes[0][a].annotate(appr_strings[app1], xy=(0, 0.5), xytext=(-axes[0][a].yaxis.labelpad - 5, 0),
                xycoords=axes[0][a].yaxis.label, textcoords='offset points',
                fontsize=18, ha='center', va='center', rotation='vertical', fontweight='bold')
            axes[0][a].set_ylabel(f'Knowledge Gain', fontsize=16)
        if app2 is None:
            axes[0][a].set_xlabel('Knowledge Loss', fontsize=16)
        if a == 0:
            axes[0][a].legend(handles=marker_handles, loc='upper left')
        axes[0][a].set_xlim((0, x_max))
        axes[0][a].set_ylim((0, y_max))
        axes[0][a].set_title(f'{arch_strings[arch]} Student Models', fontsize=18, fontweight='bold')
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
                axes[1, a].annotate(appr_strings[app2], xy=(0, 0.5), xytext=(-axes[1, a].yaxis.labelpad - 5, 0),
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

    axes[0].annotate(appr_strings[appr], xy=(0, 0.5), xytext=(-axes[0].yaxis.labelpad - 5, 0),
                        xycoords=axes[0].yaxis.label, textcoords='offset points',
                        fontsize=18, ha='right', va='center', rotation='vertical', fontweight='bold')
    for a, arch in enumerate(['transformer', 'cnn', 'mlp']):
        tmp = data.loc[data['student_type'] == arch]
        students = all_students[arch]

        dist_deltas = np.zeros((len(teachers), len(students)))
        for t, teacher in enumerate(teachers):
            for s, student in enumerate(students):
                dist_deltas[t][s] = np.mean(tmp.loc[(tmp['teacher_id'] == teacher) & (tmp['student_id'] == student)]['dist_delta'].values)

        lower = np.nanquantile(dist_deltas, .1)
        upper = np.nanquantile(dist_deltas, .9)

        cbar_kwargs = {'label': 'Distillation Delta'} if a == 2 else None
        ax = sns.heatmap(dist_deltas, cmap=cmap[a], yticklabels=teachers, xticklabels=students, linewidth=0.5, cbar_kws=cbar_kwargs, vmin=lower, vmax=upper, ax=axes[a])
        axes[a].set_title(f'{arch_strings[arch]} Student Models', fontsize=18, fontweight='bold')
        if a == 0:
            axes[a].set_ylabel('Teacher Models', fontsize=16)
        axes[a].set_xlabel('Student Models', fontsize=16)

    fig.tight_layout()
    plt.savefig(f'images/{appr}_st_heatmap.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    student_heatmap()
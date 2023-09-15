import json
import textwrap
import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from aggregate_results import load_wandb_runs

plt.rc('font', family='Times New Roman', size=14)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def dist_delta_line_plot_v3(dataset='infograph'):
    plt.rc('font', family='Times New Roman', size=19)

    try:
        data = pd.read_csv(f'distillation_results_raw_{dataset}.csv')
    except FileNotFoundError:
        data = load_wandb_runs('2_distill_between_experts')
        all_students = {'transformer': [41, 5, 26],  # [41, 7, 5, 46, 26, 171],
                        'cnn': [131, 130, 40],  # [33, 131, 235, 132, 42, 48, 130],
                        'mlp': [2, 160]}  # [214, 2, 9, 77, 258, 72]} #160
        all_teachers = [239, 77, 302,
                        234]  # [211, 268, 182, 124, 310, 10, 302, 209, 152, 234, 145, 80, 36, 12, 239, 151, 101, 232, 291, 77] #77
        data = data.loc[[data['student_id'][i] in all_students[data['student_type'][i]] for i in data.index]]
        data = data.loc[[data['teacher_id'][i] in all_teachers for i in data.index]]
        dataset = dataset if dataset != 'infograph' else 'imagenet_subset'
        data = data.loc[[data['data'][i]['dataset'] == dataset for i in data.index]]
        if 'imagenet' not in dataset:
            data = data.loc[data['freeze'] == True]
        if dataset == 'infograph':
            data = data.loc[data['teacher_pretrain'] == 'infograph']
        data.to_csv(f'distillation_results_raw_{dataset}.csv')
    # drop nan in col dist_deltas
    data = data.dropna(subset=['dist_delta'])
    if dataset == 'infograph':
        appr_strings = {'KL_Dist': 'KL Distillation Transfer',
                        #'XEKL_Dist': 'XE-KL Distillation',
                        'XEKL+MCL_Dist': 'XE-KL-Dist. + MCL Transfer',
                        'KL+MT_Dist': 'KL-Dist. + DP Transfer'
                        }
    elif dataset == 'labelsmooth':
        appr_strings = {#'KL_Dist': 'KL Distillation Transfer',
                        'XEKL_Dist': 'XE-KL Distillation',
                        #'XEKL+MCL_Dist': 'XE-KL-Dist. + MCL Transfer',
                        'KL+MT_Dist': 'KL-Dist. + DP Transfer'
                        }
    else:
        appr_strings = {'KL_Dist': 'KL Distillation Transfer',
                        #'XEKL_Dist': 'XE-KL Distillation',
                        'XEKL+MCL_Dist': 'XE-KL-Dist. + MCL Transfer',
                        'KL+MT_Dist': 'KL-Dist. + DP Transfer'
                        }
    dist_deltas = {appr: [] for appr in appr_strings.keys()}
    markers = ['o', 's', 'D', '^']
    nbins = 8
    pp = 0.75

    students = data['student_id'].unique()
    success_rate = {appr: [] for appr in appr_strings.keys()}

    data['delta_acc_bin'] = pd.cut(data['ts_diff'], bins=nbins)
    # sort data by delta_acc_bin
    data = data.sort_values(by=['delta_acc_bin'])
    for appr in appr_strings.keys():
        if appr == 'XEKL_Dist' and dataset == 'labelsmooth':
            data_appr = data.loc[data['tag'] == 'XEKL_Dist']
        else:
            data_appr = data.loc[data['tag'] == appr]
        for student in students:
            tmp = data_appr.loc[data_appr['student_id'] == student]
            if len(tmp) == 0:
                success_rate[appr].append(0)
            else:
                success_rate[appr].append(np.mean(tmp['dist_delta'].values > 0))
        for bin in data['delta_acc_bin'].unique():
            tmp = data_appr.loc[data_appr['delta_acc_bin'] == bin]
            if len(tmp) == 0:
                dist_deltas[appr].append(0)
            else:
                quantile = np.quantile(tmp['dist_delta'].values, pp)
                dds = tmp.loc[tmp['dist_delta'] >= quantile]['dist_delta'].values
                dist_deltas[appr].append(np.mean(dds))
                #dist_deltas[appr].append(np.median(tmp['dist_delta'].values))

    # get the max and min dist delta for all approaches
    max_dist_delta = np.max([dist_deltas[appr] for appr in dist_deltas.keys()])
    min_dist_delta = np.min([dist_deltas[appr] for appr in dist_deltas.keys()])
    bin_centers = [bin.mid for bin in data['delta_acc_bin'].unique()]
    bin_centers = np.sort(bin_centers)

    print('Success Rate')
    for appr in appr_strings.keys():
        print(f'{appr}: {np.median(success_rate[appr])}')

    fig, axes = plt.subplots(1)
    tag_order = ['KL_Dist', 'XEKL+MCL_Dist', 'KL+MT_Dist']
    if dataset == 'labelsmooth':
        colors = ['crimson', 'darkorange']
        alphas = [0.6, 1]
        markers = ['o', 's']
    else:
        colors = ['crimson', 'dodgerblue', 'darkorange'] # ['crimson', 'limegreen', 'dodgerblue', 'darkorange']
        alphas = [0.6, 0.6, 1]
        markers = ['-o', '-D', '-^']
    axes.hlines(0, -100, 100, colors='gray', linestyle='--')
    axes.vlines(0, -100, 100, colors='gray', linestyle='--')
    for i, tag in enumerate(appr_strings.keys()):
        axes.plot(bin_centers, dist_deltas[tag], markers[i], color=colors[i], label=appr_strings[tag], alpha=alphas[i], linewidth=4, markersize=10)
        # axes[1].plot(xb[tag], ddsucc[tag], '-o', label=tag)
        # ax.errorbar(xb[tag], ddb[tag], yerr=dds[tag], label=tag)
    axes.set_xlim(np.min(bin_centers) - 0.5, np.max(bin_centers) + 0.5)
    axes.set_ylim(min_dist_delta - 0.5, max_dist_delta + 0.5)
    # ax.set_yscale('log')
    axes.tick_params(axis='both', which='major', labelsize=22)
    axes.tick_params(axis='both', which='minor', labelsize=22)
    #axes.set_title('Knowledge Transfer on CUB', fontsize=22)
    if not dataset == 'infograph' and not dataset == 'labelsmooth' and not dataset=='imagenet_subset':
        axes.set_ylim(-7.5, 5.2)
    axes.set_xlabel('Performance Difference of Teacher and Student', fontsize=22)
    axes.set_ylabel('Knowledge Transfer Delta', fontsize=22)
    from matplotlib.lines import Line2D
    if dataset == 'infograph':
        custom_lines = [Line2D([0], [0], color='crimson', lw=2, marker='o', markersize=8),
                        #Line2D([0], [0], color='limegreen', lw=2, marker='s', markersize=8),
                        Line2D([0], [0], color='dodgerblue', lw=2, marker='D', markersize=8),
                        Line2D([0], [0], color='darkorange', lw=2, marker='^', markersize=8)]
        axes.legend(custom_lines, ['KL-Dist. Transfer',
                                   #'XE-KL-Dist. Transfer',
                                   'XE-KL-Dist. + MCL Transfer',
                                   'KL-Dist. + DP Transfer'],
                    fontsize=18, handlelength=0.8, loc='lower right')
    elif dataset == 'labelsmooth':
        custom_lines = [Line2D([0], [0], color='crimson', lw=4),
                        Line2D([0], [0], color='darkorange', lw=4)]
        axes.legend(custom_lines, ['XE-KL-Dist. Transfer (LS)', 'KL-Dist. + DP Transfer'],
                    fontsize=18, handlelength=0.5)
    else:
        custom_lines = [Line2D([0], [0], color='crimson', lw=2, marker='o', markersize=8),
                        #Line2D([0], [0], color='limegreen', lw=2, marker='s', markersize=8),
                        Line2D([0], [0], color='dodgerblue', lw=2, marker='D', markersize=8),
                        Line2D([0], [0], color='darkorange', lw=2, marker='^', markersize=8)]
        axes.legend(custom_lines, ['KL-Dist. Transfer',
                                   #'XE-KL-Dist. Transfer',
                                   'XE-KL-Dist. + MCL Transfer',
                                   'KL-Dist. + DP Transfer'],
                    fontsize=18, handlelength=0.8)#, loc='lower right')
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    #fig.savefig(f'images/transfer_delta_{dataset}_{pp}.png', dpi=300)
    plt.savefig(f'images/transfer_delta_{dataset}_{pp}.pdf', bbox_inches='tight')
    plt.show()


def dist_delta_line_plot_v2():
    f = pd.read_csv('distillation_results_raw.csv')
    # %%
    tags = f['tag']
    diffs = f['ts_diff']
    dist_deltas = f['dist_delta']

    # for pp in [25, 50, 75, 95]:
    pp = 75
    x, dd = {}, {}
    for tag, diff, dist_delta in zip(tags, diffs, dist_deltas):
        if 'CRD' not in tag and 'CD' not in tag and '_u_' not in tag and 'XEKL_Dist' not in tag:
        #if 'CRD' not in tag and 'CD' not in tag and 'XEKL_Dist' not in tag:
            if tag not in x:
                x[tag] = []
                dd[tag] = []
            x[tag].append(diff)
            dd[tag].append(dist_delta)

    bins = np.linspace(-10, 13, 10)
    xb, ddb, dds, ddmax, ddsucc = {}, {}, {}, {}, {}
    for tag in x.keys():
        idcs = np.argsort(x[tag])
        x[tag] = np.array(x[tag])[idcs]
        dd[tag] = np.array(dd[tag])[idcs]
        if tag not in xb:
            xb[tag] = []
            ddb[tag] = []
            dds[tag] = []
            ddmax[tag] = []
            ddsucc[tag] = []
        bidcs = np.sum(x[tag].reshape(-1, 1) > bins.reshape(1, -1), axis=-1)
        for bid in range(len(bins)):
            ib = np.where(bidcs == bid)[0]
            if len(ib):
                cm = dd[tag][ib] > 0.05
                m = np.mean(dd[tag][ib])
                s = np.std(dd[tag][ib])
                perc = np.percentile(dd[tag][ib], pp)
                ddb[tag].append(m)
                mperc = np.mean(dd[tag][ib][dd[tag][ib] > perc])
                ddmax[tag].append(mperc)
                dds[tag].append(s)
                ddsucc[tag].append(np.mean(cm))
                # ddb[tag].append(np.mean(cm * dd[tag][ib]))
                xb[tag].append(bins[bid])

    for tag in x.keys():
        x[tag] = np.array(x[tag])
        ddb[tag] = np.array(ddb[tag])
        dds[tag] = np.array(dds[tag])

    yuse = ddmax
    fig, axes = plt.subplots(1)
    tag_order = ['KL_Dist', 'XEKL+MCL_Dist', 'KL+MT_Dist']
    colors = ['crimson', 'dodgerblue', 'darkorange']
    alphas = [0.6, 0.6, 1]
    axes.hlines(0, np.min(bins) - 0.5, np.max(bins) + 0.5, colors='gray', linestyle='--')
    axes.vlines(0, np.min(yuse['KL_Dist']) - 0.3, np.max(yuse['KL_Dist']) + 0.5, colors='gray', linestyle='--')
    for i, tag in enumerate(tag_order):
        axes.plot(xb[tag], yuse[tag], '-o', color=colors[i], label=tag, alpha=alphas[i], linewidth=4, markersize=10)
        # axes[1].plot(xb[tag], ddsucc[tag], '-o', label=tag)
        # ax.errorbar(xb[tag], ddb[tag], yerr=dds[tag], label=tag)
    axes.set_xlim(np.min(bins) - 0.5, np.max(bins) + 0.5)
    axes.set_ylim(np.min(yuse['KL_Dist']) - 0.5, np.max(yuse['KL_Dist']) + 0.5)
    # ax.set_yscale('log')
    axes.tick_params(axis='both', which='major', labelsize=22)
    axes.tick_params(axis='both', which='minor', labelsize=22)
    axes.set_xlabel('Performance Difference of Teacher and Student', fontsize=22)
    axes.set_ylabel('Knowledge Transfer Delta', fontsize=22)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='crimson', lw=4),
                    Line2D([0], [0], color='dodgerblue', lw=4),
                    Line2D([0], [0], color='darkorange', lw=4)]
    axes.legend(custom_lines, ['KL-Dist. Transfer', 'XE-KL-Dist. + MCL Transfer', 'KL-Dist. + DP Transfer'],
                fontsize=18, handlelength=0.5)
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    #fig.savefig(f'transfer_delta_cub_{pp}.png', dpi=300)
    plt.savefig(f'images/transfer_delta_{pp}.pdf', bbox_inches='tight')
    plt.show()


def dist_delta_line_plot():
    plt.rc('font', family='Times New Roman', size=19)
    data = load_wandb_runs('2_distill_between_experts')
    #data = data.dropna(subset=['dist_delta', 'ts_diff'])
    all_students = {'transformer': [41, 5, 26], # [41, 7, 5, 46, 26, 171],
                    'cnn': [131, 130, 40], # [33, 131, 235, 132, 42, 48, 130],
                    'mlp': [2, 160]} #[214, 2, 9, 77, 258, 72]} #160
    all_teachers = [239, 77, 302, 234] #[211, 268, 182, 124, 310, 10, 302, 209, 152, 234, 145, 80, 36, 12, 239, 151, 101, 232, 291, 77] #77
    data = data.loc[[data['student_id'][i] in all_students[data['student_type'][i]] for i in data.index]]
    data = data.loc[[data['teacher_id'][i] in all_teachers for i in data.index]]
    #data = data.loc[data['dist_delta'] > -20]
    data = data.loc[[data['data'][i]['dataset'] == 'cars' for i in data.index]]
    data = data.loc[data['freeze'] == True]
    #data = data.loc[data['teacher_pretrain'] == 'CUB']
    dist_deltas = {}
    data.to_csv('distillation_results_raw_cars.csv')
    appr_strings = {'KL_Dist': 'KL Distillation Transfer', #'XEKL_Dist': 'XE-KL Distillation',
                    'XEKL+MCL_Dist': 'XE-KL Dist. + MCL Transfer', 'KL+MT_Dist': 'KL Dist. + DP Transfer'}
    markers = ['o', 's', 'D', '^']
    for appr in appr_strings.keys():
        # create bins for the perfromance different of the teacher and student
        # calculate the mean and std of the feature dist delta for each bin
        data_appr = data.loc[data['mode'] == appr]
        print(f'{appr}: {len(data_appr)}')
        data_appr['delta_acc_bin'] = pd.cut(data_appr['ts_diff'], bins=5)
        #tmp = data_appr.groupby('delta_acc_bin')['dist_delta'].agg(['mean', 'std'])
        tmp = data_appr.groupby('delta_acc_bin')['dist_delta'].agg([np.median, 'std'])
        if len(tmp) < 1:
            dist_deltas[appr] = None
        else:
            tmp['x_vals'] = [tmp.index[i].mid for i in range(len(tmp.index))]
            tmp['approach'] = appr
            dist_deltas[appr] = tmp

    # append all dataframes in dist_deltas
    plot_data = pd.concat([dist_deltas[appr] for appr in appr_strings.keys()])
    #plot_data.to_csv('distillation_results_plot.csv')
    fig = plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-colorblind')
    plt.axvline(0, color='black', alpha=0.3, linestyle='--', linewidth=1)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--', linewidth=1)
    for a, appr in enumerate(appr_strings.keys()):
        plt.plot(dist_deltas[appr]['x_vals'], dist_deltas[appr]['median'], label=appr_strings[appr], linewidth=2,
                 markersize=8, marker=markers[a])
        data_appr = data.loc[data['tag'] == appr]
        #for i in data_appr.index:
        #    plt.scatter(data_appr['ts_diff'][i], data_appr['dist_delta'][i], color=f'C{a}', alpha=0.2, s=10)
        #plt.fill_between(dist_deltas[appr]['x_vals'], dist_deltas[appr]['median'] - dist_deltas[appr]['std'],
        #                 dist_deltas[appr]['median'] + dist_deltas[appr]['std'], alpha=0.2)
    #plt.xlim(-10, 10)
    #plt.ylim(-4, 2)
    plt.xlabel('Performance Difference of Teacher and Student')
    plt.ylabel('Transfer Delta [pp]')
    plt.legend()

    fig.tight_layout()
    plt.show()



def sim_heatmap():
    # highest similarity
    # sim = np.load('sims/sim_densenet121_repvgg_b3_top20p.npy')
    # sim = np.load('sims/sim_mobilenetv3_small_050_ig_resnext101_32x8d.npy')
    # sim = np.load('sims/sim_spnasnet_100_mobilenetv3_rw.npy')

    # lowest similarity
    sim = np.load('sims/sim_gluon_resnet34_v1b_resnetv2_152x2_bit_teacher_top5p.npy')
    #sim = np.load('sims/sim_wide_resnet101_2_jx_nest_base.npy')

    sim = (sim - 0.61)/0.61
    high_sim = sim - np.diag(np.diag(sim))
    # get the absolute max value
    max_val = np.max(np.abs(high_sim))

    # plot two heatmaps
    plt.xticks([])
    plt.yticks([])
    plt.imshow(high_sim, cmap='coolwarm', vmin=-max_val, vmax=max_val)
    plt.tight_layout()
    plt.savefig(f'sim_heatmap_5p.pdf', bbox_inches='tight')
    plt.show()


def unsup_v_sup():
    data = pd.read_csv('distillation_results_raw.csv')
    sup = data.loc[data['tag'] == 'KL+MT_Dist']
    sup['run'] = [f'{sup["teacher_name"][i]}-{sup["student_name"][i]}' for i in sup.index]
    unsup = data.loc[data['tag'] == 'KL+MT_u_Dist']
    unsup['run'] = [f'{unsup["teacher_name"][i]}-{unsup["student_name"][i]}' for i in unsup.index]

    assert len(np.unique(sup['run'].values)) == len(np.unique(unsup['run'].values))
    runs = np.unique(sup['run'].values)
    data = pd.DataFrame(columns=['run', 'sup_delta', 'unsup_delta', 'sup_unsup_diff', 'unsup_better', 'student_type', 'student_acc', 'student_params', 'teacher_acc', 'teacher_params', 'teacher_type', 'ts_diff', 'ts_params_diff'])
    for run in runs:
        sup_run = sup.loc[sup['run'] == run]
        unsup_run = unsup.loc[unsup['run'] == run]
        assert len(sup_run) == len(unsup_run) == 1
        data = data.append({'run': run, 'sup_delta': sup_run['dist_delta'].values[0],
                            'unsup_delta': unsup_run['dist_delta'].values[0],
                            'sup_unsup_diff': sup_run['dist_delta'].values[0] - unsup_run['dist_delta'].values[0],
                            'unsup_better': float(unsup_run['dist_delta'].values[0] > sup_run['dist_delta'].values[0]),  # True if unsupervised is better than supervised, False otherwise
                            'student_type': sup_run['student_type'].values[0], 'student_acc': sup_run['student_acc'].values[0], 'student_params': sup_run['student_params'].values[0], 'teacher_acc': sup_run['teacher_acc'].values[0], 'teacher_params': sup_run['teacher_params'].values[0], 'teacher_type': sup_run['teacher_type'].values[0], 'ts_diff': sup_run['ts_diff'].values[0], 'ts_params_diff': sup_run['ts_params_diff'].values[0]}, ignore_index=True)

    # replace the features teacher_type and student_type with three binary features
    for type in ['transformer', 'mlp', 'cnn']:
        data[f'student_{type}'] = [1 if data['student_type'][i] == type else 0 for i in data.index]
        data[f'teacher_{type}'] = [1 if data['teacher_type'][i] == type else 0 for i in data.index]

    # plot correlation heatmap of the features with the unsupervised delta (only show the correlation between the first 4 cols with the rest)
    corr = data.corr()
    corr = corr.iloc[:4, 4:]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')
    plt.tight_layout()
    plt.show()
    print(data.corr())


def cont_distillation_gain_loss_v1(student='xcit_large_24_p16_224', appr='Cont_MT'):
    """Plot the knowledge gain/loss and distillation delta for sequential continual distillation.

    :param student: Name of the student model
    :param appr: Distillation approach

    :Returns: None
    """
    plt.rc('font', family='Times New Roman', size=19)
    data = load_wandb_runs('3_continual_distillation', history=True)
    data = data.loc[(data['student_name']==student) & (data['tag']==appr)]
    data = data.loc[[data['data'][i]['dataset'] == 'imagenet_subset' for i in data.index]]
    asc = data.loc[[data['contdist'][i]['curriculum'] == 'asc' for i in data.index]]
    desc = data.loc[[data['contdist'][i]['curriculum'] == 'desc' for i in data.index]]
    dist_delta_asc = asc['dist_delta_hist'].values[0]
    dist_delta_desc = desc['dist_delta_hist'].values[0]
    x = range(len(dist_delta_desc))
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(8, 5))
    plt.axhline(0, color='black', alpha=0.3)
    plt.plot(x, dist_delta_desc, label='Descending Curriculum', color='C0', linewidth=3)
    plt.plot(x, dist_delta_asc, label='Ascending Curriculum', color='C1', linewidth=3)
    plt.text(10, -0.20, 'Teacher 1', horizontalalignment='center', fontsize=19)
    plt.text(30, 0.20, 'Teacher 2', horizontalalignment='center', fontsize=19)
    plt.text(50, 0.40, 'Teacher 3', horizontalalignment='center', fontsize=19)
    plt.axvline(20, color='black', alpha=0.3)
    plt.axvline(40, color='black', alpha=0.3)
    plt.xlabel('Transfer Epoch', fontsize=20)
    plt.ylabel('Transfer Delta [pp]', fontsize=20)
    plt.ylim(-0.7, 0.9)
    plt.xlim(0, 60)
    plt.legend(loc='lower right', handlelength=0.5)
    fig.tight_layout()
    plt.savefig(f'images/cont_dist_xcit_large.pdf', bbox_inches='tight')
    plt.show()

def cont_distillation_gain_loss_v2(student='pit_b_224', appr='Cont_MT'):
    """Plot the knowledge gain/loss and distillation delta for sequential continual distillation.

    :param student: Name of the student model
    :param appr: Distillation approach

    :Returns: None
    """
    plt.rc('font', family='Times New Roman', size=19)
    data = load_wandb_runs('3_continual_distillation', history=True)
    data = data.loc[(data['student_name']==student) & (data['tag']==appr)]
    data = data.loc[[data['contdist'][i]['curriculum'] == 'desc' for i in data.index]]
    k_gain = data['knowledge_gain_hist'].values[0]
    k_loss = data['knowledge_loss_hist'].values[0]
    dist_delta = data['dist_delta_hist'].values[0]
    x = range(len(k_gain))
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, k_gain, label='Knowledge Gain', color='C0', linewidth=3)
    plt.plot(x, k_loss, label='Knowledge Loss', color='C2', linewidth=3)
    plt.plot(x, dist_delta, label='Transfer Delta', color='C1', linewidth=3)
    plt.text(10, 0.45, 'VOLO-D2', horizontalalignment='center', fontsize=19)
    plt.text(30, 0.45, 'SWSL-ResNext', horizontalalignment='center', fontsize=19)
    plt.text(50, 0.45, 'ResMLP-36', horizontalalignment='center', fontsize=19)
    plt.axvline(20, color='black', alpha=0.3)
    plt.axvline(40, color='black', alpha=0.3)
    plt.xlabel('Distillation Epoch', fontsize=20)
    plt.ylabel('Knowledge Gain/Loss [%]', fontsize=20)
    plt.ylim(0.3, 3.8)
    plt.xlim(0, 60)
    plt.legend(loc='upper left', handlelength=0.5)
    fig.tight_layout()
    plt.savefig(f'images/cont_dist_gain_loss_desc.pdf', bbox_inches='tight')
    plt.show()


def transfer_delta_table():
    data = load_wandb_runs('2_distill_between_experts')
    #data = data.dropna(subset=['dist_delta', 'ts_diff'])
    all_students = {'transformer': [41, 5, 26],
                    'cnn': [131, 130, 40],
                    'mlp': [2, 160]}
    all_teachers = [234, 302, 77, 239]
    data = data.loc[[data['student_id'][i] in all_students[data['student_type'][i]] for i in data.index]]
    data = data.loc[[data['teacher_id'][i] in all_teachers for i in data.index]]
    #data = data.loc[data['dist_delta'] > -20]
    data = data.loc[[data['data'][i]['dataset'] == 'imagenet' for i in data.index]]
    # subset to all mode is nan
    #data = data.loc[data['mode'].isna()]
    #data = data.loc[(data['mode'].isna()) | (data['teacher_id'] == 239)]
    appr_strings = {'KL_Dist': 'KL Distillation Transfer',
                    'KL+MT_Dist': 'KL-Dist. + DP Transfer',
                    'KL+MT_u_Dist': 'KL-Dist. + DP Transfer (unsup.)',
                    }

    nbins = 3
    pp = 0.75

    students = data['student_id'].unique()

    dist_deltas = {appr: [] for appr in appr_strings.keys()}
    success_rate = {appr: [] for appr in appr_strings.keys()}

    data['delta_acc_bin'] = pd.cut(data['ts_diff'], bins=[-15, -3, -1, 1, 3, 15])
    # sort data by delta_acc_bin
    data = data.sort_values(by=['delta_acc_bin'])
    for appr in appr_strings.keys():
        data_appr = data.loc[data['tag'] == appr]
        #assert len(data_appr) == len(data_appr['student_id'].unique()) * len(data_appr['teacher_id'].unique())
        for student in students:
            tmp = data_appr.loc[data_appr['student_id'] == student]
            if len(tmp) == 0:
                success_rate[appr].append(0)
            else:
                success_rate[appr].append(np.mean(tmp['dist_delta'].values > 0))
        for bin in data['delta_acc_bin'].unique():
            tmp = data_appr.loc[data_appr['delta_acc_bin'] == bin]

            #quantile = np.quantile(tmp['dist_delta'].values, pp)
            #dds = tmp.loc[tmp['dist_delta'] >= quantile]['dist_delta'].values
            #dist_deltas[appr].append(np.mean(dds))
            dist_deltas[appr].append(list(tmp['dist_delta'].values))


    print('Mean Distillation Delta')
    for appr in appr_strings.keys():
        print(f'{appr_strings[appr]}')
        for b, bin in enumerate(data['delta_acc_bin'].unique()):
            print(f'{bin} ({len(dist_deltas[appr][b])}): {round(np.mean(dist_deltas[appr][b]), 2)} '
                  f'(+- {round(np.std(dist_deltas[appr][b]), 2)})')

    print('Success Rate')
    for appr in appr_strings.keys():
        print(f'{appr}: {np.median(success_rate[appr])}')


if __name__ == "__main__":
    #dist_delta_line_plot()
    #dist_delta_line_plot_v2()
    dist_delta_line_plot_v3('imagenet_subset')
    #unsup_v_sup()
    #cont_distillation_gain_loss()
    #cont_distillation_gain_loss_v2()
    #sim_heatmap()
    #transfer_delta_table()
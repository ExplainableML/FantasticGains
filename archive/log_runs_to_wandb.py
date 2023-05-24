import wandb
import pandas as pd
import numpy as np
from mycolorpy import colorlist as mcp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def merge_two_dicts(x, y):
    z = x.copy()  # start with keys and values of x
    z.update(y)  # modifies z with keys and values of y
    return z


def load_wandb_runs(project):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(f"luth/{project}")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    data = []
    for i in range(len(config_list)):
        tmp = merge_two_dicts(config_list[i], summary_list[i])
        data.append(merge_two_dicts({'name': name_list[i]}, tmp))
    runs = pd.DataFrame(data)
    return runs


def log_to_wandb(runs):
    runs = runs[runs['loss'].isnull()]

    for r_id in runs.index:
        print(f'Logging run {r_id}')
        config = {'teacher_name': runs["teacher_name"][r_id],
                  'student_name': runs["student_name"][r_id],
                  'ts_diff': runs["ts_diff"][r_id],
                  'dataset': runs["dataset"][r_id],
                  'seed': runs["seed"][r_id],
                  'opt': runs["opt"][r_id],
                  'lr': runs["lr"][r_id],
                  'loss': 'xekl',
                  'alpha': 1,
                  'freeze': runs["freeze"][r_id],
                  'batch_size': runs["batch_size"][r_id]}

        wandb_id = wandb.util.generate_id()
        wandb.init(id=wandb_id, project='kl-dist-imagenet', config=config)
        wandb.run.name = runs["name"][r_id]
        wandb.log({'teacher_acc': runs["teacher_acc"][r_id],
                   'student_acc': runs["student_acc"][r_id],
                   'pos_flips_rel': runs["pos_flips_rel"][r_id],
                   'neg_flips_rel': runs["neg_flips_rel"][r_id],
                   'dist_loss': runs["dist_loss"][r_id],
                   'kl_loss': runs["dist_loss"][r_id],
                   'xe_loss': 0,
                   'student_acc_diff': runs["student_acc_diff"][r_id]})
        wandb.finish()


def architectures_boxplot(runs, types, colors):
    fig = plt.figure(figsize=(10, 5))
    boxplot_data = []
    for dtype in types:
        boxplot_data.append(runs.loc[runs['distillation_type'] == dtype]['student_acc_diff'].values)
    plt.title('Distillation Delta by Architecture Types', fontsize=14)
    plt.axvline(x=0, color='red', alpha=0.2)
    bplot = plt.boxplot(boxplot_data, labels=types, vert=False, patch_artist=True)
    for c, col in enumerate(colors):
        #bplot['boxes'][c].set_color(col)
        bplot['boxes'][c].set_facecolor(col)
    spacing = 0.200
    fig.subplots_adjust(left=spacing)
    plt.xlabel('Distillation Delta', fontsize=12)
    plt.savefig('dist_type_boxplot.png')
    plt.show()


def architectures_scatter(runs, types, colors):
    fig = plt.figure(figsize=(10, 5))
    for t, dtype in enumerate(types):
        tmp = runs.loc[runs['distillation_type'] == dtype]
        plt.scatter(tmp['ts_diff'].values, tmp['student_acc_diff'].values,
                    c=colors[t], alpha=1, label=dtype)
    plt.axhline(y=0, color='red', alpha=0.2)
    plt.xlabel('Teacher-Student Performance Difference', fontsize=12)
    plt.ylabel('Distillation Delta', fontsize=12)
    plt.title('Distillation Delta by Performance Diff. and Architecture Types', fontsize=14)
    plt.legend()
    plt.minorticks_on()
    plt.grid()
    plt.savefig('dist_type_scatter.png')
    plt.show()


def topk_scatter(runs, types, colors):
    fig = plt.figure(figsize=(10, 5))
    for t, dtype in enumerate(types):
        tmp = runs.loc[runs['divergence'] == dtype]
        plt.scatter(tmp['ts_diff'].values, tmp['student_acc_diff'].values,
                    c=colors[t], alpha=1, label=dtype)
    plt.axhline(y=0, color='red', alpha=0.2)
    plt.xlabel('Teacher-Student Performance Difference', fontsize=12)
    plt.ylabel('Distillation Delta', fontsize=12)
    plt.title('Distillation Delta by Performance Diff. and Architecture Types', fontsize=14)
    plt.legend()
    plt.minorticks_on()
    plt.grid()
    plt.savefig('dist_type_scatter.png')
    plt.show()


def warmup_scatter(runs, colors):
    fig = plt.figure(figsize=(10, 5))
    tmp = runs.loc[runs['warmup'] == 1]
    plt.scatter(tmp['ts_diff'].values, tmp['student_acc_diff'].values, c=colors[6], alpha=0.6, label='warmup')
    tmp = runs.loc[runs['warmup'].isna()]
    plt.scatter(tmp['ts_diff'].values, tmp['student_acc_diff'].values, c=colors[2], alpha=0.6, label='no warmup')
    plt.axhline(y=0, color='red', alpha=0.2)
    plt.xlabel('Teacher-Student Performance Difference', fontsize=12)
    plt.ylabel('Distillation Delta', fontsize=12)
    plt.title('Distillation Delta by Performance Diff. and Warmup', fontsize=14)
    plt.legend()
    plt.minorticks_on()
    plt.grid()
    plt.savefig('warmup_scatter.png')
    plt.show()


def param_diff_scatter(runs):
    plt.rcParams['font.size'] = 13
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    s_types = ['cnn>cnn', 'cnn>transformer', 'cnn>mlp', 'transformer>cnn', 'transformer>transformer', 'transformer>mlp', 'mlp>cnn', 'mlp>transformer']
    cmap = mcp.gen_color(cmap='tab20c', n=20)
    cidxs = [0, 1, 2, 8, 9, 10, 4, 5]

    for s, s_type in enumerate(s_types):
        tmp = runs.loc[runs['dist_type']==s_type]
        axs[0, 0].scatter(tmp[f'student_params'].values, tmp['dist_delta'].values, c=cmap[cidxs[s]], label=s_type, alpha=1)
        axs[0, 1].scatter(tmp[f'student_acc'].values, tmp['dist_delta'].values, c=cmap[cidxs[s]], label=s_type, alpha=1)
        axs[1, 0].scatter(tmp[f'teacher_params'].values, tmp['dist_delta'].values, c=cmap[cidxs[s]], label=s_type, alpha=1)
        axs[1, 1].scatter(tmp[f'teacher_acc'].values, tmp['dist_delta'].values, c=cmap[cidxs[s]], label=s_type, alpha=1)
    axs[0, 0].set(xlabel=f'Student Parameters', ylabel='Distillation Delta')
    axs[1, 0].set(xlabel=f'Teacher Parameters', ylabel='Distillation Delta')
    axs[0, 1].set(xlabel=f'Student Accuracy')
    axs[1, 1].set(xlabel=f'Teacher Accuracy')
    axs[1, 0].set_xlim(0, 350)
    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    for row in axs:
        for plot in row:
            plot.axhline(y=0, color='red', alpha=0.2)
            plot.minorticks_on()
            plot.grid()
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()


def topk_div_scatter(runs, types, colors):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(3), range(3), color='red', alpha=0.8)
    for t, dtype in enumerate(types):
        tmp = runs.loc[runs['run_name'] == dtype]

        tmp_2 = tmp.loc[tmp['k'] == 10]
        plt.scatter(tmp_2['div'].values, tmp_2['topk_div'].values,
                    c=colors[t], alpha=1, label=dtype, marker='x')
        tmp_2 = tmp.loc[tmp['k'] == 100]
        plt.scatter(tmp_2['div'].values, tmp_2['topk_div'].values,
                    c=colors[t], alpha=1, marker='$c$')
    plt.xlabel('Total Divergence', fontsize=12)
    plt.ylabel('Top-k Divergence', fontsize=12)
    #plt.title('Distillation Delta by Performance Diff. and Architecture Types', fontsize=14)
    plt.legend()
    plt.minorticks_on()
    plt.grid()
    #plt.savefig('dist_type_scatter.png')
    plt.show()


def max_min_scatter(min, max, mean):
    subset = [i for i in range(len(min)) if min[i]>-2 and max[i]>-0.5]
    min, max, mean = min[subset], max[subset], mean[subset]
    fig = plt.figure(figsize=(10, 5))
    plt.axhline(0, color='red', alpha=0.5)
    plt.axvline(0, color='red', alpha=0.5)
    cmap = plt.get_cmap('winter')#, np.max(mean)-np.min(mean))
    scat = plt.scatter(min, max, c=mean, cmap=cmap, alpha=0.8)
    plt.colorbar(scat)
    plt.xlabel('Minimum Distillation Delta', fontsize=12)
    plt.ylabel('Maximum Distillation Delta', fontsize=12)
    #plt.xlim([-2,0.1])
    #plt.title('Distillation Delta by Performance Diff. and Architecture Types', fontsize=14)
    plt.legend()
    plt.minorticks_on()
    plt.grid()
    #plt.savefig('dist_type_scatter.png')
    plt.show()


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


def gain_loss_scatter(gain, loss, z, color, markers=None):
    if markers is not None:
        shapes = ['o', 'X', 'D', 'P', 'v', 's', '<', 'p']
        keys= np.unique(markers)
        marker_dict = {key:shapes[k] for k, key in enumerate(keys)}
        print(marker_dict)
        marks = [marker_dict[m] for m in markers]
    else:
        marks = ['o']*len(gain)
    z = (((z - np.min(z)) / (np.max(z) - np.min(z))) + 2) ** 4

    fig = plt.figure(figsize=(10, 5))
    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Source Sans']})
    matplotlib.rc('text', usetex=True)

    plt.axline([0.6, 0.6], [2, 2], color='red', alpha=0.5)
    #cmap = plt.get_cmap('plasma')
    cvals = [11, 22, 30, 44, 60, 86]
    colors = ["#DA4C4C", "#EDB732", "#5BC5DB", "#5387DD", "#A0C75C", "#479A5F"]
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    scat = mscatter(loss, gain, z, c=color, cmap=cmap, alpha=1, m=marks)
    cbar = plt.colorbar(scat)
    plt.xlabel('Knowledge Loss', fontsize=14)
    plt.ylabel('Knowledge Gain', fontsize=14)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('Student Parameters', rotation=270, fontsize=14)
    plt.title('Knowledge Gain by Knowledge Loss', fontsize=16, fontweight='bold')
    plt.legend()
    plt.minorticks_on()
    plt.grid()
    plt.show()


def param_search_plots(min, max, mean, gain, loss, highlight=34):
    subset = [i for i in range(len(min)) if min[i]>-2 and max[i]>-0.5]
    min_s, max_s, mean_s = min[subset], max[subset], mean[subset]
    gain_s, loss_s = gain[subset], loss[subset]
    plt.rcParams['font.size'] = 14
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.axhline(0, color='red', alpha=0.5)
    plt.axvline(0, color='red', alpha=0.5)
    cmap = plt.get_cmap('plasma')
    scat = plt.scatter(min_s, max_s, c=mean_s, cmap=cmap, alpha=1)
    if highlight is not None:
        plt.scatter(min[highlight], max[highlight], color='None', alpha=1, edgecolors='red')
    plt.xlabel('Minimum Distillation Delta', fontsize=12)
    plt.ylabel('Maximum Distillation Delta', fontsize=12)
    plt.legend()
    plt.minorticks_on()
    plt.grid()

    plt.subplot(122)
    plt.axline([1, 1], [2, 2], color='red', alpha=0.5)
    cmap = plt.get_cmap('plasma')
    scat = plt.scatter(loss_s, gain_s, c=mean_s, cmap=cmap, alpha=1)
    if highlight is not None:
        plt.scatter(loss[highlight], gain[highlight], color='None', alpha=1, edgecolors='red')
    cbar = plt.colorbar(scat)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('Mean Distillation Delta', rotation=270)
    plt.xlabel('Mean Knowledge Loss', fontsize=12)
    plt.ylabel('Mean Knowledge Gain', fontsize=12)
    plt.xlim(1, 2.6)
    plt.ylim(1, 2.6)
    plt.legend()
    plt.minorticks_on()
    plt.grid()

    plt.show()


def best_strats_plot(runs):
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(10, 5))
    student_id = np.unique(runs['student_id'].values)
    minmax = {'MaxMax': 'a1-T1-k10-tau0.995-N1-lr0.0001'}
    params = []
    plt.axhline(0, color='red', alpha=0.5)
    for s_id in student_id:
        tmp = runs.loc[runs['student_id'] == s_id].sort_values(by=['dist_delta'], ascending=False)
        plt.scatter(tmp['ts_diff'].values[0], tmp['dist_delta'].values[0], color='red', marker='x')
        if tmp['param_combination'].values[0] not in minmax.values():
            params.append(tmp['param_combination'].values[0])
    cmap = mcp.gen_color(cmap='tab20c', n=20)
    for p, param in enumerate(np.unique(params)):
        tmp = runs.loc[runs['param_combination']==param].sort_values(by=['ts_diff'])
        plt.plot(tmp['ts_diff'].values, tmp['dist_delta'].values, label=param, color=cmap[p], alpha=0.4, marker='.')
    for m, mm in enumerate(minmax.keys()):
        tmp = runs.loc[runs['param_combination'] == minmax[mm]].sort_values(by=['ts_diff'])
        plt.plot(tmp['ts_diff'].values, tmp['dist_delta'].values, label=f'{minmax[mm]}-{mm}', color=cmap[5+4*m], alpha=1, marker='.')
    plt.xlabel('Teacher-Student Performance Difference')
    plt.ylabel('Distillation Delta')
    plt.ylim((-1, 1))
    plt.minorticks_on()
    plt.grid()
    plt.legend()
    plt.show()


def correlation_heatmap(runs):
    fig = plt.figure(figsize=(12, 7))
    correlation = runs.corr(method='pearson')
    sns.heatmap(correlation, annot=True, fmt='.2f', annot_kws={"size": 11})
    plt.subplots_adjust(bottom=0.28, left=0.25)
    plt.show()


def create_model_list():
    models_list = pd.read_csv('../files/dist_runs.csv', sep=';')
    models_types = pd.read_csv('../files/timm_model_types.csv', sep=';')

    model_a_df = pd.DataFrame({'modelname': models_list['modelname_a'].values,
                              'modelparams': models_list['modelparams_a'].values,
                              'modeltop1': models_list['modeltop1_a'].values})
    model_b_df = pd.DataFrame({'modelname': models_list['modelname_b'].values,
                              'modelparams': models_list['modelparams_b'].values,
                              'modeltop1': models_list['modeltop1_b'].values})
    contdist_models = pd.concat([model_a_df, model_b_df])
    types = []
    for name in contdist_models['modelname'].values:
        type = models_types.loc[models_types['Modelname'] == name]['Modeltype'].values
        types.append(type[0] if len(type)>0 else 'None')
    contdist_models['modeltype'] = types
    contdist_models = contdist_models.loc[contdist_models['modeltype'] != 'None']
    contdist_models = contdist_models.drop_duplicates()
    contdist_models = contdist_models.reset_index(drop=True)
    contdist_models.to_csv('files/contdist_model_list.csv')


def eval_randomsearch():
    runs = load_wandb_runs('2-5_xekl_mt')
    runs['run_name'] = [f'{runs["teacher_name"][i]}>{runs["student_name"][i]}' for i in runs.index]
    params = []
    for i in runs.index:
        try:
            params.append(f'a{runs["loss"][i]["alpha"]}-T{runs["loss"][i]["kd_T"]}-k{runs["loss"][i]["k"]}-'
                          f'tau{runs["loss"][i]["tau"]}-N{runs["loss"][i]["N"]}-'
                          f'lr{runs["optimizer"][i]["lr"]}')
        except KeyError:
            params.append('None')
    runs['param_combination'] = params
    runs = runs.loc[runs['tag'] == 'random-search']
    #runs = runs.loc[runs['student_id'] != 186]
    #runs = runs.loc[[runs['student_id'][i] in [171, 132, 160] for i in runs.index]]
    #runs = runs.loc[runs['tag'] == 'random-search']
    runs_mean = runs.groupby(['param_combination']).mean()
    runs_max = runs.groupby(['param_combination']).max()
    runs_min = runs.groupby(['param_combination']).min()
    runs_sum = runs.groupby(['param_combination']).sum()

    types = np.unique(runs['run_name'].values)
    colors = mcp.gen_color(cmap='Dark2', n=len(types))
    order = [2, 4, 1, 0, 6, 3, 5]

    #topk_div_scatter(runs, types, colors)
    param_search_plots(runs_min['dist_delta'].values, runs_max['dist_delta'].values, runs_mean['dist_delta'].values,
                    runs_mean['knowledge_gain'].values, runs_mean['knowledge_loss'].values, 34)
    #best_strats_plot(runs)

    param_choices = pd.DataFrame({'params': runs_mean.index,
                                  'sum': runs_sum['dist_delta'].values,
                                  'mean_delta': runs_mean['dist_delta'].values,
                                  'max_delta': runs_max['dist_delta'].values,
                                  'min_delta': runs_min['dist_delta'].values,
                                  'mean_gain': runs_mean['knowledge_gain'].values,
                                  'max_gain': runs_max['knowledge_gain'].values,
                                  'min_gain': runs_min['knowledge_gain'].values,
                                  'mean_loss': runs_mean['knowledge_loss'].values,
                                  'max_loss': runs_max['knowledge_loss'].values,
                                  'min_loss': runs_min['knowledge_loss'].values,
                                  })
    test
    #param_choices.to_csv('files/xekl-mcp-randomsearch.csv')


def check_bad_runs():
    runs = load_wandb_runs('2-6_xekl_mt')
    runs['name'] = [f'{runs["teacher_name"][i]}>{runs["student_name"][i]}' for i in runs.index]
    params = []
    for i in runs.index:
        try:
            params.append(f'a{runs["loss"][i]["alpha"]}-T{runs["loss"][i]["kd_T"]}-k{runs["loss"][i]["k"]}-'
                          #f'tau{runs["loss"][i]["tau"]}-N{runs["loss"][i]["N"]}-'
                                 f'lr{runs["optimizer"][i]["lr"]}')
        except KeyError:
            params.append('None')
    runs['param_combination'] = params
    runs = runs.loc[(runs['tag'] == 'st-test') | (runs['tag'] == 'best-student-test')]
    #runs = runs.loc[(runs['param_combination'] == 'a0.5-T4-k100-lr0.0001')]
    runs = runs.loc[[runs['loss'][i]['strat'] == 'most-conf' for i in runs.index]]
    #runs = runs.loc[[runs['on_flip'][i]['neg'] == 'distill' and runs['on_flip'][i]['neut'] == 'distill' for i in runs.index]]
    #runs = runs.loc[['xcit_tiny' not in runs['student_name'][i] for i in runs.index]]
    #runs = runs.loc[['xcit_nano' not in runs['student_name'][i] for i in runs.index]]
    #runs = runs.loc[['xcit_small' not in runs['student_name'][i] for i in runs.index]]
    #runs = runs.loc[runs['param_combination'] == 'a0.5-T0.1-tau0.999-N100-lr0.01']

    flip_runs = load_wandb_runs('1_prediction_flips_study')
    arch_dict = {'mlp':0, 'transformer':1, 'cnn':2}
    runs = runs.loc[runs['dist_delta'] > -10]
    bad_runs = runs[['name', 'teacher_name', 'student_name', 'dist_type', 'dist_delta', 'batch_size',
                     'knowledge_gain', 'knowledge_loss', 'student_type', 'student_acc', 'student_params',
                     'teacher_type', 'teacher_params', 'teacher_acc']]
    #bad_runs['teacher_params'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['modelparams_a'].values[0] for i in bad_runs.index]
    #bad_runs['student_params'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['modelparams_b'].values[0] for i in bad_runs.index]
    #bad_runs['param_diff'] = bad_runs['teacher_params'] - bad_runs['student_params']
    #bad_runs['teacher_acc'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['modeltop1_a'].values[0] for i in bad_runs.index]
    #bad_runs['student_acc'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['modeltop1_b'].values[0] for i in bad_runs.index]
    #bad_runs['ts_diff'] = bad_runs['teacher_acc'] - bad_runs['student_acc']
    #bad_runs['posflips'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['pos_flips_rel'].values[0] for i in bad_runs.index]
    #bad_runs['negflips'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['neg_flips_rel'].values[0] for i in bad_runs.index]
    #bad_runs['teacher_type'] = [arch_dict[bad_runs['dist_type'][i].split('>')[0]] for i in bad_runs.index]
    #bad_runs['student_type'] = [arch_dict[bad_runs['dist_type'][i].split('>')[1]] for i in bad_runs.index]
    #param_diff_scatter(bad_runs)
    correlation_heatmap(bad_runs)
    gain_loss_scatter(bad_runs['knowledge_gain'].values, bad_runs['knowledge_loss'].values, bad_runs['teacher_acc'].values,
                      bad_runs['student_params'].values, bad_runs['teacher_type'].values)
    test


def analyze_mt_runs():
    runs = load_wandb_runs('2-6_xekl_mt')
    runs['name'] = [f'{runs["teacher_name"][i]}>{runs["student_name"][i]}' for i in runs.index]
    runs = runs.loc[(runs['tag'] != 'self-dist')]

    baseline = load_wandb_runs('2-5_xekl_param_search')
    baseline['name'] = [f'{baseline["teacher_name"][i]}>{baseline["student_name"][i]}' for i in baseline.index]
    ts_pairs = np.unique(runs['name'].values)
    baseline = baseline.loc[baseline['tag'] == 'agg-dist-mcl']

    flip_runs = load_wandb_runs('1_prediction_flips_study')
    arch_dict = {'mlp': 0, 'transformer': 1, 'cnn': 2}
    runs = runs.loc[runs['dist_delta'] > -10]
    check = runs[['name', 'teacher_name', 'student_name', 'dist_type', 'dist_delta',
                  'teacher_params', 'student_params', 'ts_params_diff', 'student_acc', 'teacher_acc', 'ts_diff', 'tag']]
    pos_flips, neg_flips, neut_flips = [], [], []
    for i in check.index:
        try:
            pos_flips.append(flip_runs.loc[flip_runs['name'] == check['name'][i]]['pos_flips_rel'].values[0])
            neg_flips.append(flip_runs.loc[flip_runs['name'] == check['name'][i]]['neg_flips_rel'].values[0])
            neut_flips.append(100 - pos_flips[-1] - neg_flips[-1])
        except IndexError:
            pos_flips.append(0)
            neg_flips.append(0)
            neut_flips.append(0)
    check['pos_flips'] = pos_flips
    check['neg_flips'] = neg_flips
    check['neut_flips'] = neut_flips
    check['teacher_type'] = [arch_dict[check['dist_type'][i].split('>')[0]] for i in check.index]
    check['student_type'] = [arch_dict[check['dist_type'][i].split('>')[1]] for i in check.index]

    strats = ['most-conf', 'flip-avg', 'flip-t', 'flip-st']
    data = []
    for ts in ts_pairs:
        if ts in ['regnety_120>xcit_nano_12_p16_224_dist', 'convnext_tiny_in22ft1k>hrnet_w18_small_v2']: continue
        tmp = check.loc[check['name'] == ts]
        d_most_conf = tmp.loc[tmp['tag']=='most-conf']
        row = [ts, tmp['student_params'].values[0], tmp['teacher_params'].values[0], tmp['student_acc'].values[0],
               tmp['teacher_acc'].values[0], tmp['student_type'].values[0], tmp['teacher_type'].values[0],
               tmp['pos_flips'].values[0], tmp['neg_flips'].values[0], tmp['neut_flips'].values[0]]
        for strat in strats:
            try:
                mt_delta = tmp.loc[tmp['tag'] == strat]['dist_delta']
                baseline_delta = baseline.loc[baseline['name']==ts]['dist_delta']
                delta = tmp.loc[tmp['tag'] == strat]['dist_delta'].values[0] - baseline.loc[baseline['name']==ts]['dist_delta'].values[0]
            except IndexError:
                delta = 0
            row.append(delta)
        data.append(row)
    best_strat = pd.DataFrame(data=data, columns= ['name', 's_params', 't_params', 's_acc', 't_acc', 's_type', 't_type',
                                                   'pos_flips', 'neg_flips', 'neut_flips', 'd_most_conf', 'd_flip_avg',
                                                   'd_flip_t', 'd_flip_st'])
    correlation_heatmap(best_strat)
    test


def mt_scatter(df, by='student_params'):
    plt.rcParams['font.size'] = 13

    colors = {'most-conf': 'blue', 'flip-t': 'yellow', 'flip-st': 'orange', 'flip-avg': 'red'}
    for i in df.index:
        plt.scatter(df[by][i], df['dist_delta'][i], color=colors[df['tag'][i]], label=df['tag'][i], alpha=0.75)

    plt.xlabel(f'{by}')
    plt.ylabel('Distillation Delta')

    plt.axhline(y=0, color='red', alpha=0.2)
    plt.minorticks_on()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    check_bad_runs()

import wandb
import pandas as pd
import numpy as np
from mycolorpy import colorlist as mcp
import matplotlib.pyplot as plt
import seaborn as sns


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
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    plt.subplots_adjust(bottom=0.32)

    s_types = ['cnn>cnn', 'cnn>transformer', 'cnn>mlp', 'transformer>cnn', 'transformer>transformer', 'transformer>mlp', 'mlp>cnn', 'mlp>transformer']
    cmap = mcp.gen_color(cmap='tab20c', n=20)
    cidxs = [0, 1, 2, 8, 9, 10, 4, 5]

    for s, s_type in enumerate(s_types):
        tmp = runs.loc[runs['dist_type']==s_type]
        axs[0].scatter(tmp['student_params'].values, tmp['dist_delta'].values, c=cmap[cidxs[s]], label=s_type, alpha=1)
        axs[1].scatter(tmp['negflips'].values, tmp['dist_delta'].values, c=cmap[cidxs[s]], label=s_type, alpha=1)
    axs[0].set(xlabel='Student Parameters', ylabel='Distillation Delta')
    axs[1].set(xlabel='Negative Flips')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    for ax in axs:
        ax.axhline(y=0, color='red', alpha=0.2)
        ax.minorticks_on()
        ax.grid()
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


def best_strats_plot(runs):
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(10, 5))
    student_id = [132, 160, 171]
    minmax = {'MaxMean':'a0.68-T1-tau100-lr0.001', '2ndMax': 'a0.75-T1-tau100-lr0.001', '3rdMax': 'a0.33-T1-tau100-lr0.001'}
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
    models_list = pd.read_csv('files/dist_runs.csv', sep=';')
    models_types = pd.read_csv('files/timm_model_types.csv', sep=';')

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
    runs = load_wandb_runs('xekl-randomsearch')
    runs['run_name'] = [f'{runs["teacher_name"][i]}>{runs["student_name"][i]}' for i in runs.index]
    params = []
    for i in runs.index:
        try:
            params.append(f'a{runs["loss"][i]["alpha"]}-T{runs["loss"][i]["kd_T"]}-tau{runs["loss"][i]["k"]}-'
                                 f'lr{runs["optimizer"][i]["lr"]}')
        except KeyError:
            params.append('None')
    runs['param_combination'] = params
    #runs = runs.loc[[runs['seed'][i] in [397, 235, 63, 197, 105, 92] for i in runs.index]]
    runs = runs.loc[[runs['on_flip'][i]['neg'] == 'nothing' for i in runs.index]]
    runs = runs.loc[[runs['on_flip'][i]['neut'] == 'distill' for i in runs.index]]
    runs = runs.loc[[runs['on_flip'][i]['pos'] == 'distill' for i in runs.index]]
    runs = runs.loc[runs['tag'] == 'random-search']
    runs_mean = runs.groupby(['param_combination']).mean()
    runs_max = runs.groupby(['param_combination']).max()
    runs_min = runs.groupby(['param_combination']).min()
    runs_sum = runs.groupby(['param_combination']).sum()

    types = np.unique(runs['run_name'].values)
    colors = mcp.gen_color(cmap='Dark2', n=len(types))
    order = [2, 4, 1, 0, 6, 3, 5]

    #topk_div_scatter(runs, types, colors)
    max_min_scatter(runs_min['dist_delta'].values, runs_max['dist_delta'].values, runs_mean['dist_delta'].values)
    best_strats_plot(runs)

    param_choices = pd.DataFrame({'params': runs_mean.index,
                                  'sum': runs_sum['dist_delta'].values,
                                  'mean': runs_mean['dist_delta'].values,
                                  'max': runs_max['dist_delta'].values,
                                  'min': runs_min['dist_delta'].values})
    test
    #param_choices.to_csv('files/xekl-mcp-randomsearch.csv')


def check_bad_runs():
    runs = load_wandb_runs('xekl-mcp-randomsearch')
    runs['name'] = [f'{runs["teacher_name"][i]}>{runs["student_name"][i]}' for i in runs.index]
    params = []
    for i in runs.index:
        try:
            params.append(f'a{runs["loss"][i]["alpha"]}-T{runs["loss"][i]["kd_T"]}-tau{runs["loss"][i]["tau"]}-'
                                 f'N{runs["loss"][i]["N"]}-lr{runs["optimizer"][i]["lr"]}')
        except KeyError:
            params.append('None')
    runs['param_combination'] = params
    runs = runs.loc[(runs['param_combination'] == 'a0.75-T0.1-tau0.9999-N2-lr0.01')]
    runs = runs.loc[[runs['on_flip'][i]['neg'] == 'nothing' and runs['on_flip'][i]['neut'] == 'distill' for i in runs.index]]
    runs = runs.loc[['xcit_tiny' not in runs['student_name'][i] for i in runs.index]]
    runs = runs.loc[['xcit_nano' not in runs['student_name'][i] for i in runs.index]]
    runs = runs.loc[['xcit_small' not in runs['student_name'][i] for i in runs.index]]
    #runs = runs.loc[runs['param_combination'] == 'a0.5-T0.1-tau0.999-N100-lr0.01']

    flip_runs = load_wandb_runs('flips-study-imagenet')
    arch_dict = {'mlp':0, 'transformer':1, 'cnn':2}
    runs = runs.loc[runs['dist_delta'] > -10]
    bad_runs = runs[['name', 'teacher_name', 'student_name', 'dist_type', 'dist_delta', 'batch_size']]
    bad_runs['teacher_params'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['modelparams_a'].values[0] for i in bad_runs.index]
    bad_runs['student_params'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['modelparams_b'].values[0] for i in bad_runs.index]
    bad_runs['param_diff'] = bad_runs['teacher_params'] - bad_runs['student_params']
    bad_runs['teacher_acc'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['modeltop1_a'].values[0] for i in bad_runs.index]
    bad_runs['student_acc'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['modeltop1_b'].values[0] for i in bad_runs.index]
    bad_runs['ts_diff'] = bad_runs['teacher_acc'] - bad_runs['student_acc']
    bad_runs['posflips'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['pos_flips_rel'].values[0] for i in bad_runs.index]
    bad_runs['negflips'] = [flip_runs.loc[flip_runs['name']==bad_runs['name'][i]]['neg_flips_rel'].values[0] for i in bad_runs.index]
    bad_runs['teacher_type'] = [arch_dict[bad_runs['dist_type'][i].split('>')[0]] for i in bad_runs.index]
    bad_runs['student_type'] = [arch_dict[bad_runs['dist_type'][i].split('>')[1]] for i in bad_runs.index]
    param_diff_scatter(bad_runs)
    correlation_heatmap(bad_runs)
    test


if __name__ == "__main__":
    eval_randomsearch()
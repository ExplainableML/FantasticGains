import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
from collections import Counter


def architectures_scatter(runs):
    types = np.unique(runs['distillation_type'])
    colors = mcp.gen_color(cmap='Dark2', n=len(types))
    #order = [2, 4, 1, 0, 6, 3, 5]
    #types = [types[o] for o in order]
    fig = plt.figure(figsize=(10, 5))
    for t, dtype in enumerate(types):
        tmp = runs.loc[runs['distillation_type'] == dtype]
        plt.scatter(tmp['top1_diff'].values * -1, tmp['params_diff'].values * -1,
                    c=colors[t], alpha=1, label=dtype)
    plt.axhline(y=0, color='red', alpha=0.2)
    plt.xlabel('Performance Difference', fontsize=12)
    plt.ylabel('Parameter Difference', fontsize=12)
    #plt.title('Distillation Delta by Performance Diff. and Architecture Types', fontsize=14)
    plt.legend()
    plt.minorticks_on()
    plt.grid()
    #plt.savefig('dist_type_scatter.png')
    plt.show()


def sample_small_test_set(runs, size=5, seed=1234):
    np.random.seed(seed)
    runs = runs.sort_values(by=['top1_diff'])
    runs = runs.reset_index()
    n_runs = len(runs.index)
    block_size = int(n_runs/size)

    sample = []
    for b in range(size):
        tmp = runs.index[block_size*b:block_size*(b+1)]
        sample.append(np.random.choice(tmp))

    print(f'Sample: {sample}')
    return runs.iloc[sample]


def sample_large_test_set(size=50, seed=1234):
    runs = pd.read_csv('files/dist_runs.csv', sep=';')
    model_types = pd.read_csv('files/timm_model_types.csv', sep=';')
    non_tf = [not ('tf_' in runs['modelname_a'][i] or 'tf_' in runs['modelname_b'][i]) for i in runs.index]
    runs = runs[non_tf]
    runs['student_type'] = [model_types[model_types['Modelname'] == name]['Modeltype'].values[0] for name in
                            runs['modelname_b'].values]
    runs['teacher_type'] = [model_types[model_types['Modelname'] == name]['Modeltype'].values[0] for name in
                            runs['modelname_a'].values]
    runs['distillation_type'] = [f'{runs["teacher_type"][i]}>{runs["student_type"][i]}' for i in runs.index]
    print(Counter(runs['distillation_type'].values))

    np.random.seed(seed)
    runs = runs.sort_values(by=['top1_diff'])
    runs = runs.reset_index()
    n_runs = len(runs.index)
    block_size = int(n_runs/(size/8))

    types = np.unique(runs['distillation_type'].values)
    current_type = 0

    sample = []
    for b in range(size):
        tmp = runs.iloc[runs.index[block_size*b:block_size*(b+1)]]
        done = False
        while not done:
            type_subset = tmp[tmp['distillation_type'] == types[current_type]]
            if len(type_subset) > 0:
                sample.append(np.random.choice(type_subset.index))
            current_type += 1
            if current_type >= len(types):
                current_type = 0
                done = True

    print(f'Sample: {sample}')
    return runs.iloc[sample]


def diverse_teachers(size=50, seed=1234):
    models_list = pd.read_csv('files/contdist_model_list.csv')
    np.random.seed(seed)
    models_list = models_list.sort_values(by=['modeltop1'])
    models_list = models_list.reset_index()
    n_models = len(models_list.index)
    block_size = int(n_models/(size/3))

    types = np.unique(models_list['modeltype'].values)
    current_type = 0

    sample = []
    for b in range(size):
        tmp = models_list.iloc[models_list.index[block_size*b:block_size*(b+1)]]
        done = False
        while not done:
            type_subset = tmp[tmp['modeltype'] == types[current_type]]
            if len(type_subset) > 0:
                sample.append(np.random.choice(type_subset.index))
            current_type += 1
            if current_type >= len(types):
                current_type = 0
                done = True

    print(f'Sample: {sample}')
    return models_list.iloc[sample]


def main():
    subset = diverse_teachers(20,1234)
    seeds = subset['index'].values
    print(f'Number of runs: {len(seeds)}')
    print(f'seeds: {",".join([str(s) for s in seeds])}')
    print(Counter(subset['modeltype'].values))
    test


if __name__ == '__main__':
    main()
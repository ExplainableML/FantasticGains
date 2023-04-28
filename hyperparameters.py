from sklearn.metrics._regression import r2_score

import wandb
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from log_runs_to_wandb import correlation_heatmap

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


def load_train_val_data(project='xekl-mcp-randomsearch'):
    runs = load_wandb_runs(project)
    runs['name'] = [f'{runs["teacher_name"][i]}>{runs["student_name"][i]}' for i in runs.index]
    runs = runs.loc[runs['dist_delta'].notna()]
    data = runs[['name', 'teacher_name', 'student_name', 'dist_type', 'dist_delta', 'batch_size']]
    hyperparams = {'name': [], 'alpha': [], 'kd_T': [], 'k': [], 'tau': [], 'N': []}
    for i in runs.index:
        for param in hyperparams.keys():
            try:
                hyperparams[param].append(runs['loss'][i][param])
            except KeyError:
                hyperparams[param].append(np.NaN)
    for param in hyperparams.keys():
        data[f'hp_{param}'] = hyperparams[param]
    data['hp_lr'] = [runs['optimizer'][i]['lr'] for i in runs.index]
    data = data.loc[data['hp_name']=='xekl_mcp']
    data = data.loc[data['hp_name'].notna()]

    arch_dict = {'mlp': 0, 'transformer': 1, 'cnn': 2}
    flip_runs = load_wandb_runs('flips-study-imagenet')
    data['teacher_params'] = [flip_runs.loc[flip_runs['name'] == data['name'][i]]['modelparams_a'].values[0] for
                                  i in data.index]
    data['student_params'] = [flip_runs.loc[flip_runs['name'] == data['name'][i]]['modelparams_b'].values[0] for
                                  i in data.index]
    #data['param_diff'] = data['teacher_params'] - data['student_params']
    data['teacher_acc'] = [flip_runs.loc[flip_runs['name'] == data['name'][i]]['modeltop1_a'].values[0] for i in
                               data.index]
    data['student_acc'] = [flip_runs.loc[flip_runs['name'] == data['name'][i]]['modeltop1_b'].values[0] for i in
                               data.index]
    #data['ts_diff'] = data['teacher_acc'] - data['student_acc']
    data['posflips'] = [flip_runs.loc[flip_runs['name'] == data['name'][i]]['pos_flips_rel'].values[0] for i in
                            data.index]
    data['negflips'] = [flip_runs.loc[flip_runs['name'] == data['name'][i]]['neg_flips_rel'].values[0] for i in
                            data.index]
    data['teacher_type'] = [arch_dict[data['dist_type'][i].split('>')[0]] for i in data.index]
    data['student_type'] = [arch_dict[data['dist_type'][i].split('>')[1]] for i in data.index]

    data = data.drop(columns=['dist_type'])
    data.to_csv('files/xekl_mcl_ransomsearch_results.csv')


def train_test_split(data, seed=123):
    data = data.dropna()
    names = ['convit_tiny>gluon_resnet101_v1c', 'regnety_008>resmlp_24_distilled_224',
                   'efficientnet_lite0>vit_small_patch32_224', 'swin_tiny_patch4_window7_224>pit_xs_distilled_224',
                   'resmlp_36_224>coat_lite_tiny', 'ssl_resnext50_32x4d>dla34']
    np.random.seed(seed)
    test_name = np.random.choice(names, 1)
    train_names = [name for name in names if name not in test_name]
    split = [data['name'][i] in train_names for i in data.index]
    X = data[['hp_alpha', 'hp_kd_T', 'hp_k', 'hp_tau', 'hp_N', 'hp_lr',
              'teacher_params', 'student_params', 'teacher_acc', 'student_acc', 'posflips', 'negflips',
              'teacher_type', 'student_type']].to_numpy()
    y = data['dist_delta'].to_numpy()
    X_train, X_test = X[split], X[[not s for s in split]]
    y_train, y_test = y[split], y[[not s for s in split]]

    return X_train, y_train, X_test, y_test


def support_vector_regression(X_train, y_train, X_test, y_test):
    regr = make_pipeline(StandardScaler(), SVR())
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'MSE: {mse}')


def main():
    data = pd.read_csv('files/xekl_mcl_ransomsearch_results.csv')
    correlation_heatmap(data)
    X_train, y_train, X_test, y_test = train_test_split(data)
    print(f'Data {len(data)}, Train {len(y_train)}, Test {len(y_test)}')
    #support_vector_regression(X_train, y_train, X_test, y_test)
    test


if __name__ == "__main__":
    main()

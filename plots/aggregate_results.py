import wandb

import pandas as pd
import numpy as np

arch_strings = {'transformer': 'Transformer', 'cnn': 'CNN', 'mlp': 'MLP'}
appr_strings = {'KL_Dist': 'KL Distillation', 'XEKL_Dist': 'XE-KL Distillation',
                'CD_Dist': 'CD Distillation', 'CRD_Dist': 'CRD Distillation',
                'XEKL+MCL_Dist': 'XE-KL+MCL Distillation', 'KL+MT_Dist': 'KL+MT Distillation', 'KL+MT_u_Dist': 'KL+MT-U Distillation'}
teachers = [211, 268, 234, 302, 209, 10, 152, 80, 36, 182, 310, 77, 12, 239, 151, 145, 232, 101, 291, 124]
all_students = {'transformer': [41, 7, 5, 46, 26, 171, 285, 144, 261, 63, 237],
                'cnn': [33, 131, 235, 132, 42, 130, 48, 302, 40, 139],
                'mlp': [214, 2, 9, 77, 258, 160, 72, 299, 232]}


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict.

    :param x: dict
    :param y: dict

    :Returns: dict
    """
    z = x.copy()  # start with keys and values of x
    z.update(y)  # modifies z with keys and values of y
    return z


def load_wandb_runs(project, history=False):
    """Load all runs from a wandb project.

    :param project: project name
    :param history: whether to load history or not

    :Returns: pandas dataframe
    """
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(f"luth/{project}")

    summary_list, config_list, name_list, history_list = [], [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        if history:
            metrics = ['knowledge_gain', 'knowledge_loss', 'dist_delta']
            tmp = {}
            if run.state in ['running', 'finished']:
                for m in metrics:
                    tmp[f'{m}_hist']=[]
                    for i, row in run.history().iterrows():
                        if i > 0:
                            try:
                                tmp[f'{m}_hist'].append(row[m])
                            except KeyError:
                                pass
            history_list.append(tmp)
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
        if history:
            tmp = merge_two_dicts(tmp, history_list[i])
        data.append(merge_two_dicts({'name': name_list[i]}, tmp))
    runs = pd.DataFrame(data)
    return runs


def get_summary_stats(array):
    """Get summary statistics of an array.

    :param array: array

    :Returns: list of summary statistics (max, min, mean, std, q25, q75)
    """
    return [np.max(array), np.min(array), np.mean(array), np.std(array), np.quantile(array, 0.25), np.quantile(array, 0.75)]


def aggregate_results():
    """Aggregate results from wandb runs and save to a csv file.

    :Returns: None
    """
    data = load_wandb_runs('2_distill_between_experts')
    data = data.dropna(subset=['dist_delta'])

    leaderboard = []
    for appr in appr_strings.keys():
        for arch in all_students.keys():
            for student in all_students[arch]:
                tmp = data.loc[(data['tag'] == appr) & (data['student_id'] == student)]
                if len(tmp) > 0:
                    student_stats = [tmp['student_id'].values[0], tmp['student_name'].values[0], tmp['student_params'].values[0],
                                    tmp['student_acc'].values[0], arch]
                    delta_stats = get_summary_stats(tmp['dist_delta'].values)
                    pos_delta = [sum(tmp['dist_delta'].values > 0)/20*100]
                    k_gain_stats = get_summary_stats(tmp['knowledge_gain'].values)
                    k_loss_stats = get_summary_stats(tmp['knowledge_loss'].values)

                    leaderboard.append([appr] + student_stats + delta_stats + pos_delta + k_gain_stats + k_loss_stats)

    leaderboard_df = pd.DataFrame(data=leaderboard,
                                  columns=['approach', 'student_id', 'student_name', 'student_params', 'student_acc', 'student_type',
                                           'max_dist_delta', 'min_dist_delta', 'mean_dist_delta', 'std_dist_delta', 'q25_dist_delta', 'q75_dist_delta', 'pos_dist_delta',
                                           'max_k_gain', 'min_k_gain', 'mean_k_gain', 'std_k_gain', 'q25_k_gain', 'q75_k_gain',
                                           'max_k_loss', 'min_k_loss', 'mean_k_loss', 'std_k_loss', 'q25_k_loss', 'q75_k_loss'])
    leaderboard_df.to_csv('students_leaderboard.csv')


if __name__ == "__main__":
    aggregate_results()
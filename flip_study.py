import timm
import hydra
import wandb
import time

import numpy as np
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from wandb import AlertLevel
import logging

from solo.args.pretrain import parse_cfg
from distillation.flip_study import compare_models
from analyze_flip_runs import import_wandb_runs, infer_model_names


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    # parse config
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    # import list of models from the timm library
    models_list = pd.read_csv('files/results-imagenet.csv')
    # filter model list for pretrained models
    pretrained_models = timm.list_models(pretrained=True)
    models_list = models_list.loc[[m in pretrained_models for m in models_list['model']]]
    # filter model list for image resolution 224x224
    models_list = models_list.loc[models_list['img_size'] == 224]
    models_list = models_list.reset_index()

    # start run timer
    t_start = time.time()
    # use cfg seed to draw two random models from the filtered model list
    #np.random.seed(cfg.seed)
    #i, j = np.random.choice(models_list.index, 2)
    #modelname_a, modelname_b = models_list['model'][i], models_list['model'][j]
    runs = pd.read_csv('files/dist_runs.csv', sep=';')
    try:
        current = np.load('files/current_flip_seed.npy')
    except FileNotFoundError:
        current = 0
    if cfg.seed <= current:
        cfg.seed = current + 1
    #error_runs = infer_model_names(runs.loc[runs['neg_flips_abs'].isna()])
    modelname_a, modelname_b = runs['modelname_a'].values[cfg.seed], runs['modelname_b'].values[cfg.seed]
    i = models_list.index[models_list['model'] == modelname_a].values[0]
    j = models_list.index[models_list['model'] == modelname_b].values[0]
    # initialize wandb logger
    wandb.init(project=cfg.wandb.project)
    wandb.run.name = f'{modelname_a}>{modelname_b}'

    try:
        print(f'Calculate flips for {modelname_a} -> {modelname_b}')
        # calculate flips and accuracies
        flips, accs = compare_models(modelname_a, modelname_b, cfg)
        print(f'calculated top{cfg.topn}_accs: {accs[0]}; {accs[1]}')
        print(f'reported top{cfg.topn}_accs: {models_list[f"top{cfg.topn}"][i]}; {models_list[f"top{cfg.topn}"][j]}')
        print(f'neg-flips: {flips["neg_rel"]}; pos-flips: {flips["pos_rel"]}')
        # ensure that calculated accuracy match the reported accuracy for both models
        for (idx, acc, name) in [[i, accs[0], modelname_a], [j, accs[1], modelname_b]]:
            assert abs(acc - models_list[f"top{cfg.topn}"][idx]) < 5, \
                f'Calculated accuracy and reported accuracy for {name} by {round(abs(acc - models_list[f"top{cfg.topn}"][idx]),2)}'
        # log results
        wandb.log({'modelname_a': modelname_a,
                   'modelname_b': modelname_b,
                   'modeltop1_a': models_list['top1'][i],
                   'modeltop1_b': models_list['top1'][j],
                   'modeltop5_a': models_list['top5'][i],
                   'modeltop5_b': models_list['top5'][j],
                   'modelparams_a': models_list['param_count'][i],
                   'modelparams_b': models_list['param_count'][j],
                   'top1_diff': models_list['top1'][j] - models_list['top1'][i],
                   'top5_diff': models_list['top5'][j] - models_list['top5'][i],
                   'params_diff': models_list['param_count'][j] - models_list['param_count'][i],
                   'flips_type': f'top{cfg.topn}',
                   'pos_flips_abs': flips['pos_abs'],
                   'neg_flips_abs': flips['neg_abs'],
                   'net_flips_abs': flips['pos_abs'] - flips['neg_abs'],
                   'pos_flips_rel': flips['pos_rel'],
                   'neg_flips_rel': flips['neg_rel'],
                   'net_flips_rel': flips['pos_rel'] - flips['neg_rel']
                   })
        np.save('files/current_flip_seed.npy', cfg.seed)
        wandb.alert(
            title=f'COMPLETED: Flip Study Run (seed {cfg.seed})',
            text=f'Completed calculation for {modelname_a} -> {modelname_b}, neg-flips are {flips["neg_abs"]}'
                 f' ({round(time.time() - t_start, 2)}s)',
            level=AlertLevel.INFO
        )

    except AssertionError as err:
        print(f'Error for {modelname_a} -> {modelname_b}')
        # record error message
        try:
            error_list = pd.read_csv('files/error_log_V3.csv')
            idx = error_list.index[-1]+1
        except FileNotFoundError:
            error_list = pd.DataFrame()
            idx = 0
        row = pd.Series({'modelname_a': modelname_a, 'modelname_b': modelname_b, 'error': err}, name=idx)
        new_error_list = error_list.append(row)
        new_error_list.to_csv('files/error_log_V3.csv')
        wandb.alert(
            title=f'ERROR: Flip Study Run (seed {cfg.seed})',
            text=f'Error for {modelname_a} -> {modelname_b}; {err}',
            level=AlertLevel.ERROR
        )


if __name__ == '__main__':
    main()

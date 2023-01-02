import json
from pathlib import Path
from functools import partial

import pandas as pd
import numpy as np
from scipy.stats.mstats_basic import kendalltau
import typer

from tddl.post_processing.path_utils import logdir_to_paths
from tddl.post_processing.path_utils import paths_to_df
from tddl.post_processing.kendalls_tau import split_into_run, calculate_kendalls_tau_per_run, mean_std_over_runs
from tddl.post_processing.kendalls_tau import dfs_for_bar


def load_df(
    logdir: Path,
    tt_conversion: Path,
):
    paths = logdir_to_paths(logdir)
    df = paths_to_df(paths)
    
    # rank parameter in tltorch for TT convolutions has a different meaning than CP and Tucker
    df['actual_rank'] = df['rank']

    with open(tt_conversion, 'r') as f:
        rank_conf = json.load(f)

    for k,v in rank_conf.items():
        df.loc[df['rank'].isin(v), 'actual_rank'] = float(k)

    df['actual_rank'] = df['actual_rank'].astype(float)
    df['rank'] = df['rank'].astype(float, copy=False)

    df['test_error_before_ft'] = 1 - df.test_acc_before_ft
    df['test_error'] = 1 - df.test_acc
    df['valid_error_before_ft'] = 1 - df.valid_acc_before_ft
    df['valid_error'] = 1 - df.valid_acc

    df['log_test_error_before_ft'] = np.log(df.test_error_before_ft)
    df['log_test_error'] = np.log(df.test_error)
    df['log_valid_error_before_ft'] = np.log(df.valid_error_before_ft)
    df['log_valid_error'] = np.log(df.valid_error)

    df['fact_rank'] = df['factorization'] + '-' + df['rank'].apply(str)
    df['fact_layers'] = df['factorization'] + '-' + df['layers'].apply(str)
    df['layers_fact'] = df['layers'].apply(str) + '-' + df['factorization'] 

    df = df.astype({
        'layers':"category",
        'fact_layers':"category",
        'layers_fact':"category",
    })
    return df


def main(
    logdir: Path = Path("/bigdata/cifar10/logs/decomposed"),
    tt_conversion: Path = Path("/home/jetzeschuurman/gitProjects/phd/tddl/papers/iclr_2023/configs/rn18/rn18_tt_actual_rank_to_tl_ranks.json"),
    output: Path = Path("/home/jetzeschuurman/gitProjects/phd/tddl/papers/iclr_2023/tables"),
    model: str = 'rn18',
    dataset: str = 'c10'
):
    if model == 'rn18':
        model_name = "ResNet-18"
    elif model == 'gar':
        model_name = "GaripovNet"

    if dataset == 'c10':
        dataset_name = 'CIFAR-10'
    elif dataset == 'fm':
        dataset_name = 'F-MNIST'
    
    df = load_df(
        logdir = logdir,
        tt_conversion = tt_conversion
    )

    metrics=['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
    errors=['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']

    # use Tau-a to deal with ties: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient#Tau-a
    kendalltau_a = partial(kendalltau, use_ties=False, use_missing=False, method='auto')

    factorizations = ['cp','tucker','tt']

    ##########################################################################################################

    dfs = split_into_run(df, 5)

    performs = ['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
    approxs = ['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']
    N = 5

    df_layers_ = pd.DataFrame(columns=['kt','factorization', 'run', 'performance_metric', 'approximation_error','model','dataset'])
    
    ranks = df.actual_rank.unique()
    factorizations = df.factorization.unique()
    for p, perform in enumerate(performs):
        for a, approx in enumerate(approxs):
            for i in range(len(dfs)):
                # select i-th runs
                df_i = dfs[i]
                for r in ranks:
                    df_i_r = df_i[df_i.actual_rank == r]
                    # select rows where rank == r
                    for f in factorizations:
                        df_i_r_f = df_i_r[df_i_r.factorization == f]
                        # select rows where factorization == d
                        # kt over df_layers
                        c, _ = kendalltau_a(df_i_r_f[perform],df_i_r_f[approx])
                        df_layers_ = df_layers_.append({
                            'factorization':f,
                            'performance_metric':perform,
                            'approximation_error':approx,
                            'kt': c,
                            'model': model_name,
                            'dataset': dataset_name,
                            'rank': r,
                            'run': i,
                        }, ignore_index=True)

    df_layers_.to_pickle(output / f"kta_{model}_{dataset}_across_layers_incl75-9.zip")

    #########################################################################################################

    dfs = split_into_run(df, 5)

    performs = ['log_test_error_before_ft','log_test_error'] # 'log_valid_error_before_ft','log_valid_error',
    # performs = ['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
    approxs = ['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']
    N = 5

    df_layer_fact_ = pd.DataFrame(columns=['kt','layer', 'run', 'performance_metric', 'approximation_error','model','dataset'])
    # df_means = pd.DataFrame(columns=approxs, index=performs)
    # df_mean_factorizations = pd.DataFrame(columns=['kt','run', 'performance_metric', 'approximation_error','model','dataset'])

    # array = np.zeros((len(performs),len(approxs),len(factorizations),N))

    ranks = df.actual_rank.unique()
    # layers = df.layers.unique()
    for p, perform in enumerate(performs):
        for a, approx in enumerate(approxs):
            for i in range(len(dfs)):
                # select i-th runs
                df_i = dfs[i]
                for r in ranks:
                    df_i_r = df_i[df_i.actual_rank == r]
                    # select rows where rank == r
                    # for l in layers:
                        # df_i_r_l = df_i_r[df_i_r.layers == l]
                        # select rows where factorization == d
                        # kt over df_layers
                    c, _ = kendalltau_a(df_i_r[perform],df_i_r[approx])
                    df_layer_fact_ = df_layer_fact_.append({
                        # 'layer':l,
                        'performance_metric':perform,
                        'approximation_error':approx,
                        'kt': c,
                        'model': model_name,
                        'dataset': dataset_name,
                        'rank': r,
                        'run': i,
                    }, ignore_index=True)

    df_layer_fact_.to_pickle(output / f"kta_{model}_{dataset}_across_layer_facts_incl75-9.zip")

    ###############################################################################################

    dfs = split_into_run(df, 5)

    performs = ['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
    approxs = ['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']
    N = 5

    df_facts_ = pd.DataFrame(columns=['kt','layer', 'run', 'performance_metric', 'approximation_error','model','dataset'])

    ranks = df.actual_rank.unique()
    layers = df.layers.unique()
    for p, perform in enumerate(performs):
        for a, approx in enumerate(approxs):
            for i in range(len(dfs)):
                # select i-th runs
                df_i = dfs[i]
                for r in ranks:
                    df_i_r = df_i[df_i.actual_rank == r]
                    # select rows where rank == r
                    for l in layers:
                        df_i_r_l = df_i_r[df_i_r.layers == l]
                        # select rows where factorization == d
                        # kt over df_layers
                        c, _ = kendalltau_a(df_i_r_l[perform],df_i_r_l[approx])
                        df_facts_ = df_facts_.append({
                            'layer':l,
                            'performance_metric':perform,
                            'approximation_error':approx,
                            'kt': c,
                            'model': model_name,
                            'dataset': dataset_name,
                            'rank': r,
                            'run': i,
                        }, ignore_index=True)

    df_facts_.to_pickle(output / f"kta_{model}_{dataset}_across_facts_incl75-9.zip")

    ############################################################################################
    # Exclude the few observations (layer=28, decomp={cp,tucker}) where rank is 0.75 or 0.90

    df_ex_r = df[~df['actual_rank'].isin([0.75, 0.90])]

    dfs_ex_r = split_into_run(df_ex_r, 5)
    df_kts_ex_r = calculate_kendalls_tau_per_run(dfs_ex_r, errors=errors, metrics=metrics)
    
    df_bar = dfs_for_bar(
        df_kts_ex_r, 
        dataset=dataset_name, 
        model=model_name, 
        errors=errors, 
        metrics=metrics,
    )
    df_bar.to_pickle(output / f"kta_{model}_{dataset}_ex75-9.zip")

    ##################################################################################################

    performs = ['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
    approxs = ['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']
    N = 5

    df_layers_ = pd.DataFrame(columns=['kt','factorization', 'run', 'performance_metric', 'approximation_error','model','dataset'])

    ranks = df.actual_rank.unique()
    factorizations = df.factorization.unique()
    for p, perform in enumerate(performs):
        for a, approx in enumerate(approxs):
            for i in range(len(dfs_ex_r)):
                # select i-th runs
                df_i = dfs_ex_r[i]
                for r in ranks:
                    df_i_r = df_i[df_i.actual_rank == r]
                    # select rows where rank == r
                    for f in factorizations:
                        df_i_r_f = df_i_r[df_i_r.factorization == f]
                        # select rows where factorization == d
                        # kt over df_layers
                        c, _ = kendalltau_a(df_i_r_f[perform],df_i_r_f[approx])
                        df_layers_ = df_layers_.append({
                            'factorization':f,
                            'performance_metric':perform,
                            'approximation_error':approx,
                            'kt': c,
                            'model': model_name,
                            'dataset': dataset_name,
                            'rank': r,
                            'run': i,
                        }, ignore_index=True)

    df_layers_.to_pickle(output / f"kta_{model}_{dataset}_across_layers_ex75-9.zip")

    #########################################################################################################

    performs = ['log_test_error_before_ft','log_test_error'] # 'log_valid_error_before_ft','log_valid_error',
    # performs = ['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
    approxs = ['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']
    N = 5

    df_layer_fact_ = pd.DataFrame(columns=['kt','layer', 'run', 'performance_metric', 'approximation_error','model','dataset'])

    ranks = df.actual_rank.unique()
    # layers = df.layers.unique()
    for p, perform in enumerate(performs):
        for a, approx in enumerate(approxs):
            for i in range(len(dfs_ex_r)):
                # select i-th runs
                df_i = dfs_ex_r[i]
                for r in ranks:
                    df_i_r = df_i[df_i.actual_rank == r]
                    c, _ = kendalltau_a(df_i_r[perform],df_i_r[approx])
                    df_layer_fact_ = df_layer_fact_.append({
                        # 'layer':l,
                        'performance_metric':perform,
                        'approximation_error':approx,
                        'kt': c,
                        'model': model_name,
                        'dataset': dataset_name,
                        'rank': r,
                        'run': i,
                    }, ignore_index=True)

    df_layer_fact_.to_pickle(output / f"kta_{model}_{dataset}_across_layer_facts_ex75-9.zip")

    ###############################################################################################

    performs = ['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
    approxs = ['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']
    N = 5

    df_facts_ = pd.DataFrame(columns=['kt','layer', 'run', 'performance_metric', 'approximation_error','model','dataset'])

    ranks = df.actual_rank.unique()
    layers = df.layers.unique()
    for p, perform in enumerate(performs):
        for a, approx in enumerate(approxs):
            for i in range(len(dfs_ex_r)):
                # select i-th runs
                df_i = dfs_ex_r[i]
                for r in ranks:
                    df_i_r = df_i[df_i.actual_rank == r]
                    # select rows where rank == r
                    for l in layers:
                        df_i_r_l = df_i_r[df_i_r.layers == l]
                        # select rows where factorization == d
                        # kt over df_layers
                        c, _ = kendalltau_a(df_i_r_l[perform],df_i_r_l[approx])
                        df_facts_ = df_facts_.append({
                            'layer':l,
                            'performance_metric':perform,
                            'approximation_error':approx,
                            'kt': c,
                            'model': model_name,
                            'dataset': dataset_name,
                            'rank': r,
                            'run': i,
                        }, ignore_index=True)

    df_facts_.to_pickle(output / f"kta_{model}_{dataset}_across_facts_ex75-9.zip")


if __name__ == "__main__":
    typer.run(main)

import json
from pathlib import Path
import pandas as pd
import numpy as np

import os
from datetime import datetime
from os.path import isdir


import seaborn as sns
sns.set_theme(style="darkgrid")

logdir = Path("/bigdata/cifar10/logs/decomposed")

folders = os.listdir(logdir)
print(f"number of folders: {len(folders)}")


from tddl.post_processing.path_utils import logdir_to_paths

paths = logdir_to_paths(logdir)

print(len(paths))


# baseline

baseline_path = Path("/bigdata/cifar10/logs/baselines/1646668631/rn18_18_dNone_128_adam_l0.001_g0.1_w0.0_sTrue")
# baseline_model = torch.load(baseline_path / "cnn_best.pth")
# with open(baseline_path/'results.json') as json_file:
#     baseline_result = json.load(json_file)
# baseline_result



from tddl.post_processing.path_utils import paths_to_df

df = paths_to_df(paths)
print(len(df))
df.head()

# rank parameter in tltorch for TT convolutions has a different meaning than 

df['actual_rank'] = df['rank']
rank_conf = {
    10: [
        0.16,
        0.31,
        0.18,
        0.19,
        0.45,
        0.19,
        0.19,
        0.19,
    ],
    25: [
        0.61,
        1.43,
        1.22,
        2.39,
        1.21,
        2.39,
        4.78,
        4.78,
    ],
    50: [
        2.30,
        3.52,
        4.57,
        8.98,
        2.40,
        8.98,
        17.90,
        17.90,
    ],
    75: [
        6.60,
        7.8,
        12.97,
        25.6,
        28.0,
        25.6,
        50.0,
        50.0,
    ],
    90: [
        10.3,
        12.5,
        20.0,
        40.0,
        45.0,
        40.0,
        80.0,
        80.0,
    ],
}
for k,v in rank_conf.items():
    print(k,v)
    df.loc[df['rank'].isin(v), 'actual_rank'] = k/100

# df.groupby('rank').count()
# rank=0.90: 10 observations
# rank=0.75: 10 observstions


df['test_error_before_ft'] = 1 - df.test_acc_before_ft
df['test_error'] = 1 - df.test_acc
df['valid_error_before_ft'] = 1 - df.valid_acc_before_ft
df['valid_error'] = 1 - df.valid_acc

df['log_test_error_before_ft'] = np.log(df.test_error_before_ft)
df['log_test_error'] = np.log(df.test_error)
df['log_valid_error_before_ft'] = np.log(df.valid_error_before_ft)
df['log_valid_error'] = np.log(df.valid_error)


df['rank'] = df['rank'].astype(float, copy=False)
# df['rank'].apply(float)
df['rank'].unique()
df['fact_rank'] = df['factorization'] + '-' + df['rank'].apply(str)
df['fact_layers'] = df['factorization'] + '-' + df['layers'].apply(str)
df['layers_fact'] = df['layers'].apply(str) + '-' + df['factorization'] 
df.head()


df = df.astype({
    'layers':"category",
    'fact_layers':"category",
    'layers_fact':"category",
})


##################################
# create tables








metrics=['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
errors=['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']

from tddl.post_processing.kendalls_tau import split_into_run, calculate_kendalls_tau_per_run, mean_std_over_runs

dfs = split_into_run(df, 5)
df_kts = calculate_kendalls_tau_per_run(dfs, errors=errors, metrics=metrics)
df_kt_mean_std = mean_std_over_runs(df_kts, errors=errors, metrics=metrics)
df_kt_mean_std











##################################################################


from scipy.stats.mstats_basic import kendalltau
from functools import partial

kendalltau_a = partial(kendalltau, use_ties=False, use_missing=False, method='auto')



factorizations = ['cp','tucker','tt']



dfs = split_into_run(df, 5)

model = 'ResNet-18'
dataset = 'CIFAR-10'
performs = ['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
approxs = ['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']
N = 5

df_layers_ = pd.DataFrame(columns=['kt','factorization', 'run', 'performance_metric', 'approximation_error','model','dataset'])
# df_means = pd.DataFrame(columns=approxs, index=performs)
# df_mean_factorizations = pd.DataFrame(columns=['kt','run', 'performance_metric', 'approximation_error','model','dataset'])

array = np.zeros((len(performs),len(approxs),len(factorizations),N))

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
                        'model': model,
                        'dataset': dataset,
                        'rank': r,
                        'run': i,
                    }, ignore_index=True)

df_layers_.to_pickle("./tables/kta_rn18_c10_bar_log_error_across_layers_incl75-9.zip")



#########################################################################################################


dfs = split_into_run(df, 5)

model = 'ResNet-18'
dataset = 'CIFAR-10'
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
                print(f'{approx=}', f'{r=}', f'{i=}', f'{c=}')
                df_layer_fact_ = df_layer_fact_.append({
                    # 'layer':l,
                    'performance_metric':perform,
                    'approximation_error':approx,
                    'kt': c,
                    'model': model,
                    'dataset': dataset,
                    'rank': r,
                    'run': i,
                }, ignore_index=True)

df_layer_fact_.to_pickle("./tables/kta_rn18_c10_bar_log_error_across_layer_facts_incl75-9.zip")



###############################################################################################






dfs = split_into_run(df, 5)

model = 'ResNet-18'
dataset = 'CIFAR-10'
performs = ['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
approxs = ['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']
N = 5

df_facts_ = pd.DataFrame(columns=['kt','layer', 'run', 'performance_metric', 'approximation_error','model','dataset'])
# df_means = pd.DataFrame(columns=approxs, index=performs)
# df_mean_factorizations = pd.DataFrame(columns=['kt','run', 'performance_metric', 'approximation_error','model','dataset'])

# array = np.zeros((len(performs),len(approxs),len(factorizations),N))

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
                        'model': model,
                        'dataset': dataset,
                        'rank': r,
                        'run': i,
                    }, ignore_index=True)

df_facts_.to_pickle("./tables/kta_rn18_c10_bar_log_error_across_facts_incl75-9.zip")













############################################################################################
############################################################################################
############################################################################################
# Exclude the few observations (layer=28, decomp={cp,tucker}) where rank is 0.75 or 0.90

df_ex_r = df[~df['actual_rank'].isin([0.75, 0.90])]

dfs_ex_r = split_into_run(df_ex_r, 5)
df_kts_ex_r = calculate_kendalls_tau_per_run(dfs_ex_r, errors=errors, metrics=metrics)
df_kt_mean_std_ex_r = mean_std_over_runs(df_kts_ex_r, errors=errors, metrics=metrics)
df_kt_mean_std_ex_r



from tddl.post_processing.kendalls_tau import dfs_for_bar

df_bar = dfs_for_bar(
    df_kts_ex_r, 
    dataset='CIFAR-10', 
    model='ResNet-18', 
    errors=errors, 
    metrics=metrics,
)
df_bar.to_pickle("./tables/kta_rn18_c10_bar_log_error_ex-r.zip")




model = 'ResNet-18'
dataset = 'CIFAR-10'
performs = ['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
approxs = ['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']
N = 5

df_layers_ = pd.DataFrame(columns=['kt','factorization', 'run', 'performance_metric', 'approximation_error','model','dataset'])
# df_means = pd.DataFrame(columns=approxs, index=performs)
# df_mean_factorizations = pd.DataFrame(columns=['kt','run', 'performance_metric', 'approximation_error','model','dataset'])

array = np.zeros((len(performs),len(approxs),len(factorizations),N))

ranks = df.actual_rank.unique()
factorizations = df.factorization.unique()
for p, perform in enumerate(performs):
    for a, approx in enumerate(approxs):
        for i in range(len(dfs_ex_r)):
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
                        'model': model,
                        'dataset': dataset,
                        'rank': r,
                        'run': i,
                    }, ignore_index=True)

df_layers_.to_pickle("./tables/kta_rn18_c10_bar_log_error_across_layers.zip")



#########################################################################################################



model = 'ResNet-18'
dataset = 'CIFAR-10'
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
        for i in range(len(dfs_ex_r)):
            # select i-th runs
            df_i = dfs_ex_r[i]
            for r in ranks:
                df_i_r = df_i[df_i.actual_rank == r]
                # select rows where rank == r
                # for l in layers:
                    # df_i_r_l = df_i_r[df_i_r.layers == l]
                    # select rows where factorization == d
                    # kt over df_layers
                c, _ = kendalltau_a(df_i_r[perform],df_i_r[approx])
                print(f'{approx=}', f'{r=}', f'{i=}', f'{c=}')
                df_layer_fact_ = df_layer_fact_.append({
                    # 'layer':l,
                    'performance_metric':perform,
                    'approximation_error':approx,
                    'kt': c,
                    'model': model,
                    'dataset': dataset,
                    'rank': r,
                    'run': i,
                }, ignore_index=True)

df_layer_fact_.to_pickle("./tables/kta_rn18_c10_bar_log_error_across_layer_facts.zip")



###############################################################################################







model = 'ResNet-18'
dataset = 'CIFAR-10'
performs = ['log_valid_error_before_ft','log_valid_error','log_test_error_before_ft','log_test_error']
approxs = ['relative_norm_weight','relative_norm','scaled_norm_weight','scaled_norm','diff_norm_weight','norm_diff']
N = 5

df_facts_ = pd.DataFrame(columns=['kt','layer', 'run', 'performance_metric', 'approximation_error','model','dataset'])
# df_means = pd.DataFrame(columns=approxs, index=performs)
# df_mean_factorizations = pd.DataFrame(columns=['kt','run', 'performance_metric', 'approximation_error','model','dataset'])

# array = np.zeros((len(performs),len(approxs),len(factorizations),N))

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
                        'model': model,
                        'dataset': dataset,
                        'rank': r,
                        'run': i,
                    }, ignore_index=True)

df_facts_.to_pickle("./tables/kta_rn18_c10_bar_log_error_across_facts.zip")


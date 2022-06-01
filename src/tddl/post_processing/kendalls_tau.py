# from scipy.stats import kendalltau
import pandas as pd
import numpy as np

from scipy.stats.mstats_basic import kendalltau
from functools import partial

kendalltau_a = partial(kendalltau, use_ties=False, use_missing=False, method='auto')



def split_into_run(df, n_runs):
    dfs = []
    df_ = df.copy()
    for i in range(n_runs):
        df_grouped_lfr = df_.groupby(['layers','factorization','rank'])
        idx = df_grouped_lfr.sample(n=1).index
        dfs.append(df_.loc[idx])
        df_.drop(index=idx, inplace=True)

    return dfs


def split_into_layers(df, column_name):
    df_ = df.copy()
    layers = df[column_name].unique()
    
    dfs_layers = []
    for layer in layers:
        dfs_layers.append(df_.loc[df_.nr == layer])
        
    return dfs_layers


def calculate_kendalls_tau(
    df,
    errors = ['relative_norm_weight','scaled_norm_weight','diff_norm_weight', 'relative_norm','scaled_norm','norm_diff','layers'],
    metrics = ['valid_acc_before_ft','valid_acc','test_acc'],
    **kwargs,
):
    df_kt = pd.DataFrame(index=metrics, columns=errors)

    for error in errors:
        for metric in metrics:
            corr, p = kendalltau_a(df[error], df[metric], **kwargs)
            df_kt[error][metric] = corr

    return df_kt


def calculate_kendalls_tau_per_run(
    dfs,
    *args,
    **kwargs,
):
    df_kts = []
    for df in dfs:
        df_kt = calculate_kendalls_tau(df,*args,**kwargs)
        df_kts.append(df_kt.to_numpy(dtype=np.float32))

    return df_kts


def mean_over_runs(
    df_kts,
):
    return np.stack(df_kts).mean(axis=0)


def std_over_runs(
    df_kts,
):
    return np.stack(df_kts).std(axis=0)


def mean_std_over_runs(
    df_kts,
    errors = ['relative_norm_weight','scaled_norm_weight','diff_norm_weight', 'relative_norm','scaled_norm','norm_diff','layers'],
    metrics = ['valid_acc_before_ft','valid_acc','test_acc'],
):
    df_kt_mean = pd.DataFrame(mean_over_runs(df_kts), index=metrics, columns=errors)
    df_kt_std = pd.DataFrame(std_over_runs(df_kts), index=metrics, columns=errors)
    
    return df_kt_mean.applymap(lambda s: "{:.2f}±".format(s)) + df_kt_std.applymap(lambda s: "{:.2f}".format(s))


def df_for_bar(
    df_kt,
    errors = ['relative_norm_weight','scaled_norm_weight','diff_norm_weight', 'relative_norm','scaled_norm','norm_diff','layers'],
    metrics = ['valid_acc_before_ft','valid_acc','test_acc'],
    dataset = None,
    model = None
):
    
    df = pd.DataFrame(columns=['kt','acc_type','approx_type','dataset','model'])
    
    for j,error in enumerate(errors):
        for i,metric in enumerate(metrics):
            df = df.append({
                'kt': df_kt[i,j],
                'acc_type': metric,
                'approx_type': error,
                'dataset': dataset,
                'model': model,
            }, ignore_index=True)
    return df


def dfs_for_bar(
    df_kts,
    *args,
    **kwargs,
):
    df_bar = pd.DataFrame()
    for df in df_kts:
        df_bar = df_bar.append(df_for_bar(df,*args,**kwargs))
    return df_bar

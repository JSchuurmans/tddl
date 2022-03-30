from scipy.stats import kendalltau
import pandas as pd
import numpy as np


def split_into_run(df, n_runs):
    dfs = []
    df_ = df.copy()
    for i in range(n_runs):
        df_grouped_lfr = df_.groupby(['layers','factorization','rank'])
        idx = df_grouped_lfr.sample(n=1).index
        dfs.append(df_.loc[idx])
        df_.drop(index=idx, inplace=True)

    return dfs


def calculate_kendalls_tau(
    df,
    errors = ['relative_norm_weight','scaled_norm_weight','diff_norm_weight', 'relative_norm','scaled_norm','norm_diff','layers'],
    metrics = ['valid_acc_before_ft','valid_acc','test_acc'],
):
    df_kt = pd.DataFrame(index=metrics, columns=errors)

    errors = ['relative_norm_weight','scaled_norm_weight','diff_norm_weight', 'relative_norm','scaled_norm','norm_diff','layers']
    metrics = ['valid_acc_before_ft','valid_acc','test_acc']
    for error in errors:
        for metric in metrics:
            corr, p = kendalltau(df[error], df[metric])
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

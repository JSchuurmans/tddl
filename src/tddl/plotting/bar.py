import seaborn as sns

def bar_diff(
    df,
    x = "layer_nr",
    y = "relative_norm_1",
    save_path = None,
):
    fig = sns.barplot(data=df, x=x, y=y)
    
    if save_path is not None:
        fig.get_figure().savefig(save_path)

    return fig
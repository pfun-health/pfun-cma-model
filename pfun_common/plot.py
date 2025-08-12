import matplotlib.pyplot as plt
import pandas as pd
from pfun_cma_model.misc.pathdefs import PFunDataPaths


def plot_risk_mosaic(df_local: pd.DataFrame, tcol: str = 'tod', xcol: str = 'G', save: bool = False):
    """Plot calculated risk as a mosaic plot (`maptlotlib.pyplot.subplot_mosaic`)

    Args:
        df_local (pd.DataFrame): Pandas DataFrame with time column `tcol` and data column `xcol`.
        tcol (str, optional): name of time column. Defaults to 'tod'.
        xcol (str, optional): name of input data/signal column. Defaults to 'G'.

    Returns:
        matplotlib.pyplot.Figure, matplotlib.pyplot.AxesDict: mosaic figure and axes dictionary
    """
    plt.close('all')
    import matplotlib
    matplotlib.use('agg')  # force Agg backend for simplicity
    labels = [col for col in df_local.columns if 'risk' in col]
    colors = ['lightgrey', 'orange', 'green', 'red']
    mosaic = [[xcol,]*3] + [[col,]*3 for col in labels]
    fig, axes = plt.subplot_mosaic(
        mosaic, figsize=(16, 12))  # type: ignore[arg-type]
    for j, k in enumerate(labels):
        axes[k].plot(df_local[tcol], df_local[k], label=k, color=colors[j])
        axes[k].legend(bbox_to_anchor=(1.05, 1.0),
                       loc='upper left', borderaxespad=0.0)
    axes[xcol].plot(df_local[tcol], df_local[xcol], label=f'{xcol}', color='k')
    axes[xcol].legend(bbox_to_anchor=(1.05, 1.0),
                      loc='upper left', borderaxespad=0.0)
    if save is True:
        output_fpath = PFunDataPaths().pfun_data_dirpath.joinpath(
            "..", "results", "risk_mosaic.png")
        fig.savefig(output_fpath)
    return fig, axes

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import logging
logger = logging.getLogger("pfun_cma_model")


def lineplot(df: pd.DataFrame, tcol='ts_local', ycol='sg') -> Axes:
    """Quality-of-life lineplot function for quick n dirty plots of glucose."""
    axes = sns.lineplot(df, x=tcol, y=ycol)
    return axes

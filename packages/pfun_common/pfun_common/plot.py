import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

logger = logging.getLogger("pfun_cma_model")


def lineplot(df: pd.DataFrame, tcol="ts_local", ycol="sg") -> Axes:
    """Quality-of-life lineplot function for quick n dirty plots of glucose."""
    axes = sns.lineplot(df, x=tcol, y=ycol)
    return axes

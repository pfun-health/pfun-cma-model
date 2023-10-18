

from typing import Optional, Tuple, Container, AnyStr
from dataclasses import dataclass
import logging
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from base64 import b64encode


@dataclass
class CMAPlotConfig:
    """configuration for plotting the CMA model results"""

    plot_cols: Optional[Tuple[str]] = (
        "g_0", "g_1", "g_2", "G", "c", "m", "a", "L", "I_S", "I_E", "is_meal", "g_raw")
    labels: Optional[Tuple[str]] = ("Breakfast", "Lunch", "Dinner",
                                    "Glucose", "Cortisol", "Melatonin",
                                    "Adiponectin", "Photoperiod (irradiance)",
                                    "Insulin (secreted)",
                                    "Insulin (effective)",
                                    "Meals",
                                    "Glucose (Data)")
    colors: Optional[Tuple[str]] = (
        "#ec5ef9",
        "#bd4bc7",
        "#8b3793",
        "purple",
        "cyan",
        "darkgrey",
        "m",
        'tab:orange',
        'tab:red',
        'red',
        'k',
        'darkgrey'
    )

    @classmethod
    def get_label(cls, col: Container | AnyStr):
        if not isinstance(col, str):
            return [cls.get_label(c) for c in col]
        index = cls.plot_cols.index(col)
        return cls.labels[index]

    @classmethod
    def get_color(cls, col: Container | AnyStr, rgba=False, as_hex=False, keep_alpha=False):
        if not isinstance(col, str):
            return [cls.get_color(c, rgba=rgba) for c in col]
        try:
            index = cls.plot_cols.index(col)
            c = cls.colors[index]
        except (IndexError, ValueError) as excep:
            msg = f"failed to find a plot color for: {col}"
            logging.warning(msg, exc_info=1)
            raise excep.__class__(msg)
        if rgba is True or as_hex is True:
            c = matplotlib.colors.to_rgba(c)
            if as_hex is True:
                c = matplotlib.colors.rgb2hex(c, keep_alpha=keep_alpha)
        return c

    @classmethod
    def set_global_axis_properties(cls, axs):
        """set universal axis properties (like time of day labels for x-axis)"""
        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.grid(True)
            ax.set_xticks([0, 6, 12, 18, 23], ['Midnight', '6AM', 'Noon', '6PM', '11PM'])
            ax.set_xlim([0.01, 23.99])
            ax.set_xlabel("Time (24-hours)")
        return axs
    
    @classmethod
    def set_global_axis_attributes(cls, axs):
        """alias for set_global_axis_properties..."""
        return cls.set_global_axis_properties(axs)

    @classmethod
    def plot_model_results(cls, df=None, soln=None, plot_cols=None, separate2subplots=False, as_blob=True):
        """plot the results of the model"""
        if df is None:
            raise ValueError("df is None")
        if soln is None:
            raise ValueError("soln is None")
        if plot_cols is None:
            plot_cols = cls.plot_cols
        #: drop is_meal from plot cols... (it's bool afterall)
        plot_cols = list(plot_cols)
        if "is_meal" in plot_cols:
            ismeal_ix = plot_cols.index("is_meal")
            plot_cols.pop(ismeal_ix)
        #: combine the data into a single dataframe
        df = df.set_index("t")
        soln = soln.set_index("t")
        df = pd.merge_ordered(df.copy(), soln, suffixes=("", "_soln"), on="t")
        df = df.set_index("t")
        fig, axs = plt.subplots(nrows=2 if separate2subplots is False else len(plot_cols) + 1)
        #: plot meal times, meal sizes
        ax = axs[0]
        ax = df.plot.area(y="G_soln", color='k', ax=ax, label="Estimated Meal Size")
        ax.vlines(x=df.loc[df.is_meal.astype(float).fillna(0.0) > 0].index,
                  ymin=ax.get_ylim()[0], ymax=df.G_soln.max(),
                  color='r', lw=3, linestyle='--', label='estimated mealtimes')
        ax.legend()
        #: plot other traces
        if separate2subplots is False:
            df.plot.area(y=plot_cols, color=cls.get_color(plot_cols), ax=axs[1],
                         alpha=0.2, label=cls.get_label(plot_cols), stacked=True)
        elif separate2subplots is True:
            for pcol, axi in zip(plot_cols, axs[1:]):
                axi.fill_between(
                    x=df.index, y1=df[pcol].min(), y2=df[pcol],
                    color=cls.get_color(pcol),
                    alpha=0.2,
                    label=cls.get_label(pcol)
                )
                axi.legend()
        #: set global properties for all axes...
        axs = cls.set_global_axis_properties(axs)
        #: return the figure and axes (unless this is to be a blob for the web)
        if as_blob is False:
            return fig, axs
        bio = BytesIO()
        fig.savefig(bio, format='png')
        bio.seek(0)
        bytes_value = bio.getvalue()
        img_src = 'data:image/png;base64,'
        img_src = img_src + b64encode(bytes_value).decode('utf-8')
        plt.close()
        return img_src

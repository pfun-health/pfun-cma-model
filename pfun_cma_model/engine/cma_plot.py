import logging
from base64 import b64encode
from dataclasses import dataclass, field

from io import BytesIO
from typing import Annotated, Iterable, Literal, Tuple

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["CMAPlotConfig"]


@dataclass
class CMAPlotConfig:
    """configuration for plotting the CMA model results"""
    plot_cols: Tuple[str, ...] = field(default_factory=lambda: (
        "g_0", "g_1", "g_2", "G", "c", "m", "a", "L", "I_S", "I_E", "is_meal", "value"))
    labels: Annotated[  # type: ignore
        tuple[str],
        tuple[
            Literal["Breakfast"], Literal["Lunch"], Literal["Dinner"], Literal["Glucose"], Literal["Cortisol"],
            Literal["Melatonin"], Literal["Adiponectin"], Literal[
                "Photoperiod (irradiance)"],
            Literal["Insulin (secreted)"], Literal["Insulin (effective)"], Literal["Meals"], Literal["Glucose (Data)"],
        ],
    ] = (
        "Breakfast",
        "Lunch",
        "Dinner",
        "Glucose",
        "Cortisol",
        "Melatonin",
        "Adiponectin",
        "Photoperiod (irradiance)",
        "Insulin (secreted)",
        "Insulin (effective)",
        "Meals",
        "Glucose (Data)",
    )  # type: ignore

    colors: tuple[
        Literal["#ec5ef9"],
        Literal["#bd4bc7"],
        Literal["#8b3793"],
        Literal["purple"],
        Literal["cyan"],
        Literal["darkgrey"],
        Literal["m"],
        Literal["tab:orange"],
        Literal["tab:red"],
        Literal["red"],
        Literal["k"],
        Literal["darkgrey"],
    ] = (
        "#ec5ef9",
        "#bd4bc7",
        "#8b3793",
        "purple",
        "cyan",
        "darkgrey",
        "m",
        "tab:orange",
        "tab:red",
        "red",
        "k",
        "darkgrey",
    )

    subplot_kwds: dict = field(
        default_factory=lambda: {
            "nrows": 2,
            "figsize": (14, 10),
        }
    )

    def __post_init__(self, **subplot_kwds):
        pass

    def __call__(self, **subplot_kwds):
        self.fig, self.axes = self.setup_figure_axes(
            **subplot_kwds
        )
        return self

    @property
    def axs(self):
        """alias for axes"""
        return self.axes

    @axs.setter
    def axs(self, value):
        self.axes = value

    @classmethod
    def get_label(cls, col: Iterable[str] | str):
        if not isinstance(col, str):
            return [cls.get_label(c) for c in col]
        index = cls().plot_cols.index(col)
        return cls.labels[index]

    @classmethod
    def get_color(
        cls, col: Iterable[str] | str, rgba=False, as_hex=False,
        keep_alpha=False
    ):
        if not isinstance(col, str):
            return [cls.get_color(c, rgba=rgba) for c in col]
        try:
            index = cls().plot_cols.index(col)
            c = cls.colors[index]
            c_rgba = colors.to_rgba(c)
            if as_hex is True:
                c = colors.rgb2hex(c_rgba, keep_alpha=keep_alpha)
            elif rgba is True:
                c = c_rgba  # type: ignore
        except (IndexError, ValueError) as excep:
            msg = f"failed to find a plot color for: {col}"
            logging.warning(msg, exc_info=True)
            raise excep.__class__(msg)
        return c

    @classmethod
    def set_global_axis_properties(cls, axs):
        """set universal axis properties (like time of day labels for x-axis)"""
        for ax in axs:
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.tick_params(axis="both", which="minor", labelsize=12)
            ax.grid(True)
            ax.set_xticks(
                [0, 6, 12, 18, 23], ["Midnight", "6AM", "Noon", "6PM", "11PM"]
            )
            ax.set_xlim([0.01, 23.99])
            ax.set_xlabel("Time (24-hours)")
        return axs

    @classmethod
    def set_global_axis_attributes(cls, axs):
        """alias for set_global_axis_properties..."""
        return cls.set_global_axis_properties(axs)

    @staticmethod
    def setup_input_data(df):
        """setup the input dataframe for plotting"""
        df = df.copy()
        df = df.set_index("t")
        return df

    @staticmethod
    def setup_solution(soln):
        """setup the solution dataframe for plotting"""
        df = soln.copy()
        df = df.set_index("t")
        # ensure the expected column names are present for soln
        df.rename(columns={"G": "G_soln"}, inplace=True)
        return df

    @staticmethod
    def combine_dataframes(df, soln):
        """combine the input dataframe and solution dataframe for plotting"""
        df = pd.merge_ordered(df.copy(), soln, suffixes=("", "_soln"), on="t")
        df = df.set_index("t")
        return df

    @staticmethod
    def prune_plot_cols(df, plot_cols):
        # prune plot_cols to available columns
        plot_cols = [c for c in plot_cols if c in df.columns]
        return plot_cols

    @classmethod
    def setup_figure_axes(cls, plot_cols=None, **subplot_kwds):
        if plot_cols is None:
            plot_cols = cls().plot_cols
        #: drop is_meal from plot cols... (it's bool afterall)
        plot_cols = list(plot_cols)
        if "is_meal" in plot_cols:
            ismeal_ix = plot_cols.index("is_meal")
            plot_cols.pop(ismeal_ix)
        # prepare subplots configuration
        subplot_kwds_defaults = {
            "nrows": 2,
            "figsize": (14, 10),
        }
        # ! override provided value for nrows
        if "nrows" in subplot_kwds:
            subplot_kwds["nrows"] = subplot_kwds_defaults["nrows"]
            logging.debug(
                "Provided value for nrows was overwritten. "
                "See options for 'separate2subplots'."
            )
        # include other defaults if not provided:
        for k in subplot_kwds_defaults:  # type: ignore
            if k not in subplot_kwds:
                subplot_kwds[k] = subplot_kwds_defaults[k]
        #: Instantiate the figure and subplots (axes)
        fig, axs = plt.subplots(**subplot_kwds)
        return fig, axs

    def plot(self, df, plot_cols=None, separate2subplots=False):
        """plot the given data"""
        #: plot meal times, meal sizes
        self.axs = self.plot_meals(df, self.axs)
        #: determine the secondary plot type
        secondary_plot_funcs = {
            True: self.plot_separate_subplots,
            False: self.plot_unified,
        }
        secondary_plot_func = secondary_plot_funcs[separate2subplots]
        #: plot the other traces
        self.axs = secondary_plot_func(df, plot_cols, self.axs)
        return self.fig, self.axs

    @classmethod
    def format_save_figure(cls, fig, axs, as_blob=False):
        """format, then save the figure (as a blob for the web, or as a file)"""
        #: set global properties for all axes...
        axs = cls.set_global_axis_properties(axs)
        #: return the figure and axes (not to be a blob)
        if as_blob is False:
            return fig, axs
        #: otherwise, save it as a blob for the web
        return cls.save_figure_as_blob(fig)

    @classmethod
    def plot_meals(cls, df, axs):
        """plot meal times, meal sizes."""
        ax = axs[0]
        ax = df.plot.area(y="G_soln", color="k", ax=ax,
                          label="Estimated Meal Size")
        ax.vlines(
            x=df.loc[df.is_meal.astype(float).fillna(0.0) > 0].index,
            ymin=ax.get_ylim()[0],
            ymax=df.G_soln.max(),
            color="r",
            lw=3,
            linestyle="--",
            label="estimated mealtimes",
        )
        ax.legend()
        return axs

    @classmethod
    def plot_unified(cls, df, plot_cols, axs):
        """plot the other traces as a single area chart."""
        df.plot.area(
            y=plot_cols,
            color=cls.get_color(plot_cols),
            ax=axs[1],
            alpha=0.2,
            label=cls.get_label(plot_cols),
            stacked=False,
        )
        return axs

    @classmethod
    def plot_separate_subplots(cls, df, plot_cols, axs):
        """plot the other traces in separate subplots."""
        for pcol, axi in zip(plot_cols, axs[1:]):
            axi.fill_between(
                x=df.index,
                y1=df[pcol].min(),
                y2=df[pcol],
                color=cls.get_color(pcol),
                alpha=0.2,
                label=cls.get_label(pcol),
                title=cls.get_label(pcol),
            )
            axi.legend()
        return axs

    @classmethod
    def save_figure_as_blob(cls, fig):
        """save the figure as a blob for the web."""
        bio = BytesIO()
        fig.savefig(bio, format="png")
        bio.seek(0)
        bytes_value = bio.getvalue()
        img_src = "data:image/png;base64,"
        img_src = img_src + b64encode(bytes_value).decode("utf-8")
        plt.close()
        return img_src

    @property
    def blob(self):
        """return the figure as a blob for the web."""
        return self.save_figure_as_blob(self.fig)


class CMAPlotDataConfig(CMAPlotConfig):
    """configuration for plotting the input data"""

    def plot(self, df, plot_cols=None, separate2subplots=False):
        """plot the given data"""
        df = self.setup_input_data(df)
        plot_cols = self.prune_plot_cols(df, plot_cols)
        self.fig, self.axes = super().plot(df, plot_cols, separate2subplots)
        return self.fig, self.axs


class CMAPlotSolnConfig(CMAPlotConfig):
    """configuration for plotting the model solution"""

    def plot(self, df, plot_cols=None, separate2subplots=False):
        """plot the model solution."""
        soln = self.setup_solution(df)
        plot_cols = self.prune_plot_cols(df, plot_cols)
        self.fig, self.axes = super().plot(df, plot_cols, separate2subplots)
        return self.fig, self.axes


class CMAPlotCombinedConfig(CMAPlotConfig):
    """configuration for plotting the combined data"""

    def plot_combined(self, df, soln, plot_cols=None, separate2subplots=False):
        """plot the combined data+solution."""
        df = self.setup_input_data(df)
        soln = self.setup_solution(soln)
        df_combined = self.combine_dataframes(df, soln)
        plot_cols = self.prune_plot_cols(df, plot_cols)
        self.fig, self.axes = super().plot(
            df_combined, plot_cols, separate2subplots)

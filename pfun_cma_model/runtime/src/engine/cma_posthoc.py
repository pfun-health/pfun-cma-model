import datetime
import json
import logging
from argparse import Namespace as Namespace_
from base64 import b64encode
from enum import Enum, EnumMeta
from functools import cache
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    AnyStr,
    Callable,
    Container,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template
from pydantic import BaseModel, Field, root_validator, validator
from pydantic.json import timedelta_isoformat

from pfun_cma_model.dexcom import DexcomEndpoint
from pfun_cma_model.runtime.src.engine.calc import normalize
from pfun_cma_model.runtime.src.engine.cma_plot import CMAPlotConfig
from pfun_cma_model.runtime.src.engine.data_utils import diff_tod_hours

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#: A list of colors to use for the different types of data.
mplcolors = mcolors.get_named_colors_mapping()


class Namespace(Namespace_):
    """wrapper around argparse.Namespace that implements a simple __getitem__ method.
    Examples:
    ---------
    >>> ns = Namespace(a=1, b=2)
    >>> ns['a']
    1
    >>> ns['b']
    2
    """

    def __getitem__(self, key):
        assert key in self.__dict__, f"key {key} not found in Namespace"
        return self.__dict__[key]


class DexcomRecordsModel(BaseModel):
    userId: str
    recordType: DexcomEndpoint
    queryDate: Optional[pd.Timestamp]
    tz_offset: int | float
    records: Dict | str | List

    @validator("records", allow_reuse=True, pre=True)
    def load_records_json(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            pd.Timestamp: lambda ts: ts.isoformat(),
        }


def stats_aliaser(stats_name):
    return stats_name.lower()


class CaseInsensitiveBaseModel(BaseModel):
    @root_validator(pre=True, allow_reuse=True)
    def case_insensitive_dict(cls, values):
        return {k.lower(): v for k, v in values.items()}


class ModelResultStats(CaseInsensitiveBaseModel, BaseModel):
    G_morn: float
    G_eve: float
    I_S_morn: float
    I_S_eve: float

    class Config:
        allow_extras = True
        orm_mode = True
        alias_generator = stats_aliaser


class ModelResult(BaseModel):
    """PFun CMA model result schema (web/SQL-ready)"""

    userId: str
    queryDate: pd.Timestamp | str | datetime.datetime
    message: str
    time: Container | List | dict | str
    formatted_data: Container | dict | str
    soln: Container | dict | str
    mesg: str
    popt: dict | str
    stats: ModelResultStats | Dict
    img_src: Optional[str]

    @validator("stats", pre=True, allow_reuse=True)
    def stats2model(cls, v):
        if isinstance(v, dict):
            return ModelResultStats(**v)
        return v

    @validator(
        "time", "formatted_data", "soln", "popt", "stats", pre=True, allow_reuse=True
    )
    def load_json(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @validator("time", allow_reuse=True, pre=True)
    def fixup_time(cls, value, values):
        time = pd.to_datetime(pd.Series(value, name="time"))
        time.index.rename("t", inplace=True)
        name_ix = time.index.name
        try:
            time_tmp = time.reset_index(name="time")
            time_tmp[name_ix] = time_tmp[name_ix].astype(float)
            time_tmp.set_index(name_ix, inplace=True, drop=True)
            time = pd.Series(time_tmp["time"], name="time")
        except Exception:
            logging.error("failed to fix time vector in ModelResult!", exc_info=1)
        finally:
            return time

    @validator("queryDate", allow_reuse=True)
    def fix_queryDate(cls, v):
        if not isinstance(v, str):
            v = v.isoformat()
        return v

    @validator("formatted_data", "soln", allow_reuse=True)
    def convert2dataframe(cls, v):
        def verify_dtypes(df):
            """
            Verify and update the data types of the columns in the given DataFrame.

            Parameters:
                df (DataFrame): The DataFrame to verify and update.

            Returns:
                DataFrame: The updated DataFrame with the verified and updated data types.
            """
            if df.index.name == "t":
                df.reset_index(inplace=True)
            df.dtypes.update(
                {"t": float, "tod": float, "G": float, "time": "datetime64[ns]"}
            )
            if "t" in df.columns:
                df["t"] = df["t"].astype(float)
                df.set_index("t", inplace=True)
            df.set_index(df.index.astype(float))
            df.index.rename("t", inplace=True)
            df.sort_index(inplace=True)
            return df

        if isinstance(v, pd.DataFrame):
            df = v
        elif isinstance(v, dict):
            df = pd.DataFrame(v)
        else:
            raise TypeError(f"incorrect type for {v} (ModelResult in models.py)")
        df = verify_dtypes(df)
        return df

    @validator("formatted_data", allow_reuse=True)
    def append_time_column(cls, value, values):
        is_dict = isinstance(value, dict)
        if is_dict:
            value = pd.DataFrame(value)  #: ! temporarily convert to dataframe
            values["time"] = pd.Series(values["time"])
        if "time" in values:
            time = cls.fixup_time(values["time"], None)
            if not isinstance(time, pd.Series):
                time = pd.Series(time, name="time")
                time.index.rename("t", inplace=True)
            value.index.rename("t", inplace=True)
            if "time" in value.columns:
                value.drop(columns="time", inplace=True)
            value = pd.merge_asof(
                value,
                time,
                on="t",
                tolerance=0.1,
            )
        if is_dict:
            value = value.to_dict()  #: ! convert back to original type
            values["time"] = values["time"].to_dict()
        return value

    @validator("soln", allow_reuse=True)
    def ensureFloatIndex(cls, v, values):
        soln = v
        soln.set_index(v.index.astype(float), inplace=True)
        data = values["formatted_data"]
        g_raw = (
            pd.merge_asof(soln.sort_index().reset_index(), data, on="t")[["G_x", "G_y"]]
            .mean(axis=1)
            .to_frame()
            .set_index(soln.index)
        )
        soln["g_raw"] = g_raw
        return soln

    class Config:
        arbitrary_types_allowed = True
        allow_extras = True
        orm_mode = True
        json_encoders = {
            pd.DataFrame: lambda df: df.to_json(),
            pd.Series: lambda s: s.to_json(),
            datetime.datetime: lambda v: v.timestamp(),
            datetime.timedelta: timedelta_isoformat,
            pd.Timestamp: lambda ts: ts.isoformat(),
        }


class ModelResultSafe(ModelResult, BaseModel):
    """schema model for modelresult that is safe to use with SQLAlchemy (dataframe->dict)"""

    @validator("time", allow_reuse=True, pre=False)
    def ensure_sql_compatible_time(cls, vtime):
        if isinstance(vtime, dict):
            vtime = pd.Series(vtime)
        vtime = vtime.apply(
            lambda d: d.isoformat() if hasattr(d, "isoformat") else d
        ).astype(str)
        return vtime.to_dict()

    @validator("time", allow_reuse=True, pre=True)
    def fixup_time(cls, value, values):
        #: ! override
        value = super().fixup_time(value, values)
        return value.to_dict()

    @validator("formatted_data", allow_reuse=True)
    def append_time_column(cls, v, values):
        v = super().append_time_column(v, values)
        if isinstance(v, pd.DataFrame) and "time" in v.columns:
            v["time"] = (
                v["time"].apply(lambda t: pd.Timestamp(t).isoformat()).astype(str)
            )
        return v

    @validator("queryDate", allow_reuse=True)
    def fix_queryDate(cls, v):
        v = pd.Timestamp(v).to_pydatetime()
        return v

    @root_validator(allow_reuse=True)
    def convertFromDataframe(cls, values):
        for k, v in values.items():
            if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                values[k] = v.to_dict()
        return values


from typing import Union

RecommendationEndpoints = Literal["summary"]
QualityNames = Literal["great", "decent", "improve", "poor"]
QualityTitles = Literal["Great", "Decent", "Not Good", "Poor"]
QualityColors = Union[
    Literal["success"], Literal["primary"], Literal["warning"], Literal["danger"]
]
QualityRanges = Union[
    Literal[(8, 10)], Literal[(5, 7)], Literal[(3, 4)], Literal[(1, 2)]
]


def _create_cmi_membermap():
    membermap = {}
    for qn, qt, qc, qr in zip(
        QualityNames.__args__,
        QualityTitles.__args__,
        QualityColors.__args__,
        QualityRanges.__args__,
    ):
        membermap.update({qn: Namespace(title=qt, color=qc, score_range=qr)})
    return membermap


class CMIQualityMeta(EnumMeta):
    """metaclass for CMIQuality enum.

    ref: https://www.geeksforgeeks.org/python-metaclasses/
    """

    def __new__(metacls, cls, bases, classdict, **kwds):
        membermap = _create_cmi_membermap()
        for k in QualityNames.__args__:
            classdict[k] = membermap[k]
        obj = super(CMIQualityMeta, metacls).__new__(
            metacls, cls, bases, classdict, **kwds
        )
        return obj


class CMIQuality(Enum, metaclass=CMIQualityMeta):
    @property
    def title(self):
        return self.value.title

    @property
    def color(self):
        return self.value.color

    @property
    def score_range(self):
        return self.value.score_range

    @classmethod
    def from_score(cls, score: int | float):
        """get the CMI quality corresponding to the given numerical CMI score"""
        for name, member in cls.__members__.items():
            if (score >= member.score_range[0]) and (score <= member.score_range[1]):
                return member


class EndocrineSignalNS(BaseModel):
    """
    A class representing an endocrine signal in the model.

    Attributes:
        column (str): The name of the column in the data file containing the signal data.
        label (str): The label to use for the signal in plots and other visualizations.
        color (str | Container): The color to use for the signal in plots and other visualizations.
        tmin (int | float): Default: 0. The minimum time value for the signal data.
        tmax (int | float): Default: 0. The maximum time value for the signal data.
        ymin (Optional[float | None]): Default: None. The minimum value for the signal data. Defaults to None.
        ymax (Optional[float | None]): Default: None. The maximum value for the signal data.
    """

    column: str
    label: str
    color: str | Container
    tmin: int | float
    tmax: int | float
    ymin: Optional[float] = None
    ymax: Optional[float] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_docstring(*args, **kwargs)

    def _update_docstring(self, *args, **kwargs):
        # Create a Jinja2 template for the docstring
        template = Template(
            """
        A class representing an endocrine signal in the model.

        Attributes:
            column (str): The name of the column in the data file containing the signal data.
            label (str): The label to use for the signal in plots and other visualizations.
            color (str | Container): The color to use for the signal in plots and other visualizations.
            tmin (int | float): Default: {{ tmin }}. The minimum time value for the signal data.
            tmax (int | float): Default: {{ tmax }}. The maximum time value for the signal data.
            ymin (Optional[float | None]): Default: {{ ymin }}. The minimum value for the signal data. Defaults to None.
            ymax (Optional[float | None]): Default: {{ ymax }}. The maximum value for the signal data.
        """
        )

        # Render the template with the current field values
        docstring = template.render(
            tmin=self.tmin,
            tmax=self.tmax,
            ymin=self.ymin,
            ymax=self.ymax,
        )

        # Update the docstring for the class
        self.__class__.__doc__ = docstring

    class Config:
        arbitrary_types_allowed = True


class EndocrineSignal(Enum):
    cortisol = EndocrineSignalNS(
        column="c", label="Cortisol", color=mplcolors.get("tan"), tmin=1, tmax=7
    )
    melatonin = EndocrineSignalNS(
        column="m", label="Melatonin", color=mplcolors.get("lightblue"), tmin=13, tmax=3
    )
    adiponectin = EndocrineSignalNS(
        column="a", label="Adiponectin", color=mplcolors.get("red"), tmin=3, tmax=13
    )
    breakfast = EndocrineSignalNS(
        column="g_0",
        label="Glucose (Breakfast)",
        color=mplcolors.get("purple"),
        tmin=7,
        tmax=8,
    )
    lunch = EndocrineSignalNS(
        column="g_1",
        label="Glucose (Lunch)",
        color=mplcolors.get("purple"),
        tmin=12,
        tmax=13,
    )
    dinner = EndocrineSignalNS(
        column="g_2",
        label="Glucose (Dinner)",
        color=mplcolors.get("purple"),
        tmin=17,
        tmax=18,
    )

    @property
    def tmin(self):
        return self.as_hours(self.value.tmin)

    @property
    def tmax(self):
        return self.as_hours(self.value.tmax)

    def as_hours(self, v: int | float | pd.Timedelta) -> pd.Timedelta:
        if not isinstance(v, pd.Timedelta):
            return pd.Timedelta(hours=v)
        return v

    def calc_hour_dist(
        self, v0: float | pd.Timedelta, v1: float | pd.Timedelta
    ) -> float:
        t0, t1 = self.as_hours(v0).components.hours, self.as_hours(v1).components.hours
        hours_dist = diff_tod_hours(t0, t1)
        return hours_dist

    def calc_dist(self, tmin: float, tmax: float) -> float:
        """calculate the chebyshev distance"""
        assert isinstance(tmin, float), TypeError(
            f"tmin must be a float (got {type(tmin)} instead)"
        )
        assert isinstance(tmax, float), TypeError(
            f"tmax must be a float (got {type(tmax)} instead)"
        )
        dist_raw = np.array(
            [self.calc_dist_tmin(tmin), self.calc_dist_tmax(tmax)], dtype=float
        ).max()
        #: ! limiting these to 4-hours each (in the interest of leveling out the contributions to CMI)
        dist = min(4.0, dist_raw)
        return dist

    def calc_dist_tmin(self, tmin: float) -> float:
        """calculate the absolute difference for the trough of the signal"""
        return self.calc_hour_dist(tmin, self.value.tmin)

    def calc_dist_tmax(self, tmax: float) -> float:
        """calculate the absolute difference for the peak of the signal"""
        return self.calc_hour_dist(tmax, self.value.tmax)


class ExpectedChronoDiffs(Enum):
    """the expected percent change in a given endocrine signal (morning -> evening)"""

    I_S: float = 0.14
    G: float = -0.08


class ChronometabolicDiffCheck(BaseModel):
    signal_name: Literal["I_S", "G"]
    signal_morn: float
    signal_eve: float
    nominal: Optional[bool | None] = None
    diff: Optional[float | None] = None
    expected_diff: Optional[ExpectedChronoDiffs] = None

    @classmethod
    def calc_diff(cls, values):
        """calculate the percent difference in the signal from morning to evening

        **not percent change, so if morning > evening, the result is positive.
        """
        diff = values["signal_morn"] - values["signal_eve"]
        return diff

    @classmethod
    def calc_nominal(cls, values):
        """True if the difference is within the expected range"""
        signal_name = values["signal_name"]
        expected_diff = ExpectedChronoDiffs[signal_name].value
        diff = values["diff"]
        nominal = np.isclose(diff, expected_diff, atol=0.05)
        return nominal

    @root_validator(allow_reuse=True, pre=False)
    def check_morn_eve_present(cls, values):
        kvals = ["signal_morn", "signal_eve"]
        missing = ", ".join([k for k in kvals if k not in values])
        assert all([k in values for k in kvals]), f'missing from values:\n\t"{missing}"'
        values["diff"] = cls.calc_diff(values)
        values["nominal"] = cls.calc_nominal(values)
        return values

    @validator("expected_diff", always=True, allow_reuse=True)
    def get_expected_diff(cls, value, values):
        return ExpectedChronoDiffs[values["signal_name"]]


class CMIModel(BaseModel):
    model_result: ModelResult = Field(...)
    score: Optional[float | None] = Field(default=None, ge=1.0, le=10.0)
    quality: Optional[CMIQuality | None] = None
    chrono_checks: Optional[List[ChronometabolicDiffCheck] | None] = None

    @validator("chrono_checks", always=True, allow_reuse=True, each_item=False)
    def compute_chrono_checks(cls, v, values):
        if v is not None:
            return v
        checks = []
        stats = values["model_result"].stats
        for signal_name in ["I_S", "G"]:
            morn = getattr(stats, f"{signal_name}_morn")
            eve = getattr(stats, f"{signal_name}_eve")
            check = ChronometabolicDiffCheck(
                signal_name=signal_name, signal_morn=morn, signal_eve=eve
            )
            checks.append(check)
        v = checks
        return v

    @root_validator(allow_reuse=True)
    def compute_posthoc(cls, values):
        #: compute the chronometabolic index (CMI score)
        chrono_dist = []
        model_result = values["model_result"]
        soln = pd.DataFrame(model_result.soln).sort_index()
        for member in EndocrineSignal:
            estimated_signal = soln[member.value.column]
            t_peak = estimated_signal.idxmax()
            t_trough = estimated_signal.idxmin()
            cdist = member.calc_dist(tmin=t_trough, tmax=t_peak)
            chrono_dist.append(cdist)
        #: CMI score computation...
        total_dist = np.max(chrono_dist)
        normed_cmi_dist = np.abs(total_dist) / 12.0
        #: incorporate chrono_checks (other non-cmi checks, including the morning -> evening percent differences)
        #: ...essentially this is: (1 - frequency_nominal_values)
        other_checks_dist = 1.0 - np.nanmean(
            [c.nominal for c in values["chrono_checks"]]
        )
        #: continuous checks_dist
        cont_checks_dist = np.nanmax(
            [np.abs(c.expected_diff.value - c.diff) for c in values["chrono_checks"]]
        )
        # ! take maximum distance value
        final_dist = np.nanmean([normed_cmi_dist, other_checks_dist, cont_checks_dist])
        score_01 = max(1.0 - float(final_dist), 0.0)
        #: ! CMI, defined in range of [1, 10]
        values["score"] = int(np.floor((1 + 10.0 * score_01)))
        logging.info(f"\nOverall CMI score: {values['score']:.3f}")
        logging.info(f"... score_01(raw) = {score_01:.3f}")
        logging.info(
            f"...normed_cmi_dist={normed_cmi_dist:.3f}, other_checks_dist={other_checks_dist:.3f}\n"
        )
        #: get quality from score
        values["quality"] = CMIQuality.from_score(values["score"])
        logging.info(f"CMI Quality: {values['quality']}")
        return values


class StrengthWeaknessItem(BaseModel):
    title: str
    name: Optional[str]
    color: QualityColors
    column: str
    func: Callable
    column_label: Optional[str]
    vmin: Optional[float] = 0.0
    vmax: Optional[float] = 1.0
    severity: Optional[float]
    tod_hour_goal: Optional[float | int] = -1
    tod_hour_func: Optional[Callable]
    tod_hour: Optional[float | int] = -1
    kind: Optional[Literal["strength", "weakness"]] = "strength"
    rec_short: Optional[str] = ""
    goal_short: Optional[str] = ""
    fmt_short: Optional[str] = ""
    fmt_title: Optional[str] = None

    @validator("title", allow_reuse=True)
    def fixup_title(cls, v, values):
        return v.title()

    @validator("name", allow_reuse=True)
    def fixup_name(cls, v, values):
        if v is None:
            return "_".join(values["title"].split(" ")).replace(".", "").lower()
        return v

    @validator("column_label", always=True, allow_reuse=True)
    def get_column_label(cls, v, values):
        assert "column" in values, "column missing from values"
        column = values["column"]
        cma_plot_config = CMAPlotConfig()
        try:
            assert (
                column in cma_plot_config.plot_cols
            ), f"column {column} not in cma_plot_config.plot_cols"
        except AssertionError:
            logging.warning(
                "validator get_column_label: column %s not in cma_plot_config.plot_columns"
                % column,
                exc_info=1,
            )
            return v
        else:
            return cma_plot_config.get_label(column)

    @validator("kind", always=True, allow_reuse=True)
    def calc_kind(cls, v, values):
        return "strength" if values["color"] in ["success", "primary"] else "weakness"

    @validator("fmt_short", always=True)
    def calc_fmt_string(cls, v, values):
        li_fmt = '<li class="border-0">{}</li>'
        goal_short = values.get("goal_short", "")
        if goal_short:
            goal_title = "Goal" if values["kind"] == "weakness" else "Info"
            goal_short = f'<span class="fw-bold">{goal_title}:&nbsp;</span>{goal_short}'
        rec_short = values.get("rec_short", "")
        if rec_short:
            rec_title = "Tip" if values["kind"] == "weakness" else "Recall"
            rec_short = f'<span class="fw-bold">{rec_title}:&nbsp;</span>{rec_short}'
        fmt_short = li_fmt.format(goal_short) + li_fmt.format(rec_short)
        return fmt_short

    @validator("fmt_title", always=True)
    def calc_fmt_title(cls, v, values):
        color = values["color"]
        title = values["title"]
        return f'<span class="text-bg-{color}">{title}</span>'

    def __str__(self):
        return self.fmt_string

    def __repr__(self):
        return self.__str__()


class StrengthWeakness:
    def __call__(self, *args, t=None, x=None, **kwds):
        self._x = None
        if t is not None and x is not None:
            self._x = pd.DataFrame({"t": t, "x": x}, columns=["t", "x"]).set_index(
                "t", drop=True
            )
        return self

    def check_value(self):
        """check if the condition applies.

        returns: (continuous output of the check, boolean value of check)
        """
        output = np.atleast_1d(self.value.func(self._x))
        checkflag = output.any()
        if hasattr(checkflag, "__len__"):
            checkflag = all(checkflag)
        return (output, checkflag)

    def calc_severity(self):
        """return the severity of the weakness as a score [0.0, 1.0] (or -0.0 if the weakness is not present)."""
        output, checkflag = self.check_value()
        if checkflag is False:
            s = -0.0
        else:
            s = (float(output) - self.value.vmin) / (self.value.vmax - self.value.vmin)
        if hasattr(s, "item"):
            s = float(s.item())
        return s

    def calc_tod_hour(self):
        if self.value.tod_hour_func is not None:
            thr = self.value.tod_hour_func(self._x)
            if hasattr(thr, "item"):
                thr = float(thr.item())
            return thr
        else:
            return -1


class Strength(StrengthWeakness, Enum):
    consistent_meals = StrengthWeaknessItem(
        name="consistent_meals",
        title="Consistent Meals",
        color="success",
        column="is_meal",
        func=lambda tM: (tM[tM].diff() < 7.0).all(),
    )
    not_many_highs = StrengthWeaknessItem(
        name="not_many_highs",
        title="Avoiding hyperglycemia",
        color="success",
        column="G",
        func=lambda G: (G > 0.9).mean() < 0.25,
        goal_short=(
            "You're doing a good job of avoiding excessively high blood glucose values...<br />"
            "In case you were wondering, your high for this wear period was {{ '%d'|format(stats.max_bg_value | int) }} mg/dL."
        ),
        rec_short='Keep in mind that any values above <span class="font-monospace"> 160 mg/dL </span> can result in serious health complications,'
        " including a risk of permanent pancreatic tissue damage.",
    )
    not_many_lows = StrengthWeaknessItem(
        name="not_many_lows",
        title="Avoiding hypoglycemia",
        color="success",
        column="g_raw",
        func=lambda G: ((G >= 65.0).all() & ((G >= 65.0).mean() >= 0.9)).all(),
        goal_short="Low glucose isn't the highest priority for you during the selected wear period."
        + ' Keep it up! <i class="bi bi-emoji-wink"></i>. ',
        rec_short='Any values below <span class="font-monospace"> 70 mg/dL </span> are considered dangerously low (i.e., "hypoglycemia").',
    )
    healthy_glucose_variability = StrengthWeaknessItem(
        name="healthy_glucose_variability",
        title="Healthy Glucose Variability",
        color="success",
        column="G",
        func=lambda g: g.std() < 0.2,
    )


class Weakness(StrengthWeakness, Enum):
    frequent_lows = StrengthWeaknessItem(
        name="frequent_lows",
        column="g_raw",
        title="Recent Glucose Lows",
        color="danger",
        func=lambda g: (g < 70.0).any(),
        vmin=0.0,
        vmax=1.0,
        goal_short="""Your blood glucose has dropped too low (below 70 mg/dL) during the selected CGM wear period.
         This can be very dangerous. To avoid health risks related to hypoglycemia, you'll want to make sure to keep you glucose above 70 mg/dL.""",
        rec_short="""If you're at risk of hypoglycemia during the day, make sure you have some healthy snacks (nuts, or a granola bar) with you.<br />
        To avoid low blood sugar at night, make sure you're not taking your insulin dose too close to bedtime. Be sure to ask your doctor if you have any questions.""",
    )
    frequent_highs = StrengthWeaknessItem(
        column="G",
        title="Your blood glucose is often too high.",
        color="danger",
        func=lambda g: (g >= 1.0).mean() >= 0.4,
        vmin=0.0,
        vmax=1.0,
        goal_short='Make sure to keep your blood glucose within range (i.e., between <span class="font-monospace"> 70 mg/dL</span> and <span class="font-monospace">180 mg/dL</span>).',
        rec_short="Aim to decrease the amount of sugar in your diet. Try to replace sugary foods with complex carbs such as whole-grain rice, pasta, and bread.",
    )
    inconsistent_meals = StrengthWeaknessItem(
        title="Inconsistent Meals",
        color="danger",
        column="is_meal",
        func=lambda tM: (tM[tM].index.to_series().abs().diff().diff() >= 2.5).mean(),
        vmin=0.0,
        vmax=1.0,
        goal_short="""Aim to have 3 main meals (Breakfast, lunch, dinner) no more than 3 hours apart.""",
        rec_short="""If you're hungry between meals, make sure to bring a healthy snack with you during the day (nuts, granola would be good options).""",
    )
    glucose_variability_is_too_high = StrengthWeaknessItem(
        column="G",
        title="Glucose Variability is Too High",
        color="danger",
        func=lambda g: g.var() > 0.35,
        vmin=0.0,
        vmax=1.0,
    )
    cortisol_peak_early = StrengthWeaknessItem(
        column="c",
        title="Cortisol peaks too early in the day",
        color="danger",
        tod_hour_goal=EndocrineSignal.cortisol.value.tmax,
        tod_hour_func=lambda c: c.idxmax().iloc[0],
        func=lambda c: float(c.idxmax().iloc[0])
        < float(EndocrineSignal.cortisol.value.tmax - 1.5),
        vmin=0.0,
        vmax=1.0,
        goal_short="Ideally, cortisol should peak at ~7AM.",
        rec_short="Try to reduce your exposure to bright light in the early morning (before sunrise).",
    )
    cortisol_peak_late = StrengthWeaknessItem(
        column="c",
        title="Cortisol peaks too late in the day.",
        color="danger",
        func=lambda c: float(c.idxmax().iloc[0])
        > float(EndocrineSignal.cortisol.value.tmax + 1.5),
        vmin=0.0,
        vmax=1.0,
        tod_hour_func=lambda c: float(c.idxmax().iloc[0]),
        tod_hour_goal=float(EndocrineSignal.cortisol.value.tmax),
        goal_short="""You might feel lethargic during the day... If so, it could be low cortisol levels.<br />
                                              Cortisol is a natural hormone that can help you feel awake and alert during the day.<br />
                                              Ideally, cortisol should peak soon after you wake up, ideally around sunrise.""",
        rec_short="""
                                              <ul>
                                                <li>
                                                    If possible, try to adjust your schedule to wake up as close to sunrise as possible, and aim to have breakfast within an hour of waking up.
                                                </li>
                                                <li>
                                                    For more info, <a href="https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/stress/art-20046037">check out this article from Mayo Clinic...</a>
                                                </li>
                                              </ul>""",
    )
    melatonin_peak_early = StrengthWeaknessItem(
        column="m",
        title="Melatonin peaks too early in the day.",
        color="danger",
        goal_short="Make sure you're being exposed to natural light during daylight hours to help maintain a consistent sleep schedule.",
        rec_short="Aim to spend a few minutes per hour near a well-lit window if you work indoors.<br />Best is to take a break at least once every two hours to walk outside.",
        func=lambda m: m.idxmax().iloc[0] < EndocrineSignal.melatonin.value.tmax - 2,
        vmin=0.0,
        vmax=1.0,
        tod_hour_func=lambda m: float(m.idxmax().iloc[0]),
        tod_hour_goal=float(EndocrineSignal.melatonin.value.tmax),
    )
    melatonin_peak_late = StrengthWeaknessItem(
        column="m",
        title="Melatonin peaks too late in the day.",
        goal_short="""Looks like you're staying up too late, and it's affecting your sleep schedule.<br />
                                               Aim to start winding down for bed by 9PM if possible...""",
        rec_short="""If you can, try to reduce your exposure to bright artificial lights after sundown.<br />"""
        """You can try reducing the brightness of your phone, TV, or computer monitor if you need to use any of those at night.""",
        color="danger",
        func=lambda m: float(m.idxmax().iloc[0])
        > float(EndocrineSignal.melatonin.value.tmax + 1.5),
        vmin=0.0,
        vmax=1.0,
        tod_hour_func=lambda m: float(m.idxmax().iloc[0]),
        tod_hour_goal=float(EndocrineSignal.melatonin.value.tmax),
    )


def calc_strengths_or_weaknesses(which: Literal["strength", "weakness"], values: Dict):
    """Instantiate StrengthWeaknessItem(...) for each strength/weakness (depending on which=...)."""
    items = []
    soln = values["model_result"].soln
    item_type = {"strength": Strength, "weakness": Weakness}[which]
    tvec = soln["t"] if "t" in soln.columns else soln.index
    for item in item_type:
        w = item(t=tvec, x=soln[item.value.column])
        wdict = w.value.dict()
        wdict.update({"severity": w.calc_severity(), "tod_hour": w.calc_tod_hour()})
        if len(wdict.get("rec_short", "")) > 0:
            items.append(StrengthWeaknessItem(**wdict))
    items = sorted(items, key=lambda x: x.severity, reverse=True)
    return items


def ready_data_for_plotting(model_result: ModelResult | Dict):
    """Get the model results & formatted data ready for plotting -> visual plan"""

    #: get the original "fitted" solution
    wsoln = model_result.soln
    if "t" not in wsoln:
        wsoln.reset_index(inplace=True)

    #: plot the actual glucose data (smoothed, downsampled)
    data_plot = (
        model_result.formatted_data[["tod", "G", "time"]].sort_values(by="tod").dropna()
    )
    data_ewm = (
        data_plot.set_index("time", drop=True)
        .sort_values(by="tod")
        .dropna()
        .ewm(alpha=0.5)
    )
    try:
        data_plot["gave"] = (
            data_ewm["G"].mean().reset_index(drop=True).interpolate(method="akima")
        )
    except Exception as local_ex:
        logging.warn("(gave) failed to smooth glucose...", exc_info=1)
        raise local_ex
    try:
        gstd = data_ewm["G"].std().reset_index(drop=True)
        data_plot["gmin"] = data_plot["gave"] - gstd * 2.0
        data_plot["gmax"] = data_plot["gave"] + gstd * 2.0
    except Exception as local_ex:
        logging.warn("(gmin, gmax) failed to smooth glucose...", exc_info=1)
        raise local_ex
    data_plot["markersize"] = normalize(
        data_plot["G"].to_numpy(dtype=float, na_value=0.0), 0.0, 10.0
    )
    return data_plot, wsoln


class GoalSolnNS:
    """namespace wrapper for pre-computed goal solution"""

    @classmethod
    def load_goal_soln(cls):
        #: read the goal solution
        pth = Path(__file__).parent.joinpath(
            "resources", "www", "static", "cma_goal_response.json"
        )
        nsoln = pd.DataFrame(json.loads(pth.read_text()).get("output"))
        nsoln = nsoln.set_index("t", drop=False)
        return nsoln

    @classmethod
    @property
    @cache
    def goal_soln(cls):
        nsoln = cls.load_goal_soln()
        return nsoln


class RecsSummaryModel(BaseModel):
    model_result: ModelResult | Dict
    cmi: None | CMIModel = None
    weaknesses: Optional[List[StrengthWeaknessItem | str]] = []
    strengths: Optional[List[StrengthWeaknessItem | str]] = []
    max_weakness: Optional[StrengthWeaknessItem | str] = Field(alias="weakness")
    max_strength: Optional[StrengthWeaknessItem | str] = Field(alias="strength")
    weakness_signal_color: Optional[str | Tuple | Any | Container]
    weakness_signal_color_hex: Optional[str | AnyStr]
    weakness_goal_color: Optional[str | Tuple | AnyStr | Container]
    weakness_goal_color_hex: Optional[str | AnyStr]
    visual_plan: Optional[str] = ""

    @validator("model_result", allow_reuse=True, always=True)
    def ensure_modelresult_type(cls, v, values):
        value = v
        if not isinstance(v, ModelResult):
            value = ModelResult(**v)
        return value

    @validator("cmi", always=True)
    def ensure_cmi_present(cls, value, values):
        if not isinstance(value, CMIModel):
            value = CMIModel(model_result=values["model_result"])
        return value

    @validator("weaknesses", always=True)
    def calc_weaknesses(cls, weaknesses, values):
        weaknesses = calc_strengths_or_weaknesses("weakness", values)
        values["weaknesses"] = weaknesses
        return weaknesses

    @root_validator(allow_reuse=True)
    def ensure_all(cls, values):
        values["weaknesses"] = cls.calc_weaknesses(values.get("weaknesses"), values)
        return values

    @classmethod
    def _calc_max_weakness(cls, v, values, exclude: Container | None = None):
        weaknesses = values.get("weaknesses", None)
        if weaknesses is None:
            weaknesses = cls.calc_weaknesses(v, values)
        if len(weaknesses) == 0:  # ! handle zero length case
            return None
        #: ! exclude any items specified...
        if exclude is not None:
            if hasattr(exclude[0], "column"):
                excluded_cols = [ex.column for ex in exclude]
            else:
                excluded_cols = list(exclude)
            to_remove = list(filter(lambda wk: wk.column in excluded_cols, weaknesses))
            for rms in to_remove:
                weaknesses.remove(rms)
        if len(weaknesses) == 0:
            return None
        if exclude is None:
            #: ! important ! explicitly set values.weaknesses !
            values["weaknesses"] = weaknesses
        #: get max weakness
        wk_max = max(weaknesses, key=lambda x: x.severity)
        return wk_max

    @validator("max_weakness", always=True, pre=True)
    def calc_max_weakness(cls, v, values):
        return cls._calc_max_weakness(v, values)

    @validator("strengths", always=True, pre=True)
    def calc_strengths(cls, v, values):
        strengths = calc_strengths_or_weaknesses("strength", values)
        return strengths

    @validator("max_strength", always=True, pre=True)
    def calc_max_strength(cls, v, values):
        if "strengths" not in values:
            values["strengths"] = cls.calc_strengths(v, values)
        return max(values["strengths"], key=lambda x: x.severity)

    @validator("weakness_signal_color", always=True, allow_reuse=True)
    def calc_weakness_signal_color(cls, v, values):
        if values.get("max_weakness") is None:
            values["max_weakness"] = cls.calc_max_weakness(v, values)
        if values.get("max_weakness") is not None:
            return CMAPlotConfig().get_color(values["max_weakness"].column, rgba=True)[
                :-1
            ]
        return CMAPlotConfig().get_color("G", rgba=True)[:-1]

    @validator("weakness_signal_color_hex", always=True, allow_reuse=True)
    def calc_weakness_signal_color_hex(cls, v, values):
        if values.get("max_weakness") is None:
            values["max_weakness"] = cls.calc_max_weakness(v, values)
        return mpl.colors.rgb2hex(
            CMAPlotConfig().get_color(values["max_weakness"].column, rgba=True)
        )

    @validator("weakness_goal_color", always=True, allow_reuse=True)
    def calc_weakness_goal_color(cls, v, values):
        if values.get("weakness_signal_color") is None:
            values["weakness_signal_color"] = cls.calc_weakness_signal_color(v, values)
        goal_color = mpl.colors.rgb_to_hsv(values["weakness_signal_color"])
        goal_color[1] = ((10 * (1.1 * goal_color[1])) % 10) / 10.0
        goal_color[2] *= 0.95
        return goal_color

    @validator("weakness_goal_color_hex", always=True, allow_reuse=True)
    def calc_weakness_goal_color_hex(cls, v, values):
        if values.get("weakness_goal_color") is None:
            values["weakness_goal_color"] = cls.calc_weakness_goal_color(v, values)
        if values["weakness_goal_color"] is not None:
            return mpl.colors.rgb2hex(values["weakness_goal_color"])
        return "#000000ff"

    @classmethod
    def _plot_visual_plan(
        cls, value, values, exclude=None, skip_glucose=False, return_weakness=False
    ):
        weakness = values.get("max_weakness")
        fig, axs = None, None

        def get_returns(fig, axs, weakness, return_weakness=return_weakness):
            out = (fig, axs) if return_weakness is False else (fig, axs, weakness)
            return out

        #: ! handle missing weakness or excluded plots...
        if weakness is None or exclude is not None:
            exclude = list(set([e.column for e in exclude]))
            weakness = cls._calc_max_weakness(None, values, exclude=exclude)
        there_is_weakness = all(
            [weakness.column is not None, weakness.column != "is_meal"]
        )

        if not any([there_is_weakness, skip_glucose is False]):
            #: ! handle case where no figure would be plotted...
            return get_returns(fig, axs, weakness)

        #: setup figure
        figsize = (1024, 512)
        px = 1.0 / plt.rcParams["figure.dpi"]  # pixels in inches
        fig, axs = plt.subplots(
            figsize=(int(figsize[0] * px), int(figsize[1] * px)),
            nrows=2 if (there_is_weakness is True and skip_glucose is False) else 1,
            ncols=1,
        )
        if not isinstance(axs, Container):
            axs = [
                axs,
            ]

        #: load the goal solution (pre-computed)
        nsoln = GoalSolnNS.goal_soln

        #: init plot configuration instance
        cma_plot_config = CMAPlotConfig()

        # get the model_result value
        model_result = values["model_result"]

        #: get the start, end datetimes for the CGM wear period
        tstart, tend = model_result.time.min().isoformat(
            timespec="hours"
        ), model_result.time.max().isoformat(timespec="hours")
        tstart = tstart.split("T")[0]
        tend = tend.split("T")[0]

        #: set wear period in axis title
        fig.suptitle(
            f"For CGM wear period: {tstart} to {tend}"
            if skip_glucose is False
            else f"Goal for {weakness.title}"
        )

        #: ready the data for plotting
        data_plot, wsoln = ready_data_for_plotting(model_result)

        # # # Plot data & weakness model estimates # # #

        #: plot glucose
        if skip_glucose is False:
            ax = data_plot.plot.scatter(
                x="tod",
                y="gave",
                s="markersize",
                ax=axs[0],
                color="k",
                label=cma_plot_config.get_label("G") + " (Data)",
                alpha=0.5,
                linestyle="",
                marker="o",
            )
            ax.fill_between(
                data_plot["tod"],
                data_plot["gmin"],
                data_plot["gmax"],
                color="k",
                edgecolor="k",
                alpha=0.2,
            )
            ax.set_ylabel("")

            #: plot the glucose estimate (original)
            ax = wsoln.sort_values(by="t").plot.area(
                x="t",
                y="G",
                ax=ax,
                color=cma_plot_config.get_color("G"),
                label=cma_plot_config.get_label("G") + " (Estimate)",
                alpha=0.5,
                linestyle="--",
                lw=1.0,
            )

            #: glucose axis labels
            axs[0].set_yticks([0.3, 1.0, 1.5], ["70 mg/dL", "100 mg/dL", "180+ mg/dL"])

        # # # Weakness-related plots: # # #

        if not there_is_weakness:
            #: ! handle no weakness case
            return get_returns(fig, axs, weakness)

        #: plot the weakness solution (if there is a weakness)
        ax1 = wsoln.sort_values(by="t").plot.area(
            x="t",
            y=weakness.column,
            ax=axs[-1],
            color=values.get("weakness_signal_color", "r"),
            label=cma_plot_config.get_label(weakness.column),
            alpha=0.5,
            linestyle="-",
            lw=2.0,
        )

        #: plot the weakness-goal
        goal_color = "g"  # ! explicit
        goal_col = weakness.column if weakness.column in nsoln.columns else "G"
        ax1 = nsoln[["t", goal_col]].plot.area(
            x="t",
            y=goal_col,
            ax=ax1,
            color=goal_color,
            label=cma_plot_config.get_label(weakness.column) + " (Goal)",
            alpha=0.5,
            linestyle="-",
            lw=2.0,
        )
        if weakness.tod_hour_goal not in [None, -1]:
            #: ! mark the current tod_hour and goal tod_hour
            wsoln = model_result.soln.set_index("t")
            ymax = wsoln[weakness.column].max()
            nmax = nsoln[weakness.column].max()
            ax1.axvline(
                weakness.tod_hour,
                ymin=0.0,
                ymax=wsoln.loc[weakness.tod_hour, weakness.column] / ymax,
                label="Current",
                color="k",
                linestyle="--",
                alpha=0.3,
            )
            y_text = -0.35
            ax1.annotate(
                "Now",
                xy=(weakness.tod_hour, -0.0),
                xytext=(weakness.tod_hour, y_text),
                xycoords="data",
                textcoords="data",
                fontsize=12,
                color="k",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="k"),
            )
            close_ixs = np.flatnonzero(
                np.isclose(nsoln.index, weakness.tod_hour_goal, atol=0.3)
            )
            if len(close_ixs) > 0:
                close_ix = close_ixs[0]
                ymax = nsoln.iloc[close_ix, :][weakness.column] / nmax
            else:
                logging.warn(
                    f"(models.py:visual_plan) no matching index for goal solution was found (col={weakness.column})"
                )
                ymax = 1.0  # ! in case none is close enough
            ax1.axvline(
                weakness.tod_hour_goal,
                ymin=0.0,
                ymax=ymax,
                label="Goal",
                color=goal_color,
                linestyle="-",
                alpha=1.0,
            )
            ax1.annotate(
                "Goal",
                xy=(weakness.tod_hour_goal, -0.0),
                xytext=(weakness.tod_hour_goal, y_text),
                xycoords="data",
                textcoords="data",
                fontsize=12,
                color=goal_color,
                arrowprops=dict(
                    arrowstyle="->", connectionstyle="arc3", color=goal_color
                ),
            )

        return get_returns(fig, axs, weakness)

    @validator("visual_plan", always=True)
    def calc_visual_plan(cls, value, values):
        #: plot visual plan...
        fig, axs, weakness = cls._plot_visual_plan(value, values, return_weakness=True)

        #: set global axis attributes...
        axs = CMAPlotConfig.set_global_axis_attributes(axs)

        def ready_plot_for_output(fig, fmt="png"):
            #: ready plot for output -> html
            fig.tight_layout()
            bio = BytesIO()
            fig.savefig(bio, format=fmt)
            bio.seek(0)
            bytes_value = bio.getvalue()
            fmt = "png;base64" if fmt.lower() == "png" else fmt
            img_src = f"data:image/{fmt},"
            img_src = img_src + b64encode(bytes_value).decode("utf-8")
            return img_src

        raw_img_component = '<ion-img src="{}" width="100%" height="auto" class="img-fluid" id="{}"></ion-img>'
        img_components = []

        #: first image component
        img_src = ready_plot_for_output(fig)
        img_component = raw_img_component.format(img_src, "visualPlanImage")
        img_components.append(img_component)
        plt.close()

        #: compute other components (any weaknesses)
        weaknesses = [
            weakness,
        ]
        iw = 1
        while True:
            fig, axs, weakness = cls._plot_visual_plan(
                value,
                values,
                exclude=weaknesses,
                return_weakness=True,
                skip_glucose=True,
            )
            if any([el is None for el in [fig, axs, weakness]]):
                break
            else:
                imsrc = ready_plot_for_output(fig)
                imcomp = raw_img_component.format(imsrc, f"visualPlanImage-{iw:02d}")
                img_components.append(imcomp)
                weaknesses.append(weakness)
                plt.close()
                iw = iw + 1

        #: compute the cma image component(s)...
        img_src2 = CMAPlotConfig().plot_model_results(
            df=values["model_result"].formatted_data,
            soln=values["model_result"].soln,
            plot_cols=["c", "m", "a"],
            separate2subplots=True,
            as_blob=True,
        )
        img_component2 = raw_img_component.format(img_src2, "visualPlanImage2")
        img_components.append(img_component2)

        #: ensure all figures are closed
        plt.close("all")

        return "<br />".join(img_components)

    class Config:
        allow_extras = True
        arbitrary_types_allowed = True


class RecsItemModel(BaseModel):
    kind: Literal["mealtime", "mealsize", "circadian", "medication"]
    action: Literal["sleep", "eat", "dose insulin"]
    recommended_action: Literal["increase", "decrease", "earlier", "later"]


class RecsListModel(BaseModel):
    pass

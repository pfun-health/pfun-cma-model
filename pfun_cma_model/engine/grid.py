import concurrent.futures
import logging
import os
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
import gzip
from sklearn.model_selection import ParameterGrid
from pfun_cma_model.engine.cma import CMASleepWakeModel
from pfun_cma_model.engine.cma_model_params import (
    CMAModelParams,
    BoundedCMAModelParam
)


@dataclass
class PFunCMAParamsGridResult:
    """Result object for grid search.

    Mutability consistency not guaranteed.
    """

    param_keys: list[str]
    param_values: list[float | int]
    soln: pd.DataFrame

    def __post_init__(self):
        self.param_dict = dict(
            zip(self.param_keys, self.param_values)
        )
        self.params_json = json.dumps(self.param_dict)
        self.params = CMAModelParams(**self.param_dict)
        self.params_md = self.params.generate_markdown_table("md")

    def get_markdown_document(self):
        params_md = self.params_md
        soln_md = self.soln.to_markdown()
        return 

    def get_soln_as_json(self) -> str:
        return self.soln.to_json()

    def get_soln_as_gzjson(self) -> bytes:
        s = self.get_soln_as_json()
        return gzip.compress(s)


def compute_psample(params, N) -> pd.DataFrame:
    """Compute the (CMA, Glucose) solution from a given parameter set."""
    cma = CMASleepWakeModel(config=params, N=N)
    out = cma.run()
    return out


def get_db_client():
    """get the database client"""
    from pfun_cma_model.data import get_db_path
    import duckdb
    connection = duckdb.connect(
        database = get_db_path()
    )
    return connection


def collate_results(
        results: list[PFunCMAParamsGridResult],
        collection_id: str = "cma_results"
):
    """Store results in database."""
    connection = get_db_client()
    # create a collection with given ID as the name
    df_collection = pd.DataFrame.from_dict(
        dict(
            ids=[result.params_json for result in results],
            documents=[result.get_soln_as_json() for result in results]
        )
    )
    connection.sql(
        f"CREATE TABLE IF NOT EXISTS {collection_id} as SELECT * FROM df_collection"
    )
    return connection


class PFunCMAParamsGrid:
    """Parameter grid class for analyzing the parameter space of the CMA model."""

    #: absolute upper/lower bounds for mealtimes
    tmK = ("tM0", "tM1", "tM2")
    tmL = (4, 11, 13)
    tmU = (11, 16, 22)

    def __init__(self, N=48, m=3, include_mealtimes=True, keys=None, Njobs=-1):
        self.N = N
        self.m = m
        self._Njobs = None
        self.Njobs = Njobs
        self.include_mealtimes = include_mealtimes
        cma = CMASleepWakeModel(N=self.N)
        if keys is None:
            keys = list(cma.bounded_param_keys)
            lb = list(cma.bounds.lb)
            ub = list(cma.bounds.ub)
        else:
            ixs = [list(cma.bounded_param_keys).index(k) for k in keys]
            lb = [cma.bounds.lb[ix] for ix in ixs]
            ub = [cma.bounds.ub[ix] for ix in ixs]
        plist = list(zip(keys, lb, ub))
        pdict = {}
        # create m-length parameter ranges
        pdict = {k: np.linspace(l, u, num=self.m) for k, l, u in plist}
        if self.include_mealtimes is True:
            pdict.update(
                {
                    k: list(range(l, u, self.m))
                    for k, l, u in zip(self.tmK, self.tmL, self.tmU)
                }
            )
        # defines the parameter grid to search
        self.pgrid = ParameterGrid(pdict)
        # solutions vector (temporary storage)
        self.solns = []
        # database client
        self.client = None

    @property
    def Njobs(self):
        return self._Njobs

    @Njobs.setter
    def Njobs(self, val):
        """Safely set the number of jobs (without exceeding 'os.cpu_count()').
        """
        _ncpus = os.cpu_count()
        if val < 1:
            self._Njobs = _ncpus
        elif val > _ncpus:
            logging.warning(
                "specified Njobs=%d is higher than measured cores %d. "
                "Setting to %d.",
                val,
                _ncpus,
                _ncpus,
            )
            self._Njobs = _ncpus
        else:
            self._Njobs = val

    def run(self):
        """Run the parameter grid to produce a dataframe of results."""
        logging.info("Running parameter grid of size: %02d...", len(self.pgrid))  # type: ignore

        # distribute tasks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.Njobs) as pool:  # type: ignore
            future_to_params = {
                pool.submit(compute_psample, params, N=self.N): params
                for params in self.pgrid
            }
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    self.solns.append(
                        PFunCMAParamsGridResult(
                            list(params.keys()), list(params.values()), future.result()  # type: ignore
                        )
                    )
                except Exception as exc:
                    logging.error("failed to compute", exc_info=exc)
        # collate to a single database
        logging.info("...done searching parameter grid and collating results.")
        self.client = collate_results(self.solns)
        return self

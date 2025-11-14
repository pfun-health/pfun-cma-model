import json
import os
import logging
from pfun_cma_model.engine.cma import CMASleepWakeModel
import numpy as np
import pandas as pd
import concurrent.futures
from sklearn.model_selection import ParameterGrid
from dataclasses import dataclass, asdict


@dataclass
class PFunCMAParamsGridResult:
    """result object for grid search"""
    #: json-string-ified params
    params: str
    #: json-string-ified result
    result: str


def compute_psample(params, N):
    """compute from a single sample of parameters from the grid."""
    cma = CMASleepWakeModel(config=params, N=N)
    out = cma.run()
    return out


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
            pdict.update({
                k: list(range(l, u, self.m)) for k, l, u in zip(self.tmK, self.tmL, self.tmU)
            })
        self.pgrid = ParameterGrid(pdict)
        self.df = None
        # solutions vector
        self.solns = []

    @property
    def Njobs(self):
        return self._Njobs

    @Njobs.setter
    def Njobs(self, val):
        """safely set the number of jobs (without exceeding 'os.cpu_count()')."""
        _ncpus = os.cpu_count()
        if val < 1:
            self._Njobs = _ncpus
        elif val > _ncpus:
            logging.warning(
                "specified Njobs=%d is higher than measured cores %d. "
                "Setting to %d.", val, _ncpus, _ncpus
            )
            self._Njobs = _ncpus
        else:
            self._Njobs = val

    def run(self):
        """Run the parameter grid to produce a dataframe of results.
        """
        logging.info("Running parameter grid of size: %02d...",
                     len(self.pgrid))

        # distribute tasks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.Njobs) as pool:
            future_to_params = {
                pool.submit(compute_psample, params, N=self.N): params
                for params in self.pgrid
            }
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    self.solns.append(
                        asdict(
                            PFunCMAParamsGridResult(
                                json.dumps(params),
                                future.result().to_json()
                            )
                        )
                    )
                except Exception as exc:
                    logging.error("failed to compute", exc_info=exc)

        # format results
        self.df = pd.DataFrame(self.solns, columns=["params", "result"])
        return self.df

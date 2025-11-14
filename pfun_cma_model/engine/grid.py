import os
import logging
from pfun_cma_model.engine.cma import CMASleepWakeModel
import numpy as np
import pandas as pd
from multiprocessing import Pool, Queue
from sklearn.model_selection import ParameterGrid
from dataclasses import dataclass


@dataclass
class PFunCMAParamsGridResult:
    """result object for grid search"""
    #: integer index (within the original grid)
    i: int
    #: json-string-ified params
    params: str
    #: json-string-ified result
    result: str


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
        # setup solutions queue (multiproc friendly)
        self.solns = Queue(maxsize=len(self.pgrid))

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

    def batch_pickleable_psamplers(self, batch_params) -> tuple:
        """create a single computable unit of parameter samplers."""

        # ensure mealtimes are/are not present
        if self.include_mealtimes is True:
            tM = [batch_params.pop(tmk) for tmk in self.tmK]
            batch_params["tM"] = tM

        def compute_psample(params, output_queue=self.solns):
            """compute from a single sample of parameters from the grid."""
            cma = CMASleepWakeModel(config=params, N=self.N)
            out = cma.run()
            output_queue.put(out)

        return compute_psample, (batch_params,)

    @staticmethod
    def _batch_worker(compute_psample, sample_args):
        compute_psample(*sample_args)  # solution stored in output queue

    def run(self):
        """Run the parameter grid to produce a dataframe of results.
        """
        logging.info("Running parameter grid of size: %02d...",
                     len(self.pgrid))
        # distribute tasks across the pool
        current_batch = Queue(maxsize=self.Njobs)
        pending_results = Queue(maxsize=len(self.pgrid))
        with Pool(processes=self.Njobs) as pool:
            for i, params in enumerate(self.pgrid):
                # continually batch parameters (waiting only on queue max_size)
                batched_psamples = self.batch_pickleable_psamplers(params)
                current_batch.put(batched_psamples)
                # compute the batch, include in async results
                for pres in pool.map_async(PFunCMAParamsGrid._batch_worker, current_batch):
                    pending_results.put(pres)
            while True:
                pending_results.
                if i % self.Njobs == 0:
                    logging.debug(f"Iteration ({i:03d}/{len(self.pgrid)}) ...")

        # format results
        self.df = pd.DataFrame(, columns=["params", "result"], index="i")
        return self.df

import logging
from pfun_cma_model.engine.cma import CMASleepWakeModel
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


class PFunCMAParamsGrid:

    #: absolute upper/lower bounds for mealtimes
    tmK = ("tM0", "tM1", "tM2")
    tmL = (4, 11, 13)
    tmU = (11, 16, 22)
    
    def __init__(self, N=48, m=3, include_mealtimes=True, keys=None):
        self.N = N
        self.m = m
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
        self.fit_result = []
        self.df = None

    def run(self):
        logging.info("Running parameter grid of size: %02d...", len(self.pgrid))
        for i, params in enumerate(self.pgrid):
            if i % 7 == 0:
                logging.debug(f"Iteration ({i:03d}/{len(self.pgrid)}) ...")
            if self.include_mealtimes is True:
                tM = [params.pop(tmk) for tmk in self.tmK]
                params["tM"] = tM
            cma = CMASleepWakeModel(N=self.N)
            cma.update(**params)
            out = cma.run()
            self.fit_result.append({"i": i, "params": str(params), "result": out.to_json()})
        self.df = pd.DataFrame(self.fit_result, columns=["params", "result"])
        return self.df

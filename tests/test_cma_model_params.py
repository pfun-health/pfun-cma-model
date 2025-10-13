
from . import test_base
test_base.setup_test_environment()


class TestCMAModelParams:

    def __init__(self, *args, **kwargs):
        from pfun_cma_model.engine.cma_model_params import CMAModelParams
        self.cma_model_params_ = CMAModelParams

    # Create an instance of CMAModelParams with default values.
    def test_default_values(self):

        params = self.cma_model_params_()
        assert params.t is None
        assert params.N == 24
        assert params.d == 0.0
        assert params.taup == 1.0
        assert params.taug == 1.0
        assert params.B == 0.05
        assert params.Cm == 0.0
        assert params.toff == 0.0
        assert params.tM == (7.0, 11.0, 17.5)
        assert params.seed is None
        assert params.eps == 1e-18

    # Create an instance of CMAModelParams with all parameters set.
    def test_all_parameters_set(self):
        t = [1.0, 2.0, 3.0]
        taug = [0.5, 1.0, 1.5]
        tM = (5.0, 10.0, 15.0)
        params = self.cma_model_params_(t=t, N=100, d=0.5, taup=2.0, taug=taug,
                                        B=0.1, Cm=1.0, toff=0.5, tM=tM, seed=12345, eps=1e-10)
        assert params.t == t
        assert params.N == 100
        assert params.d == 0.5
        assert params.taup == 2.0
        assert params.taug == taug
        assert params.B == 0.1
        assert params.Cm == 1.0
        assert params.toff == 0.5
        assert params.tM == tM
        assert params.seed == 12345
        assert params.eps == 1e-10

    # Create an instance of CMAModelParams with a numpy array as t.
    def test_numpy_array_as_t(self):
        import numpy as np

        t = np.array([1.0, 2.0, 3.0])
        params = self.cma_model_params_(t=t)
        params_t = np.array(params.t)
        assert np.array_equal(params_t, t)

    # Create an instance of CMAModelParams with N=0.
    def test_N_zero(self):
        params = self.cma_model_params_(N=0)
        assert params.N == 0

    # Create an instance of CMAModelParams with d=NaN.
    def test_d_nan(self):
        import math
        params = self.cma_model_params_(d=math.nan)
        assert math.isnan(params.d)

    # Create an instance of CMAModelParams with taup=0.
    def test_taup_zero(self):
        params = self.cma_model_params_(taup=0)
        assert params.taup == 0.0

    def test_cma_params_description(self):
        params = self.cma_model_params_()
        descriptions = [(params.calc_serr(b), params.describe(b))
                        for b in params.bounded.bounded_param_keys]
        assert descriptions == [(0.0, 'Time zone offset (hours) (Normal)'),
                                (0.0, 'Photoperiod length (Normal)'),
                                (0.0, 'Glucose response time constant (Normal)'),
                                (0.0, 'Glucose Bias constant (Normal)'),
                                (0.0, 'Cortisol temporal sensitivity coefficient (Normal)'),
                                (0.0, 'Solar noon offset (latitude) (Normal)')]

    def test_cma_bounded_param_keys(self):
        params = self.cma_model_params_()
        assert params.bounded.bounded_param_keys == [
            'd', 'taup', 'taug', 'B', 'Cm', 'toff']

from . import test_base

test_base.setup_test_environment()


class TestCMAModelParams:
    def setup_method(self):
        from pfun_cma_model.engine.cma_model_params import CMAModelParams

        self.cma_model_params_ = CMAModelParams

    # Create an instance of CMAModelParams with default values.
    def test_default_values(self):
        import numpy as np

        params = self.cma_model_params_()
        assert isinstance(params.t, np.ndarray)
        assert params.N == 1024
        assert params.d == 0.0
        assert params.taup == 1.0
        assert params.taug == 1.0
        assert params.B == 0.05
        assert params.Cm == 0.0
        assert params.toff == 0.0
        # tM is serialized to list in json but stored as list/array.
        # Check against list values
        assert list(params.tM) == [7.0, 11.0, 17.5]
        assert params.seed is None
        assert params.eps == 1e-18

    # Create an instance of CMAModelParams with all parameters set.
    def test_all_parameters_set(self):
        import numpy as np

        taug = np.array([0.5, 1.0, 1.5])
        tM = np.array([5.0, 10.0, 15.0])
        params = self.cma_model_params_(
            N=100,
            d=0.5,
            taup=2.0,
            taug=taug,
            B=0.1,
            Cm=1.0,
            toff=0.5,
            tM=tM,
            seed=12345,
            eps=1e-10,
        )
        assert params.N == 100
        assert params.d == 0.5
        assert params.taup == 2.0
        assert np.array_equal(params.taug, taug)
        assert params.B == 0.1
        assert params.Cm == 1.0
        assert params.toff == 0.5
        assert np.array_equal(params.tM, tM)
        assert params.seed == 12345
        assert params.eps == 1e-10

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


    def test_cma_bounded_param_keys(self):
        params = self.cma_model_params_()
        assert params.bounded.bounded_param_keys == (
            "d",
            "taup",
            "taug",
            "B",
            "Cm",
            "toff",
        )

    def test_assignment_validation(self):
        import pytest
        from pydantic import ValidationError
        params = self.cma_model_params_()

        # Test valid assignment
        params.N = 100
        assert params.N == 100

        # Test invalid assignment (string instead of int for N)
        # Note: Pydantic might coerce string to int if possible, so use non-numeric string
        with pytest.raises(ValidationError):
            params.N = "invalid"

        # Test invalid assignment for float field
        with pytest.raises(ValidationError):
            params.d = "invalid"

    def test_mutable_defaults(self):
        """Verify that mutable default values are not shared across instances."""
        params1 = self.cma_model_params_()
        params2 = self.cma_model_params_()

        # Check if they are different objects
        assert params1.tM is not params2.tM

        # Modify params1.tM and verify params2.tM is unchanged
        original_value = params2.tM[0]
        params1.tM[0] = 99.9
        assert params2.tM[0] == original_value

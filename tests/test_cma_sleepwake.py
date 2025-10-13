
import pandas as pd
import numpy as np
from . import test_base
test_base.setup_test_environment()


class TestCMASleepWakeModel:

    # Test that the calculate_Gt method of the CMASleepWakeModel class returns the expected output.
    def test_calculate_Gt(self):
        from pfun_cma_model.engine.bounds import BoundsTypeError, Bounds
        from pfun_cma_model.engine.cma import CMASleepWakeModel
        model = CMASleepWakeModel()
        Gt = model.calc_Gt()
        assert isinstance(Gt, pd.DataFrame)
        assert Gt.shape[1] == (len(model.tM) + 1)

    # Test that the CMASleepWakeModel class correctly handles updating the model with invalid parameter values.

    def test_update_with_invalid_parameters(self):
        from pfun_cma_model.engine.bounds import BoundsTypeError, Bounds
        from pfun_cma_model.engine.cma import CMASleepWakeModel
        # Create an instance of CMASleepWakeModel
        model = CMASleepWakeModel()
        try:
            # Update the model with invalid parameter values
            model.update(d='invalid', taup='invalid', taug='invalid',
                         B='invalid', Cm='invalid', toff='invalid')
        except Exception as e:
            assert isinstance(e, (BoundsTypeError, ValueError, TypeError))

    def test_integrate_G_with_NaN_values(self):
        from pfun_cma_model.engine.bounds import BoundsTypeError, Bounds
        from pfun_cma_model.engine.cma import CMASleepWakeModel
        # Create an instance of CMASleepWakeModel
        model = CMASleepWakeModel()

        # Set the G signal to include NaN values
        model.G[10:20] = np.nan

        # Integrate the G signal over a time period
        result = model.integrate_signal(signal=model.G, t0=0, t1=24)

        # Check that the result is correct
        assert np.logical_not(np.isnan(result))

    def test_update_bounds(self):
        from pfun_cma_model.engine.bounds import BoundsTypeError, Bounds
        from pfun_cma_model.engine.cma import CMASleepWakeModel
        # Create an instance of CMASleepWakeModel
        model = CMASleepWakeModel()

        # Define the input parameters
        keys = ['d', 'taup']
        lb = [0.0, 1.0]
        ub = [10.0, 20.0]
        # weird pattern for satisfying mypy:
        keep_feasible = [np.bool_(x) for x in (True, False)]

        # Call the update_bounds method
        model.update_bounds(keys, lb, ub, keep_feasible)

        # Verify that the bounds are updated correctly
        expected_bounds = Bounds(
            lb=lb, ub=ub, keep_feasible=list(keep_feasible))  # type: ignore
        assert np.all(model.bounds[model.param_key_index(
            keys, only_bounded=True)] == expected_bounds)

    def test_cma_bounded_params_as_dict(self):
        from pfun_cma_model.engine.bounds import BoundsTypeError, Bounds
        from pfun_cma_model.engine.cma import CMASleepWakeModel
        model = CMASleepWakeModel()
        params = model.bounded_params_as_dict
        expected_dict = {
            'd': 0.0,
            'taup': 1.0,
            'taug': 1.0,
            'B': 0.05,
            'Cm': 0.0,
            'toff': 0.0
        }
        assert params == expected_dict

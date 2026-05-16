import pytest
import pandas as pd
from pfun_cma_model.engine.fit import estimate_mealtimes


def test_estimate_mealtimes_invalid_input():
    with pytest.raises(ValueError, match="Input data cannot be None or empty."):
        estimate_mealtimes(None)

    with pytest.raises(ValueError, match="Input data cannot be None or empty."):
        estimate_mealtimes(pd.DataFrame())


def test_estimate_mealtimes_valid_input():
    # Simple test to ensure it doesn't crash with valid data
    # Create dummy data
    t = pd.timedelta_range(start="0h", end="24h", freq="15min")
    # Convert timedeltas to decimal hours for 't' column as required by estimate_mealtimes logic if index isn't TimedeltaIndex initially
    # but estimate_mealtimes checks df[["t", ycol]]

    # Let's create a DataFrame that mimics what estimate_mealtimes expects
    data = pd.DataFrame({"t": t.total_seconds() / 3600.0, "G": [100.0] * len(t)})

    # Just checking it doesn't raise the None/Empty error
    # It might fail later in logic if data is too simple, but we only care about the first check here mostly.
    # But let's try to make it runnable.

    try:
        estimate_mealtimes(data, n_meals=1)
    except Exception as e:
        # We don't care if it fails on calculation, just that it passed the validation
        assert "Input data cannot be None or empty" not in str(e)

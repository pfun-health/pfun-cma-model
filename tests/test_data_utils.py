import pfun_path_helper as pph
import pandas as pd
from pfun_cma_model.engine.data_utils import format_data
from . import test_base
test_base.setup_test_environment()
pph.append_path(path=pph.get_lib_path('pfun_cma_model'))


class TestFormatData:
    # Tests that the function 'format_data' works correctly when given a pandas DataFrame with all required columns
    def test_input_dataframe_with_all_required_columns(self):
        # Create a sample input DataFrame
        input_df = pd.DataFrame({
            'systemTime': ['2022-01-01T00:00:00.000Z', '2022-01-01T00:30:00.000Z', '2022-01-01T01:00:00.000Z'],
            'displayTime': ['2022-01-01T00:00:00.000Z', '2022-01-01T00:30:00.000Z', '2022-01-01T01:00:00.000Z'],
            'value': [100, 110, 120],
            'sg': [100, 110, 120],
            'ts_utc': ['2022-01-01T00:00:00.000Z', '2022-01-01T00:30:00.000Z', '2022-01-01T01:00:00.000Z'],
            'ts_local': ['2022-01-01T00:00:00.000Z', '2022-01-01T00:30:00.000Z', '2022-01-01T01:00:00.000Z']
        })
        # Call the function
        output_df = format_data(input_df)
        # Check that the output is a pandas DataFrame
        assert isinstance(output_df, pd.DataFrame)
        # Check that the output has the expected columns
        assert set(output_df.columns) == {'time', 'value', 'tod', 't', 'G'}
        # Check that the output has the expected number of rows
        # 1024 samples, see 'pfun_cma_model.engine.data_utils.downsample_data'
        assert len(output_df) == 1024

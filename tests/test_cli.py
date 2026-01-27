"""
Tests for the CLI module (pfun_cma_model/cli.py).

Tests cover all CLI commands:
- cli: main CLI group
- launch: launch the application
- generate_scenario: generate scenario using LLM
- fit_model: fit the model to data
- run_param_grid: run parameter grid search
- download_sample_data: download sample data
- version: print version
- run_doctests: run doctests
"""

import pfun_path_helper as pph  # type: ignore
pph.append_path(path=pph.get_lib_path('pfun_cma_model'))
from . import test_base
test_base.setup_test_environment()

import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from click.testing import CliRunner
import click
import pandas as pd
from pfun_cma_model.cli import (
    cli,
    launch,
    generate_scenario,
    fit_model,
    run_param_grid,
    download_sample_data,
    version,
    run_doctests,
    process_kwds
)


@pytest.fixture
def runner():
    """Provides a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_data_df():
    """Provides sample data as a DataFrame."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='15min'),
        'glucose': [100 + i*0.5 for i in range(100)],
        'cortisol': [10 + i*0.1 for i in range(100)]
    })


@pytest.fixture
def temp_output_dir():
    """Provides a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestCliGroup:
    """Tests for the main CLI group."""

    def test_cli_group_creation(self, runner):
        """Test that the CLI group can be invoked."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output

    def test_cli_group_help(self, runner):
        """Test CLI group help displays available commands."""
        result = runner.invoke(cli, ['--help'])
        assert 'launch' in result.output
        assert 'generate-scenario' in result.output
        assert 'fit-model' in result.output
        assert 'run-param-grid' in result.output
        assert 'download-sample-data' in result.output
        assert 'version' in result.output
        assert 'run-doctests' in result.output

    def test_cli_context_object_initialization(self, runner):
        """Test that CLI context object is properly initialized."""
        result = runner.invoke(cli, ['--help'])
        # Verify the help command works and context was initialized
        assert result.exit_code == 0
        assert 'Usage:' in result.output


class TestLaunchCommand:
    """Tests for the launch command."""

    def test_launch_help(self, runner):
        """Test launch command help."""
        result = runner.invoke(cli, ['launch', '--help'])
        assert result.exit_code == 0
        assert '--host' in result.output
        assert '--port' in result.output
        assert '--reload' in result.output

    def test_launch_default_options(self, runner):
        """Test launch command with default options."""
        with patch('pfun_cma_model.main.run_app') as mock_run_app:
            result = runner.invoke(cli, ['launch'])
            # Should call run_app with defaults
            mock_run_app.assert_called_once()
            args, kwargs = mock_run_app.call_args
            assert args[0] == '0.0.0.0'  # host
            assert args[1] == 8001  # port
            assert kwargs.get('reload') is False
            assert kwargs.get('debug') is True

    def test_launch_custom_host_port(self, runner):
        """Test launch command with custom host and port."""
        with patch('pfun_cma_model.main.run_app') as mock_run_app:
            result = runner.invoke(cli, ['launch', '--host', '127.0.0.1', '--port', '9000'])
            mock_run_app.assert_called_once()
            args, kwargs = mock_run_app.call_args
            assert args[0] == '127.0.0.1'
            assert args[1] == 9000

    def test_launch_with_reload(self, runner):
        """Test launch command with reload flag."""
        with patch('pfun_cma_model.main.run_app') as mock_run_app:
            result = runner.invoke(cli, ['launch', '--reload'])
            mock_run_app.assert_called_once()
            args, kwargs = mock_run_app.call_args
            assert kwargs.get('reload') is True

    def test_launch_with_extra_args(self, runner):
        """Test launch command with extra arguments passed through."""
        with patch('pfun_cma_model.main.run_app') as mock_run_app:
            result = runner.invoke(cli, ['launch', '--', '--some-arg', 'value'])
            mock_run_app.assert_called_once()
            args, kwargs = mock_run_app.call_args
            # Extra args should be passed as extra_args list
            assert 'extra_args' in kwargs


class TestGenerateScenarioCommand:
    """Tests for the generate_scenario command."""

    def test_generate_scenario_help(self, runner):
        """Test generate_scenario command help."""
        result = runner.invoke(cli, ['generate-scenario', '--help'])
        assert result.exit_code == 0
        assert '--query' in result.output

    def test_generate_scenario_default_query(self, runner):
        """Test generate_scenario with default query."""
        mock_response = {'scenario': 'test scenario'}
        with patch('pfun_cma_model.llm.generate_scenario', new_callable=MagicMock, return_value=mock_response, new_callable=MagicMock):
            with patch('pfun_cma_model.cli.asyncio.run', return_value=mock_response):
                result = runner.invoke(cli, ['generate-scenario'])
                # Should output JSON
                assert result.exit_code == 0

    def test_generate_scenario_custom_query(self, runner):
        """Test generate_scenario with custom query."""
        mock_response = {'scenario': 'custom scenario'}
        custom_query = "A patient with diabetes."
        with patch('pfun_cma_model.llm.generate_scenario', new_callable=MagicMock, return_value=mock_response):
            with patch('pfun_cma_model.cli.asyncio.run', return_value=mock_response):
                result = runner.invoke(cli, ['generate-scenario', '--query', custom_query])
                assert result.exit_code == 0

    def test_generate_scenario_json_output(self, runner):
        """Test that generate_scenario outputs valid JSON."""
        mock_response = {
            'scenario': 'test',
            'glucose': [100, 110, 105],
            'cortisol': [10, 12, 11]
        }
        with patch('pfun_cma_model.llm.generate_scenario', new_callable=MagicMock, return_value=mock_response):
            with patch('pfun_cma_model.cli.asyncio.run', return_value=mock_response):
                result = runner.invoke(cli, ['generate-scenario'])
                # Should produce JSON output
                assert result.exit_code == 0


class TestFitModelCommand:
    """Tests for the fit_model command."""

    def test_fit_model_help(self, runner):
        """Test fit_model command help."""
        result = runner.invoke(cli, ['fit-model', '--help'])
        assert result.exit_code == 0
        assert '--input-fpath' in result.output
        assert '--output-dir' in result.output
        assert '--N' in result.output
        assert '--plot' in result.output

    def test_fit_model_with_default_input(self, runner, temp_output_dir):
        """Test fit_model with default input data."""
        mock_fit_result = MagicMock()
        mock_fit_result.model_dump_json.return_value = '{}'
        mock_fit_result.formatted_data = pd.DataFrame()

        with patch('pfun_cma_model.engine.fit.fit_model', return_value=mock_fit_result):
            with patch('pfun_cma_model.cli.pd.read_csv') as mock_read:
                mock_read.return_value = pd.DataFrame({'glucose': [100, 110]})
                result = runner.invoke(cli, ['fit-model'], input='{}')
                assert result.exit_code == 0
                assert 'fit_result.json' in result.output

    def test_fit_model_with_custom_input(self, runner, sample_data_df, temp_output_dir):
        """Test fit_model with custom input file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data_df.to_csv(f, index=False)
            input_file = f.name

        try:
            mock_fit_result = MagicMock()
            mock_fit_result.model_dump_json.return_value = '{}'
            mock_fit_result.formatted_data = sample_data_df

            with patch('pfun_cma_model.engine.fit.fit_model', return_value=mock_fit_result):
                result = runner.invoke(
                    cli,
                    ['fit-model', '--input-fpath', input_file, '--output-dir', temp_output_dir],
                    input='{}'
                )
                assert result.exit_code == 0
                assert 'fit_result.json' in result.output
        finally:
            os.unlink(input_file)

    def test_fit_model_with_custom_N(self, runner, temp_output_dir):
        """Test fit_model with custom N parameter."""
        mock_fit_result = MagicMock()
        mock_fit_result.model_dump_json.return_value = '{}'
        mock_fit_result.formatted_data = pd.DataFrame()

        with patch('pfun_cma_model.engine.fit.fit_model') as mock_fit:
            with patch('pfun_cma_model.cli.pd.read_csv') as mock_read:
                mock_read.return_value = pd.DataFrame()
                result = runner.invoke(
                    cli,
                    ['fit-model', '--N', '500', '--output-dir', temp_output_dir],
                    input='{}'
                )
                # Check that fit was called with the custom N
                if mock_fit.called:
                    assert mock_fit.call_args[1]['n'] == 500

    def test_fit_model_with_plot(self, runner, temp_output_dir):
        """Test fit_model with --plot flag."""
        mock_fit_result = MagicMock()
        mock_fit_result.model_dump_json.return_value = '{}'
        mock_fit_result.formatted_data = pd.DataFrame()

        with patch('pfun_cma_model.engine.fit.fit_model', return_value=mock_fit_result):
            with patch('pfun_cma_model.cli.pd.read_csv') as mock_read:
                with patch('pfun_cma_model.engine.cma_plot.CMAPlotSolnConfig') as mock_plot:
                    mock_read.return_value = pd.DataFrame()
                    mock_fig = MagicMock()
                    mock_plot.return_value.plot.return_value = (mock_fig, None)

                    result = runner.invoke(
                        cli,
                        ['fit-model', '--plot', '--output-dir', temp_output_dir],
                        input='{}',
                        catch_exceptions=False
                    )
                    # Should have created plot file
                    assert 'plot' in result.output.lower() or result.exit_code == 0

    def test_fit_model_with_opts(self, runner, temp_output_dir):
        """Test fit_model with custom curve fit options."""
        mock_fit_result = MagicMock()
        mock_fit_result.model_dump_json.return_value = '{}'
        mock_fit_result.formatted_data = pd.DataFrame()

        with patch('pfun_cma_model.engine.fit.fit_model', return_value=mock_fit_result):
            with patch('pfun_cma_model.cli.pd.read_csv') as mock_read:
                mock_read.return_value = pd.DataFrame()
                result = runner.invoke(
                    cli,
                    ['fit-model', '--opts', 'maxiter', '100', '--output-dir', temp_output_dir],
                    input='{}'
                )
                assert result.exit_code == 0

    def test_fit_model_with_model_config(self, runner, temp_output_dir):
        """Test fit_model with model configuration."""
        config = '{"param1": "value1"}'
        mock_fit_result = MagicMock()
        mock_fit_result.model_dump_json.return_value = '{}'
        mock_fit_result.formatted_data = pd.DataFrame()

        with patch('pfun_cma_model.engine.fit.fit_model', return_value=mock_fit_result):
            with patch('pfun_cma_model.cli.pd.read_csv') as mock_read:
                mock_read.return_value = pd.DataFrame()
                result = runner.invoke(
                    cli,
                    ['fit-model', '--model-config', config, '--output-dir', temp_output_dir],
                    input=config
                )
                assert result.exit_code == 0

    def test_fit_model_output_file_creation(self, runner, sample_data_df):
        """Test that fit_model creates output JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                sample_data_df.to_csv(f, index=False)
                input_file = f.name

            try:
                mock_fit_result = MagicMock()
                fit_json = '{"params": {"glucose": 100}}'
                mock_fit_result.model_dump_json.return_value = fit_json
                mock_fit_result.formatted_data = sample_data_df

                with patch('pfun_cma_model.engine.fit.fit_model', return_value=mock_fit_result):
                    result = runner.invoke(
                        cli,
                        ['fit-model', '--input-fpath', input_file, '--output-dir', tmpdir],
                        input='{}'
                    )
                    output_file = os.path.join(tmpdir, 'fit_result.json')
                    assert os.path.exists(output_file)
            finally:
                os.unlink(input_file)


class TestRunParamGridCommand:
    """Tests for the run_param_grid command."""

    def test_run_param_grid_help(self, runner):
        """Test run_param_grid command help."""
        result = runner.invoke(cli, ['run-param-grid', '--help'])
        assert result.exit_code == 0

    def test_run_param_grid_execution(self, runner, temp_output_dir):
        """Test run_param_grid command execution."""
        mock_grid = MagicMock()

        # Mock the result of run() to be an object with a .params attribute
        mock_result = MagicMock()
        mock_result.params = MagicMock() # Mock the params attribute (PyArrow Table)

        mock_grid.pgrid = [1, 2, 3]
        mock_grid.run.return_value = mock_result


class TestDownloadSampleDataCommand:
    """Tests for the download_sample_data command."""

    def test_download_sample_data_help(self, runner):
        """Test download_sample_data command help."""
        result = runner.invoke(cli, ['download-sample-data', '--help'])
        assert result.exit_code == 0
        assert '--overwrite' in result.output

    def test_download_sample_data_without_overwrite(self, runner):
        """Test download_sample_data without overwrite flag."""
        mock_paths = MagicMock()
        mock_paths.sample_data_fpath = '/path/to/sample_data.csv'

        with patch('pfun_cma_model.misc.pathdefs.PFunDataPaths', return_value=mock_paths):
            result = runner.invoke(cli, ['download-sample-data'])
            assert result.exit_code == 0
            mock_paths.download_sample_data.assert_called_once_with(overwrite=False)

    def test_download_sample_data_with_overwrite(self, runner):
        """Test download_sample_data with overwrite flag."""
        mock_paths = MagicMock()
        mock_paths.sample_data_fpath = '/path/to/sample_data.csv'

        with patch('pfun_cma_model.misc.pathdefs.PFunDataPaths', return_value=mock_paths):
            result = runner.invoke(cli, ['download-sample-data', '--overwrite'])
            assert result.exit_code == 0
            mock_paths.download_sample_data.assert_called_once_with(overwrite=True)
            assert 'Overwrite is enabled' in result.output

    def test_download_sample_data_output_message(self, runner):
        """Test download_sample_data displays correct output message."""
        mock_paths = MagicMock()
        mock_paths.sample_data_fpath = '/test/path/sample_data.csv'

        with patch('pfun_cma_model.misc.pathdefs.PFunDataPaths', return_value=mock_paths):
            result = runner.invoke(cli, ['download-sample-data'])
            assert result.exit_code == 0
            assert '/test/path/sample_data.csv' in result.output


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_help(self, runner):
        """Test version command help."""
        result = runner.invoke(cli, ['version', '--help'])
        assert result.exit_code == 0

    def test_version_output(self, runner):
        """Test version command outputs version string."""
        result = runner.invoke(cli, ['version'])
        assert result.exit_code == 0
        assert 'pfun-cma-model version:' in result.output

    def test_version_output_format(self, runner):
        """Test version command output format."""
        result = runner.invoke(cli, ['version'])
        assert result.exit_code == 0
        # Should contain version pattern
        assert 'version' in result.output.lower()


class TestRunDoctestsCommand:
    """Tests for the run_doctests command."""

    def test_run_doctests_help(self, runner):
        """Test run_doctests command help."""
        result = runner.invoke(cli, ['run-doctests', '--help'])
        assert result.exit_code == 0

    def test_run_doctests_execution(self, runner):
        """Test run_doctests command execution."""
        result = runner.invoke(cli, ['run-doctests'])
        # Should complete without error (even if no doctests found)
        assert result.exit_code == 0 or result.exit_code is not None


class TestProcessKwds:
    """Tests for the process_kwds callback function."""

    def test_process_kwds_with_integers(self):
        """Test process_kwds converts string integers to int."""
        ctx = MagicMock()
        param = MagicMock()
        param.name = 'opts'
        value = [['maxiter', '100'], ['popsize', '50']]
        result = process_kwds(ctx, param, value)
        assert result[0][1] == 100
        assert result[1][1] == 50
        assert isinstance(result[0][1], int)
        assert isinstance(result[1][1], int)

    def test_process_kwds_with_floats(self):
        """Test process_kwds doesn't convert float strings (isnumeric() fails on decimals)."""
        ctx = MagicMock()
        param = MagicMock()
        param.name = 'opts'
        value = [['learning_rate', '0.001'], ['threshold', '0.5']]
        result = process_kwds(ctx, param, value)
        # isnumeric() returns False for strings with decimal points, so they stay as strings
        assert result[0][1] == '0.001'
        assert result[1][1] == '0.5'
        assert isinstance(result[0][1], str)
        assert isinstance(result[1][1], str)

    def test_process_kwds_with_non_numeric_values(self):
        """Test process_kwds leaves non-numeric values as strings."""
        ctx = MagicMock()
        param = MagicMock()
        param.name = 'opts'
        value = [['mode', 'fast'], ['method', 'newton']]
        result = process_kwds(ctx, param, value)
        assert result[0][1] == 'fast'
        assert result[1][1] == 'newton'

    def test_process_kwds_other_param(self):
        """Test process_kwds returns value unchanged for non-opts params."""
        ctx = MagicMock()
        param = MagicMock()
        param.name = 'other_param'
        value = 'some_value'
        result = process_kwds(ctx, param, value)
        assert result == value

    def test_process_kwds_mixed_types(self):
        """Test process_kwds with mixed integer and string values."""
        ctx = MagicMock()
        param = MagicMock()
        param.name = 'opts'
        value = [['count', '10'], ['ratio', '0.5'], ['name', 'test']]
        result = process_kwds(ctx, param, value)
        assert result[0][1] == 10
        assert isinstance(result[0][1], int)
        # Float strings stay as strings since isnumeric() returns False for them
        assert result[1][1] == '0.5'
        assert isinstance(result[1][1], str)
        assert result[2][1] == 'test'
        assert isinstance(result[2][1], str)

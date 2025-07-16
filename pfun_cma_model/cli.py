import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import click
from sklearn.model_selection import ParameterGrid
import pfun_path_helper as pph
from pfun_cma_model.misc.pathdefs import PFunDataPaths
from pfun_cma_model.engine.cma_plot import CMAPlotConfig
from pfun_cma_model.engine.cma import CMASleepWakeModel
from pfun_cma_model.engine.fit import fit_model as call_fit_model
from pfun_cma_model.app import run_app


@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)
    ctx.obj["sample_data_fpath"] = PFunDataPaths().sample_data_fpath
    ctx.obj["output_dir"] = os.path.abspath(
        os.path.join(pph.get_lib_path('pfun_cma_model'), '../results')
    )


@cli.command()
@click.pass_context
def launch(ctx):
    run_app()


def process_kwds(ctx, param, value):
    if param.name != "opts":
        return value
    value = list(value)
    for i in range(len(value)):
        value[i] = list(value[i])
        if value[i][1].isnumeric():
            try:
                new = int(value[i][1])
            except ValueError:
                new = float(value[i][1])
            value[i][1] = new
    return value


fit_result_global = None


@cli.command()
@click.option('--input-fpath', '-i', type=click.Path(exists=True), default=None, required=False)
@click.option('--output-dir', '--output', '-o', type=click.Path(exists=True), default=None, required=False)
@click.option("--N", default=288, type=click.INT, help="Number of time points to produce in the final model solution.")
@click.option("--plot/--no-plot", is_flag=True, default=False)
@click.option("--opts", "--curve-fit-kwds", multiple=True, type=click.Tuple([str, click.UNPROCESSED]),
              callback=process_kwds)
@click.option("--model-config", "--config", prompt=True, default="{}", type=str)
@click.pass_context
def fit_model(ctx, input_fpath, output_dir, n, plot, opts, model_config):
    global fit_result_global
    model_config = json.loads(model_config)
    if input_fpath is None:
        input_fpath = ctx.obj["sample_data_fpath"]
    if output_dir is None:
        output_dir = ctx.obj["output_dir"]
    # read the input dataset
    data = pd.read_csv(input_fpath)
    # fit the model
    fit_result = call_fit_model(data, n=n, plot=plot, opts=opts, **model_config)
    fit_result_global = fit_result
    # write fitted model parameters (with the corresponding time-series solution) to disk
    output_fpath = os.path.join(output_dir, "fit_result.json")
    with open(output_fpath, "w", encoding='utf8') as f:
        f.write(fit_result.model_dump_json())
    click.secho(f"...wrote fitted model params to: '{output_fpath}'")
    # plot the results (if '--plot' is indicated)
    if plot is True:
        from pfun_cma_model.engine.cma_plot import CMAPlotConfig
        fig, _ = CMAPlotConfig().plot_model_results(df=fit_result.formatted_data, soln=fit_result.soln, as_blob=False)
        fig_output_fpath = os.path.join(output_dir, "fit_result.png")
        fig.savefig(fig_output_fpath)
        click.secho(f"...saved plot to: '{fig_output_fpath}'")
        click.confirm("[enter] to exit...", default=True, abort=True, show_default=False)
        plt.close('all')


@cli.command()
@click.pass_context
def run_param_grid(ctx):
    output_fpath = os.path.join(ctx.obj["output_dir"], "cma_paramgrid.feather")
    from pfun_cma_model.engine.grid import PFunCMAParamsGrid
    pfun_grid = PFunCMAParamsGrid()
    df = pfun_grid.run()
    df.to_feather(output_fpath)
    click.secho(f"...saved result to: '{output_fpath}'")
    click.secho('...done.')


@cli.command()
def run_doctests():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    cli()

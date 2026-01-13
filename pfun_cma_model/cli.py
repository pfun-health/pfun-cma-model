import asyncio
import json
import os

import click
import matplotlib.pyplot as plt
import pandas as pd

# Ignore mypy for the next line (this is my repo)
import pfun_path_helper as pph  # type: ignore
from pfun_cma_model.misc.pathdefs import PFunDataPaths

pph.get_lib_path("pfun_cma_model")


@click.group()
@click.pass_context
def cli(ctx):
    """Command line interface for the pfun-cma-model package.
    This CLI provides commands to fit the PFun CMA model, run parameter grid searches, and launch the application.
    """
    # Set up the context object with default paths
    # for sample data and output directory
    ctx.ensure_object(dict)
    ctx.obj["sample_data_fpath"] = PFunDataPaths().sample_data_fpath
    ctx.obj["output_dir"] = os.path.abspath(
        os.path.join(pph.get_lib_path("pfun_cma_model"), "../results")
    )


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option("--host", default="0.0.0.0", help="Host to run the application on.")
@click.option("--port", default=8001, help="Port to run the application on.")
@click.option(
    "--reload", is_flag=True, default=False, help="Enable auto-reload for development."
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def launch(ctx, host, port, reload, args):
    """Launch the application.

    Any additional arguments (ARGS) are passed through to the application.
    """
    from pfun_cma_model.main import run_app
    run_app(host, port, reload=reload, debug=True, extra_args=list(args))


@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.option(
    "--query",
    default="A healthy individual.",
    help="Specify a query describing the desired llm-generated scenario.",
    required=False,
)
@click.pass_context
def generate_scenario(ctx, query):
    """Generate a realistic pfun scenario (using Google AI Studio)."""
    from pfun_cma_model.llm import generate_scenario as gen_scene
    click.secho(
        f"Generating a scenario from prompt:\n\t'{query[:20]}...'\n"
    )
    try:
        loop = asyncio.get_running_loop()
        response = loop.run_until_complete(gen_scene(query=query))
    except RuntimeError:
        response = asyncio.run(gen_scene(query=query))
    click.secho(json.dumps(response, indent=4))


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
@click.option(
    "--input-fpath", "-i", type=click.Path(exists=True), default=None, required=False
)
@click.option(
    "--output-dir",
    "--output",
    "-o",
    type=click.Path(exists=True),
    default=None,
    required=False,
)
@click.option(
    "--N",
    default=288,
    type=click.INT,
    help="Number of time points to produce in the final model solution.",
)
@click.option("--plot/--no-plot", is_flag=True, default=False)
@click.option(
    "--opts",
    "--curve-fit-kwds",
    multiple=True,
    type=click.Tuple([str, click.UNPROCESSED]),
    callback=process_kwds,
)
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
    from pfun_cma_model.engine.fit import fit_model as call_fit_model
    fit_result = call_fit_model(
        data, n=n, plot=plot, opts=opts, **model_config)
    fit_result_global = fit_result
    # write fitted model parameters (with the corresponding time-series solution) to disk
    output_fpath = os.path.join(output_dir, "fit_result.json")
    with open(output_fpath, "w", encoding="utf8") as f:
        f.write(fit_result.model_dump_json())
    click.secho(f"...wrote fitted model params to: '{output_fpath}'")
    # plot the results (if '--plot' is indicated)
    if plot is True:
        from pfun_cma_model.engine.cma_plot import CMAPlotSolnConfig
        fig, _ = CMAPlotSolnConfig().plot(df=fit_result.formatted_data)
        fig_output_fpath = os.path.join(output_dir, "fit_result.png")
        fig.savefig(fig_output_fpath)
        click.secho(f"...saved plot to: '{fig_output_fpath}'")
        click.confirm(
            "[enter] to exit...", default=True, abort=True, show_default=False
        )
        plt.close("all")


@cli.command()
@click.option(
    "--output-filetype",
    "--output-ftype",
    "-F",
    type=click.Choice(('parquet', 'feather')),
    default='parquet',
    help="Output file type.",
)
@click.option(
    "-N", "-n",
    type=click.INT,
    default=100,
    help="Length of solutions vector (in number of time points).",
)
@click.option(
    "-m",
    type=click.INT,
    default=3,
    help="Parameter grid width (in span of parameter values).",
)
@click.pass_context
def run_param_grid(ctx, output_filetype, n, m):
    """Run a parameter grid search for the PFun CMA model."""
    click.secho(f"Output directory: {ctx.obj['output_dir']}")
    click.secho("Running parameter grid search for the PFun CMA model...")
    # create the output file path
    if not os.path.exists(ctx.obj["output_dir"]):
        os.makedirs(ctx.obj["output_dir"])
    output_fpath = os.path.join(
        ctx.obj["output_dir"], f"cma_paramgrid.{output_filetype}")
    from pfun_cma_model.engine.grid import PFunCMAParamsGrid
    pfun_grid = PFunCMAParamsGrid(N=n, m=m, include_mealtimes=True)
    # run the grid search
    Nparam = len(pfun_grid.pgrid)
    click.secho(f"Running a parameter grid search of size: {Nparam:02d}...")
    grid_collated = pfun_grid.run()
    # output to the specified filepath (with `output_filetype`)
    match output_filetype:
        case "parquet":
            import pyarrow.parquet as pq
            pq.write_table(grid_collated.params, output_fpath)
        case "feather":
            # df.to_feather(output_fpath)
            raise NotImplementedError('not yet implemented')
    click.secho(f"...saved result to: '{output_fpath}'")
    click.secho("...done.")


@cli.command()
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing sample data file if it exists.",
)
@click.pass_context
def download_sample_data(ctx, overwrite=False):
    """Download the sample data for the pfun-cma-model package."""
    click.secho("Downloading sample data for the pfun-cma-model package...")
    if overwrite:
        click.secho(
            "Overwrite is enabled; existing files will be replaced if they exist.",
            fg="yellow",
            bold=True,
        )
    from pfun_cma_model.misc.pathdefs import PFunDataPaths
    pfun_data_paths = PFunDataPaths()
    pfun_data_paths.download_sample_data(overwrite=overwrite)
    click.secho(
        f"...sample data downloaded to: '{pfun_data_paths.sample_data_fpath}'")


@cli.command()
def version():
    """Print the version of the pfun-cma-model package."""
    import pfun_cma_model
    click.secho(f"pfun-cma-model version: {pfun_cma_model.__version__}")


@cli.command()
def run_doctests():
    """Run the doctests for the pfun-cma-model cli."""
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    cli()

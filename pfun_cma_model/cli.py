import asyncio
from typing import Literal
import json
import os
from pathlib import Path
import click
import pandas as pd
from pfun_cma_model.misc.pathdefs import PFunDataPaths


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
    import pfun_path_helper as pph  # type: ignore

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

OutputFigureFormatType = Literal["png", "svg"]


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
    "--output-ftype",
    "--ftype",
    "-T",
    type=click.Choice(OutputFigureFormatType.__args__),
    default="png",
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
def fit_model(ctx, input_fpath, output_dir, output_ftype, n, plot, opts, model_config):
    global fit_result_global
    model_config = json.loads(model_config)
    if input_fpath is None:
        input_fpath = ctx.obj["sample_data_fpath"]
        from pfun_cma_model.misc.pathdefs import PFunDataPaths

        pfun_data_paths = PFunDataPaths()
        pfun_data_paths.download_sample_data(overwrite=True)
        click.secho(
            f"...sample data downloaded to: '{pfun_data_paths.sample_data_fpath}'",
            fg="green",
            bold=True,
        )
    if output_dir is None:
        output_dir = ctx.obj["output_dir"]
    # read the input dataset
    data = pd.read_csv(input_fpath)
    # fit the model
    from pfun_cma_model.engine.fit import fit_model as call_fit_model

    fit_result = call_fit_model(data, n=n, plot=plot, opts=opts, **model_config)
    fit_result_global = fit_result
    # write fitted model parameters (with the corresponding time-series solution) to disk
    output_fpath = os.path.join(output_dir, "fit_result.json")
    with open(output_fpath, "w", encoding="utf8") as f:
        f.write(fit_result.model_dump_json())
    click.secho(f"...wrote fitted model params to: '{output_fpath}'")
    # plot the results (if '--plot' is indicated)
    if plot is True:
        click.secho("Plotting...", bold=True)
        import matplotlib
        matplotlib.use("Agg")
        click.secho(f"Set matplotlib backend: Agg", bold=True)
        import matplotlib.pyplot as plt
        from pfun_cma_model.engine.cma_plot import CMAPlotSolnConfig, CMAPlotDataConfig
        click.secho("Formatted data (from fit_result):", bold=True)
        click.secho(fit_result.formatted_data.head().to_string())
        fig, _ = CMAPlotDataConfig().plot(df=fit_result.formatted_data, plot_cols=["G", ])
        fig_output_fpath = os.path.join(output_dir, f"fit_result.{output_ftype}")
        fig.savefig(fig_output_fpath, format=output_ftype)
        click.secho(f"...saved plot to: '{fig_output_fpath}'")
        click.confirm(
            "[enter] to exit...", default=True, abort=True, show_default=False
        )
        plt.close("all")


@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.option(
    "--query",
    default="A healthy individual.",
    help="Specify a query describing the desired llm-generated scenario.",
    required=False,
)
@click.pass_context
def generate_scenario(ctx, query):
    """Generate a realistic pfun scenario (using selected LLM backend)."""
    from pfun_cma_model.llm import generate_scenario as gen_scene

    click.secho(f"Generating a scenario from prompt:\n\t'{query[:20]}...'\n")
    try:
        loop = asyncio.get_running_loop()
        generated_scenario = loop.run_until_complete(gen_scene(query=query))
    except RuntimeError:
        # get the pydantic-validated scenario
        generated_scenario = asyncio.run(gen_scene(query=query))

    # dump to JSON, pretty-print the output for CLI
    if not hasattr(generated_scenario, "model_dump"):
        response = generated_scenario
    else:
        response = generated_scenario.model_dump()  # type: ignore
    output_json_formatted = json.dumps(response, indent=4)
    click.secho(
        output_json_formatted.encode("utf8").decode("unicode_escape"),
        fg="green",
        bold=True,
    )

    # # # ####################
    # Save result to database.
    # # # ####################

    df_result = pd.DataFrame([response], index=[0])
    df_result.to_parquet(os.path.join(ctx.obj["output_dir"], "cma_recs.parquet"))
    # save the generated params, recommendations to the results duckdb database
    from pfun_cma_model.db import save2duckdb

    db_path = Path(__file__).parent.parent.joinpath("results", "duckdb.db")
    save2duckdb(df_result, db_path=str(db_path), table_id="cma_recs")
    click.secho("...successfully saved result to the database.", fg="green", bold=True)


@cli.command()
@click.option(
    "-N",
    "-n",
    type=click.INT,
    default=6,
    help="Length of solutions vector (in number of time points).",
)
@click.option(
    "-m",
    type=click.INT,
    default=3,
    help="Parameter grid width (in span of parameter values).",
)
@click.option(
    "--params",
    "-P",
    type=click.STRING,
    multiple=True,
    callback=process_kwds,
    default=["taug", "taup", "B", "Cm"],
    help="Parameters to include as part of the grid search.",
)
@click.pass_context
def run_param_grid(ctx, n, m, params):
    """Run a parameter grid search for the PFun CMA model."""
    click.secho(f"Output directory: {ctx.obj['output_dir']}")
    click.secho("Running parameter grid search for the PFun CMA model...")
    # create the output file path
    if not os.path.exists(ctx.obj["output_dir"]):
        os.makedirs(ctx.obj["output_dir"])
    # create the parameter grid
    from pfun_cma_model.engine.grid import PFunCMAParamsGrid

    pkeys_included = params
    import logging

    logging.debug(f"{pkeys_included}")
    click.secho("Included parameter keys:", fg="yellow", bold=True)
    if not pkeys_included:
        click.secho("    + [all]", fg="yellow")
    else:
        for pkey in pkeys_included:
            click.secho(f"    + {pkey}", fg="yellow")
    pfun_grid = PFunCMAParamsGrid(
        N=n, m=m, keys=pkeys_included, include_mealtimes=True
    )  # parameter keys to include

    # run the grid search
    Nparam = len(pfun_grid.pgrid)
    click.secho(f"Running a parameter grid search of size: {Nparam:02d}...")
    pfun_grid.run()  # perform the operation in-place

    # get the grid results as a dataframe
    df_grid: pd.DataFrame = pfun_grid.collection  # type: ignore

    # save to duckdb database
    from pfun_cma_model.db import save2duckdb

    db_fpath = "results/duckdb.db"
    table_id = "cma_pgrid"
    save2duckdb(df_grid, db_path=db_fpath, table_id=table_id)
    click.secho("...done (saved to 'results/duckdb.db').", fg="green", bold=True)

    # save to parquet
    parquet_fpath = Path(ctx.obj["output_dir"]).joinpath(
        f"param_grid_{n:02d}x{m:02d}.parquet"
    )
    df_grid.to_parquet(str(parquet_fpath))
    click.secho("...done (saved to '').", fg="green", bold=True)


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
        f"...sample data downloaded to: '{pfun_data_paths.sample_data_fpath}'",
        fg="green",
        bold=True,
    )


@cli.command()
def version():
    """Print the version of the pfun-cma-model package."""
    import pfun_cma_model

    click.secho(f"pfun-cma-model version: {pfun_cma_model.__version__}", bold=True)


@cli.command()
def run_doctests():
    """Run the doctests for the pfun-cma-model cli."""
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    cli()

import click
import json
from sklearn.model_selection import ParameterGrid
from pfun_cma_model.runtime.chalicelib.engine.cma_sleepwake import CMASleepWakeModel


@click.group()
@click.pass_context
def cli(ctx):
    pass


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
            finally:
                value[i][1] = new
    return value


fit_result_global = None


@cli.command()
@click.option("--N", default=288, type=click.INT)
@click.option("--plot/--no-plot", is_flag=True, default=False)
@click.option("--opts", "--curve-fit-kwds", multiple=True, type=click.Tuple([str, click.UNPROCESSED]),
              callback=process_kwds)
@click.option("--model-config", "--config", prompt=True, default="{}", type=str)
@click.pass_context
def run_fit_model(ctx, n, plot, opts, model_config):
    global fit_result_global
    model_config = json.loads(model_config)
    fit_result = test_fit_model(n=n, plot=plot, opts=opts, **model_config)
    fit_result_global = fit_result
    if plot is True:
        click.confirm("[enter] to exit...", default=True,
                      abort=True, show_default=False)


@cli.command()
@click.pass_context
def run_param_grid(ctx):
    global fit_result_global
    fit_result_global = []
    keys = list(CMASleepWakeModel.param_keys)
    lb = list(CMASleepWakeModel.bounds.lb)
    ub = list(CMASleepWakeModel.bounds.ub)
    tmK = ["tM0", "tM1", "tM2"]
    tmL, tmU = [0, 11, 13], [13, 17, 24]
    plist = list(zip(keys, lb, ub))
    pdict = {}
    pdict = {"tM0": [7, ], "tM1": [12, ], "tM2": [
        18, ], "d": [-3.0, -2.0, 0.0, 1.0, 2.0], }
    # pdict = {k: np.linspace(l, u, num=3) for k, l, u in plist}
    # pdict.update({k: list(range(l, u, 3)) for k, l, u in zip(tmK, tmL, tmU)})
    pgrid = ParameterGrid(pdict)
    cma = CMASleepWakeModel(N=48)
    for i, params in enumerate(pgrid):
        print(f"Iteration ({i:03d}/{len(pgrid)}) ...")
        tM = [params.pop(tmk) for tmk in tmK]
        params["tM"] = tM
        cma.update(**params)
        out = cma.run()
        fit_result_global.append([params, out])
    print('...done.')


@cli.command()
def run_doctests():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    cli()

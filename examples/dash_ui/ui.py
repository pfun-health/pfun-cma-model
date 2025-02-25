from typing import Any, Sequence

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from pathlib import Path
import pfun_path_helper as pph
pph.append_path(Path(__file__).parent.parent.parent)
from pfun_cma_model.engine.cma import (
    CMAModelParams,
    CMASleepWakeModel,
)

cma = CMASleepWakeModel()

app = dash.Dash(__name__)
app.scripts.config.serve_locally = True

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Button("Toggle Menu", id="toggle-button"),
                html.Div(
                    [
                        html.Label("d: Time zone offset"),
                        dcc.Slider(
                            id="d-slider",
                            min=-12.0,
                            max=14.0,
                            value=0.0,
                            marks={i: str(i) for i in range(-12, 15)},
                            step=0.1,
                            tooltip={"always_visible": True, "placement": "bottom"},
                            updatemode="drag",
                        ),
                        html.Label("taup: Photoperiod"),
                        dcc.Slider(
                            id="taup-slider",
                            min=0.5,
                            max=3.0,
                            value=1.0,
                            marks={i / 10: str(i / 10) for i in range(5, 31)},
                            step=0.1,
                            tooltip={"always_visible": True, "placement": "bottom"},
                            updatemode="drag",
                        ),
                        html.Br(),
                        html.Label("taug: Glucose response"),
                        dcc.Slider(
                            id="taug-slider",
                            min=0.1,
                            max=3.0,
                            value=1.0,
                            marks={i / 10: str(i / 10) for i in range(1, 31)},
                            step=0.1,
                            tooltip={"always_visible": True, "placement": "bottom"},
                            updatemode="drag",
                        ),
                        html.Br(),
                        html.Label("B: Bias"),
                        dcc.Slider(
                            id="B-slider",
                            min=0.0,
                            max=1.0,
                            value=0.05,
                            marks={i / 10: str(i / 10) for i in range(0, 11)},
                            step=0.05,
                            tooltip={"always_visible": True, "placement": "bottom"},
                            updatemode="drag",
                        ),
                        html.Br(),
                        html.Label("Cm: Cortisol sensitivity"),
                        dcc.Slider(
                            id="Cm-slider",
                            min=0.0,
                            max=2.0,
                            value=0.0,
                            marks={i: str(i) for i in range(0, 3)},
                            step=0.1,
                            tooltip={"always_visible": True, "placement": "bottom"},
                            updatemode="drag",
                        ),
                        html.Br(),
                        html.Label("toff: Circadian phase offset"),
                        dcc.Slider(
                            id="toff-slider",
                            min=-3.0,
                            max=3.0,
                            value=0.0,
                            marks={i: str(i) for i in range(-3, 4)},
                            step=0.1,
                            tooltip={"always_visible": True, "placement": "bottom"},
                            updatemode="drag",
                        ),
                        html.Br(),
                        html.Label("Choose Y-axis Output"),
                        dcc.Dropdown(
                            id="yaxis-column",
                            options=[
                                {"label": "G", "value": "G"},
                                {"label": "C", "value": "c"},
                                {"label": "M", "value": "m"},
                                {"label": "A", "value": "a"},
                            ],
                            value="G",
                        ),
                        html.Br(),
                        html.Label("Choose X-axis Output"),
                        dcc.Dropdown(
                            id="xaxis-column",
                            options=[
                                {"label": "G", "value": "G"},
                                {"label": "C", "value": "c"},
                                {"label": "M", "value": "m"},
                                {"label": "A", "value": "a"},
                                {"label": "t", "value": "t"},
                            ],
                            value="t",
                        ),
                        html.Button("Update Params", id="update-button"),
                    ],
                    id="sliders-menu",
                    style={"display": "block"},
                ),
            ],
            style={"width": "30%", "float": "left"},
        ),
        html.Div([dcc.Graph(id="cma-plot")], style={"width": "70%", "float": "left"}),
    ],
    style={"overflow": "auto", "clear": "both"},
)


colors = {
    "G": "rgb(255, 0, 0)",
    "c": "rgb(0, 255, 0)",
    "m": "rgb(0, 0, 255)",
    "a": "rgb(255, 255, 0)",
}


@app.callback(
    Output("cma-plot", "figure"),
    [
        Input("update-button", "n_clicks"),
        Input("d-slider", "value"),
        Input("taup-slider", "value"),
        Input("taug-slider", "value"),
        Input("B-slider", "value"),
        Input("Cm-slider", "value"),
        Input("toff-slider", "value"),
        Input("xaxis-column", "value"),
        Input("yaxis-column", "value"),
    ],
)
def update_params(n, d, taup, taug, B, Cm, toff, xaxis_column_name, yaxis_column_name):
    if n is None:
        # Avoids running on app initialization
        return dash.no_update

    params = {"d": d, "taup": taup, "taug": taug, "B": B, "Cm": Cm, "toff": toff}
    cma.update(params, inplace=True)
    df = cma.run()
    # Create the plotly figure
    figure = {
        "data": [
            {
                "x": df[xaxis_column_name],
                "y": df[yaxis_column_name],
                "type": "scatter",
                "mode": "markers",
                "fill": "tozeroy",  # This makes it a filled area chart
                "fillcolor": colors[yaxis_column_name],
                "name": f"{yaxis_column_name} vs. {xaxis_column_name}",
            }
        ],
        "layout": {
            "title": f"{yaxis_column_name} vs. {xaxis_column_name} Plot",
            "xaxis": {"title": xaxis_column_name},
            "yaxis": {"title": yaxis_column_name},
        },
    }
    return figure


if __name__ == "__main__":
    app.run_server(
        debug=True,
        host="0.0.0.0",
        port=8050,
        dev_tools_serve_dev_bundles=True
    )

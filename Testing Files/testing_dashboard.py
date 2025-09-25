from dash import Dash, dcc, html, Input, Output, ctx
import random

app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(id='my-dropdown', options=[{'label': f'Option {i}', 'value': i} for i in range(1, 4)], value=1),
    html.Div(id='selected-value'),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])


@app.callback(
    Output('my-dropdown', 'options'),
    Output('my-dropdown', 'value'),
    Output('selected-value', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('my-dropdown', 'value')
)
def update_dropdown_and_capture_selection(n_intervals, selected_value):
    trigger = ctx.triggered_id  # checks which input triggered callback

    # Generate new options
    new_options = [{'label': f'Item {i}', 'value': i} for i in range(1, random.randint(3, 6))]

    if trigger == 'interval-component':
        # Interval triggered → update dropdown with new options
        new_value = new_options[0]['value']
        display = f'Dropdown updated automatically to {new_value}'
        return new_options, new_value, display
    else:
        # User triggered → keep their selection
        display = f'You selected: {selected_value}'
        return new_options, selected_value, display


if __name__ == '__main__':
    app.run(debug=True)

'''
import dash
from dash import dcc, html, Input, Output, State

app = dash.Dash(__name__)

# --- Dropdown options ---
entry_logic_options = [{"label": "Entry A", "value": "A"}, {"label": "Entry B", "value": "B"}]
exit_logic_options = [{"label": "Exit X", "value": "X"}, {"label": "Exit Y", "value": "Y"}]
strike_price_options = [{"label": "17000", "value": "17000"}, {"label": "17100", "value": "17100"}]
right_list_options = [{"label": "call", "value": "call"}, {"label": "put", "value": "put"}]
qty_options = [{"label": "75", "value": "75"}, {"label": "150", "value": "150"}]

# --- Layout ---
app.layout = html.Div([
    html.H2("Independent Dashboard (Dropdowns + Toggle Buttons)", style={"textAlign": "center"}),

    # Dropdowns
    dcc.Dropdown(id="Entry_Logic", options=entry_logic_options, value="A"),
    dcc.Dropdown(id="Exit_Logic", options=exit_logic_options, value="X"),
    dcc.Dropdown(id="strike_price_list", options=strike_price_options, value="17000"),
    dcc.Dropdown(id="right_list", options=right_list_options, value="call"),
    dcc.Dropdown(id="qty_list", options=qty_options, value="75"),

    html.Br(),

    # Buttons (initially grey)
    html.Button("Block Entry (0)", id="Entry_Block-btn", n_clicks=0, style={"backgroundColor": "lightgrey"}),
    html.Button("Block Sq. Off (0)", id="Sq_off_Block-btn", n_clicks=0, style={"backgroundColor": "lightgrey"}),
    html.Button("Liquidate All (0)", id="Liquidate_All-btn", n_clicks=0, style={"backgroundColor": "lightgrey"}),
    html.Button("Shutdown (0)", id="Shutdown-btn", n_clicks=0, style={"backgroundColor": "lightgrey"}),
    html.Button("Place Order (0)", id="manual_order", n_clicks=0, style={"backgroundColor": "lightgrey"}),
])


# --- Callback for buttons ---
@app.callback(
    [
        Output("Entry_Block-btn", "children"),
        Output("Entry_Block-btn", "style"),

        Output("Sq_off_Block-btn", "children"),
        Output("Sq_off_Block-btn", "style"),

        Output("Liquidate_All-btn", "children"),
        Output("Liquidate_All-btn", "style"),

        Output("Shutdown-btn", "children"),
        Output("Shutdown-btn", "style"),

        Output("manual_order", "children"),
        Output("manual_order", "style"),
    ],
    [
        Input("Entry_Block-btn", "n_clicks"),
        Input("Sq_off_Block-btn", "n_clicks"),
        Input("Liquidate_All-btn", "n_clicks"),
        Input("Shutdown-btn", "n_clicks"),
        Input("manual_order", "n_clicks"),
    ]
)
def toggle_buttons(entry_clicks, sq_clicks, liq_clicks, shutdown_clicks, manual_clicks):
    """ Toggle button color and update text with clicks count """

    def style_button(name, clicks):
        # Toggle color: even clicks = grey, odd clicks = green
        color = "lightgreen" if clicks % 2 == 1 else "lightgrey"
        return f"{name} ({clicks})", {"backgroundColor": color, "padding": "10px", "margin": "5px"}

    return (
        *style_button("Block Entry", entry_clicks),
        *style_button("Block Sq. Off", sq_clicks),
        *style_button("Liquidate All", liq_clicks),
        *style_button("Shutdown", shutdown_clicks),
        *style_button("Place Order", manual_clicks),
    )


if __name__ == "__main__":
    app.run(debug=True)
'''
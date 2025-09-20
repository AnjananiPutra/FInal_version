import pandas as pd
from Lib.Lib_Create_Dashboard_v6_Dash import Dashboard
import dash
from dash import dcc, html, Input
from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html

def launch_dashboard():
    temp_df = pd.DataFrame(columns=['N =','uptrend count','downtrend count','VWAP height'])
    temp_df2= pd.DataFrame(columns=['current_time','Expiry_date','Tick_count','Buffer_count'])
    temp_df3 = pd.DataFrame(columns=['Session_P&L'])

    # Mock data generation
    def generate_mock_df(seed):
        np.random.seed(seed)
        x = pd.date_range(start='2023-01-01', periods=50, freq='T')
        price = np.cumsum(np.random.randn(50)) + 100 + seed * 5
        volume = np.random.randint(1000, 5000, size=50)
        return pd.DataFrame({'timestamp': x, 'price': price, 'volume': volume})

    # Create 5 mock dataframes
    dataframes = [generate_mock_df(i) for i in range(5)]

    app = dash.Dash(__name__)

    plot_count = 5
    checkbox_inputs = [Input(f'SP{i}_{opt}', 'value') for i in range(1, plot_count + 1) for opt in
                       ['call_buy', 'call_sell', 'put_buy', 'put_sell']]

    Entry_dropdown_option_df = Dashboard.create_dropdown_list("Entry Logic", 4)
    Exit_dropdown_option_df = Dashboard.create_dropdown_list("Exit Logic", 4)

    app.layout = html.Div([

                    html.Div([

                        # Left column: dropdowns and buttons
                        html.Div([
                            Dashboard.create_dash_label('Label 1', 'New NIFTY Index Data'),


                            Dashboard.create_dash_dropdown("Entry_Logic", Entry_dropdown_option_df, 0),
                            Dashboard.create_dash_dropdown("Exit_Logic", Exit_dropdown_option_df, 0),
                       ], style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'alignItems': 'flex-start',
                                    'gap': '10px',
                                    'marginRight': '40px'  # spacing between columns
                                }),

                        # Left column: dropdowns and buttons
                        html.Div([
                            Dashboard.create_dash_buttons('Entry_Block-btn', 'Entry Block'),
                            Dashboard.create_dash_buttons('Exit_Block-btn', 'Exit Block'),
                            Dashboard.create_dash_buttons('Liquidate_All-btn', 'Liquidate All'),
                            Dashboard.create_dash_buttons('Code_deactivate-btn', 'Main Program OFF'),
                        ], style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'alignItems': 'flex-start',
                            'gap': '10px',
                            'marginRight': '40px'  # spacing between columns
                        }),
                        html.Div([

                                    Dashboard.create_dash_table('table_NIFTY_Index', temp_df, "100%", 10),
                                        ], style={'display': 'inline-block', 'marginLeft': '10px', 'marginRight': '10px',
                                          'marginBottom': '5px'}),

                        html.Div([
                            Dashboard.create_dash_table('table_NIFTY_Stocks', temp_df2, "100%", 10),
                        ], style={'display': 'inline-block', 'marginLeft': '10px', 'marginRight': '10px',
                                  'marginBottom': '5px'}),

                        html.Div([
                            Dashboard.create_dash_table('table_NIFTY_Stocks', temp_df3, "100%", 10),
                        ], style={'display': 'inline-block', 'marginLeft': '10px', 'marginRight': '10px',
                                  'marginBottom': '5px'}),
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'row',
                        'alignItems': 'flex-start',
                        'marginBottom': '20px'
                    }),

                    html.Div([
                        Dashboard.create_dash_candlestick_chart('candlestick-chart1'),
                        Dashboard.create_dash_candlestick_chart('candlestick-chart2'),
                    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%'}),

                    html.Div([
                        dcc.Checklist(
                            options=[{'label': 'Freeze_Focus_List_Update', 'value': True}],
                            id='Freeze_Focus_List_Update',
                            value=[],
                            style={
                                'margin-bottom': '5px',
                                'fontSize': '15px',
                                'color': 'white',
                                'transform': 'scale(1.5)'
                            },
                            inputStyle={'margin-right': '10px'}
                        ),
                    ], style={
                        'display': 'flex',
                        'justifyContent': 'center',
                        'alignItems': 'center',
                        'width': '90%',
                        'marginTop': '15px',
                        'marginBottom': '15px',
                        'padding': '15px',
                        'border': '2px solid grey',
                        'backgroundColor': '#2C5F2D',
                        'borderRadius': '5px'
                    }),

                    #html.Div([
                    #    Dashboard.create_dual_line_plot_set(plot_count),
                    #], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%'}),
                    html.Div([
                              Dashboard.create_price_volume_chart("call_display",dataframes,"timestamp","price","volume"),
                              Dashboard.create_price_volume_chart("put_display", dataframes, "timestamp", "price", "volume")
                    ],style={'display': 'flex', 'flexDirection': 'row', 'width': '90%'}),

                    html.Div([
                        html.Div([
                            html.H2("Option Chain Tree", style={'textAlign': 'center'}),
                            Dashboard.create_dash_table('Option_Chain_Tree', temp_df, "100%", 10),
                        ], style={'flex': '1'}),

                        html.Div([
                            html.H2("NIFTY Stocks List", style={'textAlign': 'center'}),
                            Dashboard.create_dash_table('NIFTY_Stocks_List', temp_df, "100%", 10),
                        ], style={'flex': '1'}),
                    ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start',
                              'justifyContent': 'space-around', 'width': '100%', 'marginBottom': '20px'}),

                    html.Div([
                        Dashboard.create_interval('interval-component', 1)
                    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%'})
    ])

    app.run(debug=True, port=8059)

# Call this function to launch the dashboard
launch_dashboard()



'''
def create_price_volume_chart(chart_id, dataframes, x_col, y_col, volume_col):
    traces = []

    # Define distinct colors for each line/bar pair
    color_palette = [
        'rgba(0, 117, 44, 1)',     # Dark Emerald Green
        'rgba(255, 233, 0, 0.9)',  # Bright Yellow
        'rgba(0, 117, 179, 0.9)',  # Brilliant Blue
        'rgba(255, 6, 0, 0.8)',    # Bright Red
        'rgba(134, 134, 134, 1)'   # Medium Gray
    ]

    # Filter each DataFrame to keep only the last 10 minutes of data
    filtered_dataframes = []
    for df in dataframes:
        latest_time = df[x_col].max()
        time_threshold = latest_time - timedelta(minutes=10)
        df_filtered = df[df[x_col] >= time_threshold]
        filtered_dataframes.append(df_filtered)

    # Create traces for each filtered DataFrame
    for i, df in enumerate(filtered_dataframes):
        color = color_palette[i]

        # Price line trace
        traces.append(go.Scatter(
            x=df[x_col],                 # X-axis: timestamp
            y=df[y_col],                 # Y-axis: price
            mode='lines',                # Display as line chart
            name=f'Price Line {i+1}',    # Legend label
            line=dict(
                color=color,             # Line color
                width=3                  # Line thickness
            ),
            yaxis='y1',                  # Assign to primary Y-axis
            opacity=0.6                  # Line transparency
        ))

        # Volume bar trace
        traces.append(go.Bar(
            x=df[x_col],                 # X-axis: timestamp
            y=df[volume_col],            # Y-axis: volume
            name=f'Volume {i+1}',        # Legend label
            yaxis='y2',                  # Assign to secondary Y-axis
            opacity=0.6,                 # Bar transparency
            marker=dict(color=color),    # Bar color
            offsetgroup=f'group{i}'      # Grouping for overlay
        ))

    # Calculate max price and round up to next multiple of 50
    max_price = max(df[y_col].max() for df in filtered_dataframes)
    y_max = math.ceil(max_price / 50) * 50

    # Calculate max volume and add buffer
    max_volume = max(df[volume_col].max() for df in filtered_dataframes)

    layout = go.Layout(
        title=chart_id,  # Chart title

        # X-axis configuration
        xaxis=dict(
            title=x_col,               # X-axis label
            type='date',              # Treat x-axis as datetime
            tickformat='%H:%M',       # Format ticks as hour:minute
            dtick=60000,              # Tick spacing: 1 minute (60,000 ms)
            showgrid=True,            # Enable gridlines
            gridcolor='lightgrey',    # Gridline color
            gridwidth=0.5             # Gridline thickness
        ),

        # Primary Y-axis (price)
        yaxis=dict(
            title=y_col,              # Y-axis label
            range=[0, y_max],         # Y-axis range from 0 to rounded max
            tickmode='linear',        # Evenly spaced ticks
            tick0=0,                  # Start ticks from 0
            dtick=50,                 # Tick spacing: 50 units
            showticklabels=True,      # Show tick labels
            tickformat=',',           # Format numbers with commas
            showgrid=True,            # Enable gridlines
            gridcolor='lightgrey',    # Gridline color
            gridwidth=0.5             # Gridline thickness
        ),

        # Secondary Y-axis (volume)
        yaxis2=dict(
            title=volume_col,         # Y-axis label
            overlaying='y',           # Overlay on primary Y-axis
            side='right',             # Position on right side
            showgrid=True,            # Enable gridlines
            tickmode='linear',        # Evenly spaced ticks
            tick0=0,                  # Start ticks from 0
            dtick=20000,              # Tick spacing: 20,000 units
            tickformat=',',           # Format numbers with commas
            range=[0, max_volume + 20000],  # Y-axis range with buffer
            gridcolor='lightgrey',    # Gridline color
            gridwidth=0.5             # Gridline thickness
        ),

        # Legend configuration
        legend=dict(
            x=0.5,                    # Center horizontally
            y=-0.2,                   # Position below plot area
            xanchor='center',         # Anchor legend to center
            orientation='h'           # Horizontal layout
        ),

        # Layout margins
        margin=dict(
            l=40, r=40, t=40, b=120   # Extra bottom space for legend
        ),

        height=600,                  # Chart height in pixels
        barmode='overlay',           # Overlay bars on same x-axis
        plot_bgcolor='rgba(237, 237, 237, 1)',  # Plot area background
        paper_bgcolor='rgba(237, 237, 237, 1)'  # Entire figure background
    )

    return dcc.Graph(id=chart_id, figure=go.Figure(data=traces, layout=layout))
# Dash app setup
app = Dash(__name__)
app.layout = html.Div([
    create_price_volume_chart('Sample Price & Volume Chart', dataframes, 'timestamp', 'price', 'volume')
])

if __name__ == '__main__':
    app.run(debug=True, port=8051)
# Dash app setup

'''
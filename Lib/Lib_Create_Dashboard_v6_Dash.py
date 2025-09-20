#================================================================================
import os
import sys
import glob
import dash
import warnings
import pandas as pd
from time import sleep
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, time,timedelta
from dash import Dash,dcc, html, dash_table, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
import math
import pandas as pd
import dash
from dash import dcc, html, Input
from datetime import timedelta

#import dash_core_components as dcc

class Dashboard():
    
    def __init__(self):
        return


    @staticmethod
    def get_app_layout(nif_index_value_df,entry_logic_drop_down_list,exit_drop_down_layout_list,tick_stats,nif_vwap_df,p_and_l_df,focus_call_df_list,focus_put_df_list,opt_chain_disp_tree_df,nif_stocks_disp_tree_df):
        try:
           #Entire Page container layout
           layout = html.Div(
                            [
                                  #First top vertical container carrying headers
                                  html.Div(
                                  [

                                            # header 1 column 1 First Object containing NIFTY 50 Value Labels and dropdowns
                                            html.Div(
                                            [
                                                        Dashboard.create_dash_table('table_nif_value',nif_index_value_df, "100%", 10),
                                                        Dashboard.create_dash_dropdown("Entry_Logic", entry_logic_drop_down_list, 0),
                                                        Dashboard.create_dash_dropdown("Exit_Logic", exit_drop_down_layout_list, 0),

                                                    ], style={
                                                                'display': 'flex',              # Enables Flexbox layout for child elements
                                                                'flexDirection': 'column',      # Stacks child elements vertically (top to bottom)
                                                                'alignItems': 'flex-start',     # Aligns children to the start of the cross-axis (left edge)
                                                                'gap': '10px',                  # Adds 10px vertical spacing between stacked children
                                                                'marginRight': '20px'           # Adds 20px space to the right of this container (useful for column separation)

                                                              }
                                                    ),

                                            # Header 1:column 2 :It contains buttons  used to control program parameters
                                            html.Div(
                                            [
                                                        Dashboard.create_dash_buttons('Entry_Block-btn', 'Block Entry'),
                                                        Dashboard.create_dash_buttons('Sq_off_Block-btn', 'Hold all Trades'),
                                                        Dashboard.create_dash_buttons('Liquidate_All-btn', 'Liquidate All'),
                                                        Dashboard.create_dash_buttons('Shutdown-btn', 'Shutdown code'),
                                                    ], style={
                                                                'display': 'flex',              # Enables Flexbox layout for child elements
                                                                'flexDirection': 'column',      # Stacks child elements vertically (top to bottom)
                                                                'alignItems': 'flex-start',     # Aligns children to the start of the cross-axis (left edge)
                                                                'gap': '10px',                  # Adds 10px vertical spacing between stacked children
                                                                'marginRight': '20px'           # Adds 20px space to the right of this container (useful for column separation)

                                                              },
                                                    ),
                                            html.Div(
                                            [

                                                        Dashboard.create_dash_table('table_tick_stats', tick_stats, "100%", 10),
                                                    ], style={
                                                                'display': 'inline-block',     # Allows the element to sit inline with others, but still accept width/height styling
                                                                'marginLeft': '10px',          # Adds 10px spacing to the left of the element (separates it from its left neighbor)
                                                                'marginRight': '10px',         # Adds 10px spacing to the right of the element (separates it from its right neighbor)
                                                                'marginBottom': '5px'          # Adds 5px spacing below the element (useful for vertical stacking or padding)
                                                             }
                                                    ),

                                            html.Div(
                                            [
                                                        Dashboard.create_dash_table('table_NIFTY_vwap', nif_vwap_df, "100%", 10),
                                                    ], style={
                                                                'display': 'inline-block',     # Allows the element to sit inline with others, but still accept width/height styling
                                                                'marginLeft': '10px',          # Adds 10px spacing to the left of the element (separates it from its left neighbor)
                                                                'marginRight': '10px',         # Adds 10px spacing to the right of the element (separates it from its right neighbor)
                                                                'marginBottom': '5px'          # Adds 5px spacing below the element (useful for vertical stacking or padding)
                                                             },
                                                    ),

                                            # need to Put P&L summary here in df format
                                            html.Div(
                                            [
                                                        Dashboard.create_dash_table('table_P_and_L', p_and_l_df, "100%", 10),
                                                    ], style={
                                                                'display': 'inline-block',     # Allows the element to sit inline with others, but still accept width/height styling
                                                                'marginLeft': '10px',          # Adds 10px spacing to the left of the element (separates it from its left neighbor)
                                                                'marginRight': '10px',         # Adds 10px spacing to the right of the element (separates it from its right neighbor)
                                                                'marginBottom': '5px'          # Adds 5px spacing below the element (useful for vertical stacking or padding)
                                                             },
                                                    ),

                                            ], style={
                                                        'display': 'flex',              # Activates Flexbox layout, allowing child elements to be aligned and spaced dynamically
                                                        'flexDirection': 'row',         # Arranges child elements horizontally from left to right
                                                        'alignItems': 'flex-start',     # Aligns children to the top of the container (start of the cross-axis)
                                                        'marginBottom': '20px'          # Adds 20px spacing below this container, useful for separating stacked sections
                                                      }
                                            ),
                                  #Second vertical container storing NIFTY 50 historical performance and current live trends
                                  html.Div(
                                  [
                                                Dashboard.create_dash_candlestick_chart('candlestick-chart1'),
                                                Dashboard.create_dash_candlestick_chart('candlestick-chart2'),
                                           ], style={
                                                        'display': 'flex',              # Activates Flexbox layout, allowing child elements to be aligned and spaced dynamically
                                                        'flexDirection': 'row',         # Arranges child elements horizontally from left to right
                                                        'alignItems': 'flex-start',     # Aligns children to the top of the container (start of the cross-axis)
                                                        'marginBottom': '20px'          # Adds 20px spacing below this container, useful for separating stacked sections
                                                      },
                                            ),
                                  #Third Vertical Container storing line charts of call and Put data of closet 5 strike prices
                                  html.Div(
                                  [
                                                Dashboard.create_price_volume_chart("call_display", focus_call_df_list, "datetime", "close", "volume"),
                                                Dashboard.create_price_volume_chart("put_display", focus_put_df_list, "datetime", "close", "volume")
                                          ], style={
                                                        'display': 'flex',           # Enables Flexbox layout, allowing flexible alignment and spacing of child elements
                                                        'flexDirection': 'row',      # Arranges child elements in a horizontal line (left to right)
                                                        'width': '90%'               # Sets the container width to 90% of its parent, leaving some margin on either side
                                                    },

                                          ),
                                  #Fourth vertical container for display of Option chain tree and NIFTY 50 Stock dynamics
                                  html.Div(
                                  [
                                                html.Div(
                                                [
                                                           html.H2("Option Chain Tree", style={'textAlign': 'center'}),
                                                           Dashboard.create_dash_table('Option_Chain_Tree', opt_chain_disp_tree_df, "100%", 10),
                                                         ], style={'flex': '1'}),

                                                html.Div(
                                                [
                                                           html.H2("NIFTY Stocks List", style={'textAlign': 'center'}),
                                                           Dashboard.create_dash_table('NIFTY_Stocks_List', nif_stocks_disp_tree_df, "100%", 10),
                                                        ], style={'flex': '1'}),
                                          ], style={
                                                        'display': 'flex',               # Enables Flexbox layout, allowing flexible arrangement of child elements
                                                        'flexDirection': 'row',          # Aligns child elements horizontally from left to right
                                                        'alignItems': 'flex-start',      # Vertically aligns children to the top of the container
                                                        'justifyContent': 'space-around',# Distributes child elements evenly with space around them (left/right padding between items)
                                                        'width': '100%',                 # Makes the container span the full width of its parent
                                                        'marginBottom': '20px'           # Adds spacing below the container to separate it from the next section
                                                    },

                                            ),
                                  #Fifth Invisible container used to update entire page at regular interval
                                  html.Div(
                                  [
                                              Dashboard.create_interval('interval-component', 1)
                                          ], style={
                                                        'display': 'flex',           # Enables Flexbox layout, allowing child elements to be aligned and spaced dynamically
                                                        'flexDirection': 'row',      # Arranges child elements horizontally from left to right
                                                        'width': '90%'               # Sets the container width to 90% of its parent, leaving 10% margin for visual balance or responsiveness
                                                    },
                                          ),

                            ]
                           )

           return layout
        except:

            return None



    @staticmethod
    def create_checkbox_list(names_list, id_prefix="checkbox", default_checked_list=None):
            """
            Creates individual HTML checkboxes with unique IDs for each name, aligned vertically to the right of the chart.
        
            Parameters:
            -----------
            names_list : list of str :Names to be used as labels and suffixes for IDs.
            id_prefix : str           :Prefix for checkbox IDs. Final ID = f"{id_prefix}_{name}"
            default_checked_list : list of str or None : List of names to be checked by default.
        
            Returns:
            --------
            html.Div                A Dash Div containing all individual checkboxes stacked vertically.
            """
            if default_checked_list is None:
                default_checked_list = []
        
            checkboxes = []
        
            for name in names_list:
                checkbox_id = f"{id_prefix}_{name}"
                checked_value = [True] if name in default_checked_list else []
        
                checkboxes.append(
                    html.Div(
                        dcc.Checklist(
                            options=[{'label': name.replace('_', ' ').title(), 'value': True}],
                            id=checkbox_id,
                            value=checked_value,
                            style={'margin-bottom': '5px'}
                        ),
                        style={'marginBottom': '4px'}
                    )
                )
        
            return html.Div(
                checkboxes,
                style={
                    'width': '20%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'paddingLeft': '10px'
                }
            )

    @staticmethod
    def create_dual_line_plot(chart_id):
        
        
        return html.Div(
            dcc.Graph(
                id=chart_id,
                style={
                    'width': '100%',
                    'height': '300px',
                    'border': '1px solid #ccc'
                }
            ),
            style={
                'width': '75%',
                'display': 'inline-block',
                'verticalAlign': 'top'
            }
        )
    @staticmethod
    def create_line_plot(chart_id):


        return dcc.Graph(
                            id      =chart_id,
                            style   ={
                                        'width': '100%',
                                        'height': '95%',
                                        'border': '1px solid gray'
                                    }
                        )




    @staticmethod
    def create_dual_line_plot_set(count):
        
            children_components = []
        
            for i in range(1, count + 1):
                chart_id = f'L{i}'
                checkbox_id_prefix = f'SP{i}'
        
                # Create one chart + checkbox row
                row = html.Div(
                    [
                        Dashboard.create_dual_line_plot(chart_id),
                        Dashboard.create_checkbox_list(['call_buy', 'call_sell', 'put_buy', 'put_sell'], checkbox_id_prefix,None)
                    ],
                    style={
                        'display': 'flex',
                        'flexDirection': 'row',
                        'alignItems': 'flex-start',
                        'marginBottom': '20px',
                        'borderBottom': '1px dashed #ddd',
                        'paddingBottom': '10px'
                    }
                )
        
                children_components.append(row)
        
            return html.Div(
                id='chart-container',
                children=children_components,
                style={'width': '100%', 'padding': '10px'}
            )

    @staticmethod
    def create_dash_table(table_id,df,td_width,page_size):
        
        
        #1 convert all data to text which can be read by plotly and send to browser for display
        df  =   df.astype(str)
        
        table =   dash_table.DataTable(
                                           id       =   table_id,
                                           columns  =   [{"name": i, "id": i} for i in df.columns],
                                           data     =   df.to_dict('records'),
                             
                                           style_table = {
                                                              'overflowX': 'auto',      # Enable horizontal scrolling
                                                              'width': td_width         # Set width to 25% of the page,'100%'
                                                          },
                                           style_header= {
                                                            'fontWeight': 'bold',       # Make headers bold
                                                            'textAlign': 'center',      # Center-align header text
                                                         },
                                           style_cell  =  {
                                                            'minWidth': '100px',          # Set minimum width for cells
                                                            'maxWidth': '200px',          # Set maximum width for cells
                                                            'overflow': 'hidden',         # Hide content that overflows the cell
                                                            'textOverflow': 'ellipsis',   # Show ellipsis for truncated text
                                                            'textAlign': 'center',        # Center-align cell content 
                                                           },
                                           
                                           style_data_conditional=[
                                                                    {
                                                                        'if': {'row_index': 'odd'},
                                                                        'backgroundColor': 'rgb(248, 248, 248)'
                                                                    }
                                                                  ],
                                           
                                           page_size = page_size,  # Number of rows per page
                                           
                                        )
      
        return table

    @staticmethod
    def publish_table(df):
          
        #converting dataframe data to text which can be read and shown by plotly to browser
        
          df        =  df.astype(str)
          
          table     =  df.to_dict('records') 
          
          
          return table

    @staticmethod
    def create_interval(interval_id,refresh_rate):
        
        count   =   round(1000/refresh_rate,0)
        
        obj     =   dcc.Interval(
                                      id='interval-component',
                                      interval=1 * count,           # No. of times to be refreshed in second
                                      n_intervals=0
                                  )
        
        
        
        return obj

    @staticmethod
    def create_dropdown_list(Label_name,label_max_count):
        
          # Create a list of logic names
          logic_names = [f"{Label_name} {i}" for i in range(1, label_max_count+1)]
        
          # Create a list of corresponding values
          values = list(range(1, label_max_count+1))
        
          # Create a dictionary to store the data
          data = {'label': logic_names, 'value': values}
        
          # Create the DataFrame
          df = pd.DataFrame(data)
          
          return df

    @staticmethod
    def create_dash_dropdown(dropdown_id,dropdown_df,default_option):
      
      dropdown_records  =   dropdown_df.to_dict('records')
      
      default_value     =   dropdown_records[default_option]['value']
      
      drop_down         =   dcc.Dropdown(
                                              id        =   dropdown_id,                                             
                                              options   =   dropdown_records,                                                             
                                              value     =   default_value,                       # Default selected value                                             
                                              style     =   {   'marginRight'   : '5px',
                                                                'marginBottom'  : '20px',
                                                                'marginLeft'    : '5px',
                                                                'width'         : '200px',       # Set width
                                                                'height'        : '10px',        # Set height (if needed)
                                                                'fontSize'      : '12px'         # Font size of the options
                                                              }
                                          )
      
      return drop_down

    @staticmethod
    def create_dash_candlestick_chart(cs_chart_id):
      
      cs_chart =  dcc.Graph(id  =   cs_chart_id, style={'height': '95%','width': '95%','border': '1px solid gray'})  
      
      return cs_chart

    @staticmethod
    def create_dash_buttons(button_id,button_text):
        
        button =  html.Button(
                                  button_text, 
                                  id       =   button_id, 
                                  n_clicks =   0,
                                  style    =   {
                                                'marginLeft'    : '5px'}
                              )
        
        return button

    @staticmethod
    def create_dash_label(label_id,label_text):
        
        label =  html.Label(
                                label_text                    ,
                                id    =  label_id             ,
                                style =  {'border': '1px solid black', 'padding': '10px','marginLeft': '20px'}
                            ) 
        
        
        return label

    @staticmethod
    def get_chart_axis_range(relayout_data):
        
        #gets the range data if the user has made any selection
        range_data    =  relayout_data.get('xaxis.range', None) if relayout_data else None 
        
        
        return range_data
    
    @staticmethod
    def plot_nifty_50_live_chart(df_sec,df_min,secondary_df,x_axis_filtered = None,band_plot=None, autofit=False):

        '''
        Parameters
        ----------
        df_sec              : DataFrame with OHLC and datetime columns in seconds it will be plotted as line chart or candlestick chart
        df_min5             : DataFrame with OHLC and datetime columns in seconds it will be plotted as 30 min candlestick chart
        
        secondary_df        : It will plot VWAP line diagram on the chart
        areas               : List of dicts with 'y0', 'y1', 'color' keys for plotting support and resistance areas
        
        
        autofit             : Boolean to autofit layout (responsive size)
        x_range             : decided automatically as per the presented data of primary data
        x_axis_filtered     : filter range captured through range slider position
        primary_y_range     : decided automatically as per the presented data
        secondary_y_range   : decided automatically as per the presented data
        
        
        Returns
        -------
        fig : Plotly candlestick chart figure
        '''

        fig = go.Figure()

        # Check if df_sec and df_min5 are valid
        if df_sec is None or df_min is None or df_sec.empty or df_min.empty:
            print("[plot_nifty_50_live_chart] ❌ df_sec or df_min5 is empty or None. Returning blank figure.")
            fig.update_layout(title="⚠️ No data to display", paper_bgcolor='#fcf6f4', plot_bgcolor='#fcf6f4')
            return fig
        
        df_sec = df_sec[df_sec['datetime']>df_min['datetime'].iloc[-1]] if len(df_min) >0 else df_sec    
        combined_df = pd.concat([df_sec, df_min]).sort_values('datetime').reset_index(drop=True)
        
        # initalise variables
        raw_x_min = combined_df['datetime'].min()                 #finding the oldest data
        raw_x_max = combined_df['datetime'].max()                 #finding the latest data
        raw_x_max = raw_x_max + timedelta(seconds=30)                                   #adding time delta for better visiblity
        
        x_axis_range =  x_axis_filtered if x_axis_filtered else [raw_x_min, raw_x_max]
        
        
        
        last_price = df_sec['close'].iloc[-1]
        raw_y_min = combined_df['low'].min()
        raw_y_max = combined_df['high'].max()
        
        # Include last_price in range
        y_min = min(raw_y_min, last_price) - 100
        y_max = max(raw_y_max, last_price) + 100
        y_axis_range = [y_min, y_max]
        
        layout_kwargs = {
                            'margin'                    : dict(l=0, r=0, t=0, b=0),
                            'paper_bgcolor'             : '#fcf6f4',
                            'plot_bgcolor'              : '#fcf6f4',
                            'font'                      : dict(color='black'),
                            'title'                     : 'NIFTY 50 multi timeframe plot',
                            'yaxis_title'               : 'NIFTY 50 live price',
                            'xaxis_rangeslider_visible' : True,
                            'showlegend'                : False,
                            'hovermode'                :'x unified',
                            'dragmode'                 : 'zoom',
                        }



        

        # Step 3: 1-sec candlestick
        fig.add_trace(
                        go.Candlestick(
                                        x       =   combined_df['datetime'],
                                        open    =   combined_df['open'],
                                        high    =   combined_df['high'],
                                        low     =   combined_df['low'],
                                        close   =   combined_df['close'],
                                        name    =   "NIFTY 1 second plot"
                                    )
                      )

        # Plot bands if available
        shapes = []
  
                        
        if band_plot:
            #print(f"plot_count: {len(band_plot)}")
            #print(f"{band_plot}")
            
            for plot in band_plot:
                
                    shapes.append({
                                    'type'      : 'rect',
                                    'xref'      : 'paper',
                                    'yref'      : 'y',
                                    'x0'        :  0,
                                    'x1'        : 1,
                                    'y0'        : float(plot['low']),
                                    'y1'        : float(plot['high']),
                                    'fillcolor' : plot['color'],
                                    'layer'     : 'above',
                                    'line'      : {'width': 0},
                                })
                    
                    
        
                
            if shapes:  # Only add shapes if the list is not empty
                layout_kwargs['shapes'] = shapes
            else:
                print("No valid shapes to add.")


        # Step 5: X-axis range
        try:
            

      
                layout_kwargs['xaxis'] = dict(range         =   x_axis_range, 
                                              tickformat    =   '%H:%M',
                                              showspikes    =   True,
                                              spikemode     =   'across',
                                              spikesnap     =   'cursor',
                                              spikedash     =   'dot',
                                              spikethickness=   1,
                                              spikecolor    =   'grey',
                                              showticklabels=   True,
                                              )
                
                layout_kwargs['yaxis'] = dict(range         =   y_axis_range,
                                              showspikes    =   True,
                                              spikemode     =   'across', 
                                              spikesnap     =   'cursor', 
                                              spikedash     =   'dot',
                                              spikethickness=   1, 
                                              spikecolor    =   'grey',
                                              showticklabels=   True,
                                              )
                
        except Exception as e:
            
            print(f"[plot_nifty_50_live_chart] X-axis range error: {e}")
            print(f"[plot_nifty_50_live_chart] Y-axis range error: {e}")


        # Step 7: Last price line — only if df_sec has data
        try:
            
            
            
            fig.add_hline(
                            y                   =   last_price,
                            line_width          =   2.5,
                            line_dash           =   'dash',
                            line_color          =   'blue',
                            layer               =   'above',
                            annotation_text     =   f"last price: {last_price}",
                            annotation_position =   'top left',
                            annotation          =   dict(
                                                            font        =   dict(color="black", size=12),
                                                            bgcolor     =   "white",
                                                            bordercolor =   "blue"
                                                        )
                        )
        except IndexError:
            print("[plot_nifty_50_live_chart] Warning: Cannot add last price line, df_sec is empty") 

        # Optional: secondary y-axis for VWAP or other overlays
        # if secondary_df is not None and not secondary_df.empty:
        #     pass
        
        # Final layout update
        fig.update_layout(**layout_kwargs)

        return fig

    
    @staticmethod
    def plot_candlestick_chart(df, x_range1=None, areas=None, x_range=None, y_range=None,
                               df_secondary_data=None, autofit=False):
        '''
        Parameters
        ----------
        df            : DataFrame with OHLC and datetime columns
        x_range1        : Default x-axis range
        areas         : List of dicts with 'y0', 'y1', 'color' keys
        x_range       : Optional x-axis range
        y_range       : Optional y-axis range dict(domain=[0.3, 1])
        secondary_data: Optional dict to plot a secondary Y-axis line
                        Format: {
                            "y": series or list,
                            "name": "Series Name",
                            "line_color": "blue"
                        }
        autofit       : Boolean to autofit layout (responsive size)
        'legend': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        
        Returns
        -------
        fig : Plotly candlestick chart figure
        '''

        # Initialize an empty Plotly figure object
        fig = go.Figure()
    
        # Primary candlestick plot
        fig.add_trace(
                        go.Candlestick(
                                            x=df['datetime'],                        # X-axis: datetime values
                                            open=df['open'],                         # Opening prices
                                            high=df['high'],                         # High prices
                                            low=df['low'],                           # Low prices
                                            close=df['close'],                       # Closing prices
                                            increasing_line_color='green',           # Color for bullish candles
                                            decreasing_line_color='rgb(139, 0, 0)',  # Color for bearish candles
                                            name="OHLC"                              # Legend name
                                        )
                    )
    
            # Area highlight shapes
        shapes = []                 # Initialize list to hold area highlight shapes
        
        if areas:
            for area in areas:
                shapes.append(
                                {
                                'type': 'rect',                 # Shape type: rectangle
                                'xref': 'paper',                # X-axis reference: full width of plot
                                'yref': 'y',                    # Y-axis reference: price axis
                                'x0': 0,                        # Start at left edge
                                'x1': 1,                        # End at right edge
                                'y0': area['low'],              # Bottom of rectangle
                                'y1': area['high'],             # Top of rectangle
                                'fillcolor': area['color'],     # Fill color for the area
                                'layer': 'below',               # Draw below chart elements
                                'line': {'width': 0},           # No borderline
                                }
                                )
    
        # Define base layout properties for the chart
        layout_kwargs = {
                            'margin': dict(l=0, r=0, t=0, b=0), # Remove all margins
                            'paper_bgcolor': '#fcf6f4',         # Background color of the figure
                            'plot_bgcolor': '#fcf6f4',          # Background color of the plot area
                            'font': dict(color='black'),        # Font color
                            'shapes': shapes,                   # Add any highlight shapes

                        }
    
        #IF autofit is not enabled then use simple plotting for primary y-axis else enable autoscale of y-axis
        if not autofit:
    
                      layout_kwargs['yaxis'] = dict(
                                                        overlaying='y',     # Overlay on default Y-axis
                                                        side='left',        # Position on left side
                                                        showgrid=False      # Hide grid lines
                                                    )
    
    
        else:
    
                    layout_kwargs['yaxis'] = dict(
                                                        overlaying='y',     # Overlay on default Y-axis
                                                        side='left',        # Position on left side
                                                        showgrid=False,     # Hide grid lines
                                                        autorange=True,     # Enable auto-scaling
                                                        fixedrange=False    # Allow zooming/panning
                                                    )
        # Secondary Y-axis plot (optional)
        if df_secondary_data:
    
            fig.add_trace(
                          go.Scatter(
                                      x=df_secondary_data['datetime'],
                                      y=df_secondary_data['y'],
                                      name= 'VWAP_50',
                                      line=dict(color= 'blue'),
                                      yaxis='y2'
                                    )
                          )
            #IF autofit is not enabled then use simple plotting for secondary y-axis else enable autoscale of y-axis
    
            if not autofit:
    
                      layout_kwargs['yaxis2'] = dict(
                                                        overlaying='y',
                                                        side='right',
                                                        showgrid=False
                                                    )
    
    
            else:
    
                    layout_kwargs['yaxis2'] = dict(
                                                        overlaying='y',
                                                        side='right',
                                                        showgrid=False,
                                                        autorange=True,
                                                        fixedrange=False
                                                    )
    
    
    
    
    
    
    
        # Axis range configuration
        xaxis_range = x_range if x_range else x_range1
    
        if xaxis_range:
            layout_kwargs['xaxis'] = dict(range=xaxis_range)
            
            
        if y_range:
            layout_kwargs['yaxis'] = dict(range=y_range)
    
      
    
        fig.update_layout(**layout_kwargs)
    
        return fig

    @staticmethod
    def plot_dual_chart_object_list(object_list):
        '''
        Parameters
        ----------
        object_list : This this contains objects which has call and put df in it

        Returns 
        -------
        list of chart plots that needs to be plotted

        '''
        plot_obj_list   =   []
        
        for each_SP_object in object_list:
            
            call_df         =   each_SP_object.call_df_sec.copy()
            put_df          =   each_SP_object.call_df_sec.copy()
            Strike_price    =   each_SP_object.strike_price
            call_cutoff_time=   (call_df['datetime'].iloc[-1] - timedelta(minutes=5))
            put_cutoff_time =   (put_df['datetime'].iloc[-1] - timedelta(minutes=5))
            
            call_df         =   call_df[call_df['datetime'] > call_cutoff_time]
            put_df          =   put_df[put_df['datetime'] >  put_cutoff_time]
            
            chart_plot = Dashboard.plot_dual_chart(call_df, put_df, f"{Strike_price}", 'green', 'red')
            
            plot_obj_list.append(chart_plot)
            
        return plot_obj_list

    @staticmethod
    def plot_dual_chart(call_df,put_df,title,call_color,put_color): 
        
        
            fig = go.Figure()
            
            call_cutoff_time=   (call_df['datetime'].iloc[-1] - timedelta(minutes=5))
            put_cutoff_time =   (put_df['datetime'].iloc[-1] - timedelta(minutes=5))
            
            call_df         =   call_df[call_df['datetime'] > call_cutoff_time]
            put_df          =   put_df[put_df['datetime'] >  put_cutoff_time]
            
            fig.add_trace(go.Scatter(x=call_df['datetime'], y=call_df['close'], mode='lines', name="call price", line=dict(color=call_color,width=2)))
            fig.add_trace(go.Scatter(x=put_df['datetime'], y=put_df['close'], mode='lines', name="put price", line=dict(color=put_color,width=2)))

            fig.add_trace(go.Bar(x=call_df['datetime'], y=call_df['volume'], name='call volume', yaxis='y2', marker_color='blue',opacity=0.9, width=500))
            fig.add_trace(go.Bar(x=put_df['datetime'], y=put_df['volume'], name='put volume', yaxis='y2', marker_color='yellow',opacity=0.9, width=500))
            
            
            fig.add_shape(
                                type='line',
                                xref='x',
                                yref='y2',
                                x0=min(call_df['datetime'].min(), put_df['datetime'].min()),
                                x1=max(call_df['datetime'].max(), put_df['datetime'].max()),
                                y0=100000,
                                y1=100000,
                                line=dict(color='gray', width=1, dash='dot'),
                                name="CF Volume Line"
                            )
            
            
            call_price  =   call_df.iloc[-1]["close"] if len(call_df)> 0 else 0
            put_price   =   put_df.iloc[-1]["close"]  if len(put_df) > 0 else 0
           
            fig.update_layout(
                                title       =   f'{title} call:{call_price},put:{put_price}',
                                yaxis       =   dict(title      = 'Price'),
                                yaxis2      =   dict(title      = 'Volume', 
                                                     overlaying ='y',
                                                     side       ='right',
                                                     range      =[10000, 300000],  # Set visible range
                                                     tickvals   =[10000, 50000, 100000, 200000, 300000],  # Exact tick labels
                                                     tickfont   =dict(size=10)),
                                xaxis       =   dict(title='Datetime'),
                                showlegend  =   True,
                                title_x     =   0.5,
                                legend      =   dict(
                                                title=dict(text='Data Type'),
                                                orientation='h',
                                                yanchor='bottom',
                                                y=1.05,        # slightly above the plot area
                                                xanchor='center',
                                                x=0.5          # center aligned with title
                                            )
                               ) 
            return fig
        
        
        
    @staticmethod
    def create_price_volume_chart(chart_id, dataframes, x_col, y_col, volume_col):
        traces = []

        # Define distinct colors for each line/bar pair
        color_palette = [
            'rgba(0, 117, 44, 1)',          # Dark Emerald Green
            'rgba(255, 233, 0, 0.9)',       # Bright Yellow
            'rgba(0, 117, 179, 0.9)',       # Brilliant Blue
            'rgba(255, 6, 0, 0.8)',         # Bright Red
            'rgba(134, 134, 134, 1)'        # Medium Gray
        ]

        # Filter each DataFrame to keep only the last 10 minutes of data
        filtered_dataframes = []
        try:
            if dataframes.empty:
                print("⚠️ Skipping: DataFrame is empty")
                return None  # or just return

        except:
            if dataframes is None:
                print("⚠️ Skipping: No Dataframe for plotting")
                return None  # or just return


        # rest of your processing code here
        for df in dataframes:
            latest_time    = df[x_col].max()
            time_threshold = latest_time - timedelta(minutes=10)
            df_filtered    = df[df[x_col] >= time_threshold]
            filtered_dataframes.append(df_filtered)

        # Create traces for each filtered DataFrame
        for i, df in enumerate(filtered_dataframes):
            color = color_palette[i]

            # Price line trace
            traces.append(
                go.Scatter(
                    x       =   df[x_col],              # X-axis: timestamp
                    y       =   df[y_col],              # Y-axis: price
                    mode    =   'lines',                # Display as line chart
                    name    =   f'Price Line {i + 1}',  # Legend label
                    line    =   dict(
                                    color=color,        # Line color
                                    width=3             # Line thickness
                                    ),
                    yaxis   =   'y1',                   # Assign to primary Y-axis
                    opacity =   0.6                     # Line transparency
                ))

            # Volume bar trace
            traces.append(
                go.Bar(
                        x           =df[x_col],          # X-axis: timestamp
                        y           =df[volume_col],     # Y-axis: volume
                        name        =f'Volume {i + 1}',  # Legend label
                        yaxis       ='y2',               # Assign to secondary Y-axis
                        opacity     =0.6,                # Bar transparency
                        marker      =dict(color=color),  # Bar color
                        offsetgroup =f'group{i}'         # Grouping for overlay
                    ))

        # Calculate max price and round up to next multiple of 50
        max_price = max(df[y_col].max() for df in filtered_dataframes)
        y_max     = math.ceil(max_price / 50) * 50

        # Calculate max volume and add buffer
        max_volume = max(df[volume_col].max() for df in filtered_dataframes)

        layout = go.Layout(
                    title   =chart_id,                      # Chart title

                    # X-axis configuration
                    xaxis   =dict(
                                title       =x_col,         # X-axis label
                                type        ='date',        # Treat x-axis as datetime
                                tickformat  ='%H:%M',       # Format ticks as hour:minute
                                dtick       =60000,         # Tick spacing: 1 minute (60,000 ms)
                                showgrid    =True,          # Enable gridlines
                                gridcolor   ='lightgrey',   # Gridline color
                                gridwidth   =0.5            # Gridline thickness
                            ),

            # Primary Y-axis (price)
            yaxis=dict(
                        title           =y_col,         # Y-axis label
                        range           =[0, y_max],    # Y-axis range from 0 to rounded max
                        tickmode        ='linear',      # Evenly spaced ticks
                        tick0           =0,             # Start ticks from 0
                        dtick           =50,            # Tick spacing: 50 units
                        showticklabels  =True,          # Show tick labels
                        tickformat      =',',           # Format numbers with commas
                        showgrid        =True,          # Enable gridlines
                        gridcolor       ='lightgrey',   # Gridline color
                        gridwidth       =0.5            # Gridline thickness
                    ),

            # Secondary Y-axis (volume)
            yaxis2=dict(
                        title       =volume_col,    # Y-axis label
                        overlaying  ='y',           # Overlay on primary Y-axis
                        side        ='right',       # Position on right side
                        showgrid    =True,          # Enable gridlines
                        tickmode    ='linear',      # Evenly spaced ticks
                        tick0       =0,             # Start ticks from 0
                        dtick       =20000,         # Tick spacing: 20,000 units
                        tickformat  =',',           # Format numbers with commas
                        range       =[0, max_volume + 20000],  # Y-axis range with buffer
                        gridcolor   ='lightgrey',   # Gridline color
                        gridwidth   =0.5            # Gridline thickness
                    ),

            # Legend configuration
            legend=dict(
                        x           =0.5,           # Center horizontally
                        y           =-0.2,          # Position below plot area
                        xanchor     ='center',      # Anchor legend to center
                        orientation ='h'            # Horizontal layout
                    ),

            # Layout margins
            margin  =dict(
                            l=40, r=40, t=40, b=120  # Extra bottom space for legend
                        ),

            height       =600,                       # Chart height in pixels
            barmode      ='overlay',                 # Overlay bars on same x-axis
            plot_bgcolor ='rgba(237, 237, 237, 1)',  # Plot area background
            paper_bgcolor='rgba(237, 237, 237, 1)'   # Entire figure background
        )

        return dcc.Graph(id     =chart_id,
                         figure =go.Figure(data=traces, layout=layout),
                         style  ={'height': '95%', 'width': '95%', 'border': '1px solid gray'}  # 👈 Chart container styling
                        )
    @staticmethod
    def update_t_count_stats(Time,T_count,t_pipeline):

        temp = pd.DataFrame(columns=['Particulars','Value'])

        try:
            new_rows = pd.DataFrame([
                                        {'Particulars': 'Timestamp', 'Value': Time.strftime('%dth %b, %Y %H:%M:%S')},
                                        {'Particulars': 'Total Count', 'Value': T_count},
                                        {'Particulars': 'Pipeline Steps', 'Value': ', '.join(map(str, t_pipeline))}
                                    ])

            temp    = pd.concat([temp, new_rows], ignore_index=True)

            return temp

        except Exception as e:

            print(f"Error updating t_count stats: {e}")
            return temp

    @staticmethod
    def compute_profit_and_loss(order_book_df):

        try:


            pass


        except Exception as e:


            return

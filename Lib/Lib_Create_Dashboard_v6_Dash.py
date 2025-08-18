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

#import dash_core_components as dcc

class Dashboard():
    
    def __init__(self):
        return

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
    
        fig = go.Figure()
    
        # Primary candlestick plot
        fig.add_trace(
                        go.Candlestick(
                                            x=df['datetime'],
                                            open=df['open'],
                                            high=df['high'],
                                            low=df['low'],
                                            close=df['close'],
                                            increasing_line_color='green',
                                            decreasing_line_color='rgb(139, 0, 0)',
                                            name="OHLC"
                                        )
                    )
    
            # Area highlight shapes
        shapes = []
        
        if areas:
            for area in areas:
                shapes.append(
                                {
                                'type': 'rect',
                                'xref': 'paper',
                                'yref': 'y',
                                'x0': 0,
                                'x1': 1,
                                'y0': area['low'],
                                'y1': area['high'],
                                'fillcolor': area['color'],
                                'layer': 'below',
                                'line': {'width': 0},
                                }
                                )
    
        # Base layout
        layout_kwargs = {
                            'margin': dict(l=0, r=0, t=0, b=0),
                            'paper_bgcolor': '#fcf6f4',
                            'plot_bgcolor': '#fcf6f4',
                            'font': dict(color='black'),
                            'shapes': shapes,
                        }
    
        #IF autofit is not enabled then use simple plotting for primary y-axis else enable autoscale of y-axis
        if not autofit:
    
                      layout_kwargs['yaxis'] = dict(
                                                        overlaying='y',
                                                        side='left',
                                                        showgrid=False
                                                    )
    
    
        else:
    
                    layout_kwargs['yaxis'] = dict(
                                                        overlaying='y',
                                                        side='left',
                                                        showgrid=False,
                                                        autorange=True,
                                                        fixedrange=False
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
        
        
        
        
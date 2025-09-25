import datetime
from datetime import datetime
import dash
from dash import html
import pandas as pd
from Lib.Lib_Create_Dashboard_v6_Dash import Dashboard

# Step 1: Create dummy DataFrames for testing
dummy_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
dummy_dropdown_df = pd.DataFrame({'label': ['Logic A', 'Logic B'], 'value': ['A', 'B']})
strike_price_list = pd.DataFrame({'label': ['17500', '17600', '17700'], 'value': ['17500', '17600', '17700']})

result_df       =  pd.DataFrame([{'Current Time': datetime.now().strftime("%d-%b-%Y %H:%M:%S"),  # e.g., 25-Sep-2025 15:32:19
                                  'T_Count': '1,222',
                                  'Queue Length': '0'
                                 }
                                ])

print(result_df)

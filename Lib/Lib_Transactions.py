import os
import sys
import glob
import math
import urllib
import zipfile
import openpyxl
import warnings
import traceback
import numpy as np
import pandas as pd
from time import sleep
from functools import reduce
from pandas import ExcelWriter
from datetime import datetime, date, time,timedelta
from Lib.Lib_Global_Functions_V3 import Global_function
""

class Transactions():
    
    Ac_bal:float    =   0.0


        
    def __init__(self,OC_object ,right,T_count,quantity):
        return

    @staticmethod
    def display_P_and_L(trade_list):
        try:
            
           trade_df     = pd.DataFrame(trade_list["Success"])
           
           if len(trade_df)>0:

               converter_list =   {
                                    'brokerage_amount' : "numeric",
                                    'strike_price'     : "numeric",
                                    'quantity'         : "numeric",
                                    'average_cost'     : "numeric",
                                    'total_taxes'      : "numeric",
                                    'trade_date'       : "datetime",
                                    'expiry_date'      : "datetime" ,
                                }
               trade_df     = Global_function.optimise_df_data_format(trade_df,converter_list)
               brokerage    = trade_df['brokerage_amount'].sum()
               trade_val    = Transactions.trade_value(trade_df)  
               taxes        = trade_df['total_taxes'].sum()  
               print('======================================================================')
               print(f'FROM ::{trade_df["trade_date"].min().strftime("%d %B %Y")}       TO ::{trade_df["trade_date"].max().strftime("%d %B %Y")}')
               print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
               print(f'total no of transactions :: {len(trade_df)}')
               print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
               print(f'Profit/Loss              :: {trade_val-brokerage:,.2f}')
               print(f'Brokerage Paid           :: {brokerage:,.2f}')
               print(f'Taxes Paid               :: {taxes:,.2f}')
               print('======================================================================')
           
               
           else:
               print('======================================================================')
               
               print(f'total no of transactions CHK POINT :: {len(trade_df)}')
               print(f'      No Trades executed today       ')
               
               print('======================================================================')
               
        except Exception as e:
            
            print(f'Error while extracting information \nError Description::{e}')
            txt     =   f'{traceback.format_exc()}'
            print(txt)

    @staticmethod
    def convert_data_format(df):
        
        # Convert relevant columns to numeric, handling errors
        for col in ['brokerage_amount', 'strike_price', 'quantity', 'average_cost']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns to datetime objects
        for col in ['trade_date', 'expiry_date']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df

    @staticmethod
    def trade_value(df):
        """Calculates a custom sumproduct based on the 'action' column.

        Args:
            df: The input DataFrame.

        Returns:
            The calculated sumproduct.
        """
        sum_product = 0
        for index, row in df.iterrows():
            
            
          if row['action'] == 'Sell':
              
            # Check for NaN values before multiplication
            if not pd.isna(row['average_cost']) and not pd.isna(row['quantity']):
                sum_product += row['average_cost'] * row['quantity']
                
                
          elif row['action'] == 'Buy':
              
            # Check for NaN values before multiplication
            if not pd.isna(row['average_cost']) and not pd.isna(row['quantity']):
                sum_product -= row['average_cost'] * row['quantity']
                
                
        return sum_product

#================================================================================
#      Importing libraries for the code
#================================================================================
import os
import sys
import glob
import json
import traceback
import threading
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt
from datetime import datetime, date, time,timedelta
import pandas as pd
import time

from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from io import StringIO

#================================================================================
#         storage location of important folders
#================================================================================

KPI_folder              = "E:/Algo_Trading_V3/Input Files/KPI/"
existing_pre_market_data= "E:/Algo_Trading_V3/Input Files/Instruments/pre_open_market_data.csv"
NSE_Stock_script        = "E:/Algo_Trading_V3/Input Files/Instruments/NSEScripMaster.txt"
NIF_50_Stock_list       = "E:/Algo_Trading_V3/Input Files/Instruments/NIFTY50_Stock_List.csv"


class Global_function():
    #used for data conversation

    
    def __init__(self,stock_name1):
        
        return
    
    @staticmethod  
    def read_csv(location,file_name):
           data = 0
           
           print(f"{file_name}")
           
           return data
       
    @staticmethod   
    def read_txt(location,file_name):
           data = 0
           
           return data
    
    

    @staticmethod
    def update_SP_flags(SP_obj,call_buy_flag,call_sell_flag,put_buy_flag,put_sell_flag):
        
        call_buy_flag       = call_buy_flag[0]   if call_buy_flag  else False
        call_sell_flag      = call_sell_flag[0]  if call_sell_flag else False
        put_buy_flag        = put_buy_flag[0]    if put_buy_flag   else False
        put_sell_flag       = put_sell_flag[0]   if put_sell_flag  else False
        
        SP_obj.call_buy_flag    = call_buy_flag
        SP_obj.call_sell_flag   = call_sell_flag
        SP_obj.put_buy_flag     = put_buy_flag
        SP_obj.put_sell_flag    = put_sell_flag
        
        txt = f"Updated Status {SP_obj.strike_price} call_buy_flag {SP_obj.call_buy_flag} || call_sell_flag {SP_obj.call_sell_flag} || put_buy_flag {SP_obj.put_buy_flag} || put_sell_flag {SP_obj.put_sell_flag}" 
        #print(txt)
        
        return txt
    
    @staticmethod      
    def convert_to_datetime(variable):
          """
          Converts a string to a datetime object if it's a string,
          otherwise returns the variable as is.
        
          Args:
            variable: The variable to convert.
        
          Returns:
            A datetime object if the input was a string, otherwise the original variable.
          """
          if isinstance(variable, str):
            # Assuming the string is in a common format like 'YYYY-MM-DD HH:MM:SS'
            # You might need to adjust the format string based on your actual data
            try:
              return datetime.strptime(variable, '%Y-%m-%d %H:%M:%S')
            except ValueError:
              # Handle cases where the string format is different
              print(f"Warning: Could not parse string '{variable}' into datetime. Returning original variable.")
              return variable
          elif isinstance(variable, datetime):
            return variable
          else:
            print(f"Warning: Variable is not a string or datetime object. Returning original variable.")
            return variable


    @staticmethod  
    def read_excel(location,file_name,sheet_name,col_range,header_offset,skip_rows):
        '''
        header      = 1   : This tells read_excel to use the second row (index 1) as the header row for your DataFrame.
        skiprows    = 1   : This tells read_excel to skip the first row (index 0) entirely when reading the data.
        usecols     ='A:D': This argument specifies that you only want to read columns from A to D (inclusive). 
                        You can use this range notation for consecutive columns.
        '''
        data = pd.read_excel(location+file_name+".xlsx",sheet_name = sheet_name, usecols=col_range, header=header_offset, skiprows=skip_rows)
        
        
           
            
        
        return data
    
    @staticmethod
    def update_dataframe(df1,df2):
            
               if len(df1) >0:
                   
                   if not df1.iloc[-1, 1:].equals(df2.iloc[-1, 1:]):
                       
                       df1    =   pd.concat([df1, df2], ignore_index=True)
                   
               else:
                       df1    =   pd.concat([df1, df2], ignore_index=True)
               
               
               
               
               return df1

    @staticmethod
    def load_ticks_with_calc_in_df(df,tick_df,MVA_period,EMV_period,VWAP,index_weight):

    
           data_df =   pd.concat([df, tick_df], ignore_index=True, sort=False)       
           # tick['Avg_vol'].iloc[-1]           =   data_df['volume'].rolling(MVA_period, min_periods=1).mean().iloc[-1]
           # tick['volume_factor'].iloc[-1]     =   df['volume'].iloc[-1]  /   df['Avg_vol'].iloc[-1]
           # tick['MVA_n'].iloc[-1]             =   df['close'].rolling(window= MVA_period, min_periods=1).mean().iloc[-1]
           # tick['EMV_n'].iloc[-1]             =   df['close'].ewm(span= EMV_period, adjust=False).mean().iloc[-1]
           # tick['VWAP'].iloc[-1]              =   VWAP
           # tick['VWAP_height'].iloc[-1]       =   tick['close'].iloc[-1] - tick['VWAP'].iloc[-1]
           # tick['Adj_VWAP_H'].iloc[-1]        =   tick['VWAP_height'].iloc[-1] * index_weight  
           
           df_len = len(data_df)
           
           if df_len > 0:
               

   
               # Calculate rolling and EWM values on the combined DataFrame
               tick_df.loc[:, 'Avg_vol']   = data_df['volume'].rolling(MVA_period, min_periods=1).mean().iloc[-1]
               tick_df.loc[:, 'MVA_n']     = data_df['close'].rolling(window=MVA_period, min_periods=1).mean().iloc[-1]
               tick_df.loc[:, 'EMV_n']     = data_df['close'].ewm(span=EMV_period, adjust=False).mean().iloc[-1]
   
           else:
               
               # Handle the case when data_df is empty
               tick_df.loc[:,'Avg_vol']   = tick_df['volume']
               tick_df.loc[:,'MVA_n']     = tick_df['close']
               tick_df.loc[:,'EMV_n']     = tick_df['close']
   
           # Calculate dependent values
           tick_df.loc[:, 'volume_factor'] = tick_df['volume'].iloc[-1] / tick_df['Avg_vol'].iloc[-1] if tick_df['Avg_vol'].iloc[-1] != 0 else np.nan
           tick_df.loc[:, 'VWAP']          = VWAP
           tick_df.loc[:,'VWAP_height']      = tick_df['close'].iloc[-1] - tick_df['VWAP'].iloc[-1]
           tick_df.loc[:,'Adj_VWAP_H']       = tick_df['VWAP_height'].iloc[-1] * index_weight  
           
           return tick_df

        
    @staticmethod       
    def find_band_levels(ohlc,band_type):
       
       try:
               '''
                   Nomenclature
                   S = Support level getting evaluated
                   S_n = list of support levels identified by above code
                   L = Candle Low
                   O = Candle Open
                   C = Candle Close
                   H = Candle High
                   Points which validate an strength of the support level getting evaluated:
                   1) O is higher than S and C within the tolerance level of S(example +-5 points) and L below S
                   2) O is Lower than S and H is Higher than S and C is within tolerance level
                   of S
                   3) O and C both are within the tolerance range of S(example +- 5 points)
                   Steps to go ahead
                   Support band
                   Assign the strength to band based on above algo
                   Prepare a rating matrix for the support levels/Bands
               '''
               
               ohlc['day_swing']  = ohlc['high'] - ohlc['low']
               max_day_delta    = ohlc['day_swing'].max()

               section         = '03.06.04' 
               period          = 7
               dt_offset       = 0                                         
               start_date      = ohlc['datetime'].iloc[0]
               end_date        = ohlc['datetime'].iloc[-1]
               data_days       = (end_date - start_date).days
               band_level_list = []
               
               section   =  '03.06.02' 
               current_market_level = ohlc['close'].iloc[-1]

               filter_market_level = (current_market_level + max_day_delta) if band_type == 'support' else (current_market_level - max_day_delta)
               df_filtered         = ohlc[ohlc['close'] < filter_market_level]  if band_type == 'support' else ohlc[ohlc['close'] > filter_market_level]
               
               section   =  '03.06.05' 
               for date1 in  pd.date_range(start_date, end_date,freq='7D') :
                   
                   str_dt    =   date1
                   end_dt    =   date1  +   timedelta(days=7)
                   
                   section   =  '03.06.06' 
                   temp = df_filtered[(df_filtered['datetime'] > str_dt) &   (df_filtered['datetime'] <= end_dt)]
                   band_level_list.append(temp['close'].min()) if band_type == 'support' else band_level_list.append(temp['close'].max())

                   
               section   =  '03.06.07'     
               #removes any Nan values in the list and keeps only valid entry    
               band_level_list = np.array(band_level_list)[~np.isnan(band_level_list)].tolist()    
               
               section   =  '03.06.08' 
               #filtering values which are not in the desired range
               band_level_list =  [x for x in band_level_list if (current_market_level  -   max_day_delta) < x < (current_market_level     +   max_day_delta)]    
               
               
               return band_level_list
           
       except Exception as e:
               
           return [] 
     
    @staticmethod
    def datetime_iso8601(dt: datetime) -> str:
        
        """Return dt in Breeze’s ‘YYYY‑MM‑DDTHH:MM:SS.000Z’ format."""
        fmt = "%Y-%m-%dT%H:%M:%S.000Z"
        return dt.strftime(fmt)    
     
        
    @staticmethod
    def create_folder(base_folder_loc,new_folder_name):

        try:
            section             =   1
            destination_folder  =   os.path.join(base_folder_loc,new_folder_name)
            os.makedirs(destination_folder,exist_ok=True)
            print(f"folder created succesfully in {destination_folder =}.")
            section             =   2
            destination_folder  =   destination_folder +"/"

        except FileExistsError:
            print(f"Directory '{destination_folder=}' already exists.")
            destination_folder  =   destination_folder +"/"

        except Exception as e:
            
            txt    =   f'Info msg:-Error while folder in create folder function'
            print(txt)
            
            txt    =   f'{traceback.format_exc()}'
            print(txt)

        return destination_folder
        
        
       
        
    @staticmethod      
    def create_bands(band_level_list,period_market_ohlc,band_delta):
    
          df    = pd.DataFrame(columns=['datetime','price','high','low','strength','relevance'])
          delta = 5                 #as per band_delta input
      
          for each_price in band_level_list:
      
             row    = len(df)
             df.loc[row, 'datetime']              = datetime.now().replace(microsecond = 0)
             df.loc[row, 'price']                 = each_price
             df.loc[row, 'band']                  = f'{each_price - delta} - {each_price + delta}'
             df.loc[row, "high"]                  = each_price + delta
             df.loc[row, 'low']                   = each_price - delta
             df.loc[row, 'relevance']             = 0


          return df
      
    @staticmethod    
    def evaluate_band_relevance(market_ohlc,bands_df,band_type):
          
          print(f"Evaluating band relevance for {band_type} bands")
          
          count_of_bands  = len(bands_df)
      
          if band_type == 'resistance':
      
            for each_band_index in range(0,count_of_bands,1):
      
                for each_ohlc_index in range(0,len(market_ohlc),1):
      
                    bands_df.loc[each_band_index,'relevance'] += Global_function.calculate_resistance_band_relevance(market_ohlc.iloc[each_ohlc_index],bands_df.iloc[each_band_index])
      
      
          elif band_type == 'support':
      
             for each_band_index in range(0,count_of_bands,1):
      
                for each_ohlc_index in range(0,len(market_ohlc),1):
      
                    bands_df.loc[each_band_index,'relevance'] += Global_function.calculate_support_band_relevance(market_ohlc.iloc[each_ohlc_index],bands_df.iloc[each_band_index])            
                    
          return bands_df
      
    @staticmethod  
    def evaluate_band_strength(market_ohlc, bands_df, band_type):

      print(f"Evaluating band strength for {band_type} bands")

      count_of_bands = len(bands_df)
      new_strengths = [0] * count_of_bands  # Initialize a list to store new strength values

      if band_type == 'resistance':
          for each_band_index in range(0, count_of_bands, 1):
              for each_ohlc_index in range(0, len(market_ohlc), 1):
                  temp = Global_function.calculate_resistance_band_relevance(market_ohlc.iloc[each_ohlc_index], bands_df.iloc[each_band_index])
                  new_strengths[each_band_index] += temp

      elif band_type == 'support':
          for each_band_index in range(0, count_of_bands, 1):
              for each_ohlc_index in range(0, len(market_ohlc), 1):
                  temp = Global_function.calculate_support_band_relevance(market_ohlc.iloc[each_ohlc_index], bands_df.iloc[each_band_index])
                  new_strengths[each_band_index] += temp

      bands_df['strength'] = new_strengths  # Update the 'strength' column after the loops

      return bands_df
  
    
    @staticmethod
    def calculate_resistance_band_relevance(ohlc,band):

        condition_1 = ohlc["open"]    < band["high"]                                    #ohlc candle opens anywhere below the band upper cutoff price
        condition_2 = ohlc["high"]    > band["low"]     or ohlc["high"] > band["high"]  #ohlc candle high either cross band high or atleast cross band low
        condition_3 = ohlc["close"]   < band["high"]                                    #ohlc candle closes below band_high
    
        if condition_1 == True and condition_2 == True and condition_3 == True:
            return 1
        else:
            return 0
        
    @staticmethod
    def calculate_support_band_relevance(ohlc,band):

        condition_1 = ohlc["high"]   > band["high"]                                    #ohlc candle high crosses above  band upper cutoff price atleast once
        condition_2 = ohlc["low"]    < band["low"]                                     #ohlc candle low  crosses below  band low   cutoff price atleast once
        condition_3 = ohlc["close"]  > band["low"]                                     #ohlc candle close        above  band high
    
        if condition_1 == True and condition_2 == True and condition_3 == True:
            return 1
        else:
            return 0 
        
    @staticmethod    
    def combine_close_bands(df, column_name, tolerance=25):
            """Combines price bands within a specified tolerance.
        
            Args:
                df: DataFrame containing the price bands.
                column_name: Name of the column containing the price band center.
                tolerance: The maximum difference between band centers to be considered close.
        
            Returns:
                DataFrame with combined price bands.
            """
        
        
            combined_bands = []
            i = 0
            while i < len(df):
                
                current_band = df.iloc[i].copy() # Create a copy to avoid SettingWithCopyWarning
                
                j = i + 1
                while j < len(df):
                    
                    next_band = df.iloc[j].copy() # Create a copy to avoid SettingWithCopyWarning
                    
                    if abs(current_band[column_name] - next_band[column_name]) <= tolerance:
                        
                        # Combine bands
                        current_band['band'] = f"{min(current_band['low'], next_band['low'])}-{max(current_band['high'], next_band['high'])}"
                        current_band[column_name] = (current_band[column_name] + next_band[column_name]) / 2
                        current_band['strength'] += next_band['strength']
                        current_band['relevance'] += next_band['relevance']
                        
                        # Update Support_low and Support_high
                        current_band['low']  = min(current_band['low'], next_band['low'])
                        current_band['high'] = max(current_band['high'], next_band['high'])
        
                        df.drop(index=df.index[j], inplace=True)  # Remove the merged row from df
                        df.reset_index(drop=True, inplace=True)  # Reset index after dropping
        
                    else:
        
                        j+=1
        
        
                combined_bands.append(current_band)
                i += 1
        
            combined_df = pd.DataFrame(combined_bands)
            
            # Select desired columns for the final output
            combined_df = combined_df[['price', 'band', 'low','high','strength','relevance']]
            
            return combined_df     
    
    
    @staticmethod
    def combine_files(folder_path, keywords, output_file):
        
        """
        Combines CSV and XLSX files in a folder containing specific keywords in their filenames.
    
        Args:
            folder_path: The path to the folder containing the CSV and XLSX files.
            keywords: A list of keywords to search for in the filenames.
            output_file: The path to the output file where the combined data will be written (CSV format).
        """
        try:
            combined_df = pd.DataFrame()
            all_dfs = []
    
            # Combine CSV and XLSX files
            for extension in ['*.csv', '*.xlsx']:
                for filepath in glob.glob(os.path.join(folder_path, extension)):
                    filename = os.path.basename(filepath)
    
                    # Check if all keywords are present in the filename
                    if all(keyword.lower() in filename.lower() for keyword in keywords):
                        try:
                            # Read file based on extension
                            if filepath.lower().endswith('.csv'):
                                df = pd.read_csv(filepath)
                            elif filepath.lower().endswith('.xlsx'):
                                df = pd.read_excel(filepath)
    
                            all_dfs.append(df)
    
                        except Exception as e:
                            print(f"Error reading file {filename}: {e}")
    
            if not all_dfs:
                print("No matching files found.")
                return combined_df
    
            # Concatenate all DataFrames
            combined_df = pd.concat(all_dfs, ignore_index=True)
    
            # Convert date/time columns
            for col in combined_df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        combined_df[col] = pd.to_datetime(combined_df[col])
                        combined_df = combined_df.sort_values(by=[col], ascending=True)
                    except Exception:
                        print(f"Could not convert column '{col}' to datetime.")
    
            # Convert common numeric columns
            for col in ['close', 'open', 'high', 'low', 'volume']:
                if col in combined_df.columns:
                    try:
                        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                    except Exception:
                        print(f"Could not convert column '{col}' to numeric.")
    
            # Save final combined DataFrame to output CSV file
            #combined_df.to_csv(output_file, index=False)
            #print(f"Combined data saved to {output_file}")
    
        except Exception:
            print("Error while executing combine_files")
            print(traceback.format_exc())
    
        return combined_df
    

        
    @staticmethod
    def optimise_df_data_format(df: pd.DataFrame, col_type_map: dict):
            """
            Converts specified columns of a DataFrame to the given datatypes.
            
            Parameters:
            - df: DataFrame containing the data
            - col_type_map: Dictionary {column_name: target_dtype} 
              Supported types: 'numeric', 'datetime', 'timedelta', 'bool', 'string', 'category'
              
            Returns:
            - A new DataFrame with converted dtypes where possible.
            
            converted = convert_columns_dtype(df, {
                "qty":      "numeric",
                "flag":     "bool",
                "start":    "datetime",
                "duration": "timedelta",
                "code":     "category",
                "mixed":    "numeric",
            })
            """
            
            # Make a copy to avoid modifying original
            df = df.copy()
            
            for col, target_type in col_type_map.items():
                if col not in df.columns:
                    print(f"⚠️  Column '{col}' not found in DataFrame. Skipping.")
                    continue
        
                try:
                    if target_type == "numeric":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif target_type == "datetime":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif target_type == "timedelta":
                        df[col] = pd.to_timedelta(df[col], errors="coerce")
                    elif target_type == "bool":
                        df[col] = df[col].astype("boolean")
                    elif target_type == "string":
                        df[col] = df[col].astype("string")
                    elif target_type == "category":
                        df[col] = df[col].astype("category")
                    else:
                        print(f"⚠️  Unsupported type '{target_type}' for column '{col}'. Skipping.")
                except Exception as e:
                    print(f"❌  Failed to convert column '{col}' to '{target_type}': {e}")
            
            return df
    
    @staticmethod    
    def generate_timestamps(start_time, end_time, interval_minutes):
        
              """Generates timestamps with a specified interval."""
              
              # Convert start and end times to datetime objects
              start_time = pd.to_datetime(start_time)
              end_time = pd.to_datetime(end_time)
          
              # Generate timestamps using pandas.date_range
              timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}min')
              
              
              timestamps = pd.DataFrame(timestamps,columns=['datetime'])
              
              return timestamps
     
    @staticmethod
    def identify_plot_band(support_df,resistance_df,nifty_df_sec,nifty_df_min,nifty_df_days):
        
        #empty list for return value
        data    =   [] 
        
        if nifty_df_sec is None or nifty_df_days is None or nifty_df_sec.empty or nifty_df_days.empty:
        
            print(f"nifty_df_sec and nifty_df_days does not have sufficient data")
            
            return data
        
        else:
            
        #if len(nifty_df_sec) > 0 and len(nifty_df_days) > 0:
            
                nifty_now     =   nifty_df_sec['close'].iloc[-1]              # nifty_price_right now
                nif_yest_high =   nifty_df_days['high'].iloc[-1]              # yesterday high price
                nif_yest_low  =   nifty_df_days['low'].iloc[-1]               # yesterday low price
                
                nif_today_low   = min(nifty_df_sec['low'].min() ,nifty_df_min['low'].min()) -1000
                nif_today_high  = max(nifty_df_sec['high'].max(),nifty_df_min['high'].max()) +1000
                
                upper_limit     =   max(nif_yest_high,nif_today_high)         
                lower_limit     =   min(nif_yest_low,nif_today_low)
                
                s_df    =   support_df[(support_df['price'] < nifty_now+10) & (support_df['price']>lower_limit)]
                r_df    =   resistance_df[(resistance_df['price']>nifty_now-10) & (resistance_df['price']<upper_limit)]
                
                s_df    =   s_df.sort_values(by='price', ascending=False)
                r_df    =   r_df.sort_values(by='price', ascending=True)
                
                s_df    =   s_df.head(5)
                r_df    =   r_df.head(5)
                
                support_mean    = support_df['strength'].mean()
                resistance_mean = resistance_df['strength'].mean()
                
                s_df['color']    =   s_df.apply(lambda row: 'rgba(20, 219, 80, 0.8)' if row['strength'] > support_mean else 'rgba(20, 219, 80, 0.4)', axis=1)
                r_df['color']    =   r_df.apply(lambda row: 'rgba(204, 18, 55, 0.8)' if row['strength'] > resistance_mean else 'rgba(204, 18, 55, 0.4)', axis=1)
                
                s_df             =  s_df[['low','high','color']]   
                r_df             =  r_df[['low','high','color']]
                
                s_df_list       =   s_df.to_dict('records')
                r_df_list       =   r_df.to_dict('records')
                

                #padding data with dummy record so we dont miss on necessary plot
                new_record = {'low': 15000, 'high': 15001, 'color': 'blue'}

                
                data =   s_df_list     +   r_df_list
                
                # Add the new record to the data list
                data.append(new_record)


                # Sort the data list by the 'low' value
                data.sort(key=lambda x: x['low'])
        
        return data         
     
        
    @staticmethod        
    def band_plot_creator(support_df, resistance_df, current_NIFTY_price,plot_df):
        '''
        this is an outdated function now updated with identify_plot_band function
        1)filter first support above the NIFTY Index value
        2)filter first resistance below the NIFTY value
        3)filter 2 support below NIFTY value by strength remove very weak support
        4)filter 2 resistance bands above NIFTY value by strength remove very weak resistance
        5) Create a list of dictionaries for plotting areas with 'low', 'high', and 'color'
        '''
        
        try:
            r_df = resistance_df.copy()
            s_df = support_df.copy()
            
            plot_high   = plot_df['high'].max()+50
            plot_low    = plot_df['low'].min()-50
            plot_close  = plot_df['close'].iloc[-1]
            
            # Filter resistance bands and create dictionaries
            resistance_below = r_df[r_df['high'] < plot_close].head(1)
            resistance_above = r_df[(r_df['low'] > plot_close) & (r_df['strength'] > r_df['strength'].mean())].head(2)
    
            resistance_areas = resistance_below.apply(lambda row: {'low': row['low'], 'high': row['high'], 'color': 'rgba(204, 18, 55, 0.4)'}, axis=1).tolist()
            resistance_areas.extend(resistance_above.apply(lambda row: {'low': row['low'], 'high': row['high'], 'color': 'rgba(204, 18, 55, 0.8)'}, axis=1).tolist())
    
            # Filter support bands and create dictionaries
            support_above = s_df[s_df['low'] > current_NIFTY_price].head(1)
            support_below = s_df[(s_df['high'] < current_NIFTY_price) & (s_df['strength'] < s_df['strength'].mean())]
    
            support_areas = support_above.apply(lambda row: {'low': row['low'], 'high': row['high'], 'color': 'rgba(20, 219, 80, 0.4)'}, axis=1).tolist()
            support_areas.extend(support_below.apply(lambda row: {'low': row['low'], 'high': row['high'], 'color': 'rgba(20, 219, 80, 0.8)'}, axis=1).tolist())
    
            # Combine all areas
            plot_areas = resistance_areas + support_areas
    
            return plot_areas
    
        except Exception as e:
            
            txt     =   f'{traceback.format_exc()}'
            print(txt)
            
            return
        
        
    @classmethod
    def calculate_time_delta(cls,interval,start_datetime,end_datetime):
         
         start_datetime  =   pd.to_datetime(start_datetime)   
         end_datetime    =   pd.to_datetime(end_datetime)  
         time_diff       =       0
         
         if interval     ==   '1day':
             
            time_diff = (end_datetime - start_datetime).days
            
            
         elif  interval     ==  '30minute':
            
            time_diff = (end_datetime - start_datetime).total_seconds() /1800
            
         elif interval     ==   '5minute':
            
             time_diff = (end_datetime - start_datetime).total_seconds() /300
            

         elif interval     ==   '1minute':
            
             time_diff = (end_datetime - start_datetime).total_seconds() /60
            
         elif interval     ==   '1second':
            
             time_diff = (end_datetime - start_datetime).total_seconds()
     
     
         return time_diff   
     
        
    @classmethod
    def update_new_time(cls,interval,start_datetime,time_delta):
         
         updated_time    =   start_datetime
         
         if interval     ==   '1day':
             
            updated_time = start_datetime + timedelta(days = time_delta)
            
            
         elif  interval     ==  '30minute':
            
            updated_time = start_datetime + timedelta(hours = time_delta/2)
        
         elif interval     ==   '1minute':
            
             updated_time = start_datetime + timedelta(minutes = time_delta)
            
         elif interval     ==   '1second':
            
            updated_time = start_datetime + timedelta(seconds = time_delta)
     
     
         return updated_time
    
        
    @staticmethod
    def current_time_match(timestamps,n):
           """
           Checks every second if the current time matches the first entry in the timestamps list.
           """
           try:
               pointer      = n
               current_time = datetime.now().replace(microsecond=0)
               #current_datetime = datetime.combine(datetime.date.today(), current_time)
               timestamp = timestamps['datetime'].iloc[pointer]
               
               if current_time.time() >= timestamp.time():
                   
                   return "Success"
   
               else:

                   return "Wait"
               
               
           except Exception as e:
               
               return e   
           
            
    @staticmethod       
    def read_NIFTY_preopen():

        url = "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market"

        # Setup Chrome options
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        driver = webdriver.Chrome(options=options)

        df = pd.DataFrame()
        try:
            driver.get(url)
            time.sleep(5)  # Let page load and JS execute
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, "//table[@id='livePreTable']"))
            )

            table = driver.find_element(By.XPATH, "//table[@id='livePreTable']")
            html = table.get_attribute("outerHTML")
            df = pd.read_html(StringIO(html))[0]
            print(f"✅ Fetched {df.shape[0]} rows.")

        except Exception as e:
            print(f"⚠️ Error fetching data: {e}")

        finally:
            driver.quit()

        return df

    @staticmethod
    def fetch_NIFTY_data():

          global existing_pre_market_data
          counter   =   0  
          df        =   pd.DataFrame()

          while len(df) < 50 and counter < 5:
              
              print(f"Attempt {counter}: Reading the Pre-open market data of NIFTY 50 ")
              df    =   Global_function.read_NIFTY_preopen()
              counter   +=1   
              
          if len(df) >50:
              
              df = df.rename(columns={'FFM CAP(₹ Crores)': 'FFM CAP(in Crores)'})
              df.to_csv(existing_pre_market_data, index=False)
              
              
          else:
              
              #read old market data
              
              df    =   pd.read_csv(existing_pre_market_data)
              
          return df

    @staticmethod
    def Get_pre_open_stock_list_NIFTY50():

          global NSE_Stock_script,NIF_50_Stock_list


          #reading stored data
          ICICI_database          = pd.read_csv(NSE_Stock_script)
          
          #Cleaning headers with unnecessary comma and spaces
          ICICI_database.columns  = ICICI_database.columns.str.strip().str.replace('"', '').str.replace(' ', '')

          #Filtering required columns      
          ICICI_database      = ICICI_database[['ShortName', 'Series', 'CompanyName','Symbol', 'ExchangeCode']]
          
          #Filtering required rows
          ICICI_database      = ICICI_database[ICICI_database['Series']== 'EQ']
          
          #Reading pre-open market data from website
          df_NIFTY50          = Global_function.fetch_NIFTY_data()      
          
          #filtering required columns for website
          df_NIFTY50          = df_NIFTY50[['Symbol','FFM CAP(in Crores)']]      

          #doing v-lookup from the master data of ICICI direct and removing unnecessary rows
          merged_df           = pd.merge(ICICI_database,df_NIFTY50, left_on='ExchangeCode', right_on='Symbol', how='right') 
          merged_df           = merged_df.dropna(subset=['ShortName'])
          
          #calculating index weight from free float market capital
          merged_df.loc[:, 'index_weg']  = round((merged_df['FFM CAP(in Crores)']/merged_df['FFM CAP(in Crores)'].sum())*100,2)
          
          #filtering required columns from merged data
          merged_df   =   merged_df[['ExchangeCode','ShortName','CompanyName','index_weg','FFM CAP(in Crores)']]
          
          
          #Sorting data by index weight
          merged_df   =   merged_df.sort_values(by='index_weg', ascending=False)
          
          #rename colume names
          merged_df   =  merged_df.rename(columns={'ShortName': 'stock_name'}) 
          
          merged_df.to_csv(NIF_50_Stock_list, index=False)
          
          return merged_df


    @staticmethod
    def reset_flag_conditionally(df, flag_col, time_col, threshold=3, wait_sec=7, stale_sec=10, key_col='strike_price'):

       try:

                """
                Reset flags in a DataFrame based on count threshold and staleness.
        
                Parameters:
                - df         : pandas DataFrame containing the data
                - flag_col   : column name for the boolean flag (e.g. 'call_buy_flag')
                - time_col   : column name for the timestamp associated with the flag (e.g. 'call_buy_flag_time')
                - threshold  : max number of active flags before full reset
                - wait_sec   : time to wait before resetting if threshold exceeded
                - stale_sec  : max duration a flag can stay active before being cleared
                - key_col    : column to uniquely identify rows (e.g. 'strike_price')
                """

                if df[flag_col].sum() > threshold:
                    sleep(wait_sec)
                    df[flag_col] = False

                else:
                    cutoff = datetime.now() - timedelta(seconds=stale_sec)
                    stale_mask = (df[flag_col] == 1) & (df[time_col] > cutoff)
                    stale_keys = df.loc[stale_mask, key_col].unique()
                    df.loc[df[key_col].isin(stale_keys), flag_col] = False

                return "Success"

       except Exception as e:

           return f"Error resetting flags: {e}"



a   = Global_function.read_NIFTY_preopen()
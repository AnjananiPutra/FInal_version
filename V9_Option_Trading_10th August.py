'''
================================================================================
      Importing libraries for the code
================================================================================
'''
import dash
import math
import queue
import urllib
import zipfile
import warnings
import threading
import traceback
import numpy as np
import pandas as pd
import multiprocessing
import plotly.graph_objects as go

from time import sleep
from Lib.Lib_Logging import Log
from Lib.Indicators import Indicators
from Lib.Lib_Transactions import Transactions
from dash import  dcc, html, Input, Output, State
from Lib.Lib_Connect_Breeze import Connect_Breeze
from datetime import datetime, date, time, timedelta
from Lib.Lib_Create_Dashboard_v6_Dash import Dashboard
from Lib.Lib_Global_Functions_V3 import Global_function

warnings.filterwarnings('ignore')

'''
==================================================================================
                    Global Variables Declaration
==================================================================================
'''
lock = threading.Lock()

code_start_time     = time(9, 1)
mkt_start_time      = time(9, 15)
code_end_time       = time(15, 30)
ticks_queue         = 0
option_chain_queue  = 0
nif_index_queue     = 0
nif_stocks_queue    = 0
queue_length        = 0
start_events        = 0
pause_events        = 0
ticks_queue_size    = 0
breeze              = 0
junk = '------------------------------------------------------'

'''
================================================================================
         storage location of important folders
================================================================================
'''

mkt_folder          = "E:/Algo_Trading_V3/Market Data/"
instrument_folder   = "E:/Algo_Trading_V3/Input Files/Instruments/"
KPI_folder          = "E:/Algo_Trading_V3/Input Files/KPI/"
lg_folder           = "E:/Algo_Trading_V3/Log File/"
NIFTY_hourly_folder = mkt_folder + "NIFTY_Historic_Data/Hourly_data/"
NIFTY_daily_folder  = mkt_folder + "NIFTY_Historic_Data/Daily_data/"
market_folder       = 0
log_folder          = 0
stk_data_folder     = 0
opt_data_folder     = 0
summary_folder      = 0
summary_opt_folder  = 0
summary_stk_folder  = 0
summary_log_folder  = 0

'''
---------------------------------------------------------------------------------
          creating folder and Updating destination location
---------------------------------------------------------------------------------
'''


def create_folders():

    global market_folder, log_folder, stk_data_folder, opt_data_folder
    global summary_folder, summary_opt_folder, summary_stk_folder, summary_log_folder

    market_folder       = Global_function.create_folder(mkt_folder, datetime.now().strftime("%B_%d_%Y"))
    log_folder          = Global_function.create_folder(lg_folder, datetime.now().strftime("%B_%d_%Y"))
    stk_data_folder     = Global_function.create_folder(market_folder, "Stock Data")
    opt_data_folder     = Global_function.create_folder(market_folder, "Option Data")
    summary_folder      = Global_function.create_folder(market_folder, "summary_folder")
    summary_opt_folder  = Global_function.create_folder(summary_folder, "Option_summary")
    summary_stk_folder  = Global_function.create_folder(summary_folder, "Stock_summary")
    summary_log_folder  = Global_function.create_folder(log_folder, "Log_summary")

    return


'''
==================================================================================
'''

Log.debug_msg("White", junk, False)
Log.info_msg("Green", junk, False)
Log.warning_msg("Blue", junk, False)
Log.error_msg("Yellow", junk, False)
Log.critical_msg("Red", junk, False)


def main():
    # Declaration of Multiprocessing Queue
    global ticks_queue, option_chain_queue, nif_index_queue, nif_stocks_queue

    # Declaration of event flags
    global start_events, pause_events

    # Declaration of breeze API related variables
    global breeze

    # Declaration of global variables
    global code_start_time, mkt_start_time, code_end_time, queue_length

    # Initialisation of variables
    code_start_time     = time(5, 1)
    mkt_start_time      = time(5, 15)
    code_end_time       = time(15, 30)
    ticks_queue         = multiprocessing.Queue()
    option_chain_queue  = multiprocessing.Queue()
    nif_index_queue     = multiprocessing.Queue()
    nif_stocks_queue    = multiprocessing.Queue()
    queue_length        = 0

    thread_names = ['ticks_manager', 'entry_scanner','trade_entry', 'square_off', 'data_archiever', 'program_parameters']
    process_names = ['consumer']

    start_events = {name: multiprocessing.Event() for name in thread_names + process_names}
    pause_events = {name: multiprocessing.Event() for name in thread_names + process_names}

    # create the necessary folders
    create_folders()

    while datetime.now().time() < code_start_time:
        sleep(1)
        txt = f'Waiting for market to open.current time {datetime.now().time()}'
        print(txt)

    # Login to breeze api app
    breeze = Connect_Breeze.login(breeze)
    breeze.on_ticks = Ticks.importer(ticks_queue)
    # det = Order_book.calc_P_and_L(datetime(2024, 1, 1, 9, 15),datetime(2025, 7, 13, 9, 15))
    option_chain = breeze.get_option_chain_quotes(
                                                    stock_code      ="NIFTY",
                                                    exchange_code   ="NFO",
                                                    product_type    ="options",
                                                    expiry_date     ="2025-08-28T06:00:00.000Z",  # ISO8601 format
                                                    right           ="call",  # or "put"
                                                    strike_price    =""  # leave blank to get fullchain
                                                )
    holdings = breeze.get_portfolio_positions()

    # initialize the important classes
    IP.init_parameters()
    Order_book.init_fund_mgmt(breeze)
    Option_Chain.init_class_variables()
    NIFTY_Index.init_class_variables()
    NIFTY_Stocks.initiate_class()


    threads,processes = Thread_Scheduler.start_threads_and_processes(start_events,
                                                                     pause_events,
                                                                     ticks_queue,
                                                                     option_chain_queue,
                                                                     nif_stocks_queue,
                                                                     nif_index_queue,
                                                                     queue_length)

    # Wait for the market to open
    while datetime.now().time() < mkt_start_time:
        sleep(1)
        txt = f'Waiting for market to open.current time {datetime.now().time()}'
        Log.debug_msg("Blue", txt, True)

    # start execution of thread and process to load ticks
    start_events['ticks_manager'].set()
    start_events['consumer'].set()
    start_events['data_archiever'].set()
    start_events['entry_scanner'].set()
    start_events['trade_entry'].set()

    # clear pause execution flag
    pause_events['ticks_manager'].set()
    pause_events['consumer'].set()
    pause_events['data_archiever'].clear()
    pause_events['entry_scanner'].set()
    pause_events['trade_entry'].set()

    # starting dash app
    txt = "Starting Dash App..."
    Log.info_msg("Green", txt, True)
    app = Display.built_dashboard()

    t1 = threading.Thread(target=Display.run_app, args=(app,))
    t1.start()

    # Thread List
    '''
    TH1 :Ticks Manager
    TH2 :Input and output parameters update and utility tasks of Option chain like update focus SP list 
    TH3 :Entry Scanners
    TH4 :Exit Scanners
    TH5 :Auto archival of Option Chain and NIFTY Stocks
    TH6 :Codes to be executed every second like VWAP,square off detection  
    '''

    Thread_Scheduler.start_threads(['TH2'])

    while datetime.now().time() < code_end_time and IP.ctrl_ptrs['Main_Program'].iloc[-1] == 0:

        try:

            txt = f'waiting for all threads to get over{datetime.now().time()}'

            # Log.info_msg("Green",txt,False)

            Option_Chain.utility_and_update(option_chain_queue,
                                              nif_stocks_queue ,
                                              nif_index_queue)

            # print({Display.update_thread_status()})

            sleep(10)
            print(f'Ticks Count :: {Ticks.tick_count}')
            # update queue length
            queue_length = max(ticks_queue.qsize(),
                               option_chain_queue.qsize(),
                               nif_index_queue.qsize(),
                               nif_stocks_queue.qsize())

            Log.info_msg("Green",f"Queue Length::{queue_length}",True)

            txt = f'Transaction Details::{Order_book.Trx_list}'
            Log.info_msg("Green",txt,True)


        except KeyboardInterrupt:

            for Th_no in Thread_Scheduler.Thread_List.index:
                Thread_Scheduler.stop_threads([Th_no])

    # making graceful exit from the main function
    for Th_no in Thread_Scheduler.Thread_List.index:
        Thread_Scheduler.stop_threads([Th_no])

    # stop execution of a process and thread to load ticks
    start_events['ticks_manager'].clear()
    start_events['consumer'].clear()
    start_events['data_archiever'].clear()

    # For today P&L
    Order_book.calc_P_and_L()

    # For date specific P&L
    # Order_book.calc_P_and_L(datetime(2024, 1, 1, 9, 15),datetime(2025, 7, 13, 9, 15))
    NIFTY_Stocks.VWAP_df_sec.to_excel(summary_folder + "VWAP_Info_sec_ticks.xlsx", index=False)

    return


class IP():
    '''
    Below code reads the Dataframe creation info from a file saved in KPI folder
    Then it creates Dataframe Variable for storing Entry and Exit parameters En_L1,En_L2,....
    Then it creates Dataframe Variables for storing
    '''

    start_time = datetime.now().replace(microsecond=0)
    ctrl_ptrs = pd.DataFrame()

    # defining Variables En_L1,to L4 and Ex_L1 to L4
    prefixes = ['En', 'Ex']
    for prefix in prefixes:
        for i in range(1, 5):
            vars()[f'{prefix}_L{i}'] = pd.DataFrame()

    Active_en = ""
    Active_ex = ""

    def __init__(self):
        return

    @staticmethod
    def update_entry_and_exit_logic():
        try:

            global KPI_folder

            # Reading Entry Parameters from excel
            data = Global_function.read_excel(KPI_folder, "Input_Parameters", "Entry_Parameters", 'A:F', 0, 1)
            data = data[data['Symbol'].notna()].transpose()
            data = data.rename(columns=data.loc['Symbol'])
            data = data.loc[['Entry Logic 1', 'Entry Logic 2', 'Entry Logic 3', 'Entry Logic 4']]


            # Update Function for dataframes
            IP.En_L1 = Global_function.update_dataframe(IP.En_L1, pd.DataFrame(data.loc['Entry Logic 1']).T)
            IP.En_L2 = Global_function.update_dataframe(IP.En_L2, pd.DataFrame(data.loc['Entry Logic 2']).T)
            IP.En_L3 = Global_function.update_dataframe(IP.En_L3, pd.DataFrame(data.loc['Entry Logic 3']).T)
            IP.En_L4 = Global_function.update_dataframe(IP.En_L4, pd.DataFrame(data.loc['Entry Logic 4']).T)


            # Reading Exit parameters from Excel
            data1 = Global_function.read_excel(KPI_folder, "Input_Parameters", "Exit_Parameters", 'A:F', 0, 1)
            data1 = data1[data1['Symbol'].notna()].transpose()
            data1 = data1.rename(columns=data1.loc['Symbol'])
            data1 = data1.loc[['Exit Logic 1', 'Exit Logic 2', 'Exit Logic 3', 'Exit Logic 4']]


            # Update Function for dataframes
            IP.Ex_L1 = Global_function.update_dataframe(IP.Ex_L1, pd.DataFrame(data1.loc['Exit Logic 1']).T)
            IP.Ex_L2 = Global_function.update_dataframe(IP.Ex_L2, pd.DataFrame(data1.loc['Exit Logic 2']).T)
            IP.Ex_L3 = Global_function.update_dataframe(IP.Ex_L3, pd.DataFrame(data1.loc['Exit Logic 3']).T)
            IP.Ex_L4 = Global_function.update_dataframe(IP.Ex_L4, pd.DataFrame(data1.loc['Exit Logic 4']).T)

        except Exception as e:
            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return "Success"

    @staticmethod
    def update_program_parameters():
        try:

            global KPI_folder

            df = Global_function.read_excel(KPI_folder, "Input_Parameters", "Entry_Parameters", 'I:J', 0, 1)
            df = df[df['Control Parameters'].notna()].transpose()
            df = df.rename(columns=df.loc['Control Parameters'])


            df.loc['Block'] = df.loc['Block'].astype(int)


            # Update Function for Dataframes
            IP.ctrl_ptrs = Global_function.update_dataframe(IP.ctrl_ptrs, df)

        except Exception as e:
            txt = f'update_program_parameters ran into some issues \nError Details ::{e} \n Traceback ::{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return "Success"

    @staticmethod
    def init_parameters():

        IP.update_entry_and_exit_logic()

        IP.update_program_parameters()

        IP.Active_en = IP.find_entry_logic(1)
        IP.Active_ex = IP.find_exit_logic(1)

        return "Success"

    @staticmethod
    def find_exit_logic(Logic_no):

        if Logic_no == 1:

            df = IP.Ex_L1

        elif Logic_no == 2:

            df = IP.Ex_L2

        elif Logic_no == 3:

            df = IP.Ex_L3

        elif Logic_no == 4:

            df = IP.Ex_L4

        return df

    @staticmethod
    def find_entry_logic(Logic_no):

        if Logic_no == 1:

            df = IP.En_L1

        elif Logic_no == 2:

            df = IP.En_L2

        elif Logic_no == 3:

            df = IP.En_L3

        elif Logic_no == 4:

            df = IP.En_L4

        return df

    @staticmethod
    def update_parameters(args):
        '''args are written to facilitate for loop'''
        try:

            global code_end_time
            txt = f'{Thread_Scheduler.Thread_List.loc["TH2", "Name"]} Thread started'
            Log.info_msg("Green", txt, True)

            # check if the flag is set to running state at the time of entry
            while Thread_Scheduler.Thread_List.loc["TH2", 'En_Flag'] == 0 \
                    and Thread_Scheduler.Thread_List.loc["TH2", 'Ex_Flag'] == 0 \
                    and datetime.now().time() < code_end_time:


                IP.update_entry_and_exit_logic()

                sleep(30)

            Thread_Scheduler.Thread_List.loc["TH2", 'En_Flag'] = 1
            Thread_Scheduler.Thread_List.loc["TH2", 'Ex_Flag'] = 1

            txt = f'{Thread_Scheduler.Thread_List.loc["TH2", "Name"]} Thread has been terminated succesfully'
            Log.info_msg("Green", txt, True)

        except Exception as e:

            Thread_Scheduler.Thread_List.loc["TH2", 'En_Flag'] = 1
            Thread_Scheduler.Thread_List.loc["TH2", 'Ex_Flag'] = 1

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)
            return "Success"

    @staticmethod
    def determine_exit_logic():
        pass


class Option_Chain():
    SP_list = []  # float
    SP_df = pd.DataFrame(
        columns=['Address', 'strike_price', 'focus_SP', 'call_buy_flag', 'put_buy_flag', 'call_sell_flag',
                 'put_sell_flag', 'call_tick_count', 'put_tick_count'])
    lot_size = 75
    expiry_date = datetime.now()

    def __init__(self, stock_code, strike_price, expiry_date):

        # Object wise variables required for storing data of particular strike price

        self.product_type       = "options"
        self.exchange_code      = "NFO"
        self.strike_price       = strike_price
        self.stock_code         = stock_code  # NIFTY by default
        self.call_df_sec        = pd.DataFrame(columns=['datetime', 'interval', 'strike_price',
                                                        'right_type', 'open', 'high', 'low', 'close',
                                                        'volume','Avg_vol', 'volume_factor', 'MVA_n', 'EMV_n',
                                                        "open_interest"])
        self.put_df_sec         = pd.DataFrame(columns=self.call_df_sec.columns)
        self.expiry_date        = expiry_date  # datetime.date(2024, 12, 19)
        self.expiry_date_iso    = expiry_date.isoformat()[:10] + 'T06:00:00.000Z'  # "2024-12-19T06:00:00.000Z"
        self.expiry_date_txt    = expiry_date.strftime("%d-%b-%Y")  # 19-Dec-2024
        self.call_VWAP_total    = 0
        self.call_Vol_total     = 0
        self.put_VWAP_total     = 0
        self.put_Vol_total      = 0
        self.call_lock          = threading.Lock()
        self.put_lock           = threading.Lock()
        self.call_entry_lock    = 0
        self.put_entry_lock     = 0
        self.call_buy_flag      = False
        self.put_buy_flag       = False
        self.call_sell_flag     = False
        self.put_sell_flag      = False

        # print(f'Object created for:{strike_price},{right},{expiry date}')

        return

        '''
        ==============================================================================================================================
          Function Name        *     Inputs         *          Function
        ------------------------------------------------------------------------------------------------------------------------------
        append_row             *   right            * This function will add data recieved recently into dataframe of call or put
                               *   data_list        *  based on function input return value is success/failure

        trim_df_head           *  right             * This function will crop the head of dataframe before the cut-off time supplied
                               *  cut_off_time      * to the function

        archieve_df_head       *  right             * This function will archieve/save dataframe information before the cutoff time
                               *  cut_off_time      * in the designated folder


        ===============================================================================================================================
        '''

    @staticmethod
    def init_class_variables():

        try:


            # 1)Get nearest expiry date
            expiry_date = Option_Chain.get_nearest_expiry("NIFTY", True)


            # 2)Get strike price list
            strike_price_list = Option_Chain.get_strike_price_list("NIFTY", "Call", expiry_date, 300, 10)


            # 3)Create objects and load the details in SP_df dataframe
            for strike_price in strike_price_list:
                status = Option_Chain.add_strike_price_and_object(strike_price, expiry_date, "NIFTY")
                print(
                    f'Failed to subscribe and add strike price: {strike_price} || {expiry_date}') if status != "Success" else None


            # 4)update focus SP list
            status = Option_Chain.update_focus_SP_and_object_list()

            if status != "Success":
                Log.error_msg("Yellow", status, True)


        except Exception as e:

            txt = f'Error while executing init_class_variables of Option Chain \nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return "Success"

    @staticmethod
    def dowonload_instruments():

        try:

            print('instruments download starting')
            url = 'https://directlink.icicidirect.com/NewSecurityMaster/SecurityMaster.zip'
            urllib.request.urlretrieve(url, instrument_folder + "SecurityMaster.zip")

            with zipfile.ZipFile(instrument_folder + "SecurityMaster.zip", 'r') as zip_ref:
                zip_ref.extractall(instrument_folder)
            print('instruments downloaded')

        except Exception as e:

            txt = f'Exception while downloading instruments \nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    # Instance Method
    def get_historical_data(self, interval, right, start_datetime, end_datetime):

        try:


            start_datetime_iso = start_datetime.isoformat() + ".000Z"
            end_datetime_iso = end_datetime.isoformat() + ".000Z"


            # expiry date format is changed then the guided format but need to check if used again

            data = breeze.get_historical_data_v2(interval       =   interval,
                                                 from_date      =   start_datetime_iso,
                                                 to_date        =   end_datetime_iso,
                                                 exchange_code  =   "NFO",
                                                 stock_code     =   self.stock_code,
                                                 product_type   =   self.product_type,
                                                 expiry_date    =   self.expiry_date,
                                                 strike_price   =   int(self.strike_price),
                                                 right          =   right)  # right is either 'call' or 'put'

            df = pd.DataFrame(data["Success"])


            if df.empty:

                print(f"No historical data for {self.strike_price} || {right} || {interval} between {start_datetime} and {end_datetime}")


            else:

                df['datetime'] = pd.to_datetime(df['datetime'])
                df["date"] = df['datetime'].dt.date
                df["time"] = df['datetime'].dt.time
                df = df[["date", "time", "stock_code", "expiry_date", "right", "strike_price",
                         "open", "high", "low", "close", "volume", "open_interest"]]


                print(f"Historical data downloaded for {str(self.strike_price)} || {right} ||{interval}")
                df["SMA_26"] = df['close'].rolling(26, min_periods=1).mean()
                df["EMA_26"] = df['close'].ewm(span=26, adjust=False).mean()
                df["EMA_12"] = df['close'].ewm(span=12, adjust=False).mean()

        except Exception as e:

            txt = f'Exception while downloading historical data \nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return df

    @staticmethod
    def get_nearest_expiry(index_name, offset_flag):

        try:

            #Dowload Option Chain Tree for Trading purpose
            Option_Chain.dowonload_instruments()

            txt = 'get_nearest_expiry starting'
            Log.debug_msg("Blue", txt, True)

            #Read the downloaded file
            scrip           = pd.read_csv(instrument_folder + "FONSEScripMaster.txt")
            scrip           = scrip[['Token', 'InstrumentName', 'ShortName',
                                     'Series', 'ExpiryDate',    'StrikePrice',
                                     'OptionType', 'ExchangeCode']]

            # Filter the read data by Index name in the dataframe
            scrip = scrip[scrip.ShortName == index_name]

            #Filter data for Option Chain data
            scrip = scrip[scrip.Series == 'OPTION']

            #convert the datetime text data to date format
            scrip.ExpiryDate = pd.to_datetime(scrip.ExpiryDate).dt.date

            #remove all duplicates
            expiry_list = scrip.ExpiryDate.drop_duplicates()

            #sort all dates
            expiry_list = sorted(expiry_list)

            #set cutoff date as next day to switch trading to next expiry if asset is expiring today or tomorrow
            cut_off_date = date.today() + timedelta(days=1)  # dont trade on last day of expiry shift to next expiry


            txt = 'get_nearest_expiry completed'
            Log.debug_msg("Blue", txt, True)

            # Shifting to Next week expiry if the expiry is today
            if offset_flag:

                for exp_date in expiry_list:

                    if exp_date >= cut_off_date:
                        expiry_date = exp_date
                        break
                    

            else:
                expiry_date     =  expiry_list[0]

            Option_Chain.expiry_date = expiry_date


        except Exception as e:

            txt = f'Exception while downloading expiry date from option chain Tree \nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

        return expiry_date

    # Instance Method
    def subscribe_ticks(self, interval):

        try:

            #Initialize Variables
            global breeze
            counter     = 0
            call_status = {'message': None}
            put_status  = {'message': None}

            #txt = f"1:{self.exchange_code} 2:{self.stock_code} 3:{self.product_type} 4:{self.expiry_date_txt} 5:{str(int(self.strike_price))} 6:{interval}"
            #Log.debug_msg("Blue", txt, True)

            while call_status.get('message') != "Stock NIFTY subscribed successfully" and counter < 5:
                txt = f"Attempt {counter}:Subscribing to CALL ||{self.exchange_code} || {self.stock_code} || {self.product_type} || {self.expiry_date_txt} || {str(int(self.strike_price))} || {interval}"
                Log.debug_msg("Blue", txt, False)
                call_status = breeze.subscribe_feeds(exchange_code  =   self.exchange_code,
                                                     stock_code     =   self.stock_code,
                                                     product_type   =   self.product_type,
                                                     expiry_date    =   self.expiry_date_txt,
                                                     strike_price   =   str(int(self.strike_price)),
                                                     right          =   "call",
                                                     interval       =   interval)

                counter += 1

            sleep(0.1)

            counter = 0

            while put_status.get('message') != "Stock NIFTY subscribed successfully" and counter < 5:
                txt = f"Attempt {counter}:Subscribing to PUT  ||{self.exchange_code} || {self.stock_code} || {self.product_type} || {self.expiry_date_txt} || {str(int(self.strike_price))} || {interval}"
                Log.debug_msg("Blue", txt, False)
                put_status = breeze.subscribe_feeds(exchange_code   =   self.exchange_code,
                                                    stock_code      =   self.stock_code,
                                                    product_type    =   self.product_type,
                                                    expiry_date     =   self.expiry_date_txt,
                                                    strike_price    =   str(int(self.strike_price)),
                                                    right           =   "put",
                                                    interval        =   interval)

                counter += 1

            call_status = call_status.get('message')
            put_status = put_status.get('message')

            Log.debug_msg("Blue", f"call status :: {call_status}", True)
            Log.debug_msg("Blue", f"put  status :: {put_status}", True)

            sleep(0.1)

            return [call_status, put_status]

        except Exception as e:

            txt = f'Exception while subscribing Option Chain SP{self.strike_price} \nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            sleep(0.1)

            return ["",""]

    # Instance Method
    def un_subscribe_ticks(self, interval):

        try:
            global breeze
            counter = 0
            breeze.unsubscribe_feeds(exchange_code  =   self.exchange_code,
                                     stock_code     =   self.stock_code,
                                     product_type   =   self.product_type,
                                     expiry_date    =   self.expiry_date_txt,
                                     strike_price   =   str(int(self.strike_price)),
                                     right          =   "call",
                                     interval       =   interval)

            counter += 1

            msg = f"Un-subscription done for {self.strike_price} || CALL || {self.expiry_date_txt}"
            Log.debug_msg("Blue", msg, True)

            breeze.unsubscribe_feeds(exchange_code=self.exchange_code,
                                     stock_code=self.stock_code,
                                     product_type=self.product_type,
                                     expiry_date=self.expiry_date_txt,
                                     strike_price=str(int(self.strike_price)),
                                     right="put",
                                     interval=interval)
            counter += 1
            msg = f"Un-subscription done for {self.strike_price} || PUT || {self.expiry_date_txt}"
            sleep(0.1)
            Log.debug_msg("Blue", msg, True)

        except Exception as e:

            txt = f'Exception while unsubscribing Option Chain{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            sleep(0.1)
            return counter

    def re_subscribe_ticks(self, interval):

        status1 = self.un_subscribe_ticks('1second')
        status2 = self.subscribe_ticks('1second')

        return "Success"

    @staticmethod
    def get_strike_price_list(stock_code, right, expiry_date, max_LTP, min_LTP):

        global breeze

        txt = f' starting download for Option Chain Tree: {stock_code} || {right} ||{expiry_date.strftime("%d-%b-%Y")}'
        Log.debug_msg("Blue", txt, True)

        expiry_date_iso = expiry_date.isoformat()[:10] + 'T05:30:00.000Z'

        try:

            data = breeze.get_option_chain_quotes(stock_code    =   stock_code,
                                                  exchange_code =   "NFO",
                                                  product_type  =   "options",
                                                  expiry_date   =   expiry_date_iso,
                                                  right         =   right)

            data_df = pd.DataFrame(data["Success"])

            if data_df.empty:

                txt = f'Did not receive any data for Option Chain Tree: {stock_code} || {right} ||{expiry_date.strftime("%d-%b-%Y")}'
                Log.debug_msg("Blue", txt, True)

                return []

            else:


                data_df = data_df.reset_index()

                txt = f'Get_option_chain download completed for:{stock_code} || {right} || {expiry_date.strftime("%d-%b-%Y")}'
                Log.debug_msg("Blue", txt, True)

                data_df = data_df[(data_df.ltp > min_LTP) & (data_df.ltp < max_LTP)]
                SP_list = data_df["strike_price"].astype(float)
                SP_list = SP_list.sort_values(ascending=True)
                SP_list = SP_list.to_list()

            return SP_list

        except Exception as e:

            txt = f'Failed to get strike price list.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return []

    @staticmethod
    def add_strike_price_and_object(strike_price, expiry_date, stock_code):

        try:

            SP_Object = Option_Chain("NIFTY", strike_price, expiry_date)

            txt = f'Attempting to add SP{strike_price}|| {expiry_date}||{stock_code}'
            Log.debug_msg("Blue", txt, True)


            # check if it has successfully subscribed to atleast call or put then update the object in list
            status      = SP_Object.subscribe_ticks("1second")
            call_status = status[0]
            put_status  = status[1]

            successful_subscribe_msg = "Stock NIFTY subscribed successfully"

            txt = f'Call Subscription::{call_status == successful_subscribe_msg}|| Put Subscription::{put_status == successful_subscribe_msg}'
            Log.debug_msg("Blue", txt, True)

            if call_status == successful_subscribe_msg or put_status == successful_subscribe_msg:

                new_row_df = pd.DataFrame([{

                                            'Address': SP_Object,
                                            'strike_price': strike_price,
                                            'focus_SP': None,
                                            'call_buy_flag': None,
                                            'put_buy_flag': None,
                                            'call_sell_flag': None,
                                            'put_sell_flag': None,
                                            'call_tick_count': None,
                                            'put_tick_count': None

                                            }])

                Option_Chain.SP_df = pd.concat([Option_Chain.SP_df, new_row_df], ignore_index=True)
                Option_Chain.SP_df = Option_Chain.SP_df.sort_values(by='strike_price')

                txt = f"Subscription done for {SP_Object.strike_price} || CALL/PUT || {SP_Object.expiry_date_txt}"
                Log.debug_msg("Blue", txt, True)

                return "Success"

            else:

                return "Failed"

        except Exception as e:

            txt = f'Exception while adding SP:{strike_price} in option chain tree.Error Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    @staticmethod
    def remove_strike_price_and_object(strike_price, expiry_date, stock_code):

        try:

            df = Option_Chain.SP_df

            remove_SP_object = df[df['strike_price'] == strike_price]['Address'].iloc[0]

            remove_SP_object.un_subscribe_ticks('1second')

            Option_Chain.SP_df = df[df['strike_price'] != strike_price]

            return "Success"

        except Exception as e:

            txt = f'Exception while removing SP:{strike_price} from option chain SP_df.Error Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    @staticmethod
    def find_add_and_remove_SP_list():

        try:


            '''
            This program is called when Active_strike price reaches near the min/max price list of
            current SP list min/max.
            The program will first download and shortlist new Strike_prices to be subscribed for getting data and then

            '''

            # testing Purpose
            # Option_Chain.SP_list =  [23000]
            new_SP_obj_list = []
            add_SP_list = []
            del_SP_list = []


            expiry_date = Option_Chain.expiry_date

            # getting the data in required format for further functioning
            # It is necessary to convert panda series to list before checking 'in' function to get desired result
            updated_SP_list = Option_Chain.get_strike_price_list("NIFTY", "Call", expiry_date, 300, 10)
            current_SP_list = Option_Chain.SP_df['strike_price'].to_list()


            for strike_price in updated_SP_list:

                if strike_price not in current_SP_list:
                    add_SP_list.append(strike_price)



            for strike_price in current_SP_list:

                if strike_price not in updated_SP_list:
                    del_SP_list.append(strike_price)

            return [add_SP_list, del_SP_list]

        except Exception as e:

            txt = f'Exception while updating stirke price list in function find_add_and_remove_SP_list.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return [add_SP_list, del_SP_list]

    @staticmethod
    def update_focus_SP_and_object_list():

        '''
        Check if the active SP is  closer to maximum SP in focus price list or closer to
        minimum SP.If yes then create a duplicate focus_SP_list and update in the focus_SP_List variable
        '''

        try:

            # Define Alias
            df = Option_Chain.SP_df


            # Sum all the true values in focus_SP  column equal to True
            if Option_Chain.SP_df['focus_SP'].sum() > 0:


                # Get Min-max value of SP

                lowest_strike_price = NIFTY_Index.Active_SP - IP.ctrl_ptrs['LL'].iloc[-1]
                highest_stike_price = NIFTY_Index.Active_SP + IP.ctrl_ptrs['UL'].iloc[-1]


                # Filter Focus SP list closest to current SP
                Option_Chain.SP_df['focus_SP'] = Option_Chain.SP_df['strike_price'].apply(
                    lambda x: lowest_strike_price <= x <= highest_stike_price)


            else:
                # Check if SP_df is loaded with strike Price before assignment

                if len(Option_Chain.SP_df) > 5:


                    # Get Min-Max value of SP
                    lowest_strike_price = Option_Chain.SP_df['strike_price'].iloc[0]
                    highest_stike_price = Option_Chain.SP_df['strike_price'].iloc[5]


                    # Filter Focus SP list closest to current SP
                    Option_Chain.SP_df['focus_SP'] = Option_Chain.SP_df['strike_price'].apply(
                        lambda x: lowest_strike_price <= x <= highest_stike_price)

                else:


                    txt = f"SP_df dataframe is does not have sufficient data and hence cannot update focus SP list"
                    Log.error_msg(txt, True)


            # msg         = f"Focus SP list updated: {df[df['focus_SP']]['strike_price']}"
            # Log.info_msg("Green",msg,True)

            return "Success"

        except Exception as e:

            txt = f"Unknown type of error happened while updating focus SP list function.\nError Details::{e}"
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return e

    @staticmethod
    def un_subscribe_all_sp(interval):

        for each_sp_obj in Option_Chain.SP_df['Address']:
            each_sp_obj.un_subscribe_ticks(interval)

        return "Success"

    @staticmethod
    def utility_and_update(OC_queue, NIF_Ind_queue, NIF_Stk_queue):

        try:

            global queue_length

            # Update queue length
            queue_length = max(
                OC_queue.qsize(),
                NIF_Ind_queue.qsize(),
                NIF_Stk_queue.qsize()
            )

            Display.update_VWAP()

            status = Option_Chain.update_focus_SP_and_object_list()

            if status != "Success":
                Log.error_msg("Yellow", status, True)

            min_SP = Option_Chain.SP_df['strike_price'].min()
            max_SP = Option_Chain.SP_df['strike_price'].max()

            if NIFTY_Index.Active_SP > 0 and \
                    (NIFTY_Index.Active_SP > (max_SP - 100) or \
                     NIFTY_Index.Active_SP < (min_SP + 100)):

                add_remove_SP_list = Option_Chain.find_add_and_remove_SP_list()

                add_SP_list = add_remove_SP_list[0]
                del_SP_list = add_remove_SP_list[1]
                expiry_date = Option_Chain.expiry_date

                for strike_price in add_SP_list:
                    Option_Chain.add_strike_price_and_object(strike_price, expiry_date, "NIFTY")

                for strike_price in del_SP_list:
                    Option_Chain.remove_strike_price_and_object(strike_price, expiry_date, "NIFTY")


        except Exception as e:

            txt = f'Exception while executing utility_and_update function \n Error details::{e} \nTrace Back::{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

        return "Success"

    @staticmethod
    def start_thread(start_event,pause_event):

        try:
            '''
            Check if thread conditions are satisfied for entry
            If Valid flag condition then start threads
            1)Perform regular evaluation of strike price
            2)evaluate all strike price buy condition
            3)evaluate all buy conditions
            '''
            start_event.wait()

            while start_event.is_set():

                pause_event.wait()

                Option_Chain.utility_and_update()


                Option_Chain.evaluate_all_buy()


                Option_Chain.evaluate_all_sell()

        except Exception as e:

            txt = f'Exception while executing start thread in Option Chain class.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

        return "Success"

    # Instance Method
    def add_tick(self, tick):

        try:

            # Object function to add data and do necessary computation
            # 1) Correct data type of information in data_list
            # 2) Convert information into dataframe
            # 3) Identify the right type and create alias of respective data frame
            # 4) add the columns and compute the data for each item
            # 5) start lock method to access the dataframe and concat latest info into the df
            # 6) close/surrender the lock and return success/failure of function

            # initialise variables
            MVA_period = int(IP.Active_en['N1'].iloc[-1])
            EMV_period = int(IP.Active_en['N2'].iloc[-1])

            # Convert information into a DataFrame with pre-defined columns
            tick_df = pd.DataFrame([tick],
                                   columns=["strike_price", "right_type", "open", "high", "low", "close", "volume",
                                            "open_interest", 'datetime', 'interval'])

            # Initialize new columns
            tick_df[["Avg_vol", "volume_factor", "MVA_n", "EMV_n", "VWAP"]] = np.nan

            # Identify the right type and prepare data

            if tick["right_type"] == "CE" and tick['interval'] == '1second':
                data_df = self.call_df_sec
                self.call_Vol_total += tick['volume']
                self.call_VWAP_total += tick['volume'] * tick['close']

            elif tick["right_type"] == "PE" and tick['interval'] == '1second':
                data_df = self.put_df_sec
                self.put_Vol_total += tick['volume']
                self.put_VWAP_total += tick['volume'] * tick['close']

            else:

                # txt = f"Invalid tick recieved in {self.strike_price} with interval ::{tick['interval']}"
                # Log.error_msg("Yellow",txt,True)

                return

            # Compute data for the new tick
            data_df_len = len(data_df)

            if data_df_len > 0:

                # Append the new tick for calculations without modifying the original dataframe yet
                combined_df = pd.concat([data_df, tick_df], ignore_index=True, sort=False)

                # Calculate rolling and EWM values on the combined DataFrame
                tick_df.loc[:, 'Avg_vol'] = combined_df['volume'].rolling(MVA_period, min_periods=1).mean().iloc[-1]
                tick_df.loc[:, 'MVA_n'] = combined_df['close'].rolling(window=MVA_period, min_periods=1).mean().iloc[-1]
                tick_df.loc[:, 'EMV_n'] = combined_df['close'].ewm(span=EMV_period, adjust=False).mean().iloc[-1]

            else:
                # Handle the case when data_df is empty
                tick_df.loc[:, 'Avg_vol'] = tick['volume']
                tick_df.loc[:, 'MVA_n'] = tick['close']
                tick_df.loc[:, 'EMV_n'] = tick['close']

            # Calculate dependent values
            tick_df.loc[:, 'volume_factor'] = tick_df['volume'].iloc[-1] / tick_df['Avg_vol'].iloc[-1] if \
            tick_df['Avg_vol'].iloc[-1] != 0 else np.nan
            tick_df.loc[:, 'VWAP'] = self.call_VWAP_total / self.call_Vol_total if tick[
                                                                                       "right_type"] == "CE" and self.call_Vol_total != 0 else (
                self.put_VWAP_total / self.put_Vol_total if tick[
                                                                "right_type"] == "PE" and self.put_Vol_total != 0 else np.nan)

            # Lock and concatenate the data
            df = Option_Chain.SP_df
            if tick["right_type"] == "CE":

                with self.call_lock:
                    self.call_df_sec = pd.concat([data_df, tick_df], ignore_index=True, sort=False)

                # update tick count in dataframe
                mask = df['strike_price'] == self.strike_price

                if not df[mask].empty:
                    # Fetch current value, set to 0 if it's None or NaN
                    current_val = df.loc[mask, 'call_tick_count'].iloc[0]

                    if pd.isna(current_val):
                        current_val = 0

                    df.loc[mask, 'call_tick_count'] = current_val + 1

            elif tick["right_type"] == "PE":

                with self.put_lock:
                    self.put_df_sec = pd.concat([data_df, tick_df], ignore_index=True, sort=False)

                # update tick count in dataframe
                mask = df['strike_price'] == self.strike_price

                if not df[mask].empty:

                    # Fetch current value, set to 0 if it's None or NaN
                    current_val = df.loc[mask, 'put_tick_count'].iloc[0]

                    if pd.isna(current_val):
                        current_val = 0

                    df.loc[mask, 'put_tick_count'] = current_val + 1

        except Exception as e:

            txt = f'Exception while adding ticks in option chain tree at section'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    @classmethod
    def fill_dataframes(cls, entry_count):

        try:

            for each_sp_obj in Option_Chain.SP_df['Address']:

                sample_tick = {'interval': '1second', 'exchange_code': 'NFO', 'stock_code': 'NIFTY',
                               'expiry_date': '03-Oct-2024', 'strike_price': '24700.0', 'right_type': 'PE',
                               'low': '18.2', 'high': '18.2', 'open': '18.2', 'close': '18.2', 'volume': '1375',
                               'oi': '4833250', 'datetime': '2024-10-02 13:21:09'}
                sample_tick['datetime'] = datetime.now().replace(microsecond=0)

                for i in range(entry_count):
                    sample_tick['strike_price'] = each_sp_obj.strike_price
                    sample_tick['datetime'] += timedelta(seconds=1)
                    sample_tick['right_type'] = "CE"
                    each_sp_obj.call_df_sec = pd.concat([each_sp_obj.call_df_sec, pd.DataFrame([sample_tick])],
                                                        ignore_index=True)
                    sample_tick['right_type'] = "PE"
                    each_sp_obj.put_df_sec = pd.concat([each_sp_obj.put_df_sec, pd.DataFrame([sample_tick])],
                                                       ignore_index=True)


        except Exception as e:

            txt = f'Error while filling dataframe with dummy data.\nError Details :: {e}'
            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    # Instance Method
    def trim_ticks_df_head(self, cut_off_time, cut_off_length):

        try:


            # crop the df at the time line
            '''
            1)Initiate all variables
            2)Validate dataframe shape and cut-off time format for archival
            3)lock the dataframe and filter data in the dataframe
            4)If there is data older than cut-off time then store it in system
            5)display and store relevant system message
            '''

            call_df         = self.call_df_sec
            put_df          = self.put_df_sec
            call_shape      = self.call_df_sec.shape[0] > 0
            put_shape       = self.put_df_sec.shape[0] > 0
            datetime_format = isinstance(cut_off_time, datetime)
            counter         = 0


            if call_shape == True and datetime_format == True:

                filtered_df = call_df[call_df['datetime'] >= cut_off_time]

                if filtered_df.empty:

                    txt = f"{self.strike_price}|| CALL cannot be trimmed as it does not have any data older than {cut_off_time}"
                    Log.error_msg("Yellow", txt, True)

                    temp = self.re_subscribe_ticks('1second')
                    txt = f'status of resubscription {temp}'
                    Log.error_msg("Yellow", txt, True)

                    txt = f"{self.strike_price}|| CALL || start_time = {call_df['datetime'].iloc[0]} end_time = {call_df['datetime'].iloc[-1]} "
                    Log.error_msg("Yellow", txt, True)

                    if len(filtered_df) > cut_off_length:
                        filtered_df = filtered_df.iloc[:min(len(filtered_df), cut_off_length)]

                        with self.call_lock:
                            self.call_df_sec = filtered_df

                        txt = f"{self.strike_price}|| CALL trimmed any data more than cut_off length of {cut_off_length} successfully"
                        counter += 1
                        Log.debug_msg("Blue", txt, True)

                else:

                    with self.call_lock:

                        self.call_df_sec = filtered_df

                    txt = f"{self.strike_price}|| CALL trimmed any data older than {cut_off_time} successfully"
                    counter += 1
                    Log.debug_msg("Blue", txt, True)

            elif not call_shape:

                txt = f"{self.strike_price} || CALL dataframe is empty"
                Log.error_msg("Yellow", txt, True)

            elif not datetime_format:

                txt = "{cut_off_time} data format is not valid. Datatype = {type(cut_off_time)}"
                Log.error_msg("Yellow", txt, True)

            if put_shape == True and datetime_format == True:

                filtered_df = put_df[put_df['datetime'] >= cut_off_time]

                if filtered_df.empty:

                    txt = f"{self.strike_price} || PUT does not have any data older than {cut_off_time}"
                    Log.error_msg("Yellow", txt, True)

                    temp = self.re_subscribe_ticks('1second')
                    Log.error_msg("Yellow", txt, True)

                    if len(filtered_df) > cut_off_length:
                        with self.put_lock:
                            self.put_df_sec = filtered_df

                        txt = f"{self.strike_price}|| CALL trimmed any data more than cut_off_length {cut_off_length}"
                        counter += 1
                        Log.debug_msg("Yellow", txt, True)


                else:

                    with self.put_lock:

                        self.put_df_sec = filtered_df

                    counter += 1

            elif not put_shape:

                txt = f"{self.strike_price} || PUT dataframe is empty"
                Log.error_msg("Yellow", txt, True)

                temp = self.re_subscribe_ticks('1second')
                Log.error_msg("Yellow", txt, True)

            elif not datetime_format:

                txt = f"{cut_off_time} data format is not valid. Datatype = {type(cut_off_time)}"
                Log.error_msg("Yellow", txt, True)

            return counter


        except Exception as e:

            txt = f'Exception while trimming Option Chain  SP{self.strike_price}.\nError Details: {e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    # Instance Method
    def save_ticks_df_head(self, cut_off_time, cut_off_length):

        try:

            '''
            1)Initiate all variables
            2)Validate dataframe shape and cut-off time format for archival
            3)lock the dataframe and filter data in the dataframe
            4)If there is data older than cut-off time then store it in system
            5)display and store relevant system message
            '''
            global opt_data_folder


            call_df         = self.call_df_sec
            put_df          = self.put_df_sec
            call_shape      = self.call_df_sec.shape[0] > 0
            put_shape       = self.put_df_sec.shape[0] > 0
            datetime_format = isinstance(cut_off_time, datetime)
            call_file_name  = str(self.strike_price) + "_CE_" + str(cut_off_time.strftime("%d-%m-%Y-%I-%M %p")) + ".xlsx"
            put_file_name   = str(self.strike_price) + "_PE_" + str(cut_off_time.strftime("%d-%m-%Y-%I-%M %p")) + ".xlsx"
            counter         = 0


            if call_shape == True and datetime_format == True:

                filtered_df = call_df[call_df['datetime'] < cut_off_time]

                if filtered_df.empty:

                    txt = f"{self.strike_price}|| CALL cant be archived as it does not have any data older than {cut_off_time}"
                    Log.error_msg("Yellow", txt, True)

                    txt = f"{self.strike_price}|| CALL || start_time = {call_df['datetime'].iloc[0]} end_time = {call_df['datetime'].iloc[-1]} "
                    Log.error_msg("Yellow", txt, True)

                    if len(filtered_df) > cut_off_length:
                        filtered_df = filtered_df.iloc[:min(len(filtered_df), cut_off_length)]

                        with self.call_lock:
                            self.call_df_sec = filtered_df

                        txt = f"{self.strike_price}|| CALL trimmed any data more than cut_off length of {cut_off_length} successfully"
                        counter += 1
                        Log.debug_msg("Blue", txt, True)

                else:


                    filtered_df.to_excel(opt_data_folder + call_file_name, index=False)

                    txt = f"{self.strike_price}|| CALL archived data older than {cut_off_time} successfully"
                    Log.debug_msg("Blue", txt, True)

                    counter += 1


            elif not call_shape:

                txt = "{self.strike_price} || CALL dataframe is empty"
                Log.error_msg("Yellow", txt, True)

            elif not datetime_format:

                txt = "{cut_off_time} data format is not valid. Datatype = {type(cut_off_time)}"
                Log.error_msg("Yellow", txt, True)

            if put_shape == True and datetime_format == True:


                filtered_df = put_df[put_df['datetime'] < cut_off_time]

                if filtered_df.empty:

                    txt = f"{self.strike_price} || PUT does not have any data older than {cut_off_time}"
                    Log.error_msg("Yellow", txt, True)

                else:


                    filtered_df.to_excel(opt_data_folder + put_file_name, index=False)
                    counter += 1

            elif not put_shape:

                txt = f"{self.strike_price} || PUT dataframe is empty"
                Log.error_msg("Yellow", txt, True)

            elif not datetime_format:

                txt = "{cut_off_time} data format is not valid. Datatype = {type(cut_off_time)}"
                Log.error_msg("Yellow", txt, True)

            return counter

        except Exception as e:

            txt = f'Exception while saving Option Chain {self.strike_price} : {e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    @classmethod
    def start_archiever(cls, cut_off_time, df_length, start_event, pause_event):

        try:

            for each_obj in Option_Chain.SP_df['Address']:

                if start_event.is_set():
                    each_obj.save_ticks_df_head(cut_off_time, df_length)
                    each_obj.trim_ticks_df_head(cut_off_time, df_length)


        except Exception as error_descriptor:

            txt = f"Auto‑archival crashed: {error_descriptor}\n{traceback.format_exc()}"
            Log.error_msg("Yellow", txt, True)

        finally:

            txt = f"Auto‑archival completed for Option chain for time {cut_off_time}"
            Log.error_msg("Yellow", txt, True)

        return

        # Instance Method

    # Instance Method
    def execute_limit_buy_order(self, right: str) -> str:
        """
        Places a limit buy order with retry logic.

        Steps:
            1. Initialize and validate parameters
            2. Check F&O balance to compute lot-wise quantity
            3. Attempt order placement up to 5 times or until successful
            4. Return order execution status or error
            5. Start square-off thread if entry order is successful
        """
        try:
            global breeze

            df = self.call_df_sec if right.lower() == 'call' else self.put_df_sec
            if df.empty:
                Log.error_msg("Yellow", "Price DataFrame is empty", True)
                return "Failed"

            min_lot_size = 75
            validity_date_iso = datetime.now().strftime('%Y-%m-%dT06:00:00.000Z')
            fno_avl_bal = Order_book.get_fno_balance().get('allocated_amt', 0)

            for attempt in range(1, 6):
                asset_last_price = df['close'].iloc[-1]
                buy_qty = Order_book.determine_buy_qty(fno_avl_bal, asset_last_price, min_lot_size)

                if buy_qty < min_lot_size:
                    Log.error_msg("Yellow", "Insufficient funds for minimum lot size", True)
                    return "Insufficient funds"

                Log.info_msg("Blue",
                             f"Attempt {attempt}: Placing limit buy order for {right.upper()} @ {asset_last_price}",
                             True)

                try:
                    order_details = breeze.place_order(
                        stock_code=self.stock_code,
                        exchange_code=self.exchange_code,
                        product=self.product_type,
                        action="buy",
                        order_type="limit",
                        stoploss="",
                        quantity=buy_qty,
                        price=asset_last_price,
                        validity="ioc",
                        validity_date=validity_date_iso,
                        disclosed_quantity="0",
                        expiry_date=self.expiry_date_iso,
                        right=right,
                        strike_price=self.strike_price
                    )
                except Exception as oe:
                    Log.error_msg("Yellow", f"API call failed: {str(oe)}", True)
                    continue

                if "Error" in order_details:
                    error_details = order_details["Error"]
                    Log.error_msg("Yellow", f"Order error: {error_details}", True)
                    continue

                order_id = order_details["Success"].get("order_id")
                order_status = Order_book.add_order_detail_and_return_status(self, right, breeze, order_id)
                Log.info_msg("Green", f"Limit order status: {order_status}", True)

                if order_status == "Success":
                    Exit_Scanners.square_off_holdings(order_id)
                    return "Success"

                Log.error_msg("Yellow", f"Order not successful. Status: {order_status}", True)

            Log.error_msg("Yellow", f"Limit order failed after 5 attempts for {right.upper()} @ {self.strike_price}",
                          True)
            return "Failed"

        except Exception as e:
            Log.error_msg("Yellow", f"Exception during limit buy for {self.strike_price} | {right}", True)
            Log.error_msg("Yellow", f"Error: {str(e)}", True)
            Log.error_msg("Yellow", traceback.format_exc(), True)
            return "Failed"


    # Instance Method
    def execute_market_buy_order(self, right: str) -> str:
        """
        Executes a market buy order with retry logic.

        Steps:
            1. Initialize necessary variables
            2. Check for available F&O balance and derive quantity
            3. Attempt to place market order up to 5 times or until successful
            4. Store successful order details in Order_book Trx list dataframe
            5. Log and return the result
        """
        try:
            global breeze

            # 1. Initialize Variables
            df = self.call_df_sec if right.lower() == 'call' else self.put_df_sec
            if df.empty:
                Log.error_msg("Yellow", "Price DataFrame is empty", True)
                return "Failed"

            asset_last_price    = df['close'].iloc[-1]
            min_lot_size        = 75
            validity_date_iso   = datetime.now().strftime('%Y-%m-%dT06:00:00.000Z')
            fno_avl_bal         = Order_book.get_fno_balance().get('allocated_amt', 0)
            buy_qty             = Order_book.determine_buy_qty(fno_avl_bal,
                                                               asset_last_price,
                                                               min_lot_size)

            # 2. Retry placing market order
            for attempt in range(5):

                if buy_qty < min_lot_size:
                    Log.error_msg("Yellow", "Insufficient funds for minimum lot size", True)
                    return "Insufficient funds"

                try:
                    order_details = breeze.place_order(
                                            stock_code          =   self.stock_code,
                                            exchange_code       =   self.exchange_code,
                                            product             =   self.product_type,
                                            action              =   "buy",
                                            order_type          =   "market",
                                            stoploss            =   "",
                                            quantity            =   buy_qty,
                                            price               =   "",
                                            validity            =   "ioc",
                                            validity_date       =   validity_date_iso,
                                            disclosed_quantity  =   "0",
                                            expiry_date         =   self.expiry_date_iso,
                                            right               =   right,
                                            strike_price        =   self.strike_price
                                        )
                except Exception as oe:
                    Log.error_msg("Yellow", f"Breeze API Exception: {oe}", True)
                    continue

                if "Error" not in order_details:
                    order_id     = order_details["Success"]['order_id']
                    order_status = Order_book.add_order_detail_and_return_status(self, right, breeze, order_id)
                    Log.info_msg("Green", f"Limit order status :: {order_status}", True)

                    if order_status == "Success":

                        Exit_Scanners.square_off_holdings(order_id)
                        return "Success"

                    else:

                        Log.error_msg("Yellow", f"Order status not successful: {order_status}", True)
                        continue
                else:
                    error_details = order_details.get("Error")
                    Log.error_msg("Yellow",
                                  f"Error placing order for {self.strike_price} | {right.upper()} at {asset_last_price}",
                                  True)
                    Log.error_msg("Yellow", f"Error Details::{error_details}", True)

            Log.error_msg("Yellow", f"Market order failed after retries for {right.upper()} @ {self.strike_price}",
                          True)
            return "Failed"

        except Exception as e:
            Log.error_msg("Yellow", f"Exception while placing market buy order for {self.strike_price} | {right}", True)
            Log.error_msg("Yellow", f"Error Details::{e}", True)
            Log.error_msg("Yellow", traceback.format_exc(), True)
            return "Failed"

    # Instance Method
    def execute_limit_sell_order(self, right):

        """
        Executes a limit sell order with retry logic.

        Steps:
            1. Initialize necessary variables
            2. Check for available F&O balance and derive quantity
            3. Attempt to place market order up to 5 times or until successful
            4. Store successful order details in Order_book Trx list dataframe
            5. Log and return the result
        """



        try:

            global breeze

            # Use appropriate dataframe
            df = self.call_df_sec if right.lower() == 'call' else self.put_df_sec

            if df.empty:
                Log.error_msg("Yellow", "Price DataFrame is empty", True)
                return "Error"

            counter             = 5
            order_status        = None
            asset_last_price    = df['close'].iloc[-1]
            min_lot_size        = 75
            validity_date_iso   = datetime.now().strftime('%Y-%m-%dT06:00:00.000Z')
            fno_avl_bal         = Order_book.get_fno_balance().get('allocated_amt', 0)
            sell_qty            = Order_book.determine_sell_qty(fno_avl_bal, self.strike_price, min_lot_size, right,
                                                     self.expiry_date_txt, self.stock_code)

            while counter > 0 and order_status != "Success" and sell_qty >= min_lot_size:

                try:

                    counter -= 1
                    order_details = breeze.place_order(
                                            stock_code                  =   self.stock_code,
                                            exchange_code               =   self.exchange_code,
                                            product                     =   self.product_type,
                                            action                      =   "sell",
                                            order_type                  =   "limit",
                                            stoploss                    =   "",
                                            quantity                    =   sell_qty,  # "50",
                                            price                       =   asset_last_price,
                                            validity                    =   "ioc",
                                            validity_date               =   str(validity_date_iso),
                                            disclosed_quantity          =   "0",
                                            expiry_date                 =   str(self.expiry_date_iso),  # "2022-09-29T06:00:00.000Z",
                                            right                       =   right,  # "call",
                                            strike_price                =   self.strike_price
                    )

                    # If order went through without any errors then get order Id and enter in Order_book entry records
                    if order_details.get("Error") is None:

                        order_id = order_details["Success"]['order_id']
                        order_status = Order_book.add_order_detail_and_return_status(self, right, breeze, order_id)

                        Log.info_msg("Green", f"Limit sell order status :: {order_status}", True)
                        return order_status

                    else:

                        Log.error_msg(
                            "Yellow",
                            f"Error placing limit sell order for {self.strike_price} | {right.upper()} at {asset_last_price}",
                            True)


                except Exception as oe:

                    Log.error_msg("Yellow", f"Breeze API Exception: {oe}", True)
                    
                    txt = f'{traceback.format_exc()}'
                    Log.error_msg("Yellow", txt, True)

                    continue

        except Exception as e:

            txt = f'Exception while placing sell limit order \nError Desc::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return "Failed"

    # Instance Method
    def execute_market_sell_order(self, right):

        try:

            global breeze

            # Use appropriate dataframe
            df = self.call_df_sec if right.lower() == 'call' else self.put_df_sec

            if df.empty:
                Log.error_msg("Yellow", "Price DataFrame is empty", True)
                return "Error"

            counter             = 5
            order_status        = None
            asset_last_price    = df['close'].iloc[-1]
            min_lot_size        = 75
            validity_date_iso   = datetime.now().strftime('%Y-%m-%dT06:00:00.000Z')
            fno_avl_bal         = Order_book.get_fno_balance().get('allocated_amt', 0)
            sell_qty            = Order_book.determine_sell_qty(fno_avl_bal, self.strike_price,
                                                     min_lot_size, right,
                                                     self.expiry_date_txt, self.stock_code)

            while counter > 0 and order_status != "Success" and sell_qty >= min_lot_size:

                try:

                    counter -= 1
                    order_details = breeze.place_order(
                                            stock_code              =   self.stock_code,
                                            exchange_code           =   self.exchange_code,
                                            product                 =   self.product_type,
                                            action                  =   "sell",
                                            order_type              =   "market",
                                            stoploss                =   "",
                                            quantity                =   sell_qty,  # "50",
                                            price                   =   "",
                                            validity                =   "ioc",
                                            validity_date           =   str(validity_date_iso),
                                            disclosed_quantity      =   "0",
                                            expiry_date             =   str(self.expiry_date_iso),  # "2022-09-29T06:00:00.000Z",
                                            right                   =   right,  # "call",
                                            strike_price            =   self.strike_price
                    )

                    # If order went through without any errors then get order Id and enter in Order_book entry records
                    if order_details.get("Error") is None:

                        order_id = order_details["Success"]['order_id']
                        order_status = Order_book.add_order_detail_and_return_status(self, right, breeze, order_id)

                        Log.info_msg("Green", f"Limit sell order status :: {order_status}", True)
                        return order_status

                    else:

                        Log.error_msg(
                            "Yellow",
                            f"Error placing limit sell order for {self.strike_price} | {right.upper()} at {asset_last_price}",
                            True)


                except Exception as oe:

                    Log.error_msg("Yellow", f"Breeze API Exception: {oe}", True)
                    continue

                    # testing purpose manual approach
                    # order_status = "COMPLETE"

        except Exception as e:

            txt = f'Exception while placing sell limit order \nError Desc::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return "Failed"

    # Instance Method
    def evaluate_SP_call_buy(self, evaluating_parameters):

        try:

            '''
            Steps

            1)This function receives evaluation parameters for making buy decision
            2)return True or False for Buy or Sell

            '''
            result = False

            validate_vol_price_action = Entry_Scanners.CE_vol_price_action_confirmation(self.call_df_sec,
                                                                                        evaluating_parameters)

            validate_NIFTY_Index_levels = Entry_Scanners.CE_NIFTY_Index_levels_confirmation(NIFTY_Index.df_sec,
                                                                                            evaluating_parameters)

            validate_NIFTY_Stocks_VWAP = True

            validation_list = [validate_vol_price_action,
                               validate_NIFTY_Index_levels,
                               validate_NIFTY_Stocks_VWAP
                               ]

            result = True if all(validation is True for validation in validation_list) else False



        except Exception as error_descriptor:

            txt = f'Error while executing evaluate_SP_call_buy for {self.strike_price}||call.\nError Details::{error_descriptor}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

        finally:

            return result

    # Instance Method
    def evaluate_SP_put_buy(self, evaluating_parameters):

        try:

            '''
            Steps

            1)This function receives evaluation parameters for making buy decision
            2)return True or False for Buy or Sell

            '''
            result = False

            validate_vol_price_action = Entry_Scanners.PE_vol_price_action_confirmation(self.put_df_sec,
                                                                                        evaluating_parameters)

            validate_NIFTY_Index_levels = Entry_Scanners.PE_NIFTY_Index_levels_confirmation(self.put_df_sec,
                                                                                            evaluating_parameters)

            validate_NIFTY_Stocks_VWAP = True

            validation_list = [validate_vol_price_action,
                               validate_NIFTY_Index_levels,
                               validate_NIFTY_Stocks_VWAP
                               ]

            result = True if all(validation is True for validation in validation_list) else False





        except Exception as error_descriptor:

            txt = f'Error while executing evaluate_SP_put_buy for {self.strike_price}|| put.\nError Details::{error_descriptor}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)


        finally:

            return result

    # Instance Method
    def evaluate_SP_call_sell(self, evaluating_parameters):

        try:

            '''
            1)This function receives evaluation parameters for making buy decision
            2)return True or False for Buy or Sell
            '''
            result = False

            validate_vol_price_action = Entry_Scanners.CE_vol_price_action_confirmation(self.call_df_sec,
                                                                                        evaluating_parameters)

            validate_CE_NIFTY_Index = Entry_Scanners.CE_NIFTY_Index_levels_confirmation(NIFTY_Index.df_sec,
                                                                                        evaluating_parameters)

            validate_PE_NIFTY_Index = not Entry_Scanners.PE_NIFTY_Index_levels_confirmation(self.put_df_sec,
                                                                                            evaluating_parameters)

            validate_NIFTY_Stocks_VWAP = False

            validation_list = [validate_vol_price_action,
                               validate_CE_NIFTY_Index,
                               validate_PE_NIFTY_Index,
                               validate_NIFTY_Stocks_VWAP
                               ]
            # we will only check if all results turn out to be negative for buy
            result = True if all(validation is False for validation in validation_list) else False


        except Exception as error_descriptor:

            txt = f'Error while executing evaluate_SP_call_sell for {self.strike_price}|| put'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)


        finally:

            return result

    # Instance Method
    def evaluate_SP_put_sell(self, evaluating_parameters):

        try:

            '''
            1)This function receives evaluation parameters for making buy decision
            2)return True or False for Buy or Sell
            '''
            result = False

            validate_vol_price_action = Entry_Scanners.PE_vol_price_action_confirmation(self.put_df_sec,
                                                                                        evaluating_parameters)

            validate_PE_NIFTY_Index = Entry_Scanners.PE_NIFTY_Index_levels_confirmation(self.put_df_sec,
                                                                                        evaluating_parameters)

            validate_CE_NIFTY_Index = not Entry_Scanners.CE_NIFTY_Index_levels_confirmation(NIFTY_Index.df_sec,
                                                                                            evaluating_parameters)

            validate_NIFTY_Stocks_VWAP = False

            validation_list = [validate_vol_price_action,
                               validate_PE_NIFTY_Index,
                               validate_CE_NIFTY_Index,
                               validate_NIFTY_Stocks_VWAP
                               ]
            # we will only check if all results turn out to be negative for buy
            result = True if all(validation is False for validation in validation_list) else False


        except Exception as error_descriptor:

            txt = f'Error while executing evaluate_SP_put_sell for {self.strike_price}|| put'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)


        finally:

            return result

    @staticmethod
    def evaluate_call_buy(SP_df, evaluation_parameters, start_flag, pause_flag):
        """
        Evaluates call buy conditions and updates flags/timestamps in SP_df.

        Parameters:
        - SP_df: pd.DataFrame with SP option data
        - evaluation_parameters: dict or relevant evaluation criteria
        - start_flag, pause_flag: controls for execution gating (currently unused)
        """

        try:
            for index, row in SP_df.iterrows():
                obj         = row['Address']
                old_result  = row['call_buy_flag']
                result      = obj.evaluate_SP_call_buy(evaluation_parameters)

                if result and not old_result:
                    txt = f'Entry Condition Satisfied for Buy || SP:{obj.strike_price} || call || {obj.expiry_date_txt}'
                    Log.info_msg("Green", txt, True)

                    SP_df.at[index, 'call_buy_flag'] = True
                    SP_df.at[index, 'call_buy_flag_time'] = datetime.now()

        except Exception as e:
            Log.error_msg("Yellow", f'Error in evaluate_call_buy: {e}', True)
            Log.error_msg("Yellow", traceback.format_exc(), True)

        finally:

            state = Global_function.reset_flag_conditionally(
                                                    SP_df,
                                                    flag_col    =   'call_buy_flag',
                                                    time_col    =   'call_buy_flag_time',
                                                    threshold   =   3,
                                                    wait_sec    =   7,
                                                    stale_sec   =   10,
                                                    key_col     =   'strike_price'
                                                )

            if state != 'Success':

                Log.error_msg("Yellow", f'Error in evaluate_call_buy: {state}', True)

    @staticmethod
    def evaluate_call_sell(SP_df, evaluation_parameters, start_flag, pause_flag):
        """
            Evaluates call sell conditions and updates flags in SP_df.
            Flags are set if new sell condition is met.
        """

        try:
                for index, row in SP_df.iterrows():
                    obj         = row['Address']
                    result      = obj.evaluate_SP_call_sell(evaluation_parameters)
                    old_result  = row['call_sell_flag']

                    if result and not old_result:
                        txt = f'Entry Condition Satisfied for Sell || SP: {obj.strike_price} || call || {obj.expiry_date_txt}'
                        Log.info_msg("Green", txt, True)

                        SP_df.at[index, 'call_sell_flag'] = True
                        SP_df.at[index, 'call_sell_flag_time'] = datetime.now()

        except Exception as e:
                Log.error_msg("Yellow", f'Error in evaluate_call_sell: {e}', True)
                Log.error_msg("Yellow", traceback.format_exc(), True)

        finally:

                state = Global_function.reset_flag_conditionally(
                                                        SP_df,
                                                        flag_col    =   'call_sell_flag',
                                                        time_col    =   'call_sell_flag_time',
                                                        threshold   =   3,
                                                        wait_sec    =   7,
                                                        stale_sec   =   10,
                                                        key_col     =   'strike_price'
                                                    )

                if state != 'Success':
                    Log.error_msg("Yellow", f'Error in evaluate_call_sell: {state}', True)

    @staticmethod
    def evaluate_put_sell(SP_df, evaluation_parameters, start_flag, pause_flag):
        """
        Evaluates put sell conditions and updates flags in SP_df.
        Flags are updated when a new sell condition is satisfied.
        """

        try:
            for index, row in SP_df.iterrows():
                obj = row['Address'][0]  # Adjust if Address isn't list-like

                result = obj.evaluate_SP_put_sell(evaluation_parameters)
                old_result = row['put_sell_flag']

                if result and not old_result:
                    txt = f'Entry Condition Satisfied for Sell || SP:{obj.strike_price} || put || {obj.expiry_date_txt}'
                    Log.info_msg("Green", txt, True)
                    SP_df.at[index, 'put_sell_flag']        = True
                    SP_df.at[index, 'put_sell_flag_time']  = datetime.now()

        except Exception as e:
            Log.error_msg("Yellow", f'Error in evaluate_put_sell: {e}', True)
            Log.error_msg("Yellow", traceback.format_exc(), True)

        finally:

                state = Global_function.reset_flag_conditionally(
                                                        SP_df,
                                                        flag_col    =   'put_sell_flag',
                                                        time_col    =   'put_sell_flag_time',
                                                        threshold   =   3,
                                                        wait_sec    =   7,
                                                        stale_sec   =   10,
                                                        key_col     =   'strike_price'
                                                    )

                if state != 'Success':
                    Log.error_msg("Yellow", f'Error in evaluate_put_sell: {state}', True)


    @staticmethod
    def evaluate_put_buy(SP_df, evaluation_parameters, start_flag, pause_flag):
        """
        Evaluates put buy conditions and updates flags in SP_df.
        Flags are set if new buy condition is met.
        """

        try:
            for index, row in SP_df.iterrows():
                obj         = row['Address'][0]  # Assumes Address holds a list-like container
                result      = obj.evaluate_SP_put_buy(evaluation_parameters)
                old_result  = row['put_buy_flag']

                if result and not old_result:
                    txt     = f'Entry Condition Satisfied for Buy || SP:{obj.strike_price} || put || {obj.expiry_date_txt}'
                    Log.info_msg("Green", txt, True)
                    SP_df.at[index, 'put_buy_flag']       = True
                    SP_df.at[index, 'put_buy_flag_time']  = datetime.now()

        except Exception as e:

            Log.error_msg("Yellow", f'Error in evaluate_put_buy: {e}', True)
            Log.error_msg("Yellow", traceback.format_exc(), True)

        finally:

                state = Global_function.reset_flag_conditionally(
                                                        SP_df,
                                                        flag_col    =   'put_buy_flag',
                                                        time_col    =   'put_buy_flag_time',
                                                        threshold   =   3,
                                                        wait_sec    =   7,
                                                        stale_sec   =   10,
                                                        key_col     =   'strike_price'
                                                    )

                if state != 'Success':
                    Log.error_msg("Yellow", f'Error in evaluate_put_buy: {state}', True)


    @staticmethod
    def evaluate_OP(start_flag, pause_flag):

        '''
        This function starts two threads to evaluate buy and sell of assets
        1)Buy thread further calls two functions each evaluating call_buy and put_buy function
        2)Sell thread similarly calls two functions to evaluate call and put sell
        3)each of call and put buy functions go through each strike price and check entry condition for in call and put
        4)each call and put buy calls a function which has set of functions to evaluate for suitable entry condition
        5)same flow is executed for call and buy sell operation
        6)result is used to update and toggle flag for buy and sell
        '''

        try:
            #Dummy Initialisation
            t1 = None
            t2 = None

            #Starting evaluation of Buy logic
            t1 = threading.Thread(target=Option_Chain.evaluate_OP_buy, args=(IP.En_L1,start_flag, pause_flag))

            #starting evaluation of sell logic
            #t2 = threading.Thread(target=Option_Chain.evaluate_OP_sell, args=(start_flag, pause_flag))

            #starting necessary threads
            t1.start()
            #t2.start()
            return [t1, t2]


        except Exception as e:

            txt = f'Error while executing evaluate_OP function.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

        finally:

                pass

    @staticmethod
    def evaluate_OP_buy(evaluation_parameters, start_flag, pause_flag):

        try:

            # wait for event start flag
            start_flag.wait()

            # while loop till start flag is set
            while start_flag.is_set():

                # store start_time
                start_time = datetime.now().time()

                # wait till Pause flag is set
                pause_flag.wait()

                #Dummy assignment of entry Logic
                active_logic = 1

                # get active logic
                if active_logic == 1:
                    evaluation_parameters = IP.En_L1

                elif active_logic == 2:
                    evaluation_parameters = IP.En_L2

                elif active_logic == 3:
                    evaluation_parameters = IP.En_L3

                elif active_logic == 4:
                    evaluation_parameters = IP.En_L4

                # get SP_df dataframe
                SP_df = Option_Chain.SP_df.copy()

                # perform call_buy evaluation
                call_r_value = Option_Chain.evaluate_call_buy(SP_df, evaluation_parameters, start_flag, pause_flag)

                # perform put_buy evaluation
                put_r_value = Option_Chain.evaluate_put_buy(SP_df, evaluation_parameters, start_flag, pause_flag)

                # wait for next second
                sleep_time = max(0, start_time.second + 1 - datetime.now().time().second)
                sleep(sleep_time)



        except Exception as e:

            txt = f'Error while executing evaluate_OP_buy function.\nError Details::{e}'
            Log.error_msg("Red", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    @staticmethod
    def evaluate_OP_sell(evaluation_parameters, start_flag, pause_flag):

        try:

            # wait for event start flag
            start_flag.wait()

            #Dummy assignment of entry Logic
            active_logic = 1
            # while loop till start flag is set
            while start_flag.is_set():

                # store start_time
                start_time = datetime.now().time()

                # wait till Pause flag is set
                pause_flag.wait()

                # get active logic
                if active_logic == 1:
                    evaluation_parameters = IP.En_L1

                elif active_logic == 2:
                    evaluation_parameters = IP.En_L2

                elif active_logic == 3:
                    evaluation_parameters = IP.En_L3

                elif active_logic == 4:
                    evaluation_parameters = IP.En_L4

                # get SP_df dataframe
                SP_df = Option_Chain.SP_df.copy()

                # perform call_buy evaluation
                r_value = Option_Chain.evaluate_call_sell(SP_df, evaluation_parameters, start_flag, pause_flag)

                # perform put_buy evaluation
                r_value = Option_Chain.evaluate_put_sell(SP_df, evaluation_parameters, start_flag, pause_flag)

                # Reset flags to remove stale triggers older than 10 seconds
                Global_function.reset_flag_conditionally(SP_df,
                                                         flag_col='put_sell_flag',
                                                         time_col='put_sell_flag_time')
                Global_function.reset_flag_conditionally(SP_df,
                                                         flag_col='call_sell_flag',
                                                         time_col='call_sell_flag_time')

                # wait for next second
                sleep_time = max(0, start_time.second + 1 - datetime.now().time().second)
                sleep(sleep_time)


        except Exception as e:

            txt = f'Error while executing evaluate_OP_sell function.\nError Details::{e}'
            Log.error_msg("Red", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    @staticmethod
    def graceful_exit():

        '''
        1)Check for time and market time is over then unsubscribe all data
        2)Save all the dataframes in each object
        '''
        global code_end_time, opt_data_folder

        try:
            # Part 1
            if datetime.now().time() > code_end_time:

                for each_sp_object in Option_Chain.SP_df["Address"]:
                    each_sp_object.un_subscribe_ticks('1second')

            # Part 2
            for each_sp_object in Option_Chain.SP_df["Address"]:

                if len(each_sp_object.call_df_sec) > 0:

                    each_sp_object.call_df_sec.to_excel(
                        f"{opt_data_folder}{each_sp_object.strike_price}_CE_{str(datetime.now().strftime('%d-%m-%Y-%I-%M %p'))}.xlsx",
                        index=False)

                else:

                    txt = f'{each_sp_object.strike_price} || CALL || 1second :: Dataframe does not have any data'
                    Log.debug_msg("Blue", txt, True)

                if len(each_sp_object.put_df_sec) > 0:

                    each_sp_object.put_df_sec.to_excel(
                        f"{opt_data_folder}{each_sp_object.strike_price}_PE_{str(datetime.now().strftime('%d-%m-%Y-%I-%M %p'))}.xlsx",
                        index=False)

                else:

                    txt = f'{each_sp_object.strike_price} || PUT  || 1second :: Dataframe does not have any data'
                    Log.debug_msg("Blue", txt, True)

        except Exception as e:

            txt = f'Exception while executing Option Chain graceful exit:{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    @staticmethod
    def summarize_strike_prices():

        try:

            global opt_data_folder

            # read and combine all call realted files
            df = Global_function.combine_files(opt_data_folder, "CE", summary_opt_folder)

            if len(df) > 0:

                df = df.sort_values(by=['strike_price', 'datetime'], ascending=[True, True])

                # find unique strike prices stored in the folder
                SP_list = df['strike_price'].drop_duplicates(keep='first')

                for each_sp in SP_list:
                    # since data is collected from the above master sheet it is believed that the dataframe contains atleast one genuine entry
                    # hence filtering and storing in an individual file for call and put

                    call_df = df[(df['strike_price'] == each_sp) & (df['right_type'] == 'CE')].sort_values(
                        by=['datetime'], ascending=True)
                    call_df.to_excel(
                        f"{summary_opt_folder}CE_{str(each_sp)}_{datetime.now().strftime('%d-%m-%Y-%I-%M %p')}.xlsx",
                        index=False)
                    txt = f'{each_sp} || CALL || Data combined and saved'
                    Log.info_msg("Green", txt, True)

                    put_df = df[(df['strike_price'] == each_sp) & (df['right_type'] == 'PE')].sort_values(
                        by=['datetime'], ascending=True)
                    put_df.to_excel(
                        f"{summary_opt_folder}PE_{str(each_sp)}_{datetime.now().strftime('%d-%m-%Y-%I-%M %p')}.xlsx",
                        index=False)
                    txt = f'{each_sp} || PUT  || Data combined and saved'
                    Log.info_msg("Green", txt, True)

                call_df = df[df['right_type'] == 'CE']
                put_df = df[df['right_type'] == 'PE']

                call_df.to_excel(
                    summary_opt_folder + "combined_CE_" + datetime.now().strftime("%d-%m-%Y-%I-%M %p") + ".xlsx",
                    index=False)
                txt = f'All CALL files have been saved'
                Log.info_msg("Green", txt, True)

                put_df.to_excel(
                    summary_opt_folder + "combined_PE_" + datetime.now().strftime("%d-%m-%Y-%I-%M %p") + ".xlsx",
                    index=False)
                txt = f'All PUT files have been saved'
                Log.info_msg("Green", txt, True)

            return "Success"

        except Exception as e:

            # traceback
            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return "Error"


class NIFTY_Index():
    Active_SP: float    = 0.0
    current_val: float  = 0.0
    support_band    = pd.DataFrame(columns=["Support_Band", "Touch_Strength", "Time_value", "Aggregate_Strength"])
    resistance_band = pd.DataFrame(columns=["Resistance_Band", "Touch_Strength", "Time_value", "Aggregate_Strength"])

    ticks_count = 0
    stock_name  = "NIFTY"
    df_sec      = pd.DataFrame(columns=['datetime', 'interval', 'exchange_code', 'stock_code',
                                   'open', 'high', 'low', 'close', 'volume'])
    df_min30    = pd.DataFrame(columns=['datetime', 'interval', 'exchange_code', 'stock_code',
                                     'open', 'high', 'low', 'close', 'volume'])
    hourly_df   = pd.DataFrame(columns=df_sec.columns)
    daily_df    = pd.DataFrame(columns=df_sec.columns)
    df_min      = pd.DataFrame(columns=df_sec.columns)
    df_lock     = threading.Lock()

    def __init__(self, stock_name1):

        return

    @classmethod
    def init_class_variables(cls, ):

        try:
            '''
            This function is suppose to generate below data
            1)read daily  OHLC data of NIFTY Index for last 3 years
            2)read hourly OHLC data of NIFTY Index for last 3 years
            3)Top up the data with latest Info of last day closing
            4)Compute support and resistance bands with the help of daily data
            5)Compute strength of support and resistance using hourly data
            6)Save this data to be used for day trading
            '''
            # initialize the variables

            global NIFTY_daily_folder, NIFTY_hourly_folder
            # add_hourly_df           = pd.DataFrame()
            add_daily_df = pd.DataFrame()


            # hourly_df   =   Global_function.combine_files(NIFTY_hourly_folder, "NIFTY","")
            daily_df = Global_function.combine_files(NIFTY_daily_folder, "NIFTY", "")


            # if (datetime.now().date()   -   hourly_df['datetime'].iloc[-1].date()).days >= 1:


            # add_hourly_df = NIFTY_Index.get_historical_data("30minute", hourly_df['datetime'].iloc[-1], datetime.now().replace(microsecond=0))

            if (datetime.now().date() - daily_df['datetime'].iloc[-1].date()).days >= 1:
                add_daily_df = NIFTY_Index.get_historical_data("1day", daily_df['datetime'].iloc[-1],
                                                               datetime.now().replace(microsecond=0))



            # if not add_hourly_df.empty:add_hourly_df.to_excel(f'{NIFTY_hourly_folder}NIFTY_Index_{hourly_df["datetime"].iloc[-1].strftime("%d-%m-%Y-%I-%M %p")}_to_{datetime.now().strftime("%d-%m-%Y-%I-%M %p")}.xlsx')
            if not add_daily_df.empty: add_daily_df.to_excel(
                f'{NIFTY_daily_folder}NIFTY_Index_{daily_df["datetime"].iloc[-1].strftime("%d-%m-%Y-%I-%M %p")}_to_{datetime.now().strftime("%d-%m-%Y-%I-%M %p")}.xlsx')


            # topping up additional data
            # hourly_df                   =   pd.concat([hourly_df,add_hourly_df],ignore_index=True)
            daily_df = pd.concat([daily_df, add_daily_df], ignore_index=True)

            support_level_list = Global_function.find_band_levels(daily_df, "support")
            resistance_level_list = Global_function.find_band_levels(daily_df, "resistance")

            support_bands = Global_function.create_bands(support_level_list, daily_df, 5)
            resistance_bands = Global_function.create_bands(resistance_level_list, daily_df, 5)

            support_bands = Global_function.combine_close_bands(support_bands, "price")
            resistance_bands = Global_function.combine_close_bands(resistance_bands, "price")

            support_bands = Global_function.evaluate_band_relevance(daily_df, support_bands, "support")
            resistance_bands = Global_function.evaluate_band_relevance(daily_df, resistance_bands, "resistance")

            # support_bands               =    Global_function.evaluate_band_strength(hourly_df,support_bands,"support")
            # resistance_bands            =    Global_function.evaluate_band_strength(hourly_df,resistance_bands,"resistance")


            NIFTY_Index.support_band = support_bands
            NIFTY_Index.resistance_band = resistance_bands

            NIFTY_Index.daily_df = daily_df
            # NIFTY_Index.hourly_df       =   hourly_df

            # subscribe to NIFTY stocks
            NIFTY_Index.subscribe_ticks("1second")
            NIFTY_Index.subscribe_ticks("1minute")

            return "Success"

        except Exception as e:

            txt = f'Exception while initialising NIFTY Index class Variables.\nError Details::{str(e)}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    @classmethod
    def subscribe_ticks(cls, interval):

        try:

            counter = 0
            status = {'message': None}

            while status != {'message':"Stock NIFTY subscribed successfully"} and counter < 5:
                status = breeze.subscribe_feeds(exchange_code="NSE", stock_code="NIFTY", interval=interval)
                sleep(0.2)
                counter += 1

            msg = f"Subscription done for NIFTY50 Index || {interval} || {status}"

            Log.debug_msg("Blue", msg, True)

            return 1


        except Exception as e:

            txt = f'Exception while subscribing NIFTY Index live stream at section'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            print(status)

            return 0

    @classmethod
    def download_historical_data(cls, interval, start_datetime, end_datetime):

        try:

            t1 = start_datetime.isoformat() + ".000Z"
            t2 = end_datetime.isoformat() + ".000Z"

            df = breeze.get_historical_data_v2(
                                                interval        =   interval,
                                                from_date       =   t1,         # "2022-08-15c
                                                to_date         =   t2,         # "2022-08-17T07:00:00.000Z"
                                                stock_code      =   "NIFTY",    # "ITC"
                                                exchange_code   =   "NSE",
                                                product_type    =   "cash"
                                            )

            df = pd.DataFrame(df["Success"]) if df['Status'] == 200 else []


            if len(df) > 0:
                df['interval'] = interval
                df = df[
                    ["datetime", "stock_code", "exchange_code", "open", "high", "low", "close", "volume", "interval"]]

                columns_to_convert = ['open', 'high', 'low', 'close', 'volume']


                for column in columns_to_convert:
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype(float)

                df['datetime'] = pd.to_datetime(df['datetime'])

            return df

        except Exception as e:

            txt = f'Exception while downloading historical data of NIFTY Index.\nError Details::{e} '
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    @classmethod
    def get_historical_data(cls, interval, start_datetime, end_datetime):

        try:

            '''
            This function downloads historic data of each of the NIFTY stocks and load it
            in the df_sec dataframe value of previous close so that analysis can be performed
            for money flow index/NIFTY weighted Avg.
            '''

            start_datetime  = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S") if type(
                                                        start_datetime) == str else start_datetime
            end_datetime    = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S") if type(
                                                            end_datetime) == str else end_datetime
            total_df        = pd.DataFrame()


            time_diff       = Global_function.calculate_time_delta(interval, start_datetime, end_datetime)
            time_limit      = int(time_diff) + 990


            if time_diff > 990:

                intermediate_datetime_1 = start_datetime
                intermediate_datetime_2 = Global_function.update_new_time(interval, start_datetime, 990)


                for time_delta in range(990, time_limit, 990):
                    print(
                        f"starting download of NIFTY_Index for time :: {intermediate_datetime_1} to {intermediate_datetime_2}")

                    df = NIFTY_Index.download_historical_data(interval, intermediate_datetime_1,
                                                              intermediate_datetime_2)



                    if len(df) > 0:
                        total_df = pd.concat([total_df, df], ignore_index=True)


                    intermediate_datetime_1 = intermediate_datetime_2
                    intermediate_datetime_2 = Global_function.update_new_time(interval, intermediate_datetime_2, 990)


            elif time_diff > 0:


                df          = NIFTY_Index.download_historical_data(interval, start_datetime, end_datetime)
                total_df    = pd.concat([total_df, df], ignore_index=True)


            else:

                print(f"difference between start_datetime and end_datetime is not sufficient")

            if len(total_df) > 0:

                total_df = total_df[(total_df['datetime'] >= start_datetime) & (total_df['datetime'] <= end_datetime)]
                total_df = total_df.drop_duplicates(keep='first')
                NIFTY_Index.downloaded_df = total_df

            return total_df

        except Exception as e:

            txt = f'Exception while downloading historical data'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    # noinspection PyMethodParameters
    @classmethod
    def trim_ticks_df_head(cls, interval, cut_off_time):

        try:

            # crop the df at the timeline
            '''
            1)Initiate all variables
            2)Validate dataframe shape and cut-off time format for archival
            3)lock the dataframe and filter data in the dataframe
            4)If there is data older than cut-off time then store it in system
            5)display and store relevant system message
            '''
            df =    []
            if interval == '1second':

                df = NIFTY_Index.df_sec

            elif interval == '1minute':

                df = NIFTY_Index.df_min

            elif interval == '30minute':

                df = NIFTY_Index.df_min30

            df_shape = df.shape[0] > 0
            datetime_format = isinstance(cut_off_time, datetime)
            counter = 0

            if df_shape == True and datetime_format == True:

                filtered_df = df[df['datetime'] >= cut_off_time]

                if filtered_df.empty:

                    txt = f"NIFTY Index does not have any data older than {cut_off_time}"
                    Log.error_msg("Red", txt, True)

                    txt = f"NIFTY Index || start_time = {df['datetime'].iloc[0]} end_time = {df['datetime'].iloc[-1]} "
                    Log.error_msg("Red", txt, True)

                else:

                    df = filtered_df

                    txt = f"NIFTY Index archived any data older than {cut_off_time} successfully for interval data of {interval} "
                    counter += 1
                    Log.info_msg("Green", txt, True)

            elif not df_shape:

                txt = "NIFTY Index dataframe is empty for interval {interval}"
                Log.error_msg("Red", txt, True)

            elif not datetime_format:

                txt = "{cut_off_time} data format is not valid. Datatype = {type(cut_off_time)}"
                Log.error_msg("Red", txt, True)

            return

        except Exception as e:

            txt = f'Error while Trimming NIFTY Index data.\n Error Details::{e}\nTrace back::{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    @classmethod
    def save_ticks_df_head(cls, interval, cut_off_time):

        try:

            global stk_data_folder
            # crop and save the data in local drive
            '''
            1)Initate all variables
            2)Validate dataframe shape and cut-off time format for archival
            3)lock the dataframe and filter data in the dataframe
            4)If there is data older than cut-off time then store it in system
            5)display and store relevant system message
            '''

            NIFTY_shape     = NIFTY_Index.df_sec.shape[0] > 0
            NIFTY_df        = NIFTY_Index.df_sec
            datetime_format = isinstance(cut_off_time, datetime)
            NIFTY_file_name = str(NIFTY_Index.stock_name) + str(cut_off_time.strftime("%d-%m-%Y-%I-%M %p")) + ".xlsx"
            counter         = 0

            if NIFTY_shape == True and datetime_format == True:

                filtered_df = NIFTY_df[NIFTY_df['datetime'] < cut_off_time]


                if filtered_df.empty:


                    txt = f"{NIFTY_Index.stock_name}|| NIFTY does not have any data older than {cut_off_time}"
                    Log.error_msg("Yellow", txt, True)
                    txt = f"{NIFTY_Index.stock_name}|| start_time = {NIFTY_df['datetime'].iloc[0]} end_time = {NIFTY_df['datetime'].iloc[-1]} "
                    Log.error_msg("Yellow", txt, True)

                else:


                    filtered_df.to_excel(stk_data_folder + NIFTY_file_name, index=False)
                    txt = f"{NIFTY_Index.stock_name}|| archieved any data older than {cut_off_time} successfully"
                    Log.info_msg("Green", txt, True)
                    counter += 1


            elif not NIFTY_shape:

                txt = "{self.strike_price} || NIFTY dataframe is empty"
                Log.error_msg("Yellow", txt, True)

            elif not datetime_format:

                txt = "{cut_off_time} data format is not valid. Datatype = {type(cut_off_time)}"

            return counter
        except Exception as e:

            txt = f'Exception while saving NIFTY Index ticks data.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return counter

    @classmethod
    def add_tick(cls, tick):

        try:

            '''
            Object function to add data and do necessary computation

            1) Correct data type of information in data_list
            2) Convert information into dataframe
            3) Identify the right type and create alias of respective data frame
            4) add the columns and compute the data for each item
            5) start lock method to access the dataframe and concat latest info into the df
            6) close/surrender the lock and return success/failure of function
            #1) Correct data type of information in data_list

            '''
            #Convert information into dataframe
            tick_df = pd.DataFrame([tick])

            # 3) Identify the right type and create alias of respective data frame

            if tick["exchange_code"] == "NSE" and tick["stock_code"] == "NIFTY" and tick["interval"] == '1second':

                #Updating NIFTY Values
                NIFTY_Index.current_val = tick["close"]
                NIFTY_Index.Active_SP = float(math.floor(tick["close"] / 50) * 50)

                with NIFTY_Index.df_lock:
                    NIFTY_Index.df_sec = pd.concat([NIFTY_Index.df_sec, tick_df], ignore_index=True, sort=False)
                    NIFTY_Index.ticks_count += 1


            elif tick["exchange_code"] == "NSE" and tick["stock_code"] == "NIFTY" and tick["interval"] == '30minute':

                with NIFTY_Index.df_lock:
                    NIFTY_Index.df_min30 = pd.concat([NIFTY_Index.df_min30, tick_df], ignore_index=True, sort=False)
                    NIFTY_Index.ticks_count += 1

            elif tick["exchange_code"] == "NSE" and tick["stock_code"] == "NIFTY" and tick["interval"] == '1minute':

                with NIFTY_Index.df_lock:
                    NIFTY_Index.df_min = pd.concat([NIFTY_Index.df_min, tick_df], ignore_index=True, sort=False)
                    NIFTY_Index.ticks_count += 1


            elif tick["exchange_code"] == "NSE" and tick["stock_code"] == "NIFTY" and tick["interval"] == '5minute':

                with NIFTY_Index.df_lock:
                    NIFTY_Index.df_min5 = pd.concat([NIFTY_Index.df_min5, tick_df], ignore_index=True, sort=False)
                    NIFTY_Index.ticks_count += 1


        except Exception as e:

            txt = f'Exception while storing tick in NIFTY Index.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    @classmethod
    def fill_dataframes(cls, entry_count):

        try:

            sample_tick = {'interval': '1second', 'exchange_code': 'NSE', 'stock_code': 'ADAENT', 'low': '24000',
                           'high': '25000', 'open': '24100', 'close': '24800', 'volume': '0',
                           'datetime': '2024-10-02 13:21:07'}
            sample_tick['datetime'] = datetime.now().replace(microsecond=0)

            for i in range(entry_count):
                sample_tick['stock_code'] = NIFTY_Index.stock_name
                sample_tick['datetime'] += timedelta(minutes=1)
                # NIFTY_Index.df_sec         =  pd.concat([NIFTY_Index.df_sec, pd.DataFrame([sample_tick])], ignore_index=True)
                Ticks.handler(sample_tick)
                sleep(0.1)
        except Exception as e:

            txt = f'Error While Loading NIFTY Index dataframe with dummy data.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    @classmethod
    def graceful_exit(cls, ):

        '''
        1)Check for time and market time is over then unsubscribe all data
        2)Save all the dataframes in each object
        '''
        global code_end_time, stk_data_folder

        try:
            # Part 1
            if datetime.now().time() > code_end_time:
                status = NIFTY_Index.un_subscribe_ticks('1second')
                status = NIFTY_Index.un_subscribe_ticks('1minute')

            # Part 2

            if len(NIFTY_Index.df_sec) > 0:

                NIFTY_Index.df_sec.to_excel(
                    f"{stk_data_folder}NIFTY50_1second_{str(datetime.now().strftime('%d-%m-%Y-%I-%M %p'))}.xlsx",
                    index=False)

            else:

                txt = f'NIFTY_Index.df_sec does not have any data'
                Log.debug_msg("Blue", txt, True)

            if len(NIFTY_Index.df_min) > 0:

                NIFTY_Index.df_min.to_excel(
                    f"{stk_data_folder}NIFTY50_1minute_{str(datetime.now().strftime('%d-%m-%Y-%I-%M %p'))}.xlsx",
                    index=False)

            else:

                txt = f'NIFTY_Index.df_min does not have any data'
                Log.debug_msg("Blue", txt, True)

            if len(NIFTY_Stocks.VWAP_indicator) > 0:

                NIFTY_Stocks.VWAP_indicator.to_excel(
                    f"{stk_data_folder}NIFTY50_VWAP_{str(datetime.now().strftime('%d-%m-%Y-%I-%M %p'))}.xlsx",
                    index=False)

            else:

                txt = f'NIFTY_Stocks.VWAP_indicator does not have any data'
                Log.debug_msg("Blue", txt, True)


        except Exception as e:

            txt = f'Exception while executing Otion Chain graceful exit:{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    @classmethod
    def un_subscribe_ticks(cls, interval):

        try:

            counter = 0
            status = {'message': None}

            while status.get('message') != "Stock NIFTY unsubscribed successfully" and counter < 5:
                status = breeze.unsubscribe_feeds(exchange_code="NSE", stock_code="NIFTY", interval=interval)

                txt = f"Attempt {counter}:Unsubscribing NIFTY50 Index||{interval}"

                Log.debug_msg("Blue", txt, True)

                counter += 1

                sleep(0.1)

            msg = f"Unsubscription done for NIFTY50 Index || {interval} || {status.get('message')}"

            Log.info_msg("Green", msg, True)

            return 1




        except Exception as e:

            txt = f'Exception while Un-subscribing NIFTY Index || {e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return 0


class NIFTY_Stocks():
    object_list = []

    stock_list = pd.DataFrame(
        columns=['tick_datetime', 'Address', "stock_name", "index_weg", "VWAP", "VWAP_height", 'close'])
    VWAP_df_sec = pd.DataFrame(
        columns=['datetime', 'tick_count', "VWAP_10", "VWAP_20", "VWAP_25", "VWAP_30", "VWAP_40", "VWAP_50"])
    VWAP_indicator = pd.DataFrame(
        columns=['datetime', 'VWAP Height(n=10)', 'stocks_downtrend count(n=10)', 'stocks_uptrend count(n=10)',
                 'VWAP Height(n=50)', 'stocks_downtrend count(n=50)', 'stocks_uptrend count(n=50)'])

    def __init__(self, stock_name1):

        self.ticks_count    = 0
        self.stock_name     = stock_name1
        self.index_weg: float()
        self.df_sec         = pd.DataFrame(columns=['datetime', 'interval', 'exchange_code', 'stock_code',
                                            'open', 'high', 'low', 'close', 'volume'])
        self.df_lock        = threading.Lock()
        self.Vol_total      = 0
        self.VWAP_total     = 0

        print(f'Object created for :{stock_name1}')

        return

    @classmethod
    def start_archiever(cls, cut_off_time, df_length, start_event, pause_event):

        try:

            for each_obj in NIFTY_Stocks.stock_list['Address']:

                if start_event.is_set():
                    each_obj.save_ticks_df_head(cut_off_time, df_length)
                    each_obj.trim_ticks_df_head(cut_off_time, df_length)


        except Exception as error_descriptor:

            Log.error_msg("Red",
                          f"NIFTY Stocks Auto‑archival crashed: {error_descriptor}\n{traceback.format_exc()}",
                          True)


        finally:

            Log.info_msg("Green",
                         f"Auto‑archival completed for NIFTY Stocks for time {cut_off_time}",
                         True)

        return

    @staticmethod
    def activate_auto_archival(cut_off_time):

        try:
            for each_obj in NIFTY_Stocks.stock_list['Address']:
                each_obj.save_ticks_df_head(cut_off_time)
                each_obj.trim_ticks_df_head(cut_off_time)

            return "Success"

        except Exception as e:

            txt = f'Error while doing Auto archival of NIFTY Stocks.Error Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return "Success"

    @classmethod
    def fill_dataframes(cls, entry_count):

        try:
            '''
            This function fills the dataframe with dummy data
            '''

            for each_obj in NIFTY_Stocks.stock_list['Address']:

                sample_tick = {'interval': '1second', 'exchange_code': 'NSE', 'stock_code': 'ADAENT', 'low': '3063.9',
                               'high': '3063.9', 'open': '3063.9', 'close': '3063.9', 'volume': '0',
                               'datetime': '2024-10-02 13:21:07'}
                sample_tick['datetime'] = datetime.now().replace(microsecond=0)

                for i in range(entry_count):
                    sample_tick['stock_code'] = each_obj.stock_name
                    sample_tick['datetime'] += timedelta(seconds=1)
                    each_obj.df_sec = pd.concat([each_obj.df_sec, pd.DataFrame([sample_tick])], ignore_index=True)

        except Exception as e:

            txt = f'Error while filling dataframe with dummy data of NIFTY Stocks.Error Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    @classmethod
    def calc_VWAP_height(cls, n_stocks):
        try:
            # Variable initialised
            height  = 0
            df      = NIFTY_Stocks.stock_list.copy()
            df      = df.fillna(0)
            n_stocks= min(49, n_stocks)

            # Calculation starts now
            for n in range(0, n_stocks, 1):
                height += (df['close'].iloc[n] - df['VWAP'].iloc[n]) * df['index_weg'].iloc[n]

            return round(height, 2)

        except Exception as e:

            txt = f'Error while calculating VWAP Height.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return 0

    @classmethod
    def VWAP_uptrend_counter(cls, n_stocks, threshold_pct):

        '''
         it shows how many stocks are above VWAP
        '''

        try:

            df      = NIFTY_Stocks.stock_list
            count   = 0

            for n in range(0, n_stocks, 1):

                stock_price      = df["close"].iloc[n]
                stock_VWAP       = df["VWAP"].iloc[n]
                VWAP_threshold   = stock_VWAP * (1 + threshold_pct)

                count += 1 if stock_price > VWAP_threshold else 0

            return count

        except  Exception as e:

            txt = f'Error while counting Uptrend Stock Count.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    @classmethod
    def VWAP_downtrend_counter(cls, n_stocks, threshold_pct):

        '''
         it shows how many stocks are below VWAP
        '''

        try:

            df = NIFTY_Stocks.stock_list
            count = 0

            for n in range(0, n_stocks, 1):
                stock_price = df["close"].iloc[n]
                stock_VWAP = df["VWAP"].iloc[n]
                VWAP_threshold = stock_VWAP * (1 + threshold_pct)

                count += 1 if stock_price < VWAP_threshold else 0

            return count

        except  Exception as e:

            txt = f'Error while counting down-trending Stocks Count.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    # Instance Method
    def add_tick(self, tick):

        try:

            if tick['interval'] == '1second':

                # 2) Convert information into dataframe
                tick_df = pd.DataFrame([tick])

                # 3) Update VWAP Values
                self.Vol_total += tick['volume']
                self.VWAP_total += tick['volume'] * tick['close']
                temp_VWAP = round(self.VWAP_total / self.Vol_total,2) if self.Vol_total > 0 else np.nan

                df = NIFTY_Stocks.stock_list
                df.loc[df['stock_name'] == self.stock_name, 'VWAP'] = temp_VWAP
                df.loc[df['stock_name'] == self.stock_name, 'datetime'] = tick['datetime']
                df.loc[df['stock_name'] == self.stock_name, 'close'] = tick['close']

                # 4) add the columns and compute the data for each item
                tick_df = Global_function.load_ticks_with_calc_in_df(self.df_sec,
                                                                     tick_df,
                                                                     int(IP.Active_en['N1'].iloc[-1]),
                                                                     int(IP.Active_en['N2'].iloc[-1]),
                                                                     temp_VWAP,
                                                                     self.index_weg)

                # 3) Identify the right type and create alias of respective data frame
                with self.df_lock:
                    self.df_sec = pd.concat([self.df_sec, tick_df], ignore_index=True, sort=False)
                    self.ticks_count += 1

            return

        except Exception as e:

            txt = f'Error while adding ticks in NIFTY Stocks.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    # Instance Method
    def trim_ticks_df_head(self, cut_off_time, cut_off_length):

        try:

            # crop the df at the timeline
            '''
            1)Initiate all variables
            2)Validate dataframe shape and cut-off time format for archival
            3)lock the dataframe and filter data in the dataframe
            4)If there is data older than cut-off time then store it in system
            5)display and store relevant system message
            '''

            stock_df        = self.df_sec
            stock_df_shape  = stock_df.shape[0] > 0
            datetime_format = isinstance(cut_off_time, datetime)
            counter         = 0

            if stock_df_shape == True and datetime_format == True:

                filtered_df = stock_df[stock_df['datetime'] >= cut_off_time]

                if filtered_df.empty:

                    txt = f"{self.stock_name} does not have any data older than {cut_off_time}"
                    Log.error_msg("Red", txt, True)

                    txt = f"{self.stock_name} || start_time = {stock_df['datetime'].iloc[0]} end_time = {stock_df['datetime'].iloc[-1]} "
                    Log.error_msg("Red", txt, True)

                else:

                    with self.df_lock:

                        self.df_sec = filtered_df

                    txt = f"{self.stock_name} trimmed any data older than {cut_off_time} successfully"
                    counter += 1
                    Log.info_msg("Green", txt, True)

            elif not stock_df_shape:

                txt = f"{self.stock_name} dataframe is empty"
                Log.error_msg("Yellow", txt, True)

            elif not datetime_format:

                txt = f"{cut_off_time} data format is not valid. Datatype = {type(cut_off_time)}"
                Log.error_msg("Yellow", txt, True)

            return

        except Exception as e:

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    # Instance Method
    def save_ticks_df_head(self, cut_off_time, cut_off_length):

        try:

            '''
            1)Initate all variables
            2)Validate dataframe shape and cut-off time format for archival
            3)lock the dataframe and filter data in the dataframe
            4)If there is data older than cut-off time then store it in system
            5)display and store relevant system message
            '''

            stock_df        = self.df_sec
            stock_df_shape  = stock_df.shape[0] > 0
            datetime_format = isinstance(cut_off_time, datetime)
            counter         = 0
            stock_file_name = str(self.stock_name) + str(cut_off_time.strftime("%d-%m-%Y-%I-%M %p")) + ".xlsx"

            if stock_df_shape == True and datetime_format == True:

                filtered_df = stock_df[stock_df['datetime'] < cut_off_time]

                if filtered_df.empty:

                    txt = f"{self.stock_name} does not have any data older than {cut_off_time}"
                    Log.error_msg("Red", txt, True)

                    txt = f"{self.stock_name}|| start_time = {stock_df['datetime'].iloc[0]} end_time = {stock_df['datetime'].iloc[-1]} "
                    Log.error_msg("Red", txt, True)

                else:

                    filtered_df.to_excel(stk_data_folder + stock_file_name, index=False)
                    txt = f"{self.stock_name} archieved any data older than {cut_off_time} successfully"
                    Log.info_msg("Green", txt, True)
                    counter += 1


            elif not stock_df_shape:

                txt = f"{self.stock_name} dataframe is empty"
                Log.error_msg("Yellow", txt, True)

            elif not datetime_format:

                txt = f"{cut_off_time} data format is not valid. Datatype = {type(cut_off_time)}"
                Log.error_msg("Yellow", txt, True)

            return counter

        except Exception as e:

            txt = f'Error while counting saving NIFTY Stocks data::{self.stock_name}.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    # Instance Method
    def download_historical_data(self, interval, start_datetime, end_datetime):

        try:

            t1 = start_datetime.isoformat() + ".000Z"
            t2 = end_datetime.isoformat() + ".000Z"

            df = breeze.get_historical_data_v2(
                                                interval        =   interval,
                                                from_date       =   t1,                 # "2022-08-15c
                                                to_date         =   t2,                 # "2022-08-17T07:00:00.000Z"
                                                stock_code      =   self.stock_name,    # "ITC"
                                                exchange_code   =   "NSE",
                                                product_type    =   "cash"
                                            )
            df = pd.DataFrame(df["Success"])

            if len(df) > 0:

                df = df[["datetime", "stock_code", "exchange_code", "open", "high", "low", "close", "volume"]]

                columns_to_convert = ['open', 'high', 'low', 'close', 'volume']

                for column in columns_to_convert:
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype(float)

                df['datetime'] = pd.to_datetime(df['datetime'])

            return df

        except Exception as e:

            txt = f'Error while downloading historic NIFTY Stocks data::{self.stock_name}.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    @classmethod
    def get_historical_data(cls, interval, start_datetime, end_datetime):

        try:

            start_datetime  = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
            end_datetime    = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S")
            total_df        = pd.DataFrame()
            time_diff       = Global_function.calculate_time_delta(interval, start_datetime, end_datetime)
            time_limit      = int(time_diff) + 990

            if len(NIFTY_Stocks.object_list) == 0:
                NIFTY_Stocks.initiate_class()

            for stock_instance in NIFTY_Stocks.object_list:

                total_df = pd.DataFrame()

                if time_diff > 990:

                    intermediate_datetime_1 = start_datetime
                    intermediate_datetime_2 = Global_function.update_new_time(interval, start_datetime, 990)

                    for time_delta in range(990, time_limit, 990):

                        df = stock_instance.download_historical_data(interval, intermediate_datetime_1,
                                                                     intermediate_datetime_2)

                        print(
                            f"starting download of {stock_instance.stock_name} stock for time :: {intermediate_datetime_1} to {intermediate_datetime_2}")

                        if len(df) > 0:
                            total_df = pd.concat([total_df, df], ignore_index=True)

                        if (time_delta + 990) > time_limit:
                            time_delta = time_limit - 991

                        intermediate_datetime_1 = intermediate_datetime_2
                        intermediate_datetime_2 = Global_function.update_new_time(interval, intermediate_datetime_2,
                                                                                  990)


                elif time_diff > 0:

                    total_df = pd.DataFrame()
                    df = stock_instance.download_historical_data(interval, start_datetime, end_datetime)
                    total_df = pd.concat([total_df, df], ignore_index=True)


                else:

                    print(f"difference between start_datetime and end_datetime is not sufficient")

                if len(total_df) > 0:
                    total_df = total_df.drop_duplicates(keep='first')
                    total_df = total_df[
                        (total_df['datetime'] >= start_datetime) & (total_df['datetime'] <= end_datetime)]
                    total_df = NIFTY_Stocks.calculate_VWAP_height(total_df)
                    total_df['Adj_VWAP'] = total_df['VWAP_height'] * stock_instance.index_weg
                    stock_instance.downloaded_df = total_df.copy()

                print(f'data successfully downloaded for ::{stock_instance.stock_name}')

            return "Success"


        except Exception as e:

            txt = f'error while downloading NIFTY stocks historical data {e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return

    @classmethod
    def initiate_class(cls):

        try:
            subscribe_list = Global_function.Get_pre_open_stock_list_NIFTY50()
            NIFTY_Stocks.stock_list = pd.concat((NIFTY_Stocks.stock_list, subscribe_list[["stock_name", "index_weg"]]),
                                                ignore_index=True, sort=False)

            for index_no in range(0, len(NIFTY_Stocks.stock_list), 1):

                try:

                    stock_name  = NIFTY_Stocks.stock_list['stock_name'].iloc[index_no]
                    index_weg   = NIFTY_Stocks.stock_list['index_weg'].iloc[index_no]

                    obj_NIF_Stocks = NIFTY_Stocks(stock_name)
                    NIFTY_Stocks.stock_list.loc[index_no, "Address"] = obj_NIF_Stocks
                    NIFTY_Stocks.stock_list['Address'].iloc[index_no].index_weg = index_weg

                    status = NIFTY_Stocks.stock_list['Address'].iloc[index_no].subscribe_ticks()


                except Exception as e:

                    txt = f"Issue with object creation and subscription of NIFTY STOCK {stock_name}.\nError Details::{e}"
                    Log.error_msg("Yellow", txt, True)

                    txt = f'{traceback.format_exc()}'
                    Log.error_msg("Yellow", txt, True)

            return

        except Exception as e:

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)
    @classmethod
    def unsubscribe_all_stocks(cls):

        try:
            stock_objects = NIFTY_Stocks.stock_list["Address"]

            for each_obj in stock_objects:
                each_obj.un_subscribe_ticks()

                sleep(0.1)

        except Exception as e:

            txt = f'Exception while unsubscribing stock data of {stock_objects.stock_name}.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    # Instance Method
    def subscribe_ticks(self,):

        try:
            counter = 0
            status = {'message': None}

            while status.get('message') != f"Stock {self.stock_name} subscribed successfully" and counter < 5:
                status = breeze.subscribe_feeds(exchange_code="NSE", stock_code=self.stock_name, interval="1second")
                sleep(0.1)
                counter += 1

                txt = f'Attempt {counter}:Subscribing {self.stock_name} || 1 second ticks data'
                Log.debug_msg("Blue", txt, True)

                # Stock BHAAIR subscribed successfully
                # Stock ITC subscribed successfully

            Log.info_msg("Yellow", status.get('message'), True)

            return status.get('message')

        except Exception as e:
            txt = f'Exception while subscribing stock data of {self.stock_name}.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

    # Instance Method
    def un_subscribe_ticks(self, ):

        try:
                counter = 0
                status = {'message': None}

                while status.get('message') != f"Stock {self.stock_name} unsubscribed successfully" and counter < 5:
                    status = breeze.unsubscribe_feeds(exchange_code="NSE", stock_code=self.stock_name, interval="1second")
                    sleep(0.1)
                    counter += 1

                txt = f"{status.get('message')}"
                Log.info_msg("Green", txt, True)

                return status.get('message')

        except Exception as e:

            txt = f'Exception while unsubscribing stock data of {self.stock_name}.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)


    @classmethod
    def graceful_exit(cls, ):

        global code_end_time, stk_data_folder

        try:
            # Part 1
            if datetime.now().time() > code_end_time:

                for each_stk_object in NIFTY_Stocks.stock_list["Address"]:
                    each_stk_object.un_subscribe_ticks()

            # Part 2
            for each_stk_object in NIFTY_Stocks.stock_list["Address"]:

                if len(each_stk_object.df_sec) > 0:

                    each_stk_object.df_sec.to_excel(
                        f"{stk_data_folder}{each_stk_object.stock_name}_{str(datetime.now().strftime('%d-%m-%Y-%I-%M %p'))}.xlsx",
                        index=False)

                else:

                    txt = f'{each_stk_object.stock_name} || 1second :: Dataframe does not have any data'
                    Log.debug_msg("Blue", txt, True)


        except Exception as e:

            txt = f'Exception while executing NIFTY_Stocks graceful exit:{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

        return

    @staticmethod
    def summarize_stock_data():

        try:

            global stk_data_folder

            for each_stock in NIFTY_Stocks.stock_list["Address"]:

                df = Global_function.combine_files(stk_data_folder, each_stock.stock_name, summary_stk_folder)

                df = df[df['stock_code'] == each_stock.stock_name]

                if len(df) > 0:

                    df = df.sort_values(by=['datetime'], ascending=True)

                    df.to_excel(
                        f'{summary_stk_folder}{each_stock.stock_name}_{datetime.now().strftime("%d-%m-%Y_%I-%M %p")}.xlsx',
                        index=False)
                    txt = f'{each_stock.stock_name} data stored in summary folder'
                    Log.info_msg("Green", txt, True)


                else:

                    txt = f'{each_stock.stock_name} dataframe does not have any data'
                    Log.info_msg("Yellow", txt, True)

            return "Success"


        except Exception as e:

            txt = f'Exception while preparing summary stock data.\nError Details::{e}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return "Failed"


# noinspection Annotator
class Order_book():
    Ac_bal: float = 0.0

    Trx_list = pd.DataFrame(columns=['OC_object', 'strike_price', 'right',
                                     '''Order entry details'''    'datetime', 'order_id', 'order_qty', 'order_type',
                                     'entry_price',
                                     '''NIFTY Index Values'''     'NIFTY_val',
                                     '''NIFTY VWAP VAL'''         'VWAP_50_val',  'VWAP_10_val',
                                     '''Order Sq.off details'''   'exit_order_id', 'exit_time', 'exit_price',
                                     '''Order Analysis details''' 'profit_pct', 'profit_amt', 'trx_cost'
                                     ]
                            )
    Active_order_list   = pd.DataFrame(columns=Trx_list.columns)
    buy_limit           = 5
    file_name           = datetime.now().strftime("%d-%m-%Y") + "_Transaction_details_" + str(
                          datetime.now().strftime("%I-%M %p")) + ".xlsx"

    def __init__(self, OC_object, right, T_count, quantity):
        return


    @staticmethod
    def add_order_detail_and_return_status(obj, right, breeze, order_id):

        '''
        return type
        1)Success
        2)Waiting to Execute
        3)Failed to Execute

        '''

        try:

            # order_id  =   202507043700000337
            # get the order order details command
            order_status = breeze.get_order_detail(exchange_code="NFO", order_id=order_id)

            # If order details are fetched correctly
            if order_status['Error'] != "No Data Found":

                # extract data
                order_details = order_status["Success"][0]

                # check if order is executed then add details in an order book else return status unsuccessful
                if order_details['status'] == 'Executed':

                    df = pd.DataFrame(columns=Order_book.Ac_Or.columns)
                    df.loc[0, 'datetime']       = datetime.now().replace(microsecond=0)
                    df.loc[0, 'entry_time']     = datetime.now().replace(microsecond=0)
                    df.loc[0, 'OC_object']      = obj
                    df.loc[0, 'strike_price']   = obj.strike_price
                    df.loc[0, 'right']          = right
                    df.loc[0, 'order_qty']      = order_details['quantity']
                    df.loc[0, 'order_type']     = order_details['action']
                    df.loc[0, 'order_id']       = order_id
                    df.loc[0, 'entry_price']    = order_details['price']
                    df.loc[0, 'NIFTY_val']      = NIFTY_Index.df_sec['close'].iloc[-1]

                    # capturing critical max information details, here max price =entry price since at time of entry all values are max price
                    df.loc[0, 'VWAP50'] = NIFTY_Stocks.VWAP_indicator['VWAP Height(n=50)'].iloc[-1]
                    df.loc[0, 'VWAP10'] = NIFTY_Stocks.VWAP_indicator['VWAP Height(n=10)'].iloc[-1]

                    Order_book.Ac_Or = pd.concat([Order_book.Ac_Or, df], ignore_index=True, sort=False)

                    txt = ' Order ID::{order_id} Executed successfully and added to Order Book'
                    Log.info_msg("Green", txt, True)

                    return "Success"

                else:

                    txt = ' Order ID::{order_id} failed to execute at limit price'
                    Log.info_msg("Red", txt, True)

                    return "Failed"

            else:

                txt = 'No Order details found for Order ID::{order_id}'
                Log.error_msg("Yellow", txt, True)

                return "Failed"


        except Exception as e:

            txt = f'Error while getting order details order ID::{order_id} \n{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)
            return "Failed"

        finally:

            '''return order status as Success or cancelled'''
            pass

    # noinspection Annotator
    @classmethod
    def calc_P_and_L(cls, start_datetime: datetime = None,
                     end_datetime: datetime = None):

        """
        Fetch trades for a time window and display P&L.

        Parameters
        ----------
        start_datetime : datetime  (optional)
            Beginning of the window. Defaults to today 00:00:00.
        end_datetime   : datetime  (optional)
            End of the window. Defaults to `datetime.now()`.
        """

        try:

            if start_datetime is None:
                today = date.today()
                start_datetime = datetime.combine(today, time.min)  # 00:00:00

            if end_datetime is None:
                end_datetime = datetime.now()

            fmt = "%Y-%m-%dT%H:%M:%S.000Z"
            from_date = start_datetime.strftime(fmt)
            end_date = end_datetime.strftime(fmt)  # this helps to capture datetime for end of day
            trade_list = breeze.get_trade_list(from_date=from_date,
                                               to_date=end_date,
                                               exchange_code="NFO",
                                               product_type="",
                                               action="",
                                               stock_code="")

            Transactions.display_P_and_L(trade_list)

        except Exception as e:

            txt = f'Error while calculating P_and_L \n{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)


        finally:

            txt = f'P_and_L calculation is now complete'
            Log.info_msg("Green", txt, True)

    @staticmethod
    def init_fund_mgmt(breeze):

        # get current fund allocation
        funds_allocation = breeze.get_funds()
        funds_allocation = funds_allocation['Success'] if funds_allocation['Status'] == 200 else 0

        # get total fno allocation
        fno_blocked_amt = funds_allocation['block_by_trade_fno'] if funds_allocation != 0 else 0
        fno_allocated_amt = funds_allocation['allocated_fno'] if funds_allocation != 0 else 0

        # check the total amount in the account
        total_fno_amt = math.floor(fno_blocked_amt + fno_allocated_amt)

        # If there are blocked funds then realloction will be done
        if fno_blocked_amt > 0:

            # Debit all the funds in FNO bucket to bank account
            status_1 = breeze.set_funds(transaction_type="debit",
                                        amount=str(total_fno_amt),
                                        segment="FNO")

            # check if reallocation is successful
            if status_1['Status'] == 200:

                txt = f"Funds transferred from FNO to Bank Account\nAmount::{total_fno_amt}"
                Log.info_msg("Green", txt, True)

                # Allocate funds back to FNO bucket so blocked amount can be used for trading
                status_2 = breeze.set_funds(transaction_type="credit",
                                            amount=str(total_fno_amt),
                                            segment="FNO")

                # checking if the reallocation is successful
                if status_2['Status'] == 200:

                    txt = f"Funds transferred back to FNO Successfully\nAmount::{total_fno_amt}"
                    Log.info_msg("Green", txt, True)


                else:

                    txt = f"Funds transfer from Bank Account to FNO Failed  \nError Reason ::  {status_1['Error']} \nAmount::{total_fno_amt}"
                    Log.error_msg("Red", txt, True)

            else:

                txt = f"Funds transfer from FNO to Bank Account Failed \nError Reason ::  {status_1['Error']} \nAmount::{total_fno_amt}"
                Log.error_msg("Red", txt, True)


        else:

            txt = f"Funds do not require reallocation"
            Log.info_msg("Green", txt, True)

    @staticmethod
    def balance_confirmation(fno_allocated_amt, buy_and_sell_price, Qty):

        try:

            # get total fno allocation
            funds_required = buy_and_sell_price * Qty

            # If neccessary funds are available
            if fno_allocated_amt > funds_required:

                return True

            else:

                return False

        except Exception as e:

            txt = f'Error while getting balance confirmation\nError details:: {e}'
            Log.error_msg("Red", txt, True)
            return False

    @classmethod
    def add_active_order_list_entry(cls, transaction_df, SP_obj):

        df = Order_book.Active_order_list
        new_row = len(df)

        df.loc[new_row, 'datetime'] = transaction_df['order_datetime']  # '05-Feb-2025 09:26'
        df.loc[new_row, 'OC_object'] = SP_obj
        df.loc[new_row, 'strike_price'] = SP_obj.strike_price
        df.loc[new_row, 'right'] = transaction_df['right']
        df.loc[new_row, 'order_id'] = transaction_df['order_id']  # '20250205N300001234'
        df.loc[new_row, 'order_qty'] = transaction_df['quantity']  # '1'
        df.loc[new_row, 'order_type'] = transaction_df['type']  # '1'
        df.loc[new_row, 'entry_price'] = transaction_df['price']  # '1'
        # df.loc[new_row,'VWAP_50']       =
        # df.loc[new_row,'VWAP_20']       =
        # df.loc[new_row,'VWAP_10']       =

        return "Success"

    @classmethod
    def update_active_orders(cls):

        df = Order_book.Active_order_list
        max_ctr = len(Order_book.Active_order_list)

        for each_entry in range(0, max_ctr, 1):
            obj = df.loc[each_entry, 'OC_object'].iloc[0]
            right = df.loc[each_entry, 'right'].iloc[0]
            price_df = obj.call_df_sec if right == "call" else obj.put_df_sec
            current_price = price_df["close"].iloc[0]
            entry_price = df.loc[each_entry, 'entry_price'].iloc[0]
            df.loc[each_entry, 'profit_pct'] = (current_price - entry_price) / entry_price
            df.loc[each_entry, 'profit_amt'] = (current_price - entry_price)

        return "Success"

    @classmethod
    def remove_active_order_list_entry(cls, order_id):

        df = Order_book.Active_order_list

        df = df[df['order_id'] != order_id]

        return "Success"

    @classmethod
    def add_transaction_item(cls, transaction_df, SP_obj):

        df = Order_book.Trx_list
        new_row = len(df)

        df.loc[new_row, 'datetime'] = transaction_df['order_datetime']  # '05-Feb-2025 09:26'
        df.loc[new_row, 'OC_object'] = SP_obj
        df.loc[new_row, 'strike_price'] = SP_obj.strike_price
        df.loc[new_row, 'right'] = transaction_df['right']
        df.loc[new_row, 'order_id'] = transaction_df['order_id']  # '20250205N300001234'
        df.loc[new_row, 'order_qty'] = transaction_df['quantity']  # '1'
        df.loc[new_row, 'order_type'] = transaction_df['type']  # '1'
        df.loc[new_row, 'entry_price'] = transaction_df['price']  # '1'

        return "Success"

    @classmethod
    def update_transaction_item(cls, order_id, transaction_df):

        df = Order_book.Trx_list
        new_row = len(df)
        SP_obj = df[df['order_id'] == order_id, 'OC_object'].iloc[0]
        df.loc[new_row, 'datetime'] = transaction_df['order_datetime']  # '05-Feb-2025 09:26'
        df.loc[new_row, 'strike_price'] = SP_obj.strike_price
        df.loc[new_row, 'right'] = transaction_df['right']
        df.loc[new_row, 'order_id'] = transaction_df['order_id']  # '20250205N300001234'
        df.loc[new_row, 'order_qty'] = transaction_df['quantity']  # '1'
        df.loc[new_row, 'order_type'] = transaction_df['type']  # '1'
        df.loc[new_row, 'entry_price'] = transaction_df['price']  # '1'
        return "Success"

    @classmethod
    def remove_transaction_item(cls, order_id):

        df = Order_book.Trx_list
        new_row = len(df)
        df = df[df['order_id'] != order_id]
        return "Success"

    @classmethod
    def determine_buy_qty(cls, balance_amount, last_price, lot_size):

        lot_Q = math.floor(balance_amount / (last_price * lot_size))
        Q = max(lot_Q, 0)
        Qty = Q * lot_size

        return Qty

    @classmethod
    def get_fno_balance(cls, ):
        try:
            global breeze

            # get current fund allocation
            funds_allocation = breeze.get_funds()
            funds_allocation = funds_allocation['Success'] if funds_allocation['Status'] == 200 else 0

            # get total fno allocation
            fno_blocked_amt = funds_allocation['block_by_trade_fno'] if funds_allocation != 0 else 0
            fno_allocated_amt = funds_allocation['allocated_fno'] if funds_allocation != 0 else 0

            return {'allocated_amt': fno_allocated_amt, 'blocked_amt': fno_blocked_amt}

        except Exception as e:

            txt = f'Error while getting fno balance from ICICI Direct server \nError Details:: {e}'
            Log.error_msg("Red", txt, True)

            return {'allocated_amt': 0, 'blocked_amt': 0}

    @classmethod
    def determine_sell_qty(cls, fno_avl_balance, strike_price, quantity, right, expiry_date_txt, stock_code, price=""):

        order_type = 'market' if price == "" else "limit"

        message = breeze.margin_calculator(
            [
                {
                    "strike_price": strike_price,
                    "quantity": quantity,
                    "right": right,
                    "product": "options",
                    "action": "sell",
                    "price": price,
                    "expiry_date": expiry_date_txt,
                    "stock_code": stock_code,
                    "cover_order_flow": "sell",
                    "fresh_order_type": order_type,
                    "cover_limit_rate": "0",
                    "cover_sltp_price": "0",
                    "fresh_limit_rate": "0",
                    "open_quantity": "0"
                }
            ], exchange_code="NFO"
        )
        # system margin requirement
        margin_req = float(message['Success']['span_margin_required'])

        # adjusted margin requirement
        margin_req *= 1.005

        # determine quantity required
        qty = math.floor(fno_avl_balance / margin_req) * quantity

        return qty

    @classmethod
    def get_avl_balance(cls):

        section = 1
        b = breeze.get_funds()

        temp = b["Success"]['allocated_fno'] + b["Success"]["block_by_trade_balance"]
        Order_book.Ac_bal = float(temp)

        return Order_book.Ac_bal


class Display():
    NIFTY_Index_data = pd.DataFrame(columns=['ClosePrice'])
    NIFTY_Stocks_data = pd.DataFrame(
        columns=['VWAP Height(n=10)', 'stocks_downtrend count(n=10)', 'stocks_uptrend count(n=10)', 'VWAP Height(n=50)',
                 'stocks_downtrend count(n=50)', 'stocks_uptrend count(n=50)'])
    Current_time = datetime.now()
    Trade_and_profit = []
    Thread_status = pd.DataFrame(columns=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'])
    Tick_count = 0
    Code_messages = pd.DataFrame(columns=['datetime', 'messages'])
    Trade_list = pd.DataFrame(
        columns=['Strike_price', 'Call/Put', 'Trx_Type', 'Qty', 'Entry_Time', 'Sq_off_time', 'Entry_price',
                 'Sq_off_price', 'P/L Amount'])
    Trigger_list = pd.DataFrame(columns=['datetime', 'Active_Logic', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'])
    button_count = pd.DataFrame(
        columns=['Entry_Block_btn', 'Exit_Block_btn', 'Liquidate_All_btn', 'Code_deactivate_btn'])
    today_daterange = datetime.now()
    today_NIF_range = []

    def __init__(self):
        return

    @staticmethod
    def run_app(app):

        app.run(debug=True, port=8050, use_reloader=False)
        return

    @classmethod
    def update_VWAP(cls):

        df = Display.NIFTY_Stocks_data

        df.loc[0, 'VWAP Height(n=10)'] = NIFTY_Stocks.calc_VWAP_height(10)
        df.loc[0, 'stocks_downtrend count(n=10)'] = NIFTY_Stocks.VWAP_downtrend_counter(10, -0.0005)
        df.loc[0, 'stocks_uptrend count(n=10)'] = NIFTY_Stocks.VWAP_uptrend_counter(10, 0.0005)
        df.loc[0, 'VWAP Height(n=50)'] = NIFTY_Stocks.calc_VWAP_height(50)
        df.loc[0, 'stocks_downtrend count(n=50)'] = NIFTY_Stocks.VWAP_downtrend_counter(50, -0.0005)
        df.loc[0, 'stocks_uptrend count(n=50)'] = NIFTY_Stocks.VWAP_uptrend_counter(50, 0.0005)

        Display.NIFTY_Stocks_data = df.copy()

        df.loc[0, 'datetime'] = datetime.now().replace(microsecond=0)

        if len(NIFTY_Stocks.VWAP_indicator) > 0:

            if NIFTY_Stocks.VWAP_indicator['datetime'].iloc[-1] < datetime.now().replace(microsecond=0):
                NIFTY_Stocks.VWAP_indicator = pd.concat([NIFTY_Stocks.VWAP_indicator, df], ignore_index=True,
                                                        sort=False)
        else:

            NIFTY_Stocks.VWAP_indicator = pd.concat([NIFTY_Stocks.VWAP_indicator, df], ignore_index=True, sort=False)

    @classmethod
    def update_charts_n_intervals(cls, relayout_data1, relayout_data2):

        try:
            # Your logic for updating charts based on the interval
            # You can fetch new data or perform calculations here
            # ...
            global queue_length

            previous_close = NIFTY_Index.daily_df['close'].iloc[-1]
            current_value = NIFTY_Index.current_val if NIFTY_Index.current_val > 0 else \
            NIFTY_Index.daily_df['close'].iloc[-1]

            up_down = round(current_value - previous_close, 2)

            Display.NIFTY_Index_data.loc[0, 'ClosePrice'] = f'{current_value}({up_down})'

            # Capture the current x-axis range from the range slider of candlesticks chart 1 and chart 2
            range1 = Dashboard.get_chart_axis_range(relayout_data1)
            range2 = Dashboard.get_chart_axis_range(relayout_data2)

            # NIFTY df update
            bands = Global_function.identify_plot_band(NIFTY_Index.support_band, NIFTY_Index.resistance_band,
                                                       NIFTY_Index.df_sec, NIFTY_Index.df_min, NIFTY_Index.daily_df)

            # filtering one second data only
            # nif_df_sec = NIFTY_Index.df_sec[NIFTY_Index.df_sec['datetime']>NIFTY_Index.df_min['datetime'].iloc[-1]] if len(NIFTY_Index.df_min) >0 else NIFTY_Index.df_sec

            # Example: Update candlestick chart 1 and chart 2
            candlestick_fig1 = Dashboard.plot_candlestick_chart(NIFTY_Index.daily_df, range1, bands, None,
                                                                Display.today_NIF_range, None, False)
            candlestick_fig2 = Dashboard.plot_nifty_50_live_chart(NIFTY_Index.df_sec, NIFTY_Index.df_min, None, range2,
                                                                  bands)

            # VWAP indicator value
            NIFTY_Stocks_info = Dashboard.publish_table(Display.NIFTY_Stocks_data)

            # NIFTY50 value of Index
            NIFTY_Index_info = Dashboard.publish_table(Display.NIFTY_Index_data)

            # Option Chain Display
            SP_list = Dashboard.publish_table(Option_Chain.SP_df)
            stock_list = Dashboard.publish_table(NIFTY_Stocks.stock_list)

            new_label1 = f"Time :: {datetime.now().strftime('%dth %b, %Y %H:%M:%S')}  T-Count ::{Ticks.tick_count} Ticks Pipeline:{queue_length}"  # Example: Update label text
            new_label2 = f"Profit/Loss :: 10,00,000"  # Example: Update label text

            SP_charts = []

            OP_objects = Option_Chain.SP_df[Option_Chain.SP_df['focus_SP'] == True]['Address'].head(5)

            for op_object in OP_objects:

                try:

                    SP_charts.append(
                        Dashboard.plot_dual_chart(op_object.call_df_sec,
                                                  op_object.put_df_sec,
                                                  op_object.strike_price,
                                                  "Green",
                                                  "red"))

                except Exception as e:

                    # Log.error_msg(f"SP Chart plot error: {e}", True)
                    SP_charts.append(go.Figure())  # or use dash.no_update

            # if SP_charts does not have more than 5 objects
            while len(SP_charts) < 5:
                SP_charts.append(go.Figure())

            return (candlestick_fig1, candlestick_fig2, new_label1, new_label2, NIFTY_Index_info, NIFTY_Stocks_info,
                    *SP_charts, SP_list, stock_list)

        except Exception as e:

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            print("Update failed:", e)
            # Return no_update for all 13 outputs to avoid error
            return [dash.no_update] * 13

    @classmethod
    def update_charts_button_click(cls, Entry_Block_btn, Exit_Block_btn, Liquidate_All_btn, Code_deactivate_btn):

        # Your logic for updating charts based on button clicks
        # ...
        # global submit_button_clicks
        # buttons keep incrementing the counter automatically once triggered

        row = len(Display.button_count)

        # Step 1: Log the button states into the DataFrame
        current_index = len(Display.button_count)

        Display.button_count.loc[current_index] = {
            'Entry_Block_btn': Entry_Block_btn,
            'Exit_Block_btn': Exit_Block_btn,
            'Liquidate_All_btn': Liquidate_All_btn,
            'Code_deactivate_btn': Code_deactivate_btn
        }

        # Step 2: Define the logic for detecting new clicks
        def is_new_click(button_name):
            if current_index == 0:
                return True  # First row — treat all buttons as newly clicked
            return Display.button_count[button_name].iloc[current_index] > Display.button_count[button_name].iloc[
                current_index - 1]

        # Step 3: Handle each button click individually

        # Entry Block button
        if is_new_click('Entry_Block_btn'):
            entry_active = Entry_Block_btn % 2 != 0
            IP.ctrl_ptrs.loc[-1, 'Call_Entry'] = int(entry_active)
            IP.ctrl_ptrs.loc[-1, 'Put_Entry'] = int(entry_active)
            print(
                f"{'Call & Put entry block activated' if entry_active else 'Call & Put entry block deactivated'} : {Entry_Block_btn}")

        # Exit Block button
        if is_new_click('Exit_Block_btn'):
            exit_active = Exit_Block_btn % 2 != 0
            IP.ctrl_ptrs.loc[-1, 'Call_Exit'] = int(exit_active)
            IP.ctrl_ptrs.loc[-1, 'Put_Exit'] = int(exit_active)
            print(
                f"{'Call & Put exit block activated' if exit_active else 'Call & Put exit block deactivated'} : {Exit_Block_btn}")

        # Code Shutdown button
        if is_new_click('Code_deactivate_btn'):
            shutdown_active = Code_deactivate_btn % 2 != 0
            IP.ctrl_ptrs.loc[-1, 'Main_Program'] = int(shutdown_active)
            print(
                f"{'Code standby mode activated' if shutdown_active else 'Code standby mode deactivated'} : {Code_deactivate_btn}")

        # Liquidate All button
        if is_new_click('Liquidate_All_btn'):
            liquidate_active = Liquidate_All_btn % 2 != 0
            Exit_Scanners.square_off_all_holdings()

            #IP.ctrl_ptrs.loc[-1, 'Liquidate'] = int(liquidate_active)
            print(
                f"{'Liquidate all assets activated' if liquidate_active else 'Liquidate all assets deactivated'} : {Liquidate_All_btn}")

        return

    @staticmethod
    def update_check_box_values(*checkbox_value):

        SP_obj_list = Option_Chain.SP_df[Option_Chain.SP_df['focus_SP'] == True]['Address'].head(5)

        print(f'check box update function activated')
        print(f'{checkbox_value}')

        if len(SP_obj_list) > 0:

            for i in range(0, len(SP_obj_list), 1):

                if isinstance(SP_obj_list[i].call_df_sec, pd.DataFrame):

                    status = Global_function.update_SP_flags(SP_obj_list[i], checkbox_value[i * 4],
                                                             checkbox_value[i * 4 + 1], checkbox_value[i * 4 + 2],
                                                             checkbox_value[i * 4 + 3])
                    Log.info_msg("Green", status, True)

                else:

                    txt = f"Focus object list is invalid :: {SP_obj_list[i].strike_price}"

                    Log.info_msg("Green", txt, True)



        else:

            txt = f"Focus_SP_List is empty"
            Log.info_msg("Green", txt, True)

        return "Success"

    @staticmethod
    def update_active_logic(entry_func_dropdown_val, exit_func_dropdown_val):

        print(f'Active Entry Logic is {entry_func_dropdown_val} and Active Exit Logic is {exit_func_dropdown_val}')

        # current entry logic
        Active_en_logic = IP.ctrl_ptrs['Active_Entry_Logic'].iloc[-1]
        Active_ex_logic = IP.ctrl_ptrs['Active_Exit_Logic'].iloc[-1]
        df = pd.DataFrame(columns=IP.ctrl_ptrs.columns)

        df.loc[0, 'datetime'] = datetime.now().replace(microsecond=0)

        # Check if new selected entry logic is not the same as old logic
        C1 = Active_en_logic != entry_func_dropdown_val
        C2 = Active_ex_logic != exit_func_dropdown_val

        if C1 == True or C2 == True:
            df.loc[0, 'Active_Entry_Logic'] = entry_func_dropdown_val
            df.loc[0, 'Active_Exit_Logic'] = exit_func_dropdown_val

            IP.ctrl_ptrs = pd.concat([IP.ctrl_ptrs, df], ignore_index=True, sort=False)

        return "Success"

    @staticmethod
    def built_dashboard():

        # Initialize Dash app
        app = dash.Dash(__name__)

        plot_count = 5

        # Dynamically generate Input list for callback
        checkbox_inputs = [Input(f'SP{i}_{opt}', 'value') for i in range(1, plot_count + 1) for opt in
                           ['call_buy', 'call_sell', 'put_buy', 'put_sell']]
        checkbox_id = [f'SP{i}_{opt}' for i in range(1, plot_count + 1) for opt in
                       ['call_buy', 'call_sell', 'put_buy', 'put_sell']]

        Entry_dropdown_option_df = Dashboard.create_dropdown_list("Entry Logic", 4)
        Exit_dropdown_option_df = Dashboard.create_dropdown_list("Exit Logic", 4)

        # Layout of the app
        app.layout = html.Div(
            [
                html.Div(
                    [
                        Dashboard.create_dash_dropdown("Entry_Logic", Entry_dropdown_option_df, 0),
                        Dashboard.create_dash_dropdown("Exit_Logic", Exit_dropdown_option_df, 0),

                        Dashboard.create_dash_buttons('Entry_Block-btn', 'Entry Block'),
                        Dashboard.create_dash_buttons('Exit_Block-btn', 'Exit Block'),
                        Dashboard.create_dash_buttons('Liquidate_All-btn', 'Liquidate All'),
                        Dashboard.create_dash_buttons('Code_deactivate-btn', 'Main Program OFF'),

                        Dashboard.create_dash_label('Label 1', 'New NIFTY Index Data'),
                        Dashboard.create_dash_label('Label 2', 'New NIFTY stock Data')

                        # n_clicks save number of times button is pressed for each buttons seperately
                    ], style={'display': 'flex', 'align-items': 'center', 'marginBottom': '20px'}
                ),

                html.Div(
                    [

                        Dashboard.create_dash_table('table_NIFTY_Index', Display.NIFTY_Index_data, "100%", 10),

                    ], style={'display': 'inline-block', 'marginLeft': '10px', 'marginRight': '10px',
                              'marginBottom': '5px'}

                ),

                html.Div(
                    [
                        Dashboard.create_dash_table('table_NIFTY_Stocks', Display.NIFTY_Stocks_data, "100%", 10),
                    ], style={'display': 'inline-block', 'marginLeft': '10px', 'marginRight': '10px',
                              'marginBottom': '5px'}
                ),

                # Candlestick Charts
                html.Div(
                    [

                        Dashboard.create_dash_candlestick_chart('candlestick-chart1'),
                        Dashboard.create_dash_candlestick_chart('candlestick-chart2'),

                    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%'}
                ),

                html.Div(
                    [
                        dcc.Checklist(
                            options=[{'label': 'Freeze_Focus_List_Update',
                                      'value': True}],
                            id='Freeze_Focus_List_Update',
                            value=[],
                            style={
                                'margin-bottom': '5px',
                                'fontSize': '15px',  # Increased font size
                                'color': 'white',
                                'transform': 'scale(1.5)'  # Enlarges checkbox
                            }, inputStyle={'margin-right': '10px'}  # Adds spacing between checkbox and label
                        ),
                    ], style={
                        'display': 'flex',
                        'justifyContent': 'center',  # Center-aligns horizontally
                        'alignItems': 'center',  # Center-aligns vertically
                        'width': '90%',
                        'marginTop': '15px',
                        'marginBottom': '15px',
                        'padding': '15px',
                        'border': '2px solid grey',
                        'backgroundColor': '#2C5F2D',
                        'borderRadius': '5px'
                    }
                ),

                html.Div(
                    [
                        Dashboard.create_dual_line_plot_set(plot_count),

                    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%'}
                ),

                html.Div(children=[

                    html.Div(
                        [html.H2("Option Chain Tree", style={'textAlign': 'center'}),

                         Dashboard.create_dash_table('Option_Chain_Tree', Option_Chain.SP_df, "100%", 10),

                         ], style={'flex': '1'}),  # Use flex: 1 for equal height and take available space

                    html.Div(
                        [html.H2("Option Chain Tree", style={'textAlign': 'center'}),

                         Dashboard.create_dash_table('NIFTY_Stocks_List', NIFTY_Stocks.stock_list, "100%", 10),

                         ], style={'flex': '1'}),

                ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start',
                          'justifyContent': 'space-around', 'width': '100%', 'marginBottom': '20px'}

                ),  # Parent Div for horizontal alignment, align items to the top)

                html.Div(
                    [  # No. of times to be refreshed in one second
                        Dashboard.create_interval('interval-component', 1)
                    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%'}
                ),

            ]
        )

        def update_charts(En_fun_sel, Ex_fun_sel, En_blk_btn, Ex_blk_btn, Liq_all_btn, de_act_btn, Frz_fcs_Lst_Updte,
                          n_intervals, relayout_data1, relayout_data2, *checkbox_values):

            # This object provides information about which input triggered the callback
            ctx = dash.callback_context

            # If no input triggered the callback (e.g., during initial rendering), it returns dash.no_update for all outputs,
            # preventing any unnecessary updates.
            if not ctx.triggered:
                return [dash.no_update] * 13

            # If the callback was triggered, this line extracts the ID of the component that triggered it.
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'interval-component':
                # print(f'printing based on regular interval component')
                return Display.update_charts_n_intervals(relayout_data1, relayout_data2)


            elif trigger_id in ['Select_logic-btn', 'Entry_Block-btn', 'Exit_Block-btn', 'Liquidate_All-btn',
                                'Code_deactivate-btn']:

                print(f'printing based on button press')
                return_value2 = Display.update_charts_button_click(En_blk_btn, Ex_blk_btn, Liq_all_btn, de_act_btn)

                return [dash.no_update] * 13


            elif trigger_id in ['Entry_Logic', 'Exit_Logic']:

                print(f'printing based on dropdown selection')
                return_value3 = Display.update_active_logic(En_fun_sel, Ex_fun_sel)

                return [dash.no_update] * 13


            elif trigger_id in checkbox_id:

                print(f'printing based on checkbox selection')
                status = Display.update_check_box_values(*checkbox_values)

                return [dash.no_update] * 13

            elif trigger_id in ['Freeze_Focus_List_Update']:

                print(f'Focus SP List Freezed')

                return [dash.no_update] * 13

            else:

                print(f'some other tirgger has been activated')

                return [dash.no_update] * 13

            return

        app.callback(
            [
                Output('candlestick-chart1', 'figure'),
                Output('candlestick-chart2', 'figure'),
                Output('Label 1', 'children'),
                Output('Label 2', 'children'),
                Output('table_NIFTY_Index', 'data'),
                Output('table_NIFTY_Stocks', 'data'),
                Output('L1', 'figure'),
                Output('L2', 'figure'),
                Output('L3', 'figure'),
                Output('L4', 'figure'),
                Output('L5', 'figure'),
                Output('Option_Chain_Tree', 'data'),
                Output('NIFTY_Stocks_List', 'data')
            ],
            [
                Input('Entry_Logic', 'value'),
                Input('Exit_Logic', 'value'),
                Input('Entry_Block-btn', 'n_clicks'),
                Input('Exit_Block-btn', 'n_clicks'),
                Input('Liquidate_All-btn', 'n_clicks'),
                Input('Code_deactivate-btn', 'n_clicks'),
                Input('Freeze_Focus_List_Update', 'value'),
                Input('interval-component', 'n_intervals'),

            ],
            [
                State('candlestick-chart1', 'relayoutData'),
                State('candlestick-chart2', 'relayoutData')
            ],

            [
                *checkbox_inputs
            ]
        )(update_charts)  # Apply the callback to the function

        return app


class Entry_Scanners():
    dic = []

    def __init__(self, OC_object, right, T_count, quantity):

        return

    @staticmethod
    def Start_scanning(start_event, pause_event):
        try:
            txt = f'Waiting for start event to set for Entry scanner'
            Log.debug_msg("Blue",txt,True)


            txt = f'Start event for Entry scanner is set beginning Execution'
            Log.debug_msg("Blue",txt,True)

            t1 = threading.Thread(target=Entry_Scanners.trade_in,
                                  args=(start_event, pause_event,)
                                  )

            t1.start()

            return t1


        except Exception as e:

            txt = f'Error while executing Entry Scanners start scanning thread function \nError Desc ::{e} \n{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

        finally:

            txt = f'Entry Scanners start scanning thread function has successfully completed the execution'
            Log.debug_msg("Blue", txt, True)

    @staticmethod
    def trade_in(start_event, pause_event):

        try:

            # create a copt of SD_df

            start_event.wait()

            global start_events, pause_events

            while start_event.is_set():

                SP_df = Option_Chain.SP_df

                # wait till pause event is not set
                pause_event.wait()

                # Thread1     =   threading.Thread(target = Entry_Scanners.buy_trade_in,args = (SP_df,))
                # Thread2     =   threading.Thread(target = Entry_Scanners.sell_trade_in,args = (SP_df,))

                # function will return True if any trade is executed else false
                ret_buy = Entry_Scanners.buy_trade_in(start_event, pause_event, SP_df)
                ret_sell = Entry_Scanners.sell_trade_in(start_event, pause_event, SP_df)

                # if any trade is executed, then some threads are paused
                if ret_buy or ret_sell == True:
                    # If entry condition is positive, then pause
                    # evaluation of entry conditions and pause entry scanner
                    # all resources should be diverted to square_off scanner

                    pause_events['evaluate_scanner'].is_set()
                    pause_events['entry_scanner'].is_set()
                    pause_events['trade_entry'].is_set()

                sleep(0.5)


        except Exception as e:

            txt = f'Trade_in function is returning error \nError Desc ::{e} \n{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)


        finally:

            txt = f'Trade_in function execution is completed'
            Log.error_msg("Yellow", txt, True)

    @staticmethod
    def buy_trade_in(start_event, pause_event, SP_df):

        try:

            call_buy_sum = SP_df['call_buy_flag'].sum()
            put_buy_sum = SP_df['put_buy_flag'].sum()

            A1 = Entry_Scanners.call_buy_trade_in(SP_df, call_buy_sum, put_buy_sum)
            A2 = Entry_Scanners.put_buy_trade_in(SP_df, call_buy_sum, put_buy_sum)


            if A1 or A2 ==  True:

                return True

            else:

                return False

        except Exception as e:

            txt = f'buy_trade_in function is returning error \nError Desc ::{e} \n{traceback.format_exc()}'
            #Log.error_msg("Yellow", txt, True)

        finally:

            txt = f'buy_trade_in function execution is completed'
            #Log.error_msg("Yellow", txt, True)

    @staticmethod
    def sell_trade_in(start_event, pause_event, SP_df):

        try:

            call_sell_sum = SP_df['call_sell_flag'].sum()
            put_sell_sum = SP_df['put_sell_flag'].sum()
            A1 = Entry_Scanners.call_sell_trade_in(SP_df, call_sell_sum, put_sell_sum)
            A2 = Entry_Scanners.put_sell_trade_in(SP_df, call_sell_sum, put_sell_sum)

            if A1 or A2 == True:

                return True

            else:

                return False


        except Exception as e:

            txt = f'sell_trade_in function is returning error \nError Desc ::{e} \n{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

        finally:

            txt = f'sell_trade_in function execution is completed'
            #Log.error_msg("Yellow", txt, True)

    @staticmethod
    def call_buy_trade_in(SP_df, call_buy_flag_sum, put_buy_flag_sum):

        try:

            buy_order_status = "Failed"

            # if call_buy_flag > 2 and put_buy_flag ==0 then
            if call_buy_flag_sum > 2 and put_buy_flag_sum == 0:

                # find the highest SP with active call buy flag
                SP_df = SP_df[SP_df['call_buy_flag'] == 1]

                # place limit buy order and wait for it to execute
                SP_max_obj = SP_df.loc[SP_df['strike_price'].idxmax(), 'Address']

                buy_order_status = SP_max_obj.execute_limit_buy_order('call')
                # place market buy order if above execution is successful

                if buy_order_status != "Success" and buy_order_status != "Insufficient Balance":
                    # if execution is successful, then pause the buy_trade_in thread
                    print(buy_order_status)
                    buy_order_status = SP_max_obj.execute_market_buy_order('call')

                if buy_order_status == "Success":
                    #pause_events['evaluate_buy'].set()
                    #pause_events['entry_scanners'].set()
                    # pause_events['trade_entry'].set()

                    txt =   f'Buying and evaluation is paused'
                    Log.debug_msg("Blue",txt,True)
                    return True

                else:

                    return False

        except Exception as e:

            txt = f"Error while execution of call_buy_trade_in.\nError Details{e}"
            Log.error_msg("Yellow", txt, True)

            Log.error_msg("Red",
                          f"{traceback.format_exc()}",
                          True)

            return "Failed"



    @staticmethod
    def put_buy_trade_in(SP_df, call_buy_flag_sum, put_buy_flag_sum):

        try:

            global pause_events


            # if put_buy_flag > 2 and call_buy_flag ==0 then
            if put_buy_flag_sum > 2 and call_buy_flag_sum == 0:

                # find the lowest SP with active call buy flag
                SP_df = SP_df[SP_df['put_buy_flag'] == 1]

                # place limit buy order and wait for it to execute
                SP_min_obj = SP_df.loc[SP_df['strike_price'].idxmin(), 'Address']

                #Log the entry Information
                Log.info_msg('Green',
                             f"Placing limit purchase order for SP::{SP_min_obj.strike_price} || PUT",
                             True)

                # place market buy order if the above execution is successful
                buy_order_status = SP_min_obj.execute_limit_buy_order('put')

                # if execution is successful, then pause the buy_trade_in thread
                if buy_order_status != "Success" and buy_order_status != "Insufficient Balance":

                    buy_order_status = SP_min_obj.execute_market_buy_order('put')

                if buy_order_status == "Success":
                    pause_events['evaluate_buy'].set()
                    pause_events['entry_scanners'].set()
                    pause_events['trade_entry'].set()

                    txt = f'Buying and evaluation is paused'
                    Log.debug_msg("Blue", txt, True)
                    return True

                else:

                    return False


        except Exception as e:

            txt = f"Error while execution of put_buy_trade_in.\nError Details::{e}"
            Log.error_msg("Yellow", txt, True)

            Log.error_msg("Red",
                          f"{traceback.format_exc()}",
                          True)




    @staticmethod
    def call_sell_trade_in(SP_df, call_sell_flag_sum, put_sell_flag_sum):

        try:

            sell_order_status = "Failed"

            # if call_buy_flag > 2 and put_buy_flag ==0 then
            if call_sell_flag_sum > 2 and put_sell_flag_sum == 0:

                # find the highest SP with active call buy flag
                SP_df = SP_df[SP_df['call_buy_flag'] == 1]

                # place limit buy order and wait for it to execute
                SP_min_obj = SP_df.loc[SP_df['strike_price'].idxmin(), 'Address']

                sell_order_status = SP_min_obj.execute_limit_sell_order('call')
                # place market buy order if above execution is successful

                if sell_order_status != "Success" and sell_order_status != "Insufficient Balance":
                    # if execution is successful, then pause the buy_trade_in thread

                    sell_order_status = SP_min_obj.execute_market_sell_order('call')


                if sell_order_status == "Success":
                    pause_events['evaluate_buy'].set()
                    pause_events['entry_scanners'].set()
                    pause_events['trade_entry'].set()

                    txt = f'Buying and evaluation is paused'
                    Log.debug_msg("Blue", txt, True)
                    return True

                else:

                    return False


        except Exception as e:

            txt = f"Error while execution of call_sell_trade_in.\nError Details::{e}"
            Log.error_msg("Yellow", txt, True)

            Log.error_msg("Red",
                          f"{traceback.format_exc()}",
                          True)
            return "Failed"



    @staticmethod
    def put_sell_trade_in(SP_df, call_sell_flag_sum, put_sell_flag_sum):

        try:
            global pause_events


            # if put_buy_flag > 2 and call_buy_flag ==0 then
            if call_sell_flag_sum > 2 and put_sell_flag_sum == 0:

                # find the highest SP with active call buy flag
                SP_df = SP_df[SP_df['call_sell_flag'] == 1]

                # place limit buy order and wait for it to execute
                SP_max_obj = SP_df.loc[SP_df['strike_price'].idxmax(), 'Address']

                # place market buy order if above execution is successful
                sell_order_status = SP_max_obj.execute_limit_sell_order('put')

                # if execution is successfully then pause the buy_trade_in thread
                if sell_order_status != "Success" and sell_order_status != "Insufficient Balance":
                    sell_order_status = SP_max_obj.execute_market_sell_order('put')

                # also pause evaluate_OP_buy flag and OP_sell flag
                if sell_order_status == "Success":
                     pause_events['evaluate_buy'].set()
                     pause_events['entry_scanners'].set()
                     pause_events['trade_entry'].set()

        except Exception as e:

            txt = f"Error while execution of buy_trade_in.\nError Details::{e}"
            Log.error_msg("Yellow", txt, True)

            Log.error_msg("Red",
                          f"{traceback.format_exc()}",
                          True)


    @staticmethod
    def CE_NIFTY_Index_levels_confirmation(NIF_df, evaluation_parameters):

        '''
        Parameters
        ----------
        df : Dataframe  || It contains the data of strike price to be evaluated
        T1 : Integer    || The first time slot for evaluation to check short trend
        T2 : Integer    || The second time slot for evaluation to check medium term trend
        T3 : Integer    || The third time slot for evaluation to check longer term continuity of trend
        VF : Float      || The sudden flux increase in volume over each second compared to Avg. volume
        Max_Vol : Float || The Absolute maximum volume in a given period
        CP_G1 : Float   || The % increase in call price in T1 duration
        CP_G2 : Float   || The % increase in call price in T2 duration
        CP_G3 : Float   || The % increase in call price in T2 duration
        VWAP_50 : Float || height of close from VWAP of 50 stocks used to find change in trend
        VWAP_20 : Float || height of close from VWAP of 20 stocks used to find change in trend
        VWAP_10 : Float || height of close from VWAP of 10 stocks used to find change in trend
        NIF_T1 :  Float || NIFTY increase in last T1 seconds
        NIF_T2 :  Float || NIFTY increase in last T2 seconds
        NIF_T3 : Float  || NIFTY increase in last T3 seconds

        Returns 1 or 0 i.e., buy or not to buy
        '''

        #Define Alias
        EP = evaluation_parameters.iloc[-1]
        T1, T2, T3 = EP.loc[['T1', 'T2', 'T3']].values.astype(int)
        max_duration = max(T1, T2, T3)

        #Define NIFTY parameter Alias
        NIF_T1,NIF_T2,NIF_T3 = EP['call_NIF_T1','call_NIF_T2','call_NIF_T3']


        # Calculate average price of put in T1, T2, T3 durations
        mean_NIF = {
                    'T1': NIF_df['close'].iloc[-T1:-2].mean(),
                    'T2': NIF_df['close'].iloc[-T2:-2].mean(),
                    'T3': NIF_df['close'].iloc[-T3:-2].mean()
                }

        # Testing conditions
        C1 = (NIF_df['close'].iloc[-1] / mean_NIF['T1']) > NIF_T1
        C2 = (NIF_df['close'].iloc[-1] / mean_NIF['T2']) > NIF_T2
        C3 = (NIF_df['close'].iloc[-1] / mean_NIF['T3']) > NIF_T3

        # encapsulate variables
        conditions = [C1, C2, C3]

        #check if all entry condition is True then return True else return False
        return True if all(condition is True for condition in conditions) else False

    @staticmethod
    def CE_vol_price_action_confirmation(call_df, evaluation_parameters):

        '''
        Parameters
        ----------
        df : Dataframe  || It contains the data of strike price to be evaluated
        T1 : Integer    || The first time slot for evaluation to check short trend
        T2 : Integer    || The second time slot for evaluation to check medium term trend
        T3 : Integer    || The third time slot for evaluation to check longer term continuity of trend
        VF : Float      || The sudden flux increase in volume over each second compared to Avg. volume
        Max_Vol : Float || The Absolute maximum volume in a given period
        CP_G1 : Float   || The % increase in call price in T1 duration
        CP_G2 : Float   || The % increase in call price in T2 duration
        CP_G3 : Float   || The % increase in call price in T3 duration
        VWAP_50 : Float || height of close from VWAP of 50 stocks used to find change in trend
        VWAP_20 : Float || height of close from VWAP of 20 stocks used to find change in trend
        VWAP_10 : Float || height of close from VWAP of 10 stocks used to find change in trend
        NIF_T1 :  Float || NIFTY increase in last T1 seconds
        NIF_T2 :  Float || NIFTY increase in last T2 seconds
        NIF_T3 : Float  || NIFTY increase in last T3 seconds

        Returns 1 or 0 i.e. buy or not to buy
        '''

        EP                  = evaluation_parameters.iloc[-1]
        T1, T2, T3          = EP.loc[['T1', 'T2', 'T3']].values.astype(int)
        max_duration        = int(max(T1, T2, T3))
        VC,VF               = EP[['VC','VF']]
        CP_G1, CP_G2, CP_G3 = EP[['PP1', 'PP2', 'PP3']]
        max_duration        = max(T1, T2, T3)
        max_VF              = call_df['volume_factor'].iloc[-max_duration:-2]
        max_Vol             = call_df['volume'].iloc[-max_duration:-2]

        mean_CP = {
                        'T1': call_df['close'].iloc[-T1:-2].mean(),
                        'T2': call_df['close'].iloc[-T2:-2].mean(),
                        'T3': call_df['close'].iloc[-T3:-2].mean()
                    }


        # Testing conditions
        C1 = (call_df['close'].iloc[-1] / mean_CP['T1']) > CP_G1
        C2 = (call_df['close'].iloc[-1] / mean_CP['T2']) > CP_G2
        C3 = (call_df['close'].iloc[-1] / mean_CP['T3']) > CP_G3

        C4 = max_VF  > VF
        C5 = max_Vol > VC

        # encapsulate variables
        conditions = [C1, C2, C3, C4, C5]

        return True if all(condition is True for condition in conditions) else False

    @staticmethod
    def PE_vol_price_action_confirmation(put_df, evaluation_parameters):

        '''
        Parameters
        ----------
        df : Dataframe  || It contains the data of strike price to be evaluated
        T1 : Integer    || The first time slot for evaluation to check short trend
        T2 : Integer    || The second time slot for evaluation to check medium term trend
        T3 : Integer    || The third time slot for evaluation to check longer term continuity of trend
        VF : Float      || The sudden flux increase in volume over each second compared to Avg. volume
        Max_Vol : Float || The Absolute maximum volume in a given period
        PP_G1 : Float   || The % increase in put price in T1 duration
        PP_G2 : Float   || The % increase in put price in T2 duration
        PP_G3 : Float   || The % increase in put price in T2 duration
        VWAP_50 : Float || height of close from VWAP of 50 stocks used to find change in trend
        VWAP_20 : Float || height of close from VWAP of 20 stocks used to find change in trend
        VWAP_10 : Float || height of close from VWAP of 10 stocks used to find change in trend
        NIF_T1 :  Float || NIFTY increase in last T1 seconds
        NIF_T2 :  Float || NIFTY increase in last T2 seconds
        NIF_T3 : Float  || NIFTY increase in last T3 seconds
        ========================

        Returns 1 or 0 i.e. buy or not to buy
        '''

        EP                  = evaluation_parameters.iloc[-1]
        T1, T2, T3          = EP.loc[['T1', 'T2', 'T3']].values.astype(int)
        VP,VF               = EP[['VP','VF']]
        PP_G1, PP_G2, PP_G3 = EP[['PP1', 'PP2', 'PP3']]
        max_duration        = int(max(T1, T2, T3))
        max_VF              = put_df['volume_factor'].iloc[-max_duration:-2]
        max_Vol             = put_df['volume'].iloc[-max_duration:-2]

        # Calculate average price of put in T1, T2, T3 durations
        mean_PP = {
                        'T1': put_df['close'].iloc[-T1:-2].mean(),
                        'T2': put_df['close'].iloc[-T2:-2].mean(),
                        'T3': put_df['close'].iloc[-T3:-2].mean()
                    }


        # Testing hypothesis by price confirmation
        C1 = (put_df['close'].iloc[-1] / mean_PP['T1']) > PP_G1
        C2 = (put_df['close'].iloc[-1] / mean_PP['T2']) > PP_G2
        C3 = (put_df['close'].iloc[-1] / mean_PP['T3']) > PP_G3

        # Testing hypothesis by volume confirmation
        C4 = max_VF  > VF
        C5 = max_Vol > VP

        # encapsulate variables
        conditions = [C1, C2, C3, C4, C5]

        return True if all(condition is True for condition in conditions) else False




class Exit_Scanners():
    dic = []

    def __init__(self):
        return

    @staticmethod
    def square_off_holdings(order_id=None):
        try:

            global start_events,pause_events
            holdings = Order_book.Trx_list[Order_book.Trx_list['order_id'] == order_id].copy()

            if len(holdings)==0:
                print(f"No holdings found for Order ID: {order_id}")
                return

            holdings    =   holdings.iloc[0].to_dict(orient='records')
            right       =   holdings['right']
            od_type     =   holdings['order_type']
            holding_obj =   holdings['Address']

            exit_logic  =   IP.determine_exit_logic()

            if right.lower() == 'call' and od_type.lower() == 'buy':

                a = threading.Thread(target = Exit_Scanners.eval_and_exc_call_buy_square_off,args = (order_id,exit_logic,start_events['square_off'],pause_events['square_off']))

            elif right.lower() == 'put' and od_type.lower() == 'buy':

                a = threading.Thread(target=Exit_Scanners.eval_and_exc_put_buy_square_off, args=(order_id, exit_logic,start_events['square_off'],pause_events['square_off']))

            elif right.lower() =='call' and od_type.lower() =='sell':

                a = threading.Thread(target=Exit_Scanners.eval_and_exc_call_sell_square_off, args=(order_id, exit_logic,start_events['square_off'],pause_events['square_off']))

            elif right.lower() == 'put' and od_type.lower() == 'sell':

                a = threading.Thread(target=Exit_Scanners.eval_and_exc_put_sell_square_off, args=(order_id, exit_logic,start_events['square_off'],pause_events['square_off']))

            else:

                txt = f'Invalid Order Type detected for Square-off'
                Log.error_msg('Yellow',txt,True)

                # Once we start a thread to square off holdings, then at the exit in finally clause
            # enable all pause flags of entry_Scanners and trade_in

        except Exception as e:

                txt =f'Error while squaring_off holdings.Error Details::{e}'
                Log.error_msg('Red',txt,True)

        finally:

            txt =f'Square-Off thread for order-id::{order_id} initiated'
            Log.debug_msg('Blue',txt,True)
    @staticmethod
    def square_off_all_holdings():
        try:
            #Steps
            #1)Get all current holdings from ICICI Direct platform
            #2)square_off all holdings one by one as market price

            #step 1: Fetch current holdings
            holdings = breeze.get_portfolio_positions()

            if not holdings or 'Success' not in holdings.get('Status', ''):
                print("Failed to fetch holdings.")
                return

            positions = holdings.get('Success', [])

            # Step 2: Filter for options holdings
            options_positions = [pos for pos in positions if pos.get('product_type') == 'options']

            if not options_positions:
                print("No options positions to square off.")
                return

            # Step 3: Square off each option position
            for pos in options_positions:
                try:
                    symbol = pos['stock_code']
                    exchange = pos['exchange_code']
                    quantity = abs(int(pos['quantity']))
                    action = 'sell' if pos['buy_or_sell'] == 'buy' else 'buy'
                    expiry_date = str(expiry_date_iso)
                    print(f"Squaring off {symbol} - {action} {quantity} units")

                    response = breeze.square_off(
                                                    stock_code          =   symbol,
                                                    exchange_code       =   exchange,
                                                    product_type        =   "Options",
                                                    action              =   action,
                                                    order_type          =   "market",
                                                    quantity            =   quantity,
                                                    price               =   0,  # Market order
                                                    validity            =   "day",
                                                    validity_date       =   "",
                                                    disclosed_quantity  =   "0",
                                                    stoploss            =   "0",
                                                    take_profit         =   "0"
                                                )

                    print("Order Response:", response)
                    time.sleep(1)  # Avoid rate limits

                except Exception as e:
                    print(f"Error squaring off {pos.get('stock_code')}: {e}")



            pass
        except Exception as e:

            pass

        finally:

            pass

    @staticmethod
    def eval_and_exc_call_buy_square_off(evaluation_logic, order_id,start_flag,pause_flag):
        try:
            # Define evaluation thresholds
            thresholds = [10, 20, 30]
            max_threshold = max(thresholds)

            # Extract order info
            order_data = Order_book.Trx_list[Order_book.Trx_list['order_id'] == order_id].copy()
            if order_data.empty:
                Log.error_msg('Red', f"No order found with ID: {order_id}", True)
                return 'Failure'

            # Extract required fields
            buy_price       = order_data.at[order_data.index[0], 'entry_price']
            nifty_entry_val = order_data.at[order_data.index[0], 'NIFTY_val']
            vwap_entry_val  = order_data.at[order_data.index[0], 'VWAP50']
            obj             = order_data.at[order_data.index[0], 'object']
            df              = obj.call_df_sec

            while start_flag.is_set():

                    # Market condition checks
                    price_uptrend       = Indicators.uptrend_confirmation(df, max_threshold, 1, "std_dev")
                    nifty_uptrend       = Indicators.uptrend_confirmation(NIFTY_Index.df_sec, max_threshold, 1, "std_dev")
                    nifty_vwap_uptrend  = Indicators.uptrend_confirmation(
                                                                            NIFTY_Stocks.VWAP_df_sec,
                                                                            max_threshold, 0.05,
                                                                      "pct", 'VWAP50',
                                                                'VWAP50'
                                                                        )

                    pause_flag.wait()

                    # Square-off logic
                    if any([price_uptrend, nifty_uptrend, nifty_vwap_uptrend]):

                        status = obj.execute_limit_buy_square_off('call', 'buy')

                        if status != 'Success':

                            status = obj.execute_market_buy_square_off('call', 'buy')

                            if status != 'Success':
                                msg = f'Error executing Square-Off for SP: {obj.strike_price} || call || buy'

                                for _ in range(20):
                                    Log.error_msg('Red', msg, True)
                                return 'Failure'
                            else:

                                start_flag.clear()
                                return 'Success'
                        else:

                            start_flag.clear()
                            return 'Success'
                        
                    Log.info_msg('Yellow', f'Square-off not required || order_id: {order_id}', True)


        except Exception as e:
            Log.debug_msg('Blue', f'Exception in call buy square_off || order_id: {order_id} || Error: {str(e)}', True)
            return 'Failure'

    @staticmethod
    def eval_and_exc_call_sell_square_off(evaluation_logic, order_id,start_flag,pause_flag):
        try:
            # Define evaluation thresholds
            thresholds = [10, 20, 30]
            max_threshold = max(thresholds)

            # Extract order info
            order_data = Order_book.Trx_list[Order_book.Trx_list['order_id'] == order_id].copy()

            if order_data.empty:
                Log.error_msg('Red', f"No order found with ID: {order_id}", True)
                return 'Failure'

            # Extract required fields
            buy_price       = order_data.at[order_data.index[0], 'entry_price']
            nifty_entry_val = order_data.at[order_data.index[0], 'NIFTY_val']
            vwap_entry_val  = order_data.at[order_data.index[0], 'VWAP50']
            obj             = order_data.at[order_data.index[0], 'object']
            df              = obj.call_df_sec

            while start_flag.is_set():

                # Market condition checks
                price_uptrend       = Indicators.sideways_confirmation(df, max_threshold, 1, "std_dev")
                nifty_uptrend       = Indicators.sideways_confirmation(NIFTY_Index.df_sec, max_threshold, 1, "std_dev")
                nifty_vwap_uptrend  = Indicators.sideways_confirmation(NIFTY_Stocks.VWAP_df_sec,
                                                                       max_threshold, 0.05, "pct",
                                                                       'VWAP50', 'VWAP50'
                                                                      )

                # Square-off logic
                if any([price_uptrend, nifty_uptrend, nifty_vwap_uptrend]):

                    status = obj.execute_limit_buy_square_off('call', 'buy')

                    if status != 'Success':
                        status = obj.execute_market_buy_square_off('call', 'buy')


                        if status != 'Success':

                                msg = f'Error executing Square-Off for SP: {obj.strike_price} || call || buy'
                                for _ in range(20):
                                    Log.error_msg('Red', msg, True)
                                return 'Failure'


                        else:

                                start_flag.clear()

                                return "Success"

                else:

                    start_flag.clear()

                    return "Success"

                Log.info_msg('Yellow', f'Square-off not required || order_id: {order_id}', True)

        except Exception as e:

            Log.debug_msg('Blue', f'Exception in call buy square_off || order_id: {order_id} || Error: {str(e)}', True)


    @staticmethod
    def eval_and_exc_put_buy_square_off(evaluation_logic, order_id,start_flag,pause_flag):

        try:
            # declare aliases for evaluation Logic parameters
            T1              =   10
            T2              =   20
            T3              =   30

            # declare aliases of order related parameters
            holdings        = Order_book.Trx_list[Order_book.Trx_list['order_id'] == order_id].copy()
            right           = holdings['right']
            od_type         = holdings['order_type']
            buy_price       = holdings['entry_price']
            obj             = holdings['object']
            df              = obj.call_df_sec if right == 'call' else obj.put_df_sec
            NIF_entry       = NIFTY_Index.current_val
            NIF_ent_VWAP    = NIFTY_Stocks.VWAP_df_sec["VWAP_50"].iloc[-1]

            while start_flag.is_set():
                    # evaluate uptrend in price confirmation
                    price_downtrend = Indicators.downtrend_confirmation(df, max(T1, T2, T3), 1, "std_dev")

                    # evaluate uptrend in NIFTY Index confirmation
                    NIF_downtrend = Indicators.downtrend_confirmation(NIFTY_Index.df_sec, max(T1, T2, T3), 1, "std_dev")

                    # evaluate uptrend in NIFTY Stocks VWAP confirmation
                    NIF_VWAP_downtrend =  True

                    pause_flag.wait()

                    # If any of the above show weakness in market then square off the asset holdings
                    Conditions = [price_downtrend, NIF_downtrend, NIF_VWAP_downtrend]

                    pause_flag.wait()

                    # Square-off logic
                    if any(Conditions):

                        status = obj.execute_limit_buy_square_off('put', 'buy')

                        if status != 'Success':

                            status = obj.execute_market_buy_square_off('put', 'buy')

                            if status != 'Success':
                                msg = f'Error executing Square-Off for SP: {obj.strike_price} || put || buy'

                                for _ in range(20):
                                    Log.error_msg('Red', msg, True)
                                return 'Failure'
                            else:

                                start_flag.clear()
                                return 'Success'
                        else:

                            start_flag.clear()
                            return 'Success'

                    Log.info_msg('Yellow', f'Square-off not required || order_id: {order_id}', True)




        except Exception as e:
            txt = f'Call buy square_off ran into error'
            Log.debug_msg('Blue', txt, True)

        finally:

            pass

    @staticmethod
    def eval_and_exc_put_sell_square_off(evaluation_logic, order_id,start_flag,pause_flag):

        try:
            # declare aliases for evaluation Logic parameters
            T1 = 10
            T2 = 20
            T3 = 30

            # declare aliases of order related parameters
            holdings            = Order_book.Trx_list[Order_book.Trx_list['order_id'] == order_id].copy()
            right               = holdings['right']
            od_type             = holdings['order_type']
            buy_price           = holdings['entry_price']
            obj                 = holdings['object']
            df                  = obj.call_df_sec if right == 'call' else obj.put_df_sec
            NIF_entry           = NIFTY_Index.current_val
            NIF_ent_VWAP        = NIFTY_Stocks.VWAP_df_sec["VWAP_50"].iloc[-1]

            # evaluate uptrend in NIFTY Index confirmation
            NIF_uptrend = Indicators.uptrend_confirmation(NIFTY_Index.df_sec, max(T1, T2, T3), 1, "std_dev")

            # evaluate uptrend in NIFTY Stocks VWAP confirmation
            NIF_VWAP_uptrend =  True

            # If any of the above show weakness in market then square off the asset holdings
            Conditions = [NIF_uptrend, NIF_VWAP_uptrend]

            # Square-off logic
            if any(Conditions):

                status = obj.execute_limit_buy_square_off('put', 'sell')

                if status != 'Success':

                    status = obj.execute_market_buy_square_off('put', 'sell')

                    if status != 'Success':
                        msg = f'Error executing Square-Off for SP: {obj.strike_price} || put || sell'

                        for _ in range(20):
                            Log.error_msg('Red', msg, True)
                        return 'Failure'
                    else:

                        start_flag.clear()
                        return 'Success'
                else:

                    start_flag.clear()
                    return 'Success'

            Log.info_msg('Yellow', f'Square-off not required || order_id: {order_id}', True)


        except Exception as e:
            txt = f'Call buy square_off ran into error'
            Log.debug_msg('Blue', txt, True)

        finally:

            pass


class Thread_Scheduler():
    Thread_List = Global_function.read_excel(KPI_folder, "Input_Parameters", "Other_DF_Col", 'A:G', 0, 0)
    Thread_List = Thread_List[Thread_List['Thread_no'].notna()]
    Thread_List = Thread_List.set_index('Thread_no')
    timer = datetime.now().replace(microsecond=0)

    def __init__(self):

        return

    def start_processing(args):

        Thread_Scheduler.timer = datetime.now().replace(microsecond=0)

        txt = f'{Thread_Scheduler.Thread_List.loc["TH6", "Name"]} Thread started'
        Log.info_msg("Green", txt, True)

        while Thread_Scheduler.Thread_List.loc["TH6", 'En_Flag'] == 0 \
                and Thread_Scheduler.Thread_List.loc["TH6", 'Ex_Flag'] == 0 \
                and datetime.now().time() < code_end_time:

            current_time = datetime.now().replace(microsecond=0)

            if current_time == Thread_Scheduler.timer + timedelta(seconds=1):

                # 1 Update VWAP
                Display.update_VWAP()

                # 2 entry scanners update

                # 3 square_off scanners update

                # updating latest timer
                Thread_Scheduler.timer = current_time

            elif current_time > Thread_Scheduler.timer + timedelta(seconds=1):

                txt = f'code is taking {current_time} - {Thread_Scheduler.timer} seconds to perform the operation'
                Log.error_msg("Yellow", txt, True)

                Thread_Scheduler.timer = datetime.now().replace(microsecond=0)

            sleep(0.2)

        # Signaling successful exit from start_exit_scanners
        Thread_Scheduler.Thread_List.loc["TH6", "Ex_Flag"] = 1
        Thread_Scheduler.Thread_List.loc["TH6", "En_Flag"] = 1

        txt = f'{Thread_Scheduler.Thread_List.loc["TH6", "Name"]} Thread has been terminated succesfully'
        Log.info_msg("Green", txt, True)

        return "Success"

    @staticmethod
    def start_threads_and_processes(start_events, pause_events, ticks_queue,option_chain_queue, nif_stocks_queue,
                                    nif_index_queue, queue_length):

        try:
            threads = []
            processes = []

            # Defining Data Achiever Thread
            t2 = threading.Thread(

                target=Thread_Scheduler.OC_STK_auto_archival,
                args=
                ("09:15:00",
                 "15:30:00",
                 5,
                 start_events['data_archiever'],
                 pause_events['data_archiever'])

            )

            # Defining threads for loading data from secondary queue to dataframe
            t1 = threading.Thread(

                target=Ticks.ticks_manager, args=(
                    option_chain_queue,
                    nif_stocks_queue,
                    nif_index_queue,
                    start_events['ticks_manager'],
                    pause_events['ticks_manager'],
                )
            )

            # Defining seperate process on a processor to load the data from main queue to secondary queue
            consumer_proc = multiprocessing.Process(

                target=Ticks.consumer, args=(
                    ticks_queue,
                    option_chain_queue,
                    nif_index_queue,
                    nif_stocks_queue,
                    start_events['consumer'],
                    pause_events['consumer'],
                )
            )

            
            # starting background threads and processes
            t1.start()
            consumer_proc.start()
            t2.start()

            t3 = Option_Chain.evaluate_OP(start_events['entry_scanner'],pause_events['entry_scanner'])
            t4 = Entry_Scanners.Start_scanning(start_events['trade_entry'],pause_events['trade_entry'])
            threads = [t1, t2] +t3
            processes = [consumer_proc]

            return threads, processes

        except Exception as oe:
            txt = f"Unknown type of error happened at start_threads_and_processes"
            print(txt)
            txt = f'Error Details::{oe}'
            print(txt)
            return threads, processes
    @staticmethod
    def OC_STK_auto_archival(start_time, end_time, time_gap,
                             start_event: threading.Event,
                             pause_event: threading.Event):

        '''
        Keep this function always in pause mode i.e. keep the pause flag clear while we want normal function
        We clear pause i.e. set the pause flag to stop normal operation and hold the functioning of this thread

        '''

        # Build all target cut‑off times
        schedule = Global_function.generate_timestamps(start_time, end_time, time_gap)['datetime']
        idx = 2  # points to “next” slot

        Log.info_msg("Green",
                     "Auto‑archival worker waiting for start_event ...", True)

        start_event.wait()  # Block until supervisor says GO
        Log.info_msg("Green", "Auto‑archival started.", True)

        try:
            while start_event.is_set():
                if idx >= len(schedule):
                    Log.info_msg("Yellow",
                                 "All time slots processed; worker exiting.", True)
                    break

                now = datetime.now()
                next_cut = schedule.iloc[idx - 1]

                # If we’re late, skip ahead
                if (schedule.iloc[idx] - now).total_seconds() <= 0:
                    idx += 1
                    continue

                # Sleep until either: a) pause lifted, or b) slot time arrives
                sleep_seconds = (schedule.iloc[idx] - now).total_seconds()

                # unpaused will wait till timeout happens or pause_flag is set to reduce memory over burden
                unpaused = pause_event.wait(timeout=sleep_seconds)

                if not unpaused:  # woke up because timeout expired
                    # Launch archivers
                    oc_thread = threading.Thread(
                        target=Option_Chain.start_archiever,
                        args=(next_cut, 900, start_event, pause_event))
                    stk_thread = threading.Thread(
                        target=NIFTY_Stocks.start_archiever,
                        args=(next_cut, 900, start_event, pause_event))

                    oc_thread.start()
                    stk_thread.start()
                    oc_thread.join()
                    stk_thread.join()

                    Log.info_msg("Green",
                                 f"Auto - archival completed for Option chain and NIFTY Stocks for time {next_cut}",
                                 True)

                    idx += 1  # advance to next slot
                else:
                    # We were paused; loop back (will spin until pause cleared)
                    continue





        except Exception as exc:
            Log.error_msg("Red",
                          f"Auto‑archival crashed: {exc}\n{traceback.format_exc()}",
                          True)
            raise
        finally:
            Log.info_msg("Yellow", "Auto‑archival thread exiting.", True)

    @staticmethod
    def Archieve_manager(args):
        ''' args for initialisation purpose only '''
        counter = 2
        start_time = "09:15:00"
        end_time = "15:29:59"
        interval_minutes = 5
        time_stamp = Global_function.generate_timestamps(start_time, end_time, interval_minutes)
        global code_end_time

        # reaching closest to time stamp for cropping data
        if (time_stamp['datetime'].iloc[counter - 1] - datetime.now()).total_seconds() < 0:

            while (time_stamp['datetime'].iloc[counter - 2] - datetime.now()).total_seconds() < 0:
                counter += 1

                if counter >= len(time_stamp) - 1:
                    txt = f'current time {datetime.now()} is outside the time slot for archival'
                    break

        print(f'cut off time set to {time_stamp["datetime"].iloc[counter]}')

        txt = f'{Thread_Scheduler.Thread_List.loc["TH5", "Name"]} Thread started'
        Log.info_msg("Green", txt, True)

        while Thread_Scheduler.Thread_List.loc["TH5", 'En_Flag'] == 0 \
                and Thread_Scheduler.Thread_List.loc["TH5", 'Ex_Flag'] == 0 \
                and datetime.now().time() < code_end_time:

            func_return = Global_function.current_time_match(time_stamp, counter)

            if func_return == "Success":

                func_status = Option_Chain.activate_auto_archival(time_stamp['datetime'].iloc[counter - 1])

                if func_status == "Success":

                    txt = f"Option Chain data successfully archived for last {interval_minutes} mins"
                    Log.info_msg("Green", txt, True)

                else:

                    txt = "Option Chain data_archiver ran into some problem"
                    Log.info_msg(txt, True)

                sleep(0.5)

                func_status = NIFTY_Stocks.activate_auto_archival(time_stamp['datetime'].iloc[counter - 1])

                if func_status == "Success":

                    txt = f"NIFTY Stocks data successfully archived for last {interval_minutes} mins"
                    Log.info_msg("Green", txt, True)

                else:

                    txt = "NIFTY Stocks data_archiver ran into some problem"
                    Log.info_msg("Green", txt, True)

                sleep(0.5)

                counter += 1

            elif func_return != "Wait":

                txt = f"error while running data_archiver return value:: {func_return}"
                Log.error_msg(txt, True)

            sleep(1)

        # Signaling successful exit from start_exit_scanners
        Thread_Scheduler.Thread_List.loc["TH5", "Ex_Flag"] = 1
        Thread_Scheduler.Thread_List.loc["TH5", "En_Flag"] = 1

        txt = f'{Thread_Scheduler.Thread_List.loc["TH5", "Name"]} Thread has been terminated succesfully'
        Log.info_msg("Green", txt, True)

        return "Success"

    def start_new_thread(Thread_no):

        # define variables
        function_name = Thread_Scheduler.Thread_List.loc[Thread_no, "Name"]
        args = Thread_Scheduler.Thread_List.loc[Thread_no, "args"]
        module_name = Thread_Scheduler.Thread_List.loc[Thread_no, "class_name"]
        target_function = getattr(globals()[module_name], function_name)

        # define new Thread
        Thread_t = threading.Thread(target=target_function, args=(args,))

        section = 1
        # store initialisation information in Threads variable
        Thread_Scheduler.Thread_List.loc[Thread_no, "Address"] = Thread_t

        section = 2
        # Reset Entry flag and start the thread to prevent duplicate start up of threads
        Thread_Scheduler.Thread_List.loc[Thread_no, "En_Flag"] = 0
        Thread_Scheduler.Thread_List.loc[Thread_no, "Ex_Flag"] = 0

        section = 3
        # start the thread
        Thread_t.start()

        section = 4
        # print success message and save in system message var
        txt = f'Thread No.{Thread_no}  :- {function_name}  of {module_name} started'

        return

    def start_threads(thread_nos):

        for Thread_no in thread_nos:

            En_flag = Thread_Scheduler.Thread_List.loc[Thread_no, 'En_Flag']
            Ex_flag = Thread_Scheduler.Thread_List.loc[Thread_no, 'Ex_Flag']
            Address = Thread_Scheduler.Thread_List.loc[Thread_no, 'Address']
            function_name = Thread_Scheduler.Thread_List.loc[Thread_no, "Name"]
            args = Thread_Scheduler.Thread_List.loc[Thread_no, "args"]
            module_name = Thread_Scheduler.Thread_List.loc[Thread_no, "class_name"]
            thread_live = Address.is_alive() if isinstance(Address, threading.Thread) else False
            target_function = getattr(globals()[module_name], function_name)
            print(f'Address ::{Address} Dtype::{type(Address)}')

            if En_flag == 1 and Ex_flag == 0 and thread_live == False:

                Thread_Scheduler.start_new_thread(Thread_no)

            elif En_flag == 1 and Ex_flag == 1 and thread_live == False and isinstance(Address,
                                                                                       threading.Thread) == True:

                Thread_Scheduler.restart_thread(Thread_no)

            elif En_flag == 0 and Ex_flag == 0 and thread_live == True:

                print(f'Thread is already active probably in sleep mode')

            elif isinstance(Address, threading.Thread) != True:

                Thread_Scheduler.start_new_thread(Thread_no)

        return

    def stop_threads(Thread_nos):
        try:
            section = 1
            for Thread_no in Thread_nos:

                # Defining Alias Name
                En_flag = Thread_Scheduler.Thread_List.loc[Thread_no, 'En_Flag']
                Ex_flag = Thread_Scheduler.Thread_List.loc[Thread_no, 'Ex_Flag']
                Address = Thread_Scheduler.Thread_List.loc[Thread_no, 'Address']
                Name = Thread_Scheduler.Thread_List.loc[Thread_no, 'Name']
                if En_flag == 0:

                    # Check if Thread is active
                    if Address.is_alive() == True:

                        # Disable Entry flag
                        Thread_Scheduler.Thread_List.loc[Thread_no, 'En_Flag'] = 1

                        # Waiting for the thread o finish operation of entire code and make safe exit
                        while Address.is_alive() == True:
                            txt = f"Waiting for Thread No.{Thread_no}  {Name} to terminate"
                            Log.info_msg("Green", txt, True)
                            sleep(0.5)

                        # Closing thread
                        txt = f"{Name} Thread terminated forcefully"
                        Log.info_msg("Green", txt, True)
                        Address.join()

                    else:

                        txt = f"{Name} Thread is already dead"
                        Log.info_msg("Green", txt, True)


                else:

                    txt = f"{Name} Thread is already inactive"
                    Log.info_msg("Green", txt, True)

            return "Success"

        except Exception as e:

            txt = f'Error in stop_threads function  at section {section}'
            Log.error_msg("Yellow", txt, True)

            txt = f'{traceback.format_exc()}'
            Log.error_msg("Yellow", txt, True)

            return e

    def restart_thread(Thread_no):

        En_flag = Thread_Scheduler.Thread_List.loc[Thread_no, 'En_Flag']
        Ex_flag = Thread_Scheduler.Thread_List.loc[Thread_no, 'Ex_Flag']
        Address = Thread_Scheduler.Thread_List.loc[Thread_no, 'Address']
        function_name = Thread_Scheduler.Thread_List.loc[Thread_no, "Name"]
        args = Thread_Scheduler.Thread_List.loc[Thread_no, "args"]
        module_name = Thread_Scheduler.Thread_List.loc[Thread_no, "class_name"]
        target_function = getattr(globals()[module_name], function_name)

        # close old thread using join method
        Address.join()

        # start new Thread
        Thread_t1 = threading.Thread(target=target_function, args=(args,))

        # store initialisation information in Threads variable
        Thread_Scheduler.Thread_List.loc[Thread_no, 'Address'] = Thread_t1

        # Reset flag to start new thread
        Thread_Scheduler.Thread_List.loc[Thread_no, 'En_Flag'] = 0
        Thread_Scheduler.Thread_List.loc[Thread_no, 'Ex_Flag'] = 0

        # start the thread with new details
        Thread_t1.start()

        # print success message and save in system message var
        txt = f'Restarted Thread No.{Thread_no}  :- {function_name}  of {module_name}'
        Log.info_msg("Green", txt, True)

    @staticmethod
    def option_chain_worker(oc_queue: multiprocessing.Queue(),
                            start_event,
                            pause_event
                            ):

        '''
        This function will recieve Queue of option chain ticks.
        It is suppose to identify the strike price and execute add_ticks operation
        '''
        # protecting from excpetional activation of this thread directly from outside the ticks manager function
        txt = f'Option Chain worker waiting for start permission.......'
        Log.info_msg("Green", txt, True)

        start_event.wait()

        txt = f'Option Chain worker started.......'
        Log.info_msg("Green", txt, True)

        while start_event.is_set():

            # variable initialisation for code

            SP_df = Option_Chain.SP_df
            was_paused = False

            if not pause_event.is_set():

                if not was_paused:
                    txt = f'Option Chain worker execution paused.......'
                    Log.info_msg("Green", txt, True)
                    was_paused = True

            pause_event.wait()

            try:

                ticks = oc_queue.get(timeout=10)

                row = SP_df[SP_df['strike_price'] == ticks['strike_price']]

                if not row.empty:

                    obj = row["Address"].iloc[0]
                    obj.add_tick(ticks)
                    txt = f'Updated Option chain Queue {datetime.now().replace(microsecond=0)}:{oc_queue.qsize()}'
                    # print(txt)


                else:
                    txt = "Received tick strike price not available in SP_df dataframe"
                    Log.error_msg('Red', txt, True)

            except:

                txt = f"Option Chain ticks not recieved.Updated Option chain Queue:{oc_queue.qsize()}"
                Log.error_msg('Red', txt, True)
                sleep(5)

        '''

            1) Is SP_df a copy?	❌ No
            2) Is it a reference/alias?	✅ Yes (to current object)
            3) Will updates in original be seen?	✅ Yes, if in-place changes
            4) Will reassignment be reflected?	❌ No, unless reassigned inside loop (which you do ✅)
            5) Is your current usage correct?	✅ Yes, safe and fresh per iteration





        '''

        return "Success"

    @staticmethod
    def nif_stock_worker(nif_stock_queue: multiprocessing.Queue(),
                         start_event,
                         pause_event
                         ):
        '''
        This function will recieve Queue of option chain ticks.
        It is suppose to identify the strike price and execute add_ticks operation
        '''

        # protecting from excpetional activation of this thread directly from outside the ticks manager function
        txt = f'NIFTY stocks worker waiting for start permission.......'
        Log.info_msg("Green", txt, True)

        start_event.wait()

        txt = f'NIFTY stocks worker started.......'
        Log.info_msg("Green", txt, True)

        while start_event.is_set():

            # variable initialisation for code

            df = NIFTY_Stocks.stock_list
            was_paused = False

            if not pause_event.is_set():

                if not was_paused:
                    txt = f'NIFTY stocks worker execution paused.......'
                    Log.info_msg("Green", txt, True)
                    was_paused = True

            pause_event.wait()

            try:

                ticks = nif_stock_queue.get(timeout=10)

                row = df[df['stock_name'] == ticks["stock_code"]]

                if not row.empty:

                    obj = row['Address'].iloc[0]
                    obj.add_tick(ticks)
                    txt = f'Updated NIFTY Stock Queue {datetime.now().replace(microsecond=0)}:{nif_stock_queue.qsize()}'
                    # print(txt)

                else:
                    txt = "Received tick stock code not available in stock_list dataframe"
                    Log.error_msg('Red', txt, True)

            except:

                txt = f"NIFTY Stocks ticks not recieved.Updated NIFTY Stock Queue:{nif_stock_queue.qsize()}"
                Log.error_msg('Red', txt, True)
                sleep(5)

        return "Success"

    @staticmethod
    def nif_index_worker(nif_index_queue: multiprocessing.Queue(),
                         start_event,
                         pause_event
                         ):

        '''
        This function will revert by appending data in the respective dataframe

        '''
        # protecting from excpetional activation of this thread directly from outside the ticks manager function
        txt = f'NIFTY Index worker waiting for start permission.......'
        Log.info_msg("Green", txt, True)

        start_event.wait()

        txt = f'NIFTY Index worker started.......'
        Log.info_msg("Green", txt, True)

        while start_event.is_set():

            # variable initialisation for code

            was_paused = False

            if not pause_event.is_set():

                if not was_paused:
                    txt = f'NIFTY stocks worker execution paused.......'
                    Log.info_msg("Green", txt, True)
                    was_paused = True

            pause_event.wait()

            try:

                ticks = nif_index_queue.get(timeout=10)
                NIFTY_Index.add_tick(ticks)
                txt = f'Updated NIFTY Index Queue {datetime.now().replace(microsecond=0)}:{nif_index_queue.qsize()}'
                # print(txt)
            except:
                txt = f"NIFTY Index ticks not recieved.Updated NIFTY Index Queue:{nif_index_queue.qsize()}"
                Log.error_msg('Red', txt, True)
                sleep(5)
                ticks = {'strike_price': 'RAM'}

        return "Success"


class Ticks:
    # Replace deque with thread-safe queue
    dic = queue.Queue(maxsize=10000)
    tick_count = 0

    def __init__(self):
        pass

    def producer(tick_q: multiprocessing.Queue(),
                 df):

        try:

            for i in range(0, len(df), 1):
                tick = df.loc[i, 'data']
                tick_q.put(tick, block=False)
                txt = f'Updated ticks Queue:{tick_q.qsize()}'
                # print(txt)

        except:

            pass

    def importer(tick_q: multiprocessing.Queue()):
        '''

        Parameters
        ----------
        tick_q : multiprocessing.Queue()
            This function will call internal function handler
            Handler function is suppose to append the incoming ticks to multiprocessing queue
            so consumer can get the incoming data

        Returns
        -------
        None.

        '''

        def handler(ticks):

            try:

                # print(f'tick recieved')
                tick_q.put(ticks, block=False)
                Ticks.tick_count += 1
                # print(f'Queue length {tick_q.qsize()}')


            except Exception as e:

                print("Error in handler:", e)

        return handler

    @staticmethod
    def consumer(tick_q: multiprocessing.Queue(),
                 option_chain_queue: multiprocessing.Queue(),
                 nif_index_queue: multiprocessing.Queue(),
                 nif_stocks_queue: multiprocessing.Queue(),
                 start_event,
                 pause_event,
                 queue_threshold=0):

        # waiting for start signal
        print(f'Ticks Consumer function is waiting for start signal', flush=True)

        start_event.wait()
        print(f'Ticks Consumer recieved the start signal', flush=True)

        while start_event.is_set():

            # variable initialisation for code

            was_paused = False

            if not pause_event.is_set():

                if not was_paused:
                    txt = f'consumer execution paused.......'
                    print(txt, flush=True)
                    was_paused = True

            pause_event.wait()

            try:

                tick = tick_q.get(timeout=10)  # Non-blocking get

                # print(f'Consumer fetched the data from queue,updated queue length {tick_q.qsize()}', flush=True)
                status = Ticks.processor(tick,
                                         option_chain_queue,
                                         nif_index_queue,
                                         nif_stocks_queue)

                # print(f'Consumer processed the data from queue,updated queue length {tick_q.qsize()}', flush=True)

            except:

                print(f"consumer function time exceeded,updated queue length {tick_q.qsize()}", flush=True)

        return "Success"

    @staticmethod
    def handler(ticks):
        try:
            Ticks.dic.put(ticks, block=False)  # Non-blocking put
            Ticks.tick_count += 1
        except queue.Full:
            txt = f'Queue is full, dropping data. Queue Size: {Ticks.dic.qsize()}'
            Log.error_msg("Yellow", txt, True)

    @staticmethod
    def associate(threshold_limit=1):
        try:
            if not Ticks.dic.empty():

                while Ticks.dic.qsize() > threshold_limit:
                    tick = Ticks.dic.get(block=False)  # Non-blocking get
                    func_status = Ticks.processor(tick)

                return "Success"

            return "Success"

        except queue.Empty:
            # No data to process
            return "Success"

        except Exception as e:

            txt = "Error in handling Ticks"
            Log.error_msg("Yellow", txt, True)
            Log.error_msg('Red', e, True)

            return e

    @staticmethod
    def ticks_manager(
            oc_queue: multiprocessing.Queue(),
            nif_stock_queue: multiprocessing.Queue(),
            nif_ind_queue: multiprocessing.Queue(),
            start_event,
            pause_event):
        try:

            was_paused = False  # to avoid repetative logs
            txt = f'ticks manager function is starting....waiting for event flag'
            Log.info_msg("Green", txt, True)

            start_event.wait()

            txt = f'ticks manager event flag is set....begining execution...'
            Log.info_msg("Green", txt, True)

            while start_event.is_set():

                was_paused = False

                if not pause_event.is_set():

                    if not was_paused:
                        txt = f'ticks manager execution paused.......'
                        Log.info_msg("Green", txt, True)
                        was_paused = True

                # waiting for code execution
                pause_event.wait()

                Th1 = threading.Thread(target=Thread_Scheduler.option_chain_worker,
                                       args=(oc_queue, start_event, pause_event,))
                Th2 = threading.Thread(target=Thread_Scheduler.nif_stock_worker,
                                       args=(nif_stock_queue, start_event, pause_event,))
                Th3 = threading.Thread(target=Thread_Scheduler.nif_index_worker,
                                       args=(nif_ind_queue, start_event, pause_event,))

                Th1.start()
                Th2.start()
                Th3.start()

                Th1.join()
                Th2.join()
                Th3.join()

            return "Success"

        except KeyboardInterrupt as e8:
            txt = "Keyboard Interrupt received. Exiting Ticks Manager Thread."
            Log.info_msg("Green", txt, True)

            return e8

        except Exception as e:
            txt = "Function ran into some problem in ticks_manager"
            Log.error_msg("Yellow", txt, True)

            txt = traceback.format_exc()
            Log.error_msg("Yellow", txt, True)

            return e

        finally:

            # final block of codes before exiting the code
            NIFTY_Stocks.unsubscribe_all_stocks()
            NIFTY_Index.un_subscribe_ticks("1second")
            NIFTY_Index.un_subscribe_ticks("1minute")
            Option_Chain.un_subscribe_all_sp("1second")

            txt = f'Ticks Manager thread has been terminated successfully'
            Log.info_msg("Green", txt, True)

    @staticmethod
    def processor(ticks,
                  option_chain_queue: multiprocessing.Queue(),
                  nif_index_queue: multiprocessing.Queue(),
                  nif_stocks_queue: multiprocessing.Queue(), ):
        try:

            txt = f'Ticks processor started'
            #Log.info_msg('Green', txt, True)

            ticks = {k: float(v) if k in ['open', 'high', 'low', 'close', 'volume', 'oi', 'strike_price'] else v for
                     k, v in ticks.items()}
            ticks['datetime'] = datetime.strptime(ticks['datetime'], '%Y-%m-%d %H:%M:%S')

            if ticks["exchange_code"] == "NFO":
                ticks['open_interest'] = ticks.pop('oi')

                option_chain_queue.put(ticks, block=False)

            elif ticks["exchange_code"] == "NSE":

                if ticks["stock_code"] == "NIFTY":

                    nif_index_queue.put(ticks, block=False)

                else:

                    nif_stocks_queue.put(ticks, block=False)

            else:
                txt = "Ticks not identified correctly"
                Log.error_msg("Yellow", txt, True)

            # txt = f'nif_stocks_queue {nif_stocks_queue.qsize()}  || nif_index_queue{nif_index_queue.qsize()} || option_chain_queue {option_chain_queue.qsize()}'
            # Log.info_msg("Green",txt,True)

            return "Success"

        except Exception as oe:

            txt = f"Unknown type of error happened in 'on_ticks' function for {ticks}"
            Log.error_msg("Red", txt, True)

            txt = f"Error Details::{oe} \n Traceback ::{traceback.format_exc()}"
            Log.error_msg("Red", txt, True)

            return


'''
==================================================================================================================

                                    Code initialisation to happen here

==================================================================================================================
'''

if __name__ == '__main__':
    main()

    # Log_msg.save_sys_messages()

    # data                =   Global_function.read_excel(mkt_folder, "Data_base_ticks_new","Data_base_ticks_new",'A:A',0,0)
    # data                =   data.iloc[:5]
    # data['data']        =   data['data'].apply(ast.literal_eval)

    # Ticks.producer(ticks_queue, data)

#  obj = Option_Chain.SP_df.loc[Option_Chain.SP_df['strike_price'] == 25500, 'Address'].iloc[0]
#  a = obj.call_df_sec
#    b = obj.put_df_sec

    txt = f'nif_stocks_queue {nif_stocks_queue.qsize()}  || nif_index_queue{nif_index_queue.qsize()} || option_chain_queue {option_chain_queue.qsize()}'
    print(txt, flush=True)

    # Ticks.ticks_manager(ticks_queue, option_chain_queue,nif_stocks_queue , nif_index_queue, start_events['ticks_manager'], pause_events['ticks_manager'])
    # Ticks.consumer(ticks_queue,option_chain_queue,nif_index_queue,nif_stocks_queue,code_end_time)





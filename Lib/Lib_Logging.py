
import logging
import os
from datetime import datetime, date, time,timedelta

'''
DEBUG   :  Blue
INFO    :  Green
WARNING :  Yellow
ERROR   :  Red
CRITICAL:  Cyan

'''


class Log():

        font_colour           =   {"Black"       : "\033[30m","Red"           : "\033[31m","Green"        : "\033[32m",
                                   "Yellow"      : "\033[33m","Blue"          : "\033[34m","Magenta"      : "\033[35m",
                                   "Cyan"        : "\033[36m","White"         : "\033[37m","Bright_Black" : "\033[90m",
                                   "Bright_Red"  : "\033[91m","Bright_Green"  : "\033[92m","Bright_Yellow": "\033[93m",
                                   "Bright_Blue" : "\033[94m","Bright_Magenta": "\033[95m","Bright_Cyan"  : "\033[96m",
                                   "Bright_White": "\033[97m",}
    
    
        bk_colour             =   {"Black"       : "\033[40m"  ,"Red"           : "\033[41m" ,"Green"        : "\033[42m",
                                   "Yellow"      : "\033[43m"  ,"Blue"          : "\033[44m" ,"Magenta"      : "\033[45m",
                                   "Cyan"        : "\033[46m"  ,"White"         : "\033[47m" }
    
        Bold	              = '\033[1m'
        Underline             =	'\033[4m'
        Reversed              =	'\033[7m'
        RESET                 = '\033[0m'
        ITALIC                = '\033[3m'
        
        # Directory to store log files
        LOG_DIR = f'E:/Algo_Trading_V3/Log File/{datetime.now().strftime("%B_%d_%Y")}/'
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Define file paths for different log levels
        LOG_FILES = {
                        "DEBUG": os.path.join(LOG_DIR, "debug.log"),
                        "INFO": os.path.join(LOG_DIR, "info.log"),
                        "WARNING": os.path.join(LOG_DIR, "warning.log"),
                        "ERROR": os.path.join(LOG_DIR, "error.log"),
                        "CRITICAL": os.path.join(LOG_DIR, "critical.log")
                    }
        
        # Create loggers for each level
        loggers = {}
        
        def __init__(self,name):
            return
        @staticmethod
        def setup_logger(level_name: str):
            
            if level_name in Log.loggers:
                
                return Log.loggers[level_name]
        
            logger = logging.getLogger(level_name)
            logger.setLevel(getattr(logging, level_name))
            logger.propagate = False  # Prevent double logging
        
            # Avoid duplicate handlers
            if not logger.handlers:
                file_handler = logging.FileHandler(Log.LOG_FILES[level_name], mode='a', encoding='utf-8')
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
            Log.loggers[level_name] = logger
            return logger
        
        # Logging functions
        @staticmethod
        def debug_msg(colour_name,msg,display_flag):
            
            logger = Log.setup_logger("DEBUG")
            logger.debug(msg)
            
            if display_flag  ==   True:
                print(f"{Log.Bold}{Log.font_colour[colour_name]}{msg}{Log.RESET}")

        @staticmethod
        def info_msg(colour_name='White',msg=None,display_flag=False):
            logger = Log.setup_logger("INFO")
            logger.info(msg)
            logger.debug(msg)

            if display_flag  ==   True:
                print(f"{Log.Bold}{Log.font_colour[colour_name]}{msg}{Log.RESET}",flush=True)

        @staticmethod
        def warning_msg(colour_name,msg,display_flag):
            
            logger = Log.setup_logger("WARNING")
            logger.warning(msg)
            logger.info(msg)
            logger.debug(msg)
        
            if display_flag  ==   True:
                print(f"{Log.Bold}{Log.font_colour[colour_name]}{msg}{Log.RESET}",flush=True)

        @staticmethod
        def error_msg(colour_name,msg,display_flag):
            
            logger = Log.setup_logger("ERROR")
            logger.error(msg)
            logger.warning(msg)
            logger.info(msg)
            logger.debug(msg)

            if display_flag  ==   True:
                print(f"{Log.Bold}{Log.font_colour[colour_name]}{msg}{Log.RESET}",flush=True)

        @staticmethod
        def critical_msg(colour_name,msg,display_flag):
            
            
            logger = Log.setup_logger("CRITICAL")
            logger.critical(msg)
            logger.error(msg)
            logger.warning(msg)
            logger.info(msg)
            logger.debug(msg)

            if display_flag  ==   True:
                print(f"{Log.Bold}{Log.font_colour[colour_name]}{msg}{Log.RESET}",flush=True)
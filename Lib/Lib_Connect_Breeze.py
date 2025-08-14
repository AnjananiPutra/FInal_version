import os
import sys
import glob
import dash
import math
import urllib
import zipfile
import openpyxl
import warnings

from time import sleep
from pyotp import TOTP
from selenium import webdriver
from breeze_connect import BreezeConnect


instrument_folder       = "E:/Algo_Trading_V3/Input Files/Instruments/"


class Connect_Breeze():
   
    def __init__(self):
        return
    
    @staticmethod
    def generate_token(api_key,userID,pwd,totp_key):
        try:
             
       
            browser         =   webdriver.Chrome()            
            #Logining Into website of ICICI Direct and Authenticate with username and password
            browser.get("https://api.icicidirect.com/apiuser/login?api_key="+urllib.parse.quote_plus(api_key))
            browser.implicitly_wait(5)
            username        =   browser.find_element("xpath",'/html/body/form/div[2]/div/div/div[1]/div[2]/div/div[1]/input')
            password        =   browser.find_element("xpath", '/html/body/form/div[2]/div/div/div[1]/div[2]/div/div[3]/div/input') 
            username.send_keys(userID)
            password.send_keys(pwd)
            
            #Checkbox
            browser.find_element("xpath", '/html/body/form/div[2]/div/div/div[1]/div[2]/div/div[4]/div/input').click()

            # Click Login Button
            browser.find_element("xpath", '/html/body/form/div[2]/div/div/div[1]/div[2]/div/div[5]/input[1]').click()
            sleep(2)
            
            #Sending OTP to authenticate
            pin             =   browser.find_element("xpath", '/html/body/form/div[2]/div/div/div[2]/div/div[2]/div[2]/div[3]/div/div[1]/input')
            totp            =   TOTP(totp_key)
            token           =   totp.now()
            pin.send_keys(token)
            
            
            #getting the token from URL
            browser.find_element("xpath", '/html/body/form/div[2]/div/div/div[2]/div/div[2]/div[2]/div[4]/input[1]').click()
            sleep(2)
            temp_token      =   browser.current_url.split('apisession=')[1][:8]
            print(f'token key = {temp_token}')
            return temp_token
        
        except:
            
            return "Error while generating token,Please try to reauthenticate on ICICI direct platform"
        
    
    @staticmethod
    def login(breeze):
        
        file            =   open(instrument_folder+"new_security.txt", "r")
        keys            =   file.read().split()  # Get a List of keys
        api_key         =   keys[0]
        key_secret      =   keys[1]
        breeze          =   BreezeConnect(api_key=api_key) 
        token           =   0
        counter         =   0 
        userID          =   keys[2]
        pwd             =   keys[3]
        totp_key        =   keys[4]        
        
        
        while counter <5:
            
              token     =   Connect_Breeze.generate_token(api_key,userID,pwd,totp_key)  
              
              if Connect_Breeze.validate_token(token) =="Success" :  
                  
                  breeze.generate_session(api_secret=key_secret,session_token=token)
                  breeze.ws_connect()
                  
                  counter     = 10
                  
              else:
                  
                  print(f"Attempt {counter}:Authentication unsuccessful.Trying Again.....")
            
              counter += 1
              
        return breeze
    
        
    
    
    @staticmethod
    def sign_out(breeze,Option_Chain,NIFTY_Stocks,NIFTY_Index):
        
       
        
        Option_Chain.unsubscribe_SP_list()
        NIFTY_Stocks.un_subscribe_ticks()
        NIFTY_Index.un_subscribe_ticks()
        
        breeze.ws_disconnect()
        
        
        return
    
    
    @staticmethod
    def log_back():
    
        try:

            msg           =   "Restarting session"
            
            
            Connect_Breeze.sign_out()
            
            Connect_Breeze.login()
   
               
            return "Success"
           
        except Exception as e:
            
            return e
    
    @staticmethod
    def validate_token(token):
        
        if len(token) > 1 and len(token) < 20:
            
            return "Success"
            
        else:
            
            return "Failed to get valid token"



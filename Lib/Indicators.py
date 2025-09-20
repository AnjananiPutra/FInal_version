import pandas as pd
import numpy as np
from datetime import datetime
import statistics
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from math import sqrt, log, exp
from scipy.optimize import brentq
from plotly.graph_objs import Indicator
from datetime import datetime, date, time, timedelta

class Indicators(object):

    def __init__(self):

        return

    @staticmethod
    def trailing_stop_loss(asset_df: pd.DataFrame,
                           stop_loss: float = 0.05,
                           mode: str = "pct",
                           eval_col_name: str = "close",
                           ref_col_name: str = "high") -> float:
        """
        Calculates a trailing stop loss level based on the selected mode and reference column.

        Parameters:
            asset_df (pd.DataFrame): DataFrame containing price columns.
            stop_loss (float, optional): Stop loss factor (percentage or multiplier). Defaults to 0.05.
            mode (str, optional): Mode of calculation ('std_dev', 'pct', or fallback). Defaults to 'pct'.
            eval_col_name (str, optional): Column used for volatility evaluation. Defaults to 'close'.
            ref_col_name (str, optional): Column used for reference level. Defaults to 'high'.

        Returns:
            float: Calculated trailing stop loss level.
        """
        try:
            # Validate required columns
            for col in [eval_col_name, ref_col_name]:
                if col not in asset_df.columns:
                    raise ValueError(f"DataFrame must contain '{col}' column")

            if ref_col_name == "high":
                ref_value = asset_df[ref_col_name].max()

                if mode == "std_dev":
                    std_dev = asset_df[eval_col_name].std()
                    trailing_level = ref_value - (stop_loss * std_dev)

                elif mode == "pct":
                    trailing_level = ref_value * (1 - stop_loss)

                else:
                    trailing_level = ref_value * 2  # Fallback logic

            elif ref_col_name == "low":
                ref_value = asset_df[ref_col_name].min()

                if mode == "std_dev":
                    std_dev = asset_df[eval_col_name].std()
                    trailing_level = ref_value + (stop_loss * std_dev)

                elif mode == "pct":
                    trailing_level = ref_value * (1 + stop_loss)

                else:
                    trailing_level = ref_value * 0.5  # Fallback logic for low

            else:
                raise ValueError(f"Unsupported ref_col_name: '{ref_col_name}'")

            return trailing_level

        except Exception as e:
            print("Error in trailing_stop_loss:")
            print(f"Details: {e}")
            return 0.0  # Safe fallback

    @staticmethod
    def count_ema_mean_up_crossovers(df: pd.DataFrame, column_name_indicator: str, threshold: float) -> int:
        """
        Counts the number of times the EMA crosses above a given threshold after being below it.

        Parameters:
            df (pd.DataFrame): DataFrame containing the EMA column.
            column_name_indicator (str): Name of the EMA column to evaluate.
            threshold (float): Threshold value to detect crossovers.

        Returns:
            int: Number of upward crossovers.
        """
        try:
            if column_name_indicator not in df.columns:
                raise ValueError(f"DataFrame must contain '{column_name_indicator}' column")

            # Boolean series: True when EMA is below a threshold
            below_threshold = df[column_name_indicator] < threshold

            count = 0
            in_below_phase = False

            for is_below in below_threshold:

                if is_below:
                    in_below_phase = True

                elif in_below_phase:
                    count += 1
                    in_below_phase = False

            return count

        except Exception as e:
            print("Error in count_ema_mean_up_crossovers:")
            print(f"Details: {e}")
            return 0  # Safe fallback

    @staticmethod
    def count_ema_mean_down_crossovers(df: pd.DataFrame, column_name_indicator: str, threshold: float) -> int:
        """
        Counts the number of times the EMA crosses below a given threshold after being above it.

        Parameters:
            df (pd.DataFrame): DataFrame containing the EMA column.
            column_name_indicator (str): Name of the EMA column to evaluate.
            threshold (float): Threshold value to detect crossovers.

        Returns:
            int: Number of downward crossovers.
        """
        try:
            if column_name_indicator not in df.columns:
                raise ValueError(f"DataFrame must contain '{column_name_indicator}' column")

            # Boolean series: True when EMA is above a threshold
            above_threshold = df[column_name_indicator] > threshold

            count = 0
            in_above_phase = False

            for is_above in above_threshold:

                if is_above:
                    in_above_phase = True

                elif in_above_phase:
                    count += 1
                    in_above_phase = False

            return count

        except Exception as e:
            print("Error in count_ema_mean_down_crossovers:")
            print(f"Details: {e}")
            return 0  # Safe fallback

    @staticmethod
    def uptrend_confirmation(df_sec, T_max, stop_loss, mode, eval_col_name='close', ref_col_name='high'):
        """
        Confirms an uptrend based on EMA crossovers and trailing stop loss logic.

        Parameters:
            df_sec (pd.DataFrame): Input DataFrame with price and indicator columns.
            T_max (int): Number of periods to evaluate.
            stop_loss (float): Stop loss percentage or value.
            mode (str): Mode for trailing stop loss calculation.
            eval_col_name (str, optional): Column used for mean evaluation. Defaults to 'close'.
            ref_col_name (str, optional): Column used for high/low reference. Defaults to 'low'.

        Returns:
            bool: True if uptrend is confirmed, False otherwise.
        """
        # Ensure proper slicing: last T_max rows
        df = df_sec.iloc[-T_max:]

        # Compute evaluation metrics
        mean_val = df[eval_col_name].mean()
        max_level = df[ref_col_name].max()
        current_high = df[ref_col_name].iloc[-1]

        # Count EMA crossovers
        count_up = Indicators.count_ema_mean_up_crossovers(df, 'EMA_n', mean_val)
        count_down = Indicators.count_ema_mean_down_crossovers(df, 'EMA_n', mean_val)

        # Trailing a stop loss threshold
        trailing_threshold = max_level - Indicators.trailing_stop_loss(df, stop_loss, mode)
        height_check = current_high > trailing_threshold

        # Uptrend confirmation logic
        return count_up >= 2 > count_down and height_check

    @staticmethod
    def downtrend_confirmation(df_sec, T_max, stop_loss, mode, eval_col_name='close', ref_col_name='low'):
        """
        Confirms a downtrend based on EMA crossovers and trailing stop loss logic.

        Parameters:
            df_sec (pd.DataFrame): Input DataFrame with price and indicator columns.
            T_max (int): Number of periods to evaluate.
            stop_loss (float): Stop loss percentage or value.
            mode (str): Mode for trailing stop loss calculation.
            eval_col_name (str, optional): Column used for mean evaluation. Defaults to 'close'.
            ref_col_name (str, optional): Column used for low/high reference. Defaults to 'low'.

        Returns:
            bool: True if downtrend is confirmed, False otherwise.
        """
        # Ensure proper slicing: last T_max rows
        df = df_sec.iloc[-T_max:]

        # Compute evaluation metrics
        mean_val = df[eval_col_name].mean()
        min_level = df[ref_col_name].min()
        current_low = df[ref_col_name].iloc[-1]

        # Count EMA crossovers
        count_up = Indicators.count_ema_mean_up_crossovers(df, 'EMA_n', mean_val)
        count_down = Indicators.count_ema_mean_down_crossovers(df, 'EMA_n', mean_val)

        # Trailing a stop loss threshold
        trailing_threshold = min_level + Indicators.trailing_stop_loss(df, stop_loss, mode)
        height_check = current_low < trailing_threshold

        # Downtrend confirmation logic
        return count_up < 2 <= count_down and height_check

    @classmethod
    def sideways_confirmation(cls, df, max_threshold, param, param1):

        try:

            pass



        except Exception as e:

                pass

        finally:

            pass

    @staticmethod
    def option_greeks_iv(Spot_price_S, Strike_price_K, Time_to_expiry_T, LT_price, option_type='call',
                         risk_free_rate_r=0.06):
        """
        Calculate option Greeks and implied volatility using the Black-Scholes model.

        Parameters:
        - Spot_price_S: Spot price of the underlying asset
        - Strike_price_K: Strike price of the option
        - Time_to_expiry_T: Time to expiry in years (if 0, it will be recalculated)
        - LT_price: Observed market premium of the option
        - option_type: 'call' or 'put'
        - risk_free_rate_r: Annualized risk-free interest rate (default = 6%)

        Returns:
        Dictionary with IV, Delta, Gamma, Theta, Vega, and Rho
        """

        # Recalculate time to expiry if T is zero
        if Time_to_expiry_T == 0:
            now                 = datetime.now()
            expiry_time         = datetime.combine(now.date(), time(15, 30))  # 3:30 PM IST
            minutes_remaining   = max((expiry_time - now).total_seconds() / 60, 1)  # avoid zero
            Time_to_expiry_T    = minutes_remaining / 525600  # convert minutes to fractional years

        # Black-Scholes pricing function
        def bs_price(sigma):

            d1 = (  log(Spot_price_S / Strike_price_K) +
                    (risk_free_rate_r + 0.5 * sigma ** 2) * Time_to_expiry_T) / (
                    sigma * sqrt(Time_to_expiry_T))

            d2 = d1 - sigma * sqrt(Time_to_expiry_T)

            if option_type == 'call':
                return Spot_price_S * norm.cdf(d1) - Strike_price_K * exp(
                    -risk_free_rate_r * Time_to_expiry_T) * norm.cdf(d2)
            else:
                return Strike_price_K * exp(-risk_free_rate_r * Time_to_expiry_T) * norm.cdf(
                    -d2) - Spot_price_S * norm.cdf(-d1)

        # The Estimate implied volatility using Brent's method
        try:
            iv = brentq(lambda sigma: bs_price(sigma) - LT_price, 0.0001, 5)
        except ValueError:
            return {
                "IV": None, "Delta": None, "Gamma": None,
                "Theta": None, "Vega": None, "Rho": None
            }

        # Calculate d1 and d2 using implied volatility
        d1 = (log(Spot_price_S / Strike_price_K) + (risk_free_rate_r + 0.5 * iv ** 2) * Time_to_expiry_T) / (
                    iv * sqrt(Time_to_expiry_T))
        d2 = d1 - iv * sqrt(Time_to_expiry_T)

        # Compute Greeks
        gamma = norm.pdf(d1) / (Spot_price_S * iv * sqrt(Time_to_expiry_T))
        vega = Spot_price_S * norm.pdf(d1) * sqrt(Time_to_expiry_T) / 100

        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (-Spot_price_S * norm.pdf(d1) * iv / (
                        2 * sqrt(Time_to_expiry_T)) - risk_free_rate_r * Strike_price_K * exp(
                -risk_free_rate_r * Time_to_expiry_T) * norm.cdf(d2)) / 365
            rho = Strike_price_K * Time_to_expiry_T * exp(-risk_free_rate_r * Time_to_expiry_T) * norm.cdf(d2) / 100
        else:
            delta = -norm.cdf(-d1)
            theta = (-Spot_price_S * norm.pdf(d1) * iv / (
                        2 * sqrt(Time_to_expiry_T)) + risk_free_rate_r * Strike_price_K * exp(
                -risk_free_rate_r * Time_to_expiry_T) * norm.cdf(-d2)) / 365
            rho = -Strike_price_K * Time_to_expiry_T * exp(-risk_free_rate_r * Time_to_expiry_T) * norm.cdf(-d2) / 100

        # Return rounded results
        return {
            "IV": round(iv * 100, 2),
            "Delta": round(delta, 4),
            "Gamma": round(gamma, 4),
            "Theta": round(theta, 4),
            "Vega": round(vega, 4),
            "Rho": round(rho, 4)
        }

    @staticmethod
    def get_option_greeks(greek_name: str,expiry_date: datetime,nifty_df_sec: pd.DataFrame,obj_list:list, right: str):
        try:

            outcome     =   []
            for each_obj in obj_list:

                price_df_sec        =  each_obj.call_df_sec if right =='call' else each_obj.put_df_sec
                Spot_price_S        = nifty_df_sec['close'].iloc[-1]                           # Spot price
                Strike_price_K      = each_obj.strike_price                                        # Strike price (choose based on strategy)
                Time_to_expiry_T    = (expiry_date.date() - datetime.now().date()).days / 365  # Time to expiry (7 days)
                risk_free_return_r  = 0.06                                                     # Risk-free rate (approximate annual)
                LT_price            = price_df_sec['close'].iloc[-1]                           # Option premium (example)
                option_type         = right.lower()                                            # right

                result              = Indicators.option_greeks_iv(Spot_price_S, Strike_price_K,
                                                                  Time_to_expiry_T, LT_price,
                                                                  option_type,risk_free_return_r)

                outcome.append(result)

            #filter greek from the result
            result_values = [item[greek_name] for item in outcome if item[greek_name] is not None]

            return result_values

        except Exception as e:

            print(f"Error in get_option_greeks: {e}")
            return []

    @staticmethod
    def generate_ohlc_data(
            min_price: float,
            max_price: float,
            start_datetime: datetime,
            frequency: str = "5min",
            num_candles: int = 20
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLC candlestick data.

        Parameters:
        - min_price: Minimum price boundary
        - max_price: Maximum price boundary
        - start_datetime: Starting datetime for the first candle
        - frequency: Time interval between candles (e.g., '5min', '15min')
        - num_candles: Number of candlesticks to generate

        Returns:
        - DataFrame with columns: timestamp, open, high, low, close
        """
        np.random.seed(42)  # For reproducibility

        # Generate timestamps
        timestamps = pd.date_range(start=start_datetime, periods=num_candles, freq=frequency)

        # Generate base prices within range
        base_prices = np.random.uniform(min_price + 10, max_price - 10, size=num_candles)

        # Generate OHLC values with realistic spreads
        opens = base_prices + np.random.uniform(-5, 5, size=num_candles)
        closes = base_prices + np.random.uniform(-5, 5, size=num_candles)
        highs = np.maximum(opens, closes) + np.random.uniform(1, 10, size=num_candles)
        lows = np.minimum(opens, closes) - np.random.uniform(1, 10, size=num_candles)

        # Clip all values to stay within bounds
        ohlc_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.clip(opens, min_price, max_price),
            "high": np.clip(highs, min_price, max_price),
            "low": np.clip(lows, min_price, max_price),
            "close": np.clip(closes, min_price, max_price)
        })

        return ohlc_df

'''
nifty_df = Indicators.generate_ohlc_data(min_price=24350, max_price=24450, start_datetime=datetime(2025, 8, 9), frequency="1min",num_candles=20)
SP_df = Indicators.generate_ohlc_data(min_price=83, max_price=90, start_datetime=datetime(2025, 8, 9), frequency="1min",num_candles=20)
print(f'test_data::{test_data}')
Spot_price_S = 24363.23  # Spot price
Strike_price_K = 24500  # Strike price (choose based on strategy)
Time_to_expiry_T = 6 / 365  # Time to expiry (7 days)
risk_free_return_r = 0.06  # Risk-free rate (approximate annual)
LT_price = 83.2  # Option premium (example)
option_type = 'call'

a = Indicators.option_greeks_iv(Spot_price_S, Strike_price_K, Time_to_expiry_T, LT_price, option_type='call',
                         risk_free_rate_r=0.06)
print(a)
'''


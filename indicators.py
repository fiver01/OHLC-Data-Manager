"""
Indicators

Description:
Calculate technical indicators for a DataFrame with Open, Close, High, Low data columns.

@author: Davide Bonanni
@Created on Fri Oct 28 18:26:23 2022

License:
This script is distributed under the GNU General Public License (GPL), Version 3, released on 29 June 2007.
You can find a copy of the license at: [Link to the GPL v3 License](https://www.gnu.org/licenses/gpl-3.0.txt).
"""

import numpy as np
import pandas as pd
import talib

class Indicators:
    open_col = 'open'
    close_col = 'close'
    high_col = 'high'
    low_col = 'low'
    volume_col = 'volume'
    date_col = 'date'

    def __init__(self, table):
        self.table = table
        self.blocks = None

    def _check_blocks(self):
        """
        Check if the blocks attribute is None and split the table into blocks if necessary.
        """
        if self.blocks == None:
            self.split_table_by_nan()

    def split_table_by_nan(self):
        """
        Split the table into blocks based on rows containing NaN values.

        Returns:
            list: List of DataFrames representing each block.
        """

        blocks = []
        current_block = []

        for _, row in self.table.iterrows():
            if row.isnull().any():
                if current_block:
                    blocks.append(pd.DataFrame(current_block))
                    current_block = []
            else:
                current_block.append(row)

        if current_block:
            blocks.append(pd.DataFrame(current_block))

        self.blocks = blocks
        return blocks

    def add_SMA(self, timeperiod=21):
        """ Add the Simple Moving Average to the dataframe. """
        timeperiod = self.check_timeperiod(timeperiod)
        for t in timeperiod:
            col_name = 'SMA_{}'.format(t)
            for b in self.blocks:
                b[col_name] = talib.SMA(b[self.close_col], timeperiod=t)
                self.update_column(b[col_name])

    def add_EMA(self, timeperiod=21):
        """ Add the Exponential Moving Average to the dataframe. """
        timeperiod = self.check_timeperiod(timeperiod)
        for t in timeperiod:
            col_name = 'EMA_{}'.format(t)
            for b in self.blocks:
                b[col_name] = talib.EMA(b[self.close_col], timeperiod=t)
                self.update_column(b[col_name])

    def add_ATR(self, timeperiod=24):
        """ Add the Average True Range to the dataframe. """
        timeperiod = self.check_timeperiod(timeperiod)
        for t in timeperiod:
            col_name = 'ATR_{}'.format(t)
            for b in self.blocks:
                b[col_name] = talib.ATR(b[self.high_col], b[self.low_col], b[self.close_col], timeperiod=t)
                self.update_column(b[col_name])

    def add_NATR(self, timeperiod=14):
        """ Add the Normalized Average True Range to the dataframe. """
        timeperiod = self.check_timeperiod(timeperiod)
        for t in timeperiod:
            col_name = 'NATR_{}'.format(t)
            for b in self.blocks:
                b[col_name] = talib.NATR(b[self.high_col], b[self.low_col], b[self.close_col], timeperiod=t)
                b[col_name] = b[col_name] / 100
                self.update_column(b[col_name])

    def add_RSI(self, timeperiod=14):
        """ Add the Relative Strength Index to the dataframe. """
        timeperiod = self.check_timeperiod(timeperiod)
        for t in timeperiod:
            col_name = 'RSI_{}'.format(t)
            for b in self.blocks:
                b[col_name] = talib.RSI(b[self.close_col], timeperiod=t)
                self.update_column(b[col_name])

    def add_STOCHRSI(self, timeperiod=14, fastk_period=14, slowk_period=5, slowd_period=3):
        """ Add the Relative Strength Index to the dataframe. """
        timeperiod = self.check_timeperiod(timeperiod)
        for t in timeperiod:
            col_name1 = f'SRSIK_{t}'
            col_name2 = f'SRSID_{t}'
            for b in self.blocks:
                rsi = talib.RSI(b[self.close_col], timeperiod=t)
                if pd.isna(rsi).all():
                    b[col_name1] = np.NaN
                    b[col_name2] = np.NaN
                else:
                    fastk, fastd = talib.STOCH(rsi, rsi, rsi, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=0,
                                               slowd_period=slowd_period, slowd_matype=0)
                    b[col_name1] = fastk
                    b[col_name2] = fastd
                self.update_column(b[col_name1])
                self.update_column(b[col_name2])

    def add_MACD(self, fastperiod=12, slowperiod=26, signalperiod=9):
        """ Add the MACD to the dataframe. """
        for b in self.blocks:
            b['MACD'], b['Signal'], b['Histogram'] = talib.MACD(b[self.close_col], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
            self.update_column(b['MACD'])
            self.update_column(b['Signal'])
            self.update_column(b['Histogram'])

    def add_BBANDS(self, timeperiod=14, nbdevup=2, nbdevdn=2):
        """ Add the Bollinger bands to the dataframe. """
        timeperiod = self.check_timeperiod(timeperiod)
        for t in timeperiod:
            for b in self.blocks:
                col_name1, col_name2, col_name3 = 'UpperBand_{}'.format(t), 'MiddleBand_{}'.format(t), 'LowerBand_{}'.format(t)
                b[col_name1], b[col_name2], b[col_name3] = talib.BBANDS(b[self.close_col], timeperiod=t, nbdevup=nbdevup, nbdevdn=nbdevdn)
                self.update_column(b[col_name1])
                self.update_column(b[col_name2])
                self.update_column(b[col_name3])

    def add_OBV(self):
        """ """
        self.table['OBV'] = talib.OBV(self.table.close, self.table.volume)

    def add_STOCH(self, fastk_period=14, slowk_period=3, **kwargs):
        """ Add the Stochastic Oscillator to the dataframe. """
        for b in self.blocks:
            k, d = talib.STOCH(b[self.high_col], b[self.low_col], b[self.close_col], fastk_period=14, **kwargs)[0]
            b[f'%K_{fastk_period}'] = k
            b[f'%D_{fastk_period}'] = d
            self.update_column(b[f'%K_{fastk_period}'])
            self.update_column(b[f'%D_{fastk_period}'])

    def update_column(self, series_to_update:pd.Series):
        """
        Update a column in the table with the values from the provided series.

        If the column already exists in the table, the values will be overwritten.
        If the column does not exist, a new column will be added to the table.

        Args:
            series_to_update (pd.Series): Series containing the values to update or add.
        """
        col_name = series_to_update.name
        if col_name in self.table.columns:
            self.table.update(series_to_update, overwrite=True)
        else:
            self.table = self.table.join(series_to_update)

    def add_hour_timefeatures(self):
        """ Add the time features for the hour OCHL. """
        self._add_hour_features()
        self._add_daily_features()
        self._add_weekly_features()
        self._add_year_features()

    def add_daily_timefeatures(self):
        """ Add the time features for the daily OCHL. """
        self._add_daily_features()
        self._add_weekly_features()
        self._add_year_features()

    def add_weekly_timefeatures(self):
        """ Add the time features for the weekly OCHL. """
        self._add_weekly_features()
        self._add_year_features()

    def _add_hour_features(self):
        """ Add the sin and cos of the hour of the day. """
        hour_of_day_sin  = np.sin(2 * np.pi * self.table['date'].dt.hour / 24.0)
        hour_of_day_cos  = np.cos(2 * np.pi * self.table['date'].dt.hour / 24.0)
        self.table = self.table.assign(sin_hour=hour_of_day_sin, cos_hour=hour_of_day_cos)

    def _add_daily_features(self):
        """ Add the sin and cos of the day of the year. """
        day_of_year_sin = np.sin(2 * np.pi * self.table['date'].dt.dayofyear / 365.0)
        day_of_year_cos = np.cos(2 * np.pi * self.table['date'].dt.dayofyear / 365.0)
        self.table = self.table.assign(sin_day=day_of_year_sin, cos_day=day_of_year_cos)

    def _add_weekly_features(self):
        """ Add the sin and cos of the day of the week. """
        day_of_week_sin = np.sin(2 * np.pi * self.table['date'].dt.dayofweek / 7.0)
        day_of_week_cos = np.cos(2 * np.pi * self.table['date'].dt.dayofweek / 7.0)
        self.table = self.table.assign(sin_week=day_of_week_sin, cos_week=day_of_week_cos)

    def _add_year_features(self, period=20):
        """ Add the sin and cos of the day of the week. """
        year_sin = np.sin(2 * np.pi * self.table['date'].dt.year / period)
        year_cos = np.cos(2 * np.pi * self.table['date'].dt.year / period)
        self.table = self.table.assign(sin_year=year_sin, cos_year=year_cos)

    @classmethod
    def check_timeperiod(cls, timeperiod):
        if isinstance(timeperiod, list):
            return timeperiod
        elif isinstance(timeperiod, int):
            return [timeperiod]
        else:
            raise ValueError(f"Invalid timeperiod format {type(timeperiod)}")

    def add_hour_indicators(self):
        """ Add standard indicators for hour OCHL data. """
        self._check_blocks()
        self.add_NATR(timeperiod=24)
        self.add_RSI(timeperiod=24)
        self.add_STOCHRSI(timeperiod=24)
        self.add_MACD()
        self.add_SMA(timeperiod=[24, 72])
        self.add_EMA(timeperiod=[24, 72])

    def add_daily_indicators(self):
        """ Add standard indicators for daily OCHL data. """
        self._check_blocks()
        self.add_NATR(timeperiod=14)
        self.add_MACD()
        self.add_BBANDS()
        self.add_RSI()
        self.add_STOCHRSI()
        self.add_STOCH()
        self.add_MACD()
        self.add_SMA(timeperiod=[20, 50, 100, 200])
        self.add_EMA(timeperiod=[20, 50, 100, 200])

    def add_weekly_indicators(self):
        """ Add standard indicators for weekly OCHL data. """
        self._check_blocks()
        self.add_RSI(timeperiod=12)
        self.add_STOCHRSI(timeperiod=12)
        self.add_SMA(timeperiod=[10, 40])
        self.add_EMA(timeperiod=[20, 40])



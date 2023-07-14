"""
OCHL Manager

Description:
OCHLManager is a Python class that facilitates the management and analysis of Open, Close, High, Low (OCHL) data in a DataFrame.

@author: Davide Bonanni
@Created on Fri Oct 28 18:26:23 2022

License:
This script is distributed under the GNU General Public License (GPL), Version 3, released on 29 June 2007.
You can find a copy of the license at: [Link to the GPL v3 License](https://www.gnu.org/licenses/gpl-3.0.txt).
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from indicators import Indicators


# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(filename)s:%(levelname)s: %(message)s',
    filename='ochl_manager.log',  # Specify the file name
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w'  # 'w' for overwrite the file on each run, 'a' to append to an existing file
)


class OCHLManager(Indicators):
    """ Manage DataFrame with Open, Close, High, Low data columns. """
    open_col = 'open'
    close_col = 'close'
    high_col = 'high'
    low_col = 'low'
    volume_col = 'volume'
    date_col = 'date'

    def __init__(self, df:pd.DataFrame, pair_token='USDT', count_volume_anomalies=False):
        """
        If multiple input dataframes, concat the OCHL dataframes without duplicates and
        check the df columns. Input tables must contain "date", "open", "close", "high" and "low"
        columns. Other columns are optional.
        """
        self.table = _check_dataframe(df)
        self.anomaly_list = None
        self.count_volume_anomalies = count_volume_anomalies
        check_ochl(self.table)
        self.rename_column(f'Volume {pair_token}', 'volume')

        Indicators.__init__(self, self.table)

    def __call__(self):
        return self.table
    
    def _check_columns_in_df(self, *columns):
        """Check if the columns are in the self.table"""
        for c in columns:
            if not c in self.table.columns:
                return False
            else:
                return True
        
    def add_table(self, new_table, start_index=None, last_index=None):
        """Append OCHL row of the arguments to the object OCHLTable"""
        
        new_table = _check_dataframe(new_table)
        
        try:
            new_table = new_table[start_index:last_index]
            if new_table.empty:
                print("Warning: adding an empty DataFrame, check input dataframe or indexes")

            self.table = pd.concat([self.table, new_table], 
                                   axis=0, 
                                   ignore_index=False,).drop_duplicates()
            
        except (TypeError, AttributeError) as t_err:
            raise Exception("Check type of added file. {}, {}".format(t_err, type(t_err)))
        except Exception as err:
            print("Something wrong with the dataframe concatenation of OCHLTable")
            raise Exception("Unexpected {}, {}".format(err, type(err)))

    def convert_column_to_datetime(self, columns='date', UTC_correction=0):
        """Convert df input date column (str) to datetime[ns] format"""
        columns = check_list(columns)
        self.table[columns] = self.table[columns].applymap(lambda x: _timestamp_to_datetime(x, UTC_correction=UTC_correction))

    def unix_to_UTC(self, unix_col='unix'):
        """ Generate a new UTC column from unix column. """
        #self.table[unix_col] = self.table[unix_col].astype(int)
        self.table['UTC'] = self.table.apply(lambda x: unix_to_datetime(x[unix_col]), axis=1)

    def rename_column(self, orig, new):
        """ Rename column if present in the table. """
        if orig in self.table.columns:
            self.table.rename(columns={orig : new}, inplace=True)

    def prepare_data(self):
        """
        Prepare the data in the table to further processing.

        This function executes a series of data preparation steps in a specific order:
        - Converts the 'date' column to datetime format.
        - Cleans the column names by removing leading/trailing spaces and converting to lowercase.
        - Interpolates missing values in the 'volume' column.
        - Orders the rows in the table by date.

        After the initial preprocessing, additional steps are performed:
        - Invalidates anomalies in the table.
        - Fixes inconsistencies in the 'open' column.
        - Fills in missing dates in the table.
        """

        self.convert_column_to_datetime()
        self.clean_columns()
        self.interpolate_column(self.volume_col)
        self.order_by_date()

        self.invalidate_anomalies()
        self.fix_inconsistency_open()
        self.fill_missing_dates()

        # Count the number of rows with at least one NaN value
        num_nan = self.table.isna().any(axis=1).sum()
        print(f"The table contains {num_nan} incomplete rows on {len(self.table)}")

    def clean_columns(self, col_to_mantain=None, col_to_drop=None):
        """
        Remove all columns except: date, open, close, high, low, volume.
        Additional columns to maintain or drop can be specified in the argument.
        """
        col_to_mantain = check_list(col_to_mantain)
        col_to_drop = check_list(col_to_drop)
        columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        columns = list(set(columns).union(col_to_mantain))
        columns = list(set(columns).difference(col_to_drop))
        self.table = self.table[columns]
        self.table = move_columns_to_front(self.table, ['date', 'open', 'close', 'high', 'low', 'volume'])

    def order_by_date(self):
        """ Reorder the rows by the descending date. """
        self.table = self.table.sort_values(self.date_col, ascending=True)
        self.table.reset_index(drop=True, inplace=True)

    def report(self):
        """
        Return a dictionary that contains information about the data.

        The dictionary includes the following information:
        - 'number_gaps': The number gaps in the date column (in hours).
        - 'max_gap': The maximum gaps in hour present in the data.
        - 'number_anomalies': The number of detected anomalies in the data.
        - 'zero_volumes': The number of rows with a volume value of 0.

        Returns:
            dict: A dictionary containing the data information.
        """
        max_gap = int(self.find_inconsecutives().total_seconds() / 3600)
        number_gaps = len(self.inconsecutives)
        anomalies = self.find_anomalies()
        number_anomalies = len(anomalies)
        zero_volumes = len(self.table[self.table[self.volume_col]==0])

        report_dict = dict(number_gaps=number_gaps, max_gap=max_gap,
                           number_anomalies=number_anomalies, zero_volumes=zero_volumes)

        for key, value in report_dict.items():
            print(f"{key}: {value}")

        return report_dict

    def invalidate_anomalies(self):
        """ Replace anomaly OCHLV values with NaN. """
        if not isinstance(self.anomaly_list, list):
            self.find_anomalies()
        for a in self.anomaly_list:
            self._invalidate_point(a)
        print(f"Removed {len(self.anomaly_list)} anomalies.")

    def _invalidate_point(self, loc):
        """ Transform OCHLV column into nan for at the argument location. """
        self.table.iloc[loc, self.table.columns.get_loc(self.open_col)] = np.NaN
        self.table.iloc[loc, self.table.columns.get_loc(self.close_col)] = np.NaN
        self.table.iloc[loc, self.table.columns.get_loc(self.high_col)] = np.NaN
        self.table.iloc[loc, self.table.columns.get_loc(self.low_col)] = np.NaN
        self.table.iloc[loc, self.table.columns.get_loc(self.volume_col)] = np.NaN

    def find_anomalies(self):
        """ Return a list with the location of the anomalies. """
        anomaly_list = []
        for i in range(len(self.table) - 1):
            # Check if the date are consecutive
            if self.check_anomaly(i):
                logging.info(f'Anomaly at {i}')
                anomaly_list.append(i)
        logging.info(f"{len(anomaly_list)} anomalies identified.")
        self.anomaly_list = anomaly_list
        return anomaly_list

    def check_anomaly(self, ref_location):
        """
        Check if there is an anomaly in the OCHL location of the table.

        Anomalies are detected based on the values in the 'open', 'close', 'high', 'low',
        and 'volume' columns at the specified reference location.

        Args:
            ref_location (int): The index of the reference location in the table.

        Returns:
            bool: True if there is an anomaly, False otherwise.

        Anomaly Detection Criteria:
        - Inconsistent Values: If there are fewer than or equal to two unique values among
          the 'open', 'close', 'high', and 'low' columns, it indicates an inconsistency.
        - Equal Open and Close Prices: If the 'open' price is equal to the 'close' price,
          it suggests no price movement during the time period, which can be considered an anomaly.
        - High Price Inconsistency: If the 'high' price is lower than the 'open', 'close',
          or 'low' price, it indicates a discrepancy in the data.
        - Low Price Inconsistency: If the 'low' price is higher than the 'open', 'close',
          or 'high' price, it signifies an inconsistency in the data.
        - Zero or Negative Volume: If the 'volume' value is zero or negative and the
          `count_volume_anomalies` flag is enabled, it is considered an anomaly.

        Note:
        Anomalies may indicate irregularities or data quality issues in the OCHL dataset.
        By identifying anomalies, you can take appropriate actions, such as data cleaning
        or further investigation, to ensure data integrity.
        """
        row = self.table.iloc[ref_location]
        selected_columns = [self.open_col, self.close_col, self.high_col, self.low_col]

        if len(row[selected_columns].unique()) <= 2:
            return True
        elif row[self.open_col] == row[self.close_col]:
            return True
        elif row[self.high_col] < row[self.open_col] or row[self.high_col] < row[self.close_col] or row[self.high_col] < row[self.low_col]:
            return True
        elif row[self.low_col] > row[self.open_col] or row[self.low_col] > row[self.close_col] or row[self.low_col] > row[self.high_col]:
            return True
        elif row[self.volume_col] <= 0 and self.count_volume_anomalies:
            return True
        else:
            return False

    def plot_anomalies(self):
        """ Show the plot of the anomalies on the data timeline. """

        # Create a list of y-coordinates for the anomalies (using a constant value for simplicity)
        anomaly_y = [0] * len(self.anomaly_list)

        # Create the line plot
        plt.plot(range(len(self.table)), [0] * len(self.table), color='blue', label='Series')
        plt.scatter(self.anomaly_list, anomaly_y, color='red', label='Anomalies')

        # Set labels and title
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Anomaly Distribution')
        plt.show()

    def fix_inconsistency_open(self, time_diff=1, time_unit='hours', tolerance=0.1):
        """
        Fixes the inconsistency in OHLC Table, where the close price at time T
        doesn't correspond to the open price at time T+1.
        """

        inconsistency_counter = 0
        for i in range(len(self.table) - 1):
            # Check if the date are consecutive
            if not self.check_consecutive_date(i, time_diff, time_unit):
                logging.info(f'Not consecutive date at {i}')
                continue

            close_t = self.table.iloc[i][self.close_col]  # Close price at time T
            open_T_plus_1 = self.table.iloc[i + 1][self.open_col]  # Open price at time T+1
            difference = abs(close_t - open_T_plus_1)

            if difference > (close_t * tolerance/100):
                logging.info(f'Fixed inconsistency at {i}')
                inconsistency_counter += 1
                self.table.iloc[i + 1][self.open_col] = close_t  # Replace the open price at T+1 with the close price at T
        print(f"Fixed {inconsistency_counter} inconsistencies. ")
        return self.table

    def find_inconsecutives(self, time_diff:int=1, time_unit='hours'):
        """ Find the points where the date shift more than the time unit. """
        max_timedelta = timedelta(0)
        self.inconsecutives = []
        for i in range(len(self.table) - 1):
            # Check if the date are consecutive
            consecutive, time_delta = self.check_consecutive_date(i, time_diff, time_unit)
            if not consecutive:
                self.inconsecutives.append(i)
                logging.info(f'Not consecutive date at {i} with {time_delta} time delta.')
                if time_delta > max_timedelta:
                    max_timedelta = time_delta
                continue
        return max_timedelta

    def fill_missing_dates(self, freq='H'):
        """ Add missing index rows. """
        starting_len = len(self.table)
        # Generate a complete date range input frequency
        date_range = pd.date_range(start=self.table[self.date_col].min(), end=self.table[self.date_col].max(), freq=freq)
        # Set the date column as the index
        self.table.set_index(self.date_col, inplace=True, drop=True)
        new_df = pd.DataFrame(index=date_range)
        df_filled = new_df.combine_first(self.table)
        df_filled = df_filled.reset_index().rename(columns={'index': 'date'})

        print(f"Filled {len(df_filled)-starting_len} missing points")
        self.table = df_filled

    def check_consecutive_date(self, ref_index, time_diff:int=1, time_unit='hours') -> bool:
        """
        Checks if two points have consecutive dates based on a given amount of time.

        Args:
            ref_index (int): The index used to retrieve the date for comparison.
            time_diff (int): The amount of time that should separate consecutive dates. Defaults to 1.
            time_unit (str): The unit of time for the time difference. Defaults to 'hours'.

        Returns:
            bool: True if the dates are consecutive, False otherwise.
        """
        current_date = self.table.iloc[ref_index][self.date_col]
        next_date = self.table.iloc[ref_index + 1][self.date_col]
        if (next_date - current_date) == timedelta(**{time_unit: time_diff}):
            return True, None
        else:
            logging.info(f'Date jump of {(next_date - current_date)}')
            return False, (next_date - current_date)

    def interpolate_column(self, interpolate_columns):
        """ Interpolate 0 values on the input columns filling the missing data. """
        interpolate_columns = check_list(interpolate_columns)
        for col in interpolate_columns:
            self.table[col] = self.table[col].replace(0, np.nan).interpolate()

    def generate_daily(self):
        """ Return a dataframe with daily OCHL from the hour data. """
        # daily_data = copy.deepcopy(self.table)
        # daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data = self.table.groupby(pd.Grouper(key='date', freq='D')).agg(
            {'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min', 'volume': 'sum'})
        daily_data.reset_index(inplace=True)
        return daily_data

    def generate_weekly(self):
        """ Return a dataframe with weekly OCHL from the hour data. """
        weekly_data = self.table.groupby(pd.Grouper(key='date', freq='W')).agg(
            {'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min', 'volume': 'sum'})
        weekly_data.reset_index(inplace=True)
        weekly_data['date'] = weekly_data['date'] + timedelta(days=1)
        return weekly_data


def check_ochl(df, columns_lst=['date', 'open', 'close', 'high', 'low']):
    """
    Check the presence of columns name in columns_lst.

    Parameters
    ----------
    columns_lst : list
        List of columns name to seek in the df

    df : pandas.DataFrame
        Input DataFrame
    """
    # Check if df is a pandas.DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input {} is not a pandas.DataFrame".format(df))

    missing_columns = []
    for col in columns_lst:
        if not col in df.columns:
            missing_columns.append(col)
    if missing_columns:
        raise MissingAttribute(missing_columns)
        # print("Column {} missing ".format(col)


class MissingAttribute(AttributeError):
    def __init__(self, *args):
        if args:
            self.message = args[0]

    def __str__(self):
        if self.message:
            return "Columns in {0} are missing from the DataFrame".format(self.message)
        else:
            return "Some columns are missing from the DataFrame"
    pass


def unix_to_datetime(unix_value):
    """ Convert Unix to datetime managing the exceptions return None. """
    try:
        return datetime.utcfromtimestamp(unix_value)
    except:
        return np.NaN


def _timestamp_to_datetime(timestp, UTC_correction=0):
    """ Convert timestp to datetime. """

    if not isinstance(timestp, str):
        return timestp

    try:
        if UTC_correction >= 0:
            return datetime.strptime(timestp, '%Y-%m-%d %I-%p') + timedelta(hours=abs(UTC_correction))
        elif UTC_correction < 0:
            return datetime.strptime(timestp, '%Y-%m-%d %I-%p') - timedelta(hours=abs(UTC_correction))
    except ValueError:
        pass
    try:
        if UTC_correction >= 0:
            return datetime.strptime(timestp, '%d/%m/%Y %H:%S') + timedelta(hours=abs(UTC_correction))
        elif UTC_correction < 0:
            return datetime.strptime(timestp, '%d/%m/%Y %H:%S') - timedelta(hours=abs(UTC_correction))
    except Exception:
        raise Exception


def move_columns_to_front(df, cols_to_move):
    """ """
    #df = df.drop(cols_to_move, axis=1)
    for col in cols_to_move[::-1]:
        #df = df.drop(col, axis=1)
        df.insert(0, col, df.pop(col))
    return df


def check_list(inp):
    """
    Check if the input is a list. Transform the input str into a list with one element.
    If the input is None return None. In all other cases return TypeError.
    """
    if isinstance(inp, list):
        return inp
    elif isinstance(inp, str):
        return [inp]
    elif inp is None:
        return list()
    else:
        raise TypeError("Wrong input format")


def _check_dataframe(i, separetor=','):
    """ Check if input is a dataframe or a path. If input is path convert it to a pd.DataFrame. """

    if isinstance(i, pd.DataFrame):
        return i
    elif isinstance(i, str):
        try:
            df = pd.read_csv(i, sep=separetor)
            return df
        except:
            raise AttributeError("Invalid path")
    else:
        return i




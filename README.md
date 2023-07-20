# OHLC-Data-Manager  

OHLCManager is a Python class that facilitates the management and analysis of Open, High, Low, Close, (OHLC) data in a DataFrame.

## Features

- **DataFrame Management:** OHLCManager allows you to easily handle OHLC data stored in a Pandas DataFrame.
- **Data Validation:** The class validates the DataFrame structure, ensuring that it contains the necessary columns: "date", "open", "close", "high", and "low".
- **Concatenation:** If multiple input DataFrames are provided, OHLCManager can concatenate the OHLC tables, removing duplicates.
- **Visualization:** OHLCManager offering convenient plotting and visualization functions for the OHLC data.
- **Indicators:** The class also inherits from the Indicators class, providing access to various technical indicators for further analysis.
- **Missing Data Detection:** OHLCManager includes functionality to identify missing points in the data, allowing you to easily find gaps or missing entries in the OHLC series.
- **Wrong Data Detection:** The class provides the ability to detect wrong data, such as inconsistent OHLC prices, helping with the cleaning process.
  
## Usage

To use OHLCManager, simply instantiate the class with your OCHL DataFrame. The class provides methods for data manipulation, visualization, and applying technical indicators.
The class was originally designed to handle 1-hour candlestick charts for cryptocurrencies but can be easily adapted to handle different timeframes and stock market data. Furthermore, the class is capable of generating daily and weekly timeframes based on hourly data.

Example usage on BTC/USDT pair OHLC data obtained from https://www.CryptoDataDownload.com.

```python
import pandas as pd
from ohlc_manager import OHLCManager

BTC_ohlc = 'sample_data/Binance_BTCUSDT_1h.csv'
ohlc_data = pd.read_csv(BTC_ohlc)

# Instantiate OHLCManager
ohlc_manager = OHLCManager(ohlc_data)

# Convert columns to datetime
self.convert_column_to_datetime()

# Remove additional columns
self.clean_columns()

# Order the data
self.order_by_date()
```
The number of gaps and anomalies in the data can checked with the report method:
```python
# Print the number of gaps and anomalies
report_dict = ohlc_manager.report()
```
Missing or wrong data can be easily handled:
```python
# Transfrom anomalies point into NaN
self.invalidate_anomalies()

# Remove large gaps between consecutives close and open price
self.fix_inconsistency_open()

# Fill missing data points
self.fill_missing_dates()
```
All this operations can be performed in once with:
```python
# Clean and prepare the data
ochl_manager.prepare_data()
```
Then, technical indicators can be implemented on the table:
```python
ochl_manager.add_SMA(timeperiod=20)
ochl_manager.add_RSI()
ochl_manager.add_STOCHRSI()
ochl_manager.add_MACD()
ochl_manager.add_BBANDS()
```
By utilizing hourly data, it is possible to generate consistent daily and weekly OHLC data across various timeframes:
```python
daily_ohlc = ohlc_manager.generate_daily()
weekly_ohlc = ohlc_manager.generate_weekly()
```





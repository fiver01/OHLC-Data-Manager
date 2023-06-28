# OCHL-Data-Manager  

OCHLManager is a Python class that facilitates the management and analysis of Open, Close, High, Low (OCHL) data in a DataFrame.

## Features

- **DataFrame Management:** OCHLManager allows you to easily handle OCHL data stored in a Pandas DataFrame.
- **Data Validation:** The class validates the DataFrame structure, ensuring that it contains the necessary columns: "date", "open", "close", "high", and "low".
- **Concatenation:** If multiple input DataFrames are provided, OCHLManager can concatenate the OCHL tables, removing duplicates.
- **Visualization:** OCHLManager extends MPLManager, offering convenient plotting and visualization functions for the OCHL data.
- **Indicators:** The class also inherits from the Indicators class, providing access to various technical indicators for further analysis.
- **Missing Data Detection:** OCHLManager includes functionality to identify missing points in the data, allowing you to easily find gaps or missing entries in the OCHL series.
- **Wrong Data Detection:** The class provides the ability to detect wrong data, such as inconsistent OCHL prices, helping with the cleaning process.
  
## Usage

To use OCHLManager, simply instantiate the class with your OCHL DataFrame. The class provides methods for data manipulation, visualization, and applying technical indicators.
The class was originally designed to handle 1-hour candlestick charts for cryptocurrencies but can be easily adapted to handle different timeframes and stock market data. Furthermore, the class is capable of generating daily and weekly timeframes based on hourly data.

Example usage on BTC/USDT pair OCHL data obtained from https://www.CryptoDataDownload.com.

```python
import pandas as pd
from ochl_manager import OCHLManager

BTC_ochl = 'sample_data/Binance_BTCUSDT_1h.csv'
ochl_data = pd.read_csv(BTC_ochl)

# Instantiate OCHLManager
ochl_manager = OCHLManager(ochl_data)

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
report_dict = ochl_manager.report()
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
By utilizing hourly data, it is possible to generate consistent daily and weekly OCHL data across various timeframes:
```python
daily_ochl = ochl_manager.generate_daily()
weekly_ochl = ochl_manager.generate_weekly()
```





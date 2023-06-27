# OCHL-Data-Manager  

OCHLManager is a Python class that facilitates the management and analysis of Open, Close, High, Low (OCHL) data in a DataFrame.

## Features

- **DataFrame Management:** OCHLManager allows you to easily handle OCHL data stored in a Pandas DataFrame.
- **Data Validation:** The class validates the DataFrame structure, ensuring that it contains the necessary columns: "date", "open", "close", "high", and "low".
- **Concatenation:** If multiple input DataFrames are provided, OCHLManager can concatenate the OCHL tables, removing duplicates.
- **Visualization:** OCHLManager extends MPLManager, offering convenient plotting and visualization functions for the OCHL data.
- **Indicators:** The class also inherits from the Indicators class, providing access to various technical indicators for further analysis.
- **Missing Data Detection:** OCHLManager includes functionality to identify missing points in the data, allowing you to easily find gaps or missing entries in the OCHL series.
- **Anomaly Detection:** The class provides the ability to detect anomalies in the data, such as abnormal price movements or volume spikes, helping you identify potential outliers or unusual patterns.
  
## Usage

To use OCHLManager, simply instantiate the class with your OCHL DataFrame. The class provides methods for data manipulation, visualization, and applying technical indicators.
The class was originally designed to handle 1-hour candlestick charts for cryptocurrencies but can be easily adapted to handle different timeframes and stock market data. Furthermore, the class is capable of generating daily and weekly timeframes based on hourly data.

Example usage:

```python
import pandas as pd
from ochl_manager import OCHLManager

BTC_ochl = 'sample_data/Binance_BTCUSDT_1h.csv'
ochl_data = pd.read_csv(BTC_ochl)

# Instantiate OCHLManager
ochl_manager = OCHLManager(ochl_data)

# Identify the anomalies and missing points
ochl_manager.report()

# Clean and prepare the data
ochl_manager.prepare_data()

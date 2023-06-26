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

To use OCHLManager, simply instantiate the class with your OCHL DataFrame. You can specify optional parameters such as the start and last index, a column for scaling, and whether to count volume anomalies. The class provides methods for data manipulation, visualization, and applying technical indicators.

Example usage:

```python
import pandas as pd
from OCHLManager import OCHLManager

# Instantiate OCHLManager
ochl_manager = OCHLManager(ochl_data)

# Access various functionalities
ochl_manager.plot_ochl()
ochl_manager.calculate_sma(20)
ochl_manager.table.head()

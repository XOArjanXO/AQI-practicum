import pandas as pd
import numpy as np
import datetime
from pathlib import Path

# Create directory
Path('data/raw').mkdir(parents=True, exist_ok=True)

# Generate synthetic data
np.random.seed(42)
num_rows = 1000
dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(hours=i) for i in range(num_rows)]

data = {
    'Timestamp': dates,
    'PM10': np.random.uniform(5, 100, num_rows),
    'Temperature': np.random.uniform(-5, 35, num_rows),
    'Humidity': np.random.uniform(20, 95, num_rows),
    'WindSpeed': np.random.uniform(0, 15, num_rows),
    'WindDirection': np.random.uniform(0, 360, num_rows),
    'Pressure': np.random.uniform(990, 1030, num_rows),
    'Rainfall': np.random.uniform(0, 5, num_rows),
}

df = pd.DataFrame(data)
df.to_csv('data/raw/airnet_raw.csv', index=False)
print("Generated synthetic 'data/raw/airnet_raw.csv'")

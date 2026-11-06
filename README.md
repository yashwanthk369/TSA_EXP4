# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 

### AIM:
To implement ARMA model in python.

### DATASET
sensor fault detection dataset

### SOFTWARE REQUIREMENT
Juypter Notebook

### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv('sensor-fault-detection.csv', sep=';')

df.columns = df.columns.str.strip().str.lower()
print("Columns found:", df.columns)

if 'sensorid' in df.columns:
    sensor_id = 1
    df = df[df['sensorid'] == sensor_id].copy()
    print(f"Filtered for sensor ID: {sensor_id}")

if df.empty:
    raise ValueError("No data found for the specified sensor!")

df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)

df = df.set_index('timestamp').sort_index()

df = df.groupby('timestamp')['value'].mean().asfreq('B').ffill().to_frame()

X = df['value']

plt.figure(figsize=(12, 6))
plt.plot(X, color='b')
plt.title('Sensor Value Time Series (Sensor ID: 1)')
plt.xlabel('Timestamp')
plt.ylabel('Sensor Value')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plot_acf(X, lags=min(30, len(X)//2), ax=plt.gca())
plt.title('ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=min(30, len(X)//2), ax=plt.gca(), method='ywm')
plt.title('PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 1, 1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']

print("ARMA(1,1) parameters:")
print("phi1 =", phi1)
print("theta1 =", theta1)

ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

arma1 = ArmaProcess(ar1, ma1).generate_sample(nsample=500)

plt.figure(figsize=(10, 4))
plt.plot(arma1, color='g')
plt.title('Simulated ARMA(1,1)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plot_acf(arma1)
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(arma1, method='ywm')
plt.show()

arma22_model = ARIMA(X, order=(2, 1, 2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']
theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']

print("\nARMA(2,2) parameters:")
print("phi1 =", phi1, "phi2 =", phi2)
print("theta1 =", theta1, "theta2 =", theta2)

ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

arma2 = ArmaProcess(ar2, ma2).generate_sample(nsample=500)

plt.figure(figsize=(10, 4))
plt.plot(arma2, color='r')
plt.title('Simulated ARMA(2,2)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plot_acf(arma2)
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(arma2, method='ywm')
plt.show()
```

### OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

<img width="1061" height="512" alt="image" src="https://github.com/user-attachments/assets/c8ad6c81-0c4e-4bba-85f8-43a48931597e" />

Partial Autocorrelation

<img width="785" height="541" alt="image" src="https://github.com/user-attachments/assets/df541360-0506-44eb-a1e7-b0fe7dba798c" />

Autocorrelation

<img width="822" height="576" alt="image" src="https://github.com/user-attachments/assets/a357d166-e334-482e-bf64-8db8c5c9b19c" />

SIMULATED ARMA(2,2) PROCESS:

<img width="1212" height="507" alt="image" src="https://github.com/user-attachments/assets/1a88ee17-342e-4c5e-a613-87e964227f64" />

Partial Autocorrelation

<img width="795" height="547" alt="image" src="https://github.com/user-attachments/assets/b6dcf141-edbf-41df-8704-ad9473dce785" />

Autocorrelation

<img width="880" height="550" alt="image" src="https://github.com/user-attachments/assets/a931a833-a793-4cec-8e9e-8c55d6ff9bea" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.

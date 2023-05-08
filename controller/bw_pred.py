import matplotlib.pyplot as plt
import numpy as np
from typing import List
import csv

def data_extraction(filename):
  
  data = []
  with open(filename, 'r') as file:
      for line in file:
          values = line.split()
          timestamp = int(values[0])
          value1 = int(values[1])
          x = float(values[2])
          y = float(values[3])
          bytes = int(values[4])
          elapsed = int(values[5])
          bandwidth = bytes/elapsed
          # Create a dictionary with the parsed values
          row = {'timestamp': timestamp, 'value1': value1, 'x': x, 'y': y, 'bytes': bytes, 'elapsed': elapsed, 'bandwidth': bandwidth}
          data.append(row)
          
  return data


#Tentative function drafts for developing
def bw_smooth(data, alpha = 0.2, stride = 1, initial_predict = 100):
  """
  Simulates a weighted smoothing solver to predict bandwidth based on observed data and generates a plot.

  Args:
      data: A list representing observed data. 
      alpha: A float representing the weight of the previous prediction in the current prediction.
      stride: An integer representing the number of steps between predictions, used to adapt different sampling frequency
  """


  N = len(data)
  predict = np.zeros(N)

  for i in range(1,N,stride):
    predict[i] = alpha * predict[i-stride] + (1-alpha) * data[i]

  return predict


def bw_kalman_filter(data: List[float], Q: float = 1e-5, R: float = 0.1, initial_estimate: float = 100, initial_error: float = 1) -> np.ndarray:
    """
    Predicts the bandwidth using a Kalman filter based on the observed data.

    Args:
        data: A list representing observed data.
        Q: A float representing the process noise covariance.
        R: A float representing the measurement noise covariance.
        initial_estimate: A float representing the initial estimate of the bandwidth.
        initial_error: A float representing the initial error estimate.

    Returns:
        A numpy array of predicted bandwidth values.
    """    
    N = len(data)
    kalman_gain = np.zeros(N)
    estimate = np.zeros(N)
    estimate[0] = initial_estimate
    error = np.zeros(N)
    error[0] = initial_error

    for i in range(1, N):
        # Predict step
        predicted_estimate = estimate[i-1]
        predicted_error = error[i-1] + Q

        # Update step
        kalman_gain[i] = predicted_error / (predicted_error + R)
        estimate[i] = predicted_estimate + kalman_gain[i] * (data[i] - predicted_estimate)
        error[i] = (1 - kalman_gain[i]) * predicted_error

    return estimate

def bw_moving_average(data: List[float], window_size: int = 10) -> np.ndarray:
    """
    Predicts the bandwidth using a moving average based on the observed data.

    Args:
        data: A list representing observed data.
        window_size: An integer representing the size of the moving window.

    Returns:
        A numpy array of predicted bandwidth values.
    """
    N = len(data)
    predicted_bandwidth = np.zeros(N)

    for i in range(window_size-1, N):
        predicted_bandwidth[i] = np.mean(data[i-window_size+1:i+1])

    return predicted_bandwidth


fname = '1.log'
data = data_extraction(fname)

history_data = []
for i in range(len(data)):
  history_data.append(data[i]['bandwidth'])
# print(history_data)
smooth_data = bw_smooth(history_data)
kmf_data = bw_kalman_filter(history_data)
moving_data = bw_moving_average(history_data)
data_list = [smooth_data, kmf_data, moving_data]

fname_idx = fname.split('.')[0]
smooth_log = f"{fname_idx}_smooth.log"
kmf_log = f"{fname_idx}_kmf.log"
moving_log = f"{fname_idx}_moving.log"
logs = [smooth_log, kmf_log, moving_log]

# print(len(history_data))
# print(len(smooth_data))
# print(len(kmf_data))
# print(len(moving_data))


for i in range(len(logs)):
   results_files = open(logs[i], "w",  newline='')
   csv_writer = csv.writer(results_files)
#    header = ("actual,predict").split(",")
#    csv_writer.writerow(header)

   for j in range(len(data_list[i])):
      stats = (f"{history_data[j]},{data_list[i][j]}").split(",")
      csv_writer.writerow(stats)

   results_files.close()

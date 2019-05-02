import numpy as np
import pandas as pd

#   Grabbing the entire data set
data_set =  pd.read_csv('events.csv')
print(data_set.head())

#   Getting the latest timestamp, which we'll use to 
#   calculate final 24 hour's worth of data
latest_time = data_set['timestamp'].max()

#   Constant defining milliseconds in a day
MILLISECOND_DAY = 86400000

#   Figuring out the cutoff between training and test  data:
time_cutoff = latest_time - MILLISECOND_DAY

#   Separating the data
training_data = data_set[data_set['timestamp'] < time_cutoff]
test_data = data_set[data_set['timestamp'] > time_cutoff]
print("Training:")
print(training_data.shape)
print("Test:")
print(test_data.shape)
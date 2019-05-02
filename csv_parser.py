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

#   Separating the data into  testing data vs training data:
training_data = data_set[data_set['timestamp'] < time_cutoff]
test_data = data_set[data_set['timestamp'] > time_cutoff]
# print("Training:")
# print(training_data.shape)
# print("Test:")
# print(test_data.shape)

#   A function to take our current DF format and return a matrix of users/items
def reshape_data(data_frame):
  #   Creating columns from unique item id's. Note that the index is for rows of userids.
  reshaped_data_frame = pd.DataFrame(0, columns = data_frame['itemid'].unique(), index=data_frame['visitorid'].unique())
  print(reshaped_data_frame.shape)

  #   Iterating through the rows of our data frame. 
  #   For each row, 
  for index, row in data_frame.iterrows():
    reshaped_data_frame[row['itemid']][row['visitorid']] = 1
  print(reshaped_data_frame.head())

reshape_data(test_data)
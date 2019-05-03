import logging
import time

import h5py
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

log = logging.getLogger(__name__)


def get_retailrocket_ecommerce():
    """Returns the Retailrocket recommender system e-commerce dataset.
    Returns a tuple of (itemids, visitorids, events) where events is a CSR matrix
    """
    filename = 'data/events.hdf5'
    log.info(f"Using dataset at '{filename}'")

    with h5py.File(filename, 'r') as f:
        m = f.get('item_user_event')
        events = csr_matrix((m.get('data'), m.get('indices'), m.get('indptr')))
        return np.array(f['item']), np.array(f['user']), events


def generate_dataset(filename, output_filename):
    """Generates a hdf5 retailrocket e-commerce data set file from the raw data file found at:
    https://www.kaggle.com/retailrocket/ecommerce-dataset#events.csv
    """
    data = _read_dataframe(filename)
    _hfd5_from_dataframe(data, output_filename)


def _read_dataframe(filename, training=True):
    """Reads the original dataset CSV as a pandas dataframe"""
    # delay importing this to avoid another dependency
    import pandas

    # read in triples of user/event/item from the input dataset
    # get a model based off the input params
    start = time.time()
    log.debug(f'reading data from {filename}')
    data = pandas.read_csv(filename, names=['time', 'user', 'event', 'item'], header=0, usecols=[0, 1, 2, 3],
                           na_filter=False)

    # Removing testing data
    latest_time = data['time'].max()
    # Constant defining milliseconds in a day
    MILLISECOND_DAY = 86_400_000
    # Figuring out the cutoff between training and test  data:
    time_cutoff = latest_time - MILLISECOND_DAY

    # Separating the data into  testing data vs training data:
    if training:
        data = data[data['time'] < time_cutoff]
    else:  # To get the test data:
        data = data[data['time'] >= time_cutoff]

    # remove users with less than 20 records
    value_counts = data['user'].value_counts()
    to_remove = value_counts[value_counts <= 20]
    data = data[~data.user.isin(to_remove)]

    # replace events with weights
    data['event'] = data['event'].replace({'view': 1.0, 'addtocart': 1.0, 'transaction': 1.0})

    # map each item and user to a unique numeric value
    data['user'] = data['user'].astype('category')
    data['item'] = data['item'].astype('category')

    # store as a CSR matrix
    log.debug(f'read data file in {time.time() - start:0.2f}s')
    return data


def _hfd5_from_dataframe(data, output_filename):
    # create a sparse matrix of all the users/events
    events = coo_matrix(
        (data['event'].astype(np.float32), (data['item'].cat.codes.copy(), data['user'].cat.codes.copy()))).tocsr()

    with h5py.File(output_filename, 'w') as f:
        g = f.create_group('item_user_event')
        g.create_dataset('data', data=events.data)
        g.create_dataset('indptr', data=events.indptr)
        g.create_dataset('indices', data=events.indices)
        dt = h5py.special_dtype(vlen=str)
        item = list(data['item'].cat.categories)
        dataset = f.create_dataset('item', (len(item),), dtype=dt)
        dataset[:] = item
        user = list(data['user'].cat.categories)
        dataset = f.create_dataset('user', (len(user),), dtype=dt)
        dataset[:] = user


if __name__ == '__main__':
    generate_dataset('data/events.csv', 'data/events.hdf5')

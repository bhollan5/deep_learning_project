import os
import logging
import time
import codecs
import tqdm

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
import numpy as np

from retailrocket import get_retailrocket_ecommerce

# for systems using OpenBLAS
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

# for systems using Intel MKL
os.environ['MKL_NUM_THREADS'] = '1'


def calculate_recommendations(output_filename):
    """Generates item recommendations for each user in the dataset"""
    # train the model based off input params
    items, users, events = get_retailrocket_ecommerce()

    # create a model from the input data
    model = AlternatingLeastSquares(factors=32, dtype=np.float32)

    # lets weight these models by bm25weight.
    logging.debug('weighting matrix by bm25_weight')
    events = bm25_weight(events, K1=100, B=0.8)

    # also disable building approximate recommend index
    model.approximate_similar_items = False

    # this is actually disturbingly expensive:
    events = events.tocsr()

    logging.debug('training model als')
    start = time.time()
    model.fit(events)
    logging.debug(f"trained model 'als' in {time.time() - start:0.2f}s")

    # generate recommendations for each user and write out to a file
    start = time.time()
    user_events = events.T.tocsr()

    with tqdm.tqdm(total=len(users)) as progress:
        with codecs.open(output_filename, 'w', 'utf8') as o:
            for user_id, username in enumerate(users):
                for item_id, score in model.recommend(user_id, user_events):
                    o.write(f'{username},{items[item_id]},{score}\n')
                progress.update(1)

    logging.debug(f'generated recommendations in {time.time() - start:0.2f}s')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    calculate_recommendations('data/recommendations.csv')

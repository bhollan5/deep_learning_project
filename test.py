import logging
from collections import defaultdict

import pandas as pd

logging.basicConfig(level=logging.DEBUG)


def get_top_k(predictions, k=10):
    top_k = defaultdict(list)

    # for user_id, item_id


def hit_rate(top_k_predicted, left_out_predictions):
    hits = 0
    total = 0

    for left_out in left_out_predictions:
        user_id = left_out[0]
        left_out_item_id = left_out[1]
        hit = False

        for item_id, predicted_rating in top_k_predicted[int(user_id)]:
            if int(left_out_item_id) == int(item_id):
                hit = True
                break

        if hit:
            hits += 1

        total += 1

    return hits / total


if __name__ == '__main__':
    import pandas as pd

    logging.basicConfig(level=logging.DEBUG)
    # calculate_recommendations('data/recommendations.csv')
    # recommendations = pd.read_csv('data/recommendations.csv')
    # print(f'Hit Rate: {hit_rate(None, None)}')

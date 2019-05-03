from collections import defaultdict
import tqdm
import logging

import pandas as pd


def get_top_k(predictions, k=10):
    top_k = defaultdict(list)
    with tqdm.tqdm(total=len(predictions)) as progress:
        for index, row in predictions.iterrows():
            top_k[int(row['user'])].append((int(row['item']), row['score']))
            progress.update(1)

    for user, scores in top_k.items():
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k[int(user)] = scores[:k]

    return top_k


def hit_rate(top_k_predicted, left_out_predictions):
    hits = 0
    total = 0

    with tqdm.tqdm(total=len(left_out_predictions)) as progress:
        for index, row in left_out_predictions.iterrows():
            user = row['user']
            left_out_item = row['item']
            hit = False

            for item, predicted_score in top_k_predicted[int(user)]:
                if int(left_out_item) == int(item):
                    hit = True
                    break

            if hit:
                hits += 1

            total += 1
            progress.update(1)

    return hits / total


if __name__ == '__main__':
    from retailrocket import _read_dataframe

    logging.basicConfig(level=logging.DEBUG)

    test_data = _read_dataframe('data/events.csv', training=False)
    test_data = test_data[['user', 'event', 'item']]  # discard time column
    all_predictions = pd.read_csv('data/recommendations.csv', header=None, names=['user', 'item', 'score'])

    # top 10
    top_10_predicted = get_top_k(all_predictions, k=10)
    print(f'Hit Rate (top 10): {hit_rate(top_10_predicted, test_data)}')

    # top 50
    top_50_predicted = get_top_k(all_predictions, k=50)
    print(f'Hit Rate (top 50): {hit_rate(top_50_predicted, test_data)}')

    # top 100
    top_100_predicted = get_top_k(all_predictions, k=100)
    print(f'Hit Rate (top 100): {hit_rate(top_100_predicted, test_data)}')

import multiprocessing
import os

import torch
import numpy as np
import json
from src.metrics import get_performance, rank_list_by_heapq, rank_list_by_sorted
from src.data_loader.data_loader import BooksDataset

print(os.getcwd())
cores = multiprocessing.cpu_count() // 5
data_dir = "./data/books"
Ks = eval("[10, 20, 50]")
with open(f'{data_dir}/train.json', 'r') as f:
    train_dict = json.load(f)

with open(f'{data_dir}/test.json', 'r') as f:
    test_dict = json.load(f)

test_set = {}
for uid, test_items in test_dict.items():
    if len(test_items) == 0:
        continue
    try:
        test_set[uid] = test_items
    except:
        continue
test_dict = test_set
with open(f'{data_dir}/validation.json', 'r') as f:
    val_dict = json.load(f)
n_items = 14790
n_users = 33962
batch_size = 1024


def test(ua_embeddings, ia_embeddings, users_to_test, batch_test_flag=False, is_val=False):
    """
    Test the performance of the model for all users

    :param ua_embeddings:  user embeddings
    :type ua_embeddings: torch.Tensor
    :param ia_embeddings:  item embeddings
    :type ia_embeddings: torch.Tensor
    :param users_to_test:  list of users to test
    :type users_to_test: list
    :param batch_test_flag:  flag to indicate if batch testing is enabled
    :type batch_test_flag: bool
    :param is_val:  flag to indicate if the test is for validation
    :type is_val: bool
    :return:  valeurs des métrics de performance
    :rtype: dict
    """
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    pool = multiprocessing.Pool(cores)

    u_batch_size = batch_size * 2
    i_batch_size = batch_size

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]
        if batch_test_flag:
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = range(i_start, i_end)
                u_g_embeddings = ua_embeddings[user_batch]
                i_g_embeddings = ia_embeddings[item_batch]
                i_rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items

        else:
            item_batch = range(n_items)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]
            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))

        rate_batch = rate_batch.detach().cpu().numpy()
        user_batch_rating_uid = zip(rate_batch, user_batch, [is_val] * len(user_batch))

        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users

    assert count == n_test_users
    pool.close()
    # make result to str for logging

    result_str = ""
    for key in result:
        if key == 'auc':
            result_str += f'{key}={result[key]:.4f}'
        else:
            for i in range(len(Ks)):
                result_str += f'{key}@{Ks[i]} = {result[key][i]:.4f} - '
        result_str += '|| '
    return result, result_str


def test_one_user(x):
    """
    Test the performance of the model for one user

    :param x:  list, user u's ratings for user u
    :return:  performance: dict
    :rtype: dict
    """
    # user u's ratings for user u
    is_val = x[-1]
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_dict[str(u)]
    except KeyError:
        training_items = []
    # user u's items in the test set
    if is_val:
        user_pos_test = val_dict[str(u)]
    else:
        user_pos_test = test_dict[str(u)]

    all_items = set(range(n_items))

    test_items = list(all_items - set(training_items))

    if "part" == 'part':
        r, auc = rank_list_by_heapq(list(user_pos_test), test_items, rating, Ks)
    else:
        r, auc = rank_list_by_sorted(list(user_pos_test), test_items, rating, Ks)

    return get_performance(r, auc, list(user_pos_test), Ks)

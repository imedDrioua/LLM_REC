import torch
import multiprocessing
import heapq
import torch
import numpy as np
from src.metrics import get_performance, rank_list_by_heapq, rank_list_by_sorted
from src.data_loader.data_loader import BooksDataset

cores = multiprocessing.cpu_count() // 5

Ks = eval("[10, 20, 50]")

ITEM_NUM = 17366
BATCH_SIZE = 1024

dataset = BooksDataset(data_dir="./data/netflix")


def test_one_user(x):
    # user u's ratings for user u
    is_val = x[-1]
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = dataset.get_dataset("train_dict")[str(u)]
    except KeyError:
        training_items = []
    # user u's items in the test set
    if is_val:
        user_pos_test = dataset.get_dataset("val_dict")[str(u)]
    else:
        user_pos_test = dataset.get_dataset("test_dict")[str(u)]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if "part" == 'part':
        r, auc = rank_list_by_heapq(list(user_pos_test), test_items, rating, Ks)
    else:
        r, auc = rank_list_by_sorted(list(user_pos_test), test_items, rating, Ks)

    return get_performance(r, auc, list(user_pos_test), Ks)


def test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val=False, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]
        if batch_test_flag:
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)
                u_g_embeddings = ua_embeddings[user_batch]
                i_g_embeddings = ia_embeddings[item_batch]
                i_rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)
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
    return result

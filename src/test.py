import src.metrics as metrics
import multiprocessing
import heapq
import torch
import numpy as np
from tqdm import tqdm

cores = multiprocessing.cpu_count() // 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Ks = [10, 20, 50]


def get_performance(r, auc, user_pos_test, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def rank_list_by_heapq(rating, user_pos_test, test_items):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def rank_list_by_sorted(rating, user_pos_test, test_items):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc


class Tester:
    def __init__(self, dataset, ks='[10]'):
        self.dataset = dataset
        self.USR_NUM, self.ITEM_NUM = dataset.n_users, dataset.n_items
        self.BATCH_SIZE = dataset.batch_size
        self.Ks = eval(ks)
        self.test_users_keys = [int(x) for x in list(self.dataset.get_dataset("test_dict").keys())]
        self.test_items = None

    def test_one_user(self, x, test_flag='part'):
        # user u's ratings for user u
        is_val = x[-1]
        rating = x[0]

        # uid
        u = x[1]
        # user u's items in the training set
        try:
            training_items = self.dataset.get_dataset("train_dict")[str(u)]
        except KeyError:
            training_items = []
        # user u's items in the test set
        if is_val:
            user_pos_test = self.dataset.get_dataset("val_dict")[str(u)]
        else:
            user_pos_test = self.dataset.get_dataset("test_dict")[str(u)]

        all_items = set(range(self.ITEM_NUM))

        test_items = list(all_items - set(training_items))

        if test_flag == 'part':
            r, auc = rank_list_by_heapq(rating, user_pos_test, test_items)
        else:
            r, auc = rank_list_by_sorted(rating, user_pos_test, test_items)

        return get_performance(r, auc, user_pos_test, Ks)

    def test(self, ua_embeddings, ia_embeddings, batch_size, is_val, batch_test_flag=False):

        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

        pool = multiprocessing.Pool(cores)

        u_batch_size = batch_size * 2

        n_test_users = len(self.test_users_keys)
        n_user_batchs = n_test_users // u_batch_size + 1
        count = 0
        for u_batch_id in tqdm(range(n_user_batchs)):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = self.test_users_keys[start: end]
            item_batch = range(self.ITEM_NUM)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]
            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))

            rate_batch = rate_batch.detach().cpu().numpy()
            user_batch_rating_uid = zip(rate_batch, user_batch, [is_val] * len(user_batch))

            batch_result = pool.map(self.test_one_user, user_batch_rating_uid)
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

    def vectoriel_test(self, ua_embeddings, ia_embeddings):
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
        user_batch = self.test_users_keys
        item_batch = range(self.ITEM_NUM)
        u_g_embeddings = ua_embeddings[user_batch]
        i_g_embeddings = ia_embeddings[item_batch]
        rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))
        rate_batch = rate_batch.detach().cpu().numpy()
        user_batch_rating_uid = zip(rate_batch, user_batch, [False] * len(user_batch))
        batch_result = list(map(self.test_one_user, user_batch_rating_uid))
        for re in batch_result:
            result['precision'] += re['precision'] / len(user_batch)
            result['recall'] += re['recall'] / len(user_batch)
            result['ndcg'] += re['ndcg'] / len(user_batch)
            result['hit_ratio'] += re['hit_ratio'] / len(user_batch)
            result['auc'] += re['auc'] / len(user_batch)
        return result

"""
Script that defines the Tester class which is used to test the model performance on the test set
"""
import multiprocessing
import torch
import numpy as np
from tqdm import tqdm
from src.metrics import get_performance, rank_list_by_heapq, rank_list_by_sorted


class Tester:
    def __init__(self, dataset, ks="[10, 20, 50]"):
        self.dataset = dataset
        self.USR_NUM, self.ITEM_NUM = dataset.n_users, dataset.n_items
        self.BATCH_SIZE = dataset.batch_size
        self.Ks = eval(ks)
        self.test_users_keys = [int(x) for x in list(self.dataset.get_dataset("test_dict").keys())]
        self.cores = multiprocessing.cpu_count() // 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_items = None

    def test_one_user(self, x, test_flag='part'):
        """
        Test the performance of the model for one user

        :param x:  list, user u's ratings for user u
        :param test_flag:  str, test flag
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
            r, auc = rank_list_by_heapq(rating, user_pos_test, test_items, self.Ks)
        else:
            r, auc = rank_list_by_sorted(rating, user_pos_test, test_items, self.Ks)

        return get_performance(r, auc, user_pos_test, self.Ks)

    def test(self, ua_embeddings, ia_embeddings, batch_size, is_val, batch_test_flag=False):
        """
        Test the performance of the model

        :param ua_embeddings:  user embeddings
        :type ua_embeddings: torch.Tensor
        :param ia_embeddings:  item embeddings
        :type ia_embeddings: torch.Tensor
        :param batch_size:  batch size
        :type batch_size: int
        :param is_val:  bool, validation flag
        :type is_val: bool
        :param batch_test_flag:  bool, batch test flag
        :type batch_test_flag: bool
        :return:  result of the model performance
        :rtype: dict
        """
        result = {'precision': np.zeros(len(self.Ks)), 'recall': np.zeros(len(self.Ks)), 'ndcg': np.zeros(len(self.Ks)),
                  'hit_ratio': np.zeros(len(self.Ks)), 'auc': 0.}

        pool = multiprocessing.Pool(self.cores)

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

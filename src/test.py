from src import metrics as metrics
import multiprocessing
import heapq
import torch
import numpy as np

cores = multiprocessing.cpu_count() // 5


class Tester:
    def __init__(self, dataset, ks='[10, 20, 50]'):
        self.dataset = dataset
        self.USR_NUM, self.ITEM_NUM = dataset.n_users, dataset.n_items
        self.BATCH_SIZE = dataset.batch_size
        self.Ks = eval(ks)
        self.test_items = None
        self.user_pos_test = None
        self.rating = None

    def rank_list_by_heapq(self):
        """
        Get the top K items for one user using heapq

        :return:  r: list, auc: float
        :rtype: list, float
        """
        item_score = {}
        for i in self.test_items:
            item_score[i] = self.rating[i]

        K_max = max(self.Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in self.user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = 0.
        return r, auc

    def get_auc(self, item_score):
        """
        Get the auc score

        :param item_score:  dict, item score
        :return:  auc: float
        :rtype: float
        """
        item_score = sorted(item_score.items(), key=lambda kv: kv[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        r = []
        for i in item_sort:
            if i in self.user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = metrics.auc(ground_truth=r, prediction=posterior)
        return auc

    def rank_list_by_sorted(self):
        """
        Get the top K items for one user using sorted

        :return:  r: list, auc: float
        :rtype: list, float
        """
        item_score = {}
        for i in self.test_items:
            item_score[i] = self.rating[i]

        K_max = max(self.Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in self.user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = self.get_auc(item_score)
        return r, auc

    def get_performance(self, r, auc):
        """
        Get the performance of the model

        :param r:  list, auc: float
        :param auc:  float
        :return:  performance: dict
        :rtype: dict
        """
        precision, recall, ndcg, hit_ratio = [], [], [], []

        for K in self.Ks:
            precision.append(metrics.precision_at_k(r, K))
            recall.append(metrics.recall_at_k(r, K, len(self.user_pos_test)))
            ndcg.append(metrics.ndcg_at_k(r, K))
            hit_ratio.append(metrics.hit_at_k(r, K))

        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

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
        self.rating = rating
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
        self.user_pos_test = user_pos_test
        all_items = set(range(self.ITEM_NUM))

        test_items = list(all_items - set(training_items))
        self.test_items = test_items

        if test_flag == 'part':
            r, auc = self.rank_list_by_heapq()
        else:
            r, auc = self.rank_list_by_sorted()

        return self.get_performance(r, auc)

    def test(self, ua_embeddings, ia_embeddings, users_to_test, batch_size, is_val, batch_test_flag=False):
        """
        Test the performance of the model

        :param ua_embeddings:  torch.Tensor, user embeddings
        :param ia_embeddings:  torch.Tensor, item embeddings
        :param users_to_test:  list, users to test
        :param batch_size:  int, batch size
        :param is_val:  bool, validation flag
        :param batch_test_flag:  bool, batch test flag
        :return:  result: dict
        :rtype: dict
        """
        Ks = self.Ks
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

        pool = multiprocessing.Pool(cores)

        u_batch_size = batch_size * 2
        test_users = [int(x) for x in users_to_test]
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1
        count = 0

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = test_users[start: end]
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

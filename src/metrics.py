"""
Script that defines the metrics which are used to evaluate the model
"""
import heapq
import numpy as np
from sklearn.metrics import roc_auc_score


def recall(rank, ground_truth, N):
    """
    calculate recall

    :param rank:  ranked list
    :type rank: list
    :param ground_truth:  ground truth list
    :type ground_truth: list
    :param N:  number of items to consider
    :type N: int
    :return:  recall
    :rtype: float
    """
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """
    calculate precision at k

    :param r:  list of relevance scores (either 1 or 0)
    :param k:  number of results to consider
    :return:  precision at k
    :rtype: float
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r, cut):
    """
    calculate average precision

    :param r:  list of relevance scores (either 1 or 0)
    :param cut:  number of results to consider
    :return:  average precision
    :rtype: float
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """
    calculate mean average precision

    :param rs:  list of relevance scores (either 1 or 0)
    :return:    mean average precision
    :rtype: float
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """
    calculate discounted cumulative gain at k

    :param r:  list of relevance scores (either 1 or 0)
    :param k:  number of results to consider
    :param method:   method to calculate dcg
    :return:  discounted cumulative gain at k
    :rtype: float
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """
   calculate normalized discounted cumulative gain at k

   :param r:  list of relevance scores (either 1 or 0)
   :param k:  number of results to consider
   :param method:  method to calculate dcg
   :return:  normalized discounted cumulative gain at k
   :rtype: float
   """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    """
  calculate recall at k

  :param r:  list of relevance scores (either 1 or 0)
  :param k:  number of results to consider
  :param all_pos_num:  number of all positive samples
  :return:  recall at k
  :rtype: float
  """
    r = np.asfarray(r)[:k]
    if all_pos_num == 0:
        return 0
    else:
        return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    """
    calculate hit at k

    :param r:  list of relevance scores (either 1 or 0)
    :param k:  number of results to consider
    :return:  hit at k
    :rtype: float
    """
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.


def F1(pre, rec):
    """
    calculate F1 score

    :param pre:  precision
    :param rec:  recall
    :return:  F1 score
    :rtype: float
    """
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def calculate_auc(ground_truth, prediction):
    """
    calculate AUC score

    :param ground_truth:  ground truth
    :param prediction:  prediction
    :return:  AUC score
    :rtype: float
    """
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def get_performance(r, auc, user_pos_test, Ks):
    """
    Get the performance of the model

    :param r:  model relevant items
    :type r: list
    :param auc:  area under the curve
    :type auc: float
    :param user_pos_test:  user positive items in the test set
    :type user_pos_test: list
    :param Ks:  list of Ks
    :type Ks: list
    :return:  metrics values
    :rtype: dict
    """
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def rank_list_by_heapq(rating, user_pos_test, test_items,Ks):
    """
    Rank the list by heapq

    :param rating:  model predicted ratings
    :type rating: dict
    :param user_pos_test:  user positive items in the test set
    :type user_pos_test: list
    :param test_items:  test items
    :type rating: dict
    :return:  r, auc
    :rtype: list, float
    """
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


def rank_list_by_sorted(rating, user_pos_test, test_items, Ks):
    """
    Rank the list by sorted

    :param rating:  model predicted ratings
    :type rating: dict
    :param user_pos_test:  user positive items in the test set
    :type user_pos_test: list
    :param test_items:  test items
    :type test_items: list
    :param Ks:  list of Ks
    :type Ks: list
    :return:  model relevant items, auc
    :rtype: list, float
    """
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
    """
    Get the area under the curve

    :param item_score:  items scores
    :type item_score: dict
    :param user_pos_test:  user positive items in the test set
    :type user_pos_test: list
    :return:  auc
    :rtype: float
    """
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
    auc = calculate_auc(ground_truth=r, prediction=posterior)
    return auc

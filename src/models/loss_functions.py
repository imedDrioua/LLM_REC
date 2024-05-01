"""
Script that defines the loss functions for the model.
"""
import torch
import numpy as np
import torch.nn.functional as f
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0):
    """
    Bayesian Personalized Ranking (BPR) loss function for user item embeddings

    :param users:  user liste
    :type users: list
    :param users_emb:  user embeddings
    :type users_emb: torch.Tensor
    :param pos_emb:  positive item embeddings
    :type pos_emb: torch.Tensor
    :param neg_emb:  negative item embeddings
    :type neg_emb: torch.Tensor
    :param userEmb0:  user initial embeddings weights
    :type userEmb0: torch.Tensor
    :param posEmb0:  positive item initial embeddings weights
    :type posEmb0: torch.Tensor
    :param negEmb0:  negative item initial embeddings weights
    :type negEmb0: torch.Tensor
    :return:  matrix factorization loss, embedding loss, regularization loss
    :rtype: float, float, float
    """
    reg_loss = (1 / 2) * (userEmb0.norm().pow(2) +
                          posEmb0.norm().pow(2) +
                          negEmb0.norm().pow(2)) / float(len(users))
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)
    maxi = f.logsigmoid(pos_scores - neg_scores + 1e-8)
    mf_loss = - prune_loss(maxi, 0.71)

    emb_loss = 1e-5 * reg_loss
    reg_loss = 0.0
    return mf_loss, emb_loss, reg_loss


def bpr_loss_aug(users, pos_items, neg_items, batch_size, prune_loss_drop_rate=0.71, decay=1e-5):
    """
    Bayesian Personalized Ranking (BPR) loss function for augmented data

    :param users:  user embeddings
    :param pos_items:  positive item embeddings
    :param neg_items:  negative item embeddings
    :param batch_size:      batch size
    :param prune_loss_drop_rate:  drop rate for pruning
    :param decay:  decay rate
    :return:  mf_loss, emb_loss, reg_loss
    :rtype: float, float, float
    """
    pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
    neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

    regularized = 1. / (2 * (users ** 2).sum() + 1e-8) + 1. / (2 * (pos_items ** 2).sum() + 1e-8) + 1. / (
            2 * (neg_items ** 2).sum() + 1e-8)
    regularized = regularized / batch_size

    maxi = f.logsigmoid(pos_scores - neg_scores + 1e-8)
    mf_loss = - prune_loss(maxi, prune_loss_drop_rate)

    emb_loss = decay * regularized
    reg_loss = 0.0
    return mf_loss, emb_loss, reg_loss


def prune_loss(prediction, drop_rate):
    """
    Prune the loss with the given drop rate

    :param prediction:  prediction
    :param drop_rate:    drop rate
    :return:  loss_update
    :rtype: float
    """
    if device == torch.device("cuda"):
        ind_sorted = np.argsort(prediction.cpu().data).cuda()
    else:
        ind_sorted = np.argsort(prediction.cpu().data).cpu()

    loss_sorted = prediction[ind_sorted]
    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))
    ind_update = ind_sorted[:num_remember]
    loss_update = prediction[ind_update]
    return loss_update.mean()

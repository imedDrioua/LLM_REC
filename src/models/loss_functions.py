# this script defines the loss functions (BPR) used in the model
# Path: src/models/loss_functions.py
import torch
import numpy as np
import torch.nn.functional as f



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bpr_loss(users, pos_items, neg_items, batch_size, prune_loss_drop_rate=0.71, decay=1e-5):
    """ Bayesian Personalized Ranking loss function
     Args:
        users: user embeddings
        pos_items: positive item embeddings
        neg_items: negative item embeddings
        batch_size: batch size
        prune_loss_drop_rate: drop rate for pruning the loss
        decay: regularization parameter
        Returns:   mf_loss, emb_loss, reg_loss
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
    """ Prune the loss by removing the lowest prediction values
    Args:
        prediction: the prediction values
        drop_rate: the drop rate for pruning
        Returns: the pruned loss
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

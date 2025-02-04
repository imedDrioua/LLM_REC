"""
Script that defines the Trainer class which is used to train the model on the training set
"""
from time import time
from datetime import datetime
import os
import torch
from tqdm import tqdm
from src.models.loss_functions import bpr_loss_aug, bpr_loss
from src.logging import Logger
from src.test import test
import torch.nn as nn


class Trainer:
    def __init__(self, dataset, model, lr=0.0001, side_info_rate=0.0001, augmentation_rate=0.012, aug_sample_rate=0.01):
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters()},
        ], lr=lr)
        self.side_info_rate = side_info_rate
        self.augmentation_rate = augmentation_rate
        self.aug_sample_rate = aug_sample_rate
        self.task_name = "%s_%s" % (datetime.now().strftime('%Y-%m-%d %H-%M-%S'), self.dataset.name)
        self.logger = Logger(filename=self.task_name, is_debug=False)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(self.dataset.describe())

    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text, feat_reg_decay=1e-5):
        """
        Calculate the feature regularization loss

        :param g_item_image:  item image embeddings
        :type g_item_image: torch.Tensor
        :param g_item_text:  item text embeddings
        :type g_item_text: torch.Tensor
        :param g_user_image:  user image embeddings
        :type g_user_image: torch.Tensor
        :param g_user_text:  user text embeddings
        :type g_user_text: torch.Tensor
        :param feat_reg_decay:  feature regularization decay
        :type feat_reg_decay: float
        :return:  feature regularization loss
        :rtype: float
        """
        feat_reg = 1. / 2 * (g_item_image ** 2).sum() + 1. / 2 * (g_item_text ** 2).sum() \
                   + 1. / 2 * (g_user_image ** 2).sum() + 1. / 2 * (g_user_text ** 2).sum()
        feat_reg = feat_reg / self.dataset.n_items
        feat_emb_loss = feat_reg_decay * feat_reg
        return feat_emb_loss

    def train(self, epochs, batch_size=1024):
        """
        Train the model

        :param epochs:  number of epochs
        :param batch_size:  batch size
        :return:  None
        :rtype: None
        """
        n_batch = self.dataset.n_train // batch_size + 1

        for epoch in range(epochs):
            start = time()
            self.model.train()
            loss = 0.
            for idx in tqdm(range(n_batch)):
                users, pos_items, neg_items = self.dataset.sample(batch_size)

                # sample augmented interactions
                users_aug, pos_items_aug, neg_items_aug = self.dataset.sample_augmented_interactions(users,
                                                                                                     self.aug_sample_rate)

                # augment the interactions
                users += users_aug
                pos_items += pos_items_aug
                neg_items += neg_items_aug
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         max_norm=1.0)
                self.optimizer.zero_grad()
                embeddings_dict = self.model(
                    users, pos_items, neg_items)
                # embeddings loss
                loss_dict = self.calculate_all_losses(embeddings_dict)
                mf_loss, emb_loss, reg = bpr_loss(users, embeddings_dict["embeddings"][0],
                                                  embeddings_dict["embeddings"][1],
                                                  embeddings_dict["embeddings"][2],
                                                  self.model.E0_user,
                                                  self.model.E0_item_pos,
                                                  self.model.E0_item_neg)
                tot = mf_loss + emb_loss
                # total loss
                total_loss = tot + self.side_info_rate * loss_dict["side_info_loss"] + \
                             loss_dict["augmentation_loss"] * self.augmentation_rate + loss_dict["side_info_reg_loss"]

                total_loss.backward(retain_graph=False)

                self.optimizer.step()
                loss += total_loss.item()
            del embeddings_dict
            test_users = self.dataset.get_dataset('test_dict')
            evaluation_results, result_str = self.evaluate(test_users)
            # print loss (not logging)
            self.logger.logging("\nEpoch %d/%d, Loss=%.4f, Time=%.2f" % (epoch + 1, epochs, loss, time() - start))
            self.logger.logging(result_str)

    def evaluate(self, test_users):
        """
        Evaluate the model on the test set

        :param test_users:  dict, test users
        :return:  evaluation results
        :rtype: dict
        """
        self.model.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings, _, _, _, _, _, _, _, _ = self.model.propagate()
            users_to_test = [int(x) for x in list(self.dataset.get_dataset("test_dict").keys())]
            return test(user_embeddings, item_embeddings, users_to_test)

    def calculate_all_losses(self, embedding_dict):
        """
        Calculate all the losses

        :param embedding_dict:  dict of all embeddings returned by the model
        :type embedding_dict: dict
        :return:  dictionary mapping loss names to loss values
        :rtype: dict
        """
        # embeddings loss
        """mf_loss, emb_loss, reg_loss = bpr_loss_aug(embedding_dict["embeddings"][0],
                                                   embedding_dict["embeddings"][1],
                                                   embedding_dict["embeddings"][2],
                                                   self.dataset.batch_size)
        embeddings_loss = mf_loss + emb_loss"""

        # image embeddings loss
        image_mf_loss, image_emb_loss, image_reg_loss = bpr_loss_aug(embedding_dict["image_embeddings"][0],
                                                                     embedding_dict["image_embeddings"][1],
                                                                     embedding_dict["image_embeddings"][2],
                                                                     self.dataset.batch_size)

        # text embeddings loss
        text_mf_loss, text_emb_loss, text_reg_loss = bpr_loss_aug(embedding_dict["text_embeddings"][0],
                                                                  embedding_dict["text_embeddings"][1],
                                                                  embedding_dict["text_embeddings"][2],
                                                                  self.dataset.batch_size)
        # side info loss
        side_info_loss = image_mf_loss + text_mf_loss

        # feature regularization loss
        side_info_reg_loss = self.feat_reg_loss_calculation(self.model.item_image_embeddings,
                                                            self.model.item_text_embeddings,
                                                            self.model.user_image_embeddings,
                                                            self.model.user_text_embeddings)

        # user profile loss
        user_profile_mf_loss, user_profile_emb_loss, _ = bpr_loss_aug(embedding_dict["attributes_embeddings"][0],
                                                                      embedding_dict["profile_embeddings"][1],
                                                                      embedding_dict["profile_embeddings"][2],
                                                                      self.dataset.batch_size)

        # item attributes loss
        item_attr_mf_loss, item_attr_emb_loss, _ = bpr_loss_aug(embedding_dict["profile_embeddings"][0],
                                                                embedding_dict["attributes_embeddings"][1],
                                                                embedding_dict["attributes_embeddings"][2],
                                                                self.dataset.batch_size)

        augmentation_loss = item_attr_mf_loss + user_profile_mf_loss

        return {"embeddings_loss": 0,
                "side_info_loss": side_info_loss,
                "augmentation_loss": augmentation_loss,
                "side_info_reg_loss": side_info_reg_loss}

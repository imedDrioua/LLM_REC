# this script defines the training process of the model

from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from src.models.loss_functions import bpr_loss_aug
from src.test import Tester


class Trainer:
    def __init__(self, dataset, model, lr=0.0001):
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.tester = Tester(self.dataset)

    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text, feat_reg_decay=1e-5):
        feat_reg = 1. / 2 * (g_item_image ** 2).sum() + 1. / 2 * (g_item_text ** 2).sum() \
                   + 1. / 2 * (g_user_image ** 2).sum() + 1. / 2 * (g_user_text ** 2).sum()
        feat_reg = feat_reg / self.dataset.n_items
        feat_emb_loss = feat_reg_decay * feat_reg
        return feat_emb_loss

    def train(self, epochs, batch_size=1024):
        n_batch = self.dataset.n_users // batch_size + 1

        for epoch in range(epochs):
            start = time()
            self.model.train()
            loss = 0.
            for idx in tqdm(range(n_batch)):
                users, pos_items, neg_items = self.dataset.sample(batch_size)

                self.optimizer.zero_grad()
                user_embeddings, pos_item_embeddings, neg_item_embeddings, user_image_embeddings, pos_item_image_embeddings, neg_item_image_embeddings, user_text_embeddings, pos_item_text_embeddings, neg_item_text_embeddings = self.model(
                    users, pos_items, neg_items)

                mf_loss, emb_loss, reg_loss = bpr_loss_aug(user_embeddings, pos_item_embeddings, neg_item_embeddings,
                                                           self.dataset.batch_size)
                embeddings_loss = mf_loss + reg_loss + emb_loss

                image_mf_loss, image_emb_loss, image_reg_loss = bpr_loss_aug(user_image_embeddings,
                                                                             pos_item_image_embeddings,
                                                                             neg_item_image_embeddings,
                                                                             self.dataset.batch_size)
                image_embeddings_loss = image_mf_loss

                text_mf_loss, text_emb_loss, text_reg_loss = bpr_loss_aug(user_text_embeddings,
                                                                          pos_item_text_embeddings,
                                                                          neg_item_text_embeddings,
                                                                          self.dataset.batch_size)
                text_embeddings_loss = text_mf_loss
                side_info_loss = image_embeddings_loss + text_embeddings_loss
                reg_feat_loss = self.feat_reg_loss_calculation(self.model.item_image_embeddings, self.model.item_text_embeddings,self.model.user_image_embeddings, self.model.user_text_embeddings)
                total_loss = embeddings_loss + side_info_loss * 0.0001 + reg_feat_loss

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         max_norm=1.0)
                self.optimizer.step()
                loss += total_loss.item()
            del user_embeddings, pos_item_embeddings, neg_item_embeddings, user_image_embeddings, pos_item_image_embeddings, neg_item_image_embeddings, user_text_embeddings, pos_item_text_embeddings, neg_item_text_embeddings
            test_users = self.dataset.get_dataset('test_dict')
            evaluation_results = self.evaluate(test_users)
            print(evaluation_results)
            print(f'Epoch {epoch} Loss {loss / n_batch} Time {time() - start}')

    def evaluate(self, test_users):
        self.model.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings, _, _, _, _ = self.model.propagate()

            return self.tester.test(user_embeddings, item_embeddings, self.dataset.batch_size,
                                    False,
                                    False)

# this script defines the training process of the model

from time import time
import torch
from tqdm import tqdm
from src.models.loss_functions import bpr_loss_aug
from src.test import Tester


class Trainer:
    def __init__(self, dataset, model, lr=0.0001):
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.tester = Tester(self.dataset)

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
                image_embeddings_loss = image_mf_loss + image_reg_loss + image_emb_loss

                text_mf_loss, text_emb_loss, text_reg_loss = bpr_loss_aug(user_text_embeddings,
                                                                          pos_item_text_embeddings,
                                                                          neg_item_text_embeddings,
                                                                          self.dataset.batch_size)
                text_embeddings_loss = text_mf_loss + text_reg_loss + text_emb_loss

                total_loss = embeddings_loss + image_embeddings_loss + text_embeddings_loss

                total_loss.backward()
                self.optimizer.step()
                loss += total_loss.item()

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

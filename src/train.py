# this script defines the training process of the model

from time import time
import torch
from tqdm import tqdm
from src.models.loss_functions import bpr_loss
from src.test import Tester


class Trainer:
    def __init__(self, dataset, model, lr=0.001):
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.tester = Tester(self.dataset)

    def train(self, epochs, batch_size=1024):
        """ Train the model for a number of epochs
        :param epochs: int, number of epochs
        :param batch_size: int, batch size
        """
        n_batch = self.dataset.n_users // batch_size + 1

        for epoch in range(epochs):
            start = time()
            self.model.train()
            loss = 0.
            for idx in tqdm(range(n_batch)):
                users, pos_items, neg_items = self.dataset.sample(batch_size)
                self.optimizer.zero_grad()
                user_embeddings, pos_item_embeddings, neg_item_embeddings = self.model(users, pos_items, neg_items)
                mf_loss, emb_loss, reg_loss = bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings,
                                                       self.dataset.batch_size)
                total_loss = mf_loss + emb_loss + reg_loss
                total_loss.backward()
                self.optimizer.step()
                loss += total_loss.item()

            test_users = self.dataset.get_dataset('test_dict')
            evaluation_results = self.evaluate(test_users)
            print(evaluation_results)
            print(f'Epoch {epoch} Loss {loss / n_batch} Time {time() - start}')

    def evaluate(self, test_users):
        """ Evaluate the model on a test set
        :param test_users: dict, test users
        :return: dict, evaluation results
        """
        self.model.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings = self.model.propagate()
            return self.tester.test(user_embeddings, item_embeddings, list(test_users.keys()), self.dataset.batch_size, False,
                                    False)



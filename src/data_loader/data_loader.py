# import the necessary libraries
import pandas as pd
import numpy as np
import os
import torch
import json


class BooksDataset:

    def __init__(self, data_dir, batch_size=1024):
        self.data_dir = data_dir
        # self.train_matrix = pd.read_csv(f'{data_dir}/train_df.csv').values
        self.images = np.load(f'{data_dir}/embed_image.npy')
        self.text = np.load(f'{data_dir}/embed_text.npy')
        self.user_profiles = np.load(f'{data_dir}/users_profiles_embeddings.npy')
        self.books_attributes = np.load(f'{data_dir}/books_attributes_embeddings.npy')
        self.interactions = torch.load(f'{data_dir}/train_matrix.pt')
        # check if the adjacency matrix exists, if not, it will be created with the model
        self.adjacency_matrix = None
        if os.path.exists(f'{data_dir}/adjacency_matrix.pt'):
            self.adjacency_matrix = torch.load(f'{data_dir}/adjacency_matrix.pt')

        self.batch_size = batch_size

        with open(f'{data_dir}/train.json', 'r') as f:
            self.train_dict = json.load(f)
        with open(f'{data_dir}/test.json', 'r') as f:
            self.test_dict = json.load(f)
        with open(f'{data_dir}/validation.json', 'r') as f:
            self.val_dict = json.load(f)

        # create a dict to map each dataset name to its corresponding data
        self.datasets = {
            # 'train_matrix': self.train_matrix,
            'images': self.images,
            'text': self.text,
            'user_profiles': self.user_profiles,
            'books_attributes': self.books_attributes,
            'train_dict': self.train_dict,
            'test_dict': self.test_dict,
            'val_dict': self.val_dict,
            'adjacency_matrix': self.adjacency_matrix,
            'interactions': self.interactions
        }
        self.n_users, self.n_items = len(self.train_dict), len(self.books_attributes)

    # return the length of all the datasets as dictionary
    def __len__(self):
        return {k: len(v) for k, v in self.datasets.items()}

    # return the dataset by name
    def get_dataset(self, dataset):
        """
        Return the dataset by name
        :param dataset: dataset name
        :type dataset: str
        :return: dataset
        :rtype: numpy.ndarray
        """
        return self.datasets[dataset]

    # return all the datasets
    def get_all_datasets(self):
        """
        Return all the datasets
        :return: all the datasets defined in the class
        :rtype: dict
        """
        return self.datasets

    # sample n_users from the train dataset, and return the users, positive and negative books
    def sample(self, n_users):
        """
        Sample n_users from the train dataset, and return the users, positive and negative books
        :param n_users: number of users to sample
        :type n_users: int
        :return:  users list, positive books list, negative books list
        :rtype: list, list, list
        """
        users = []
        pos_books = []
        neg_books = []

        # sample n_users, one positive and one negative book for each user
        for _ in range(n_users):
            user = np.random.randint(low=0, high=self.n_users, size=1)[0]
            users.append(user)
            # sample a positive book
            pos_book = np.random.choice(self.train_dict[str(user)])
            pos_books.append(pos_book)
            # sample a negative book
            neg_book = np.random.randint(self.n_items)
            while neg_book in self.train_dict[str(user)]:
                neg_book = np.random.randint(self.n_items)
            neg_books.append(neg_book)

        return users, pos_books, neg_books

    def describe(self):
        """
        Return the shape of all the datasets, number of interractions and the sparsity of the train matrix

        """
        # get the shape of all the datasets (dictionaries need special handling)
        shape = {k: v.shape if isinstance(v, np.ndarray) else len(v) for k, v in self.datasets.items()}
        # get the number of interactions in the train matrix
        n_interactions = np.count_nonzero(self.train_matrix)
        # get the sparsity of the train matrix (number of missing interactions / total interactions)
        sparsity = 1 - n_interactions / (self.train_matrix.shape[0] * self.train_matrix.shape[1])
        # print the results as two columns
        print(f"{'Dataset':<20}{'Shape':<20}")
        print('-' * 40)
        for k, v in shape.items():
            print(f"{k:<20}{v}")

        print(f"\nNumber of interactions: {n_interactions}")
        print(f"Sparsity: {sparsity:.2%}")

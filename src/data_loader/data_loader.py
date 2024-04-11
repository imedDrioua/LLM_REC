import numpy as np
import os
import torch
import json
import random


class BooksDataset:

    def __init__(self, data_dir, batch_size=1024):
        self.data_dir = data_dir
        self.name = "Netflix Dataset"
        # self.train_matrix = pd.read_csv(f'{data_dir}/train_df.csv').values
        self.images = np.load(f'{data_dir}/embed_image.npy')
        self.text = np.load(f'{data_dir}/embed_text.npy')
        self.user_profiles = np.load(f'{data_dir}/users_profiles_embeddings.npy')
        self.books_attributes = np.load(f'{data_dir}/films_attributes_embeddings.npy')
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
        with open(f'{data_dir}/augmented_interactions_dict.json', 'r') as f:
            self.augmented_interactions = json.load(f)

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

    def sample_augmented_interactions(self, users, aug_sample_rate):
        """
        Sample users from the augmented interactions, and return the users, positive and negative books

        :param users:  list of users for whom the augmented interactions will be sampled
        :type users: list
        :param aug_sample_rate:  the rate of users to sample
        :type aug_sample_rate: float
        :return:  users list, positive books list, negative books list
        :rtype: list, list, list
        """
        augmented_users = random.sample(users, int(len(users) * aug_sample_rate))
        positive_items = [self.augmented_interactions[str(user)][0] for user in augmented_users if (
                self.augmented_interactions[str(user)][0] < self.n_items and self.augmented_interactions[str(user)][1] < self.n_items)]

        neg_items_aug = [self.augmented_interactions[str(user)][1] for user in augmented_users if (
                self.augmented_interactions[str(user)][0] < self.n_items and self.augmented_interactions[str(user)][1] < self.n_items)]

        users_aug = [user for user in augmented_users if (
                self.augmented_interactions[str(user)][0] < self.n_items and self.augmented_interactions[str(user)][1] < self.n_items)]
        return users_aug, positive_items, neg_items_aug

    def describe(self):
        """
        Print the shape of all the datasets, the number of interactions in the train matrix, and the sparsity of the train matrix

        :return:  None
        :rtype: None
        """
        # get the shape of all the datasets (dictionaries need special handling)
        shape = {k: v.shape if isinstance(v, np.ndarray) else len(v) for k, v in self.datasets.items()}
        # get the number of interactions in the train matrix
        n_interactions = self.interactions._nnz()
        # get the sparsity of the train matrix (number of missing interactions / total interactions)
        sparsity = 1 - n_interactions / (self.interactions.shape[0] * self.interactions.shape[1])
        # print the results as two columns
        string = f"\n{'Dataset':<20}{'Shape':<20} \n"
        string += '-' * 40 + '\n'
        for k, v in shape.items():
            string += f"{k:<20}{v}" + '\n'

        string += f"\nNumber of interactions: {n_interactions}" + '\n'
        string += f"Sparsity: {sparsity:.2%}" + '\n'
        # convert the print statement to a return string
        return string



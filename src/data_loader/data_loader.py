# import the necessary libraries
import pandas as pd
import numpy as np
import json


class BooksDataset:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_matrix = pd.read_pickle(f'{data_dir}/train_matrix.pkl')
        self.images = np.load(f'{data_dir}/embed_image.npy')
        self.text = np.load(f'{data_dir}/embed_text.npy')
        self.user_profiles = np.load(f'{data_dir}/users_profiles_embeddings.npy')
        self.books_attributes = np.load(f'{data_dir}/books_attributes_embeddings.npy')

        with open(f'{data_dir}/train.json', 'r') as f:
            self.train_dict = json.load(f)
        with open(f'{data_dir}/test.json', 'r') as f:
            self.test_dict = json.load(f)
        with open(f'{data_dir}/validation.json', 'r') as f:
            self.val_dict = json.load(f)

        # create a dict to map each dataset name to its corresponding data
        self.datasets = {
            'train': self.train_matrix,
            'images': self.images,
            'text': self.text,
            'user_profiles': self.user_profiles,
            'books_attributes': self.books_attributes,
            'train_dict': self.train_dict,
            'test_dict': self.test_dict,
            'val_dict': self.val_dict
        }

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
        for i in range(n_users):
            user = self.train_dict[str(i)]
            pos_book = np.random.choice(self.train_dict[user])
            while True:
                neg_book = np.random.choice(self.train_dict.values())
                if neg_book not in self.train_dict[user]:
                    break
            users.append(user)
            pos_books.append(pos_book)
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

# the Mm_model class is the class that implement lightgcn model

# import the necessary packages
import torch
import torch.nn as nn
from scipy.sparse import dok_matrix
from torch.nn import init
import numpy as np
import scipy.sparse as sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MmModel(nn.Module):
    def __init__(self, n_users, n_items, embed_size, adjacency_matrix, n_layers, train_df=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_users = n_users
        self.n_items = n_items
        self.embed_size = embed_size
        self.user_item_matrix = adjacency_matrix
        self.n_layers = n_layers
        if self.user_item_matrix is None:
            self.create_adjacency_matrix(train_df)
        # optimizer
        # user and item embedding layers
        self.E0 = nn.Embedding(self.n_users + self.n_items, self.embed_size)

        # weight initialization with xavier uniform
        init.xavier_uniform_(self.E0.weight)
        # Set model Parameter
        self.model_parameters = nn.Parameter(self.E0.weight)
        # map adjacency matrix to device (GPU if available)
        self.user_item_matrix = self.user_item_matrix.to(device)

    def propagate(self):
        # get the embeddings of the users and items
        all_embeddings = [self.E0.weight]
        e_layer_weight = self.E0.weight
        for _ in range(self.n_layers):
            e_layer_weight = torch.sparse.mm(self.user_item_matrix, e_layer_weight)
            all_embeddings.append(e_layer_weight)

        # stack the embeddings
        all_embeddings = torch.stack(all_embeddings, dim=0)

        # get the mean of the embeddings
        all_embeddings_mean = torch.mean(all_embeddings, dim=0)

        # Split the embeddings of the users and items
        user_embeddings, item_embeddings = torch.split(all_embeddings_mean, [self.n_users, self.n_items], dim=0)
        return user_embeddings, item_embeddings

    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        user_embeddings, item_embeddings = self.propagate()
        # get the embeddings of the users and items
        user_embeddings = user_embeddings[user_indices]
        pos_item_embeddings = item_embeddings[pos_item_indices]
        neg_item_embeddings = item_embeddings[neg_item_indices]

        return user_embeddings, pos_item_embeddings, neg_item_embeddings

    def create_adjacency_matrix(self, train_df):
        # check if the user_item_matrix is already created
        R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        R[train_df['user_id'], train_df['item_id']] = 1.0
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = R.tolil()

        adj_mat[: self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, : self.n_users] = R.T
        adj_mat = adj_mat.todok()

        row_sum = np.array(adj_mat.sum(1))
        d_inv = np.power(row_sum + 1e-9, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(d_mat_inv)

        # Below Code is to convert the dok_matrix to sparse tensor.
        norm_adj_mat_coo = norm_adj_mat.tocoo().astype(np.float32)
        values = norm_adj_mat_coo.data
        indices = np.vstack((norm_adj_mat_coo.row, norm_adj_mat_coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = norm_adj_mat_coo.shape

        norm_adj_mat_sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))

        self.user_item_matrix = norm_adj_mat_sparse_tensor

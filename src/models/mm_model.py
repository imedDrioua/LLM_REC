"""
This module contains the implementation of the MMModel class which is a PyTorch implementation of the MMGCN model.
"""
import torch
import torch.nn as nn
from scipy.sparse import dok_matrix
from torch.nn import init
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MmModel(nn.Module):
    def __init__(self, n_users, n_items, embed_size, adjacency_matrix, interactions, interactions_t,
                 image_embeddings_data,
                 text_embeddings_data, book_attributes_data, user_profiles_data,
                 n_layers, model_cat_rate=0.02, user_cat_rate=2.8, item_cat_rate=0.005, train_df=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.E0_item_pos = None
        self.E0_item_neg = None
        self.E0_user = None
        self.user_image_embeddings = None
        self.item_image_embeddings = None
        self.user_text_embeddings = None
        self.item_text_embeddings = None
        self.user_cat_rate = user_cat_rate
        self.item_cat_rate = item_cat_rate
        self.n_users = n_users
        self.n_items = n_items
        self.embed_size = embed_size
        self.user_item_matrix = adjacency_matrix
        self.n_layers = n_layers
        self.model_cat_rate = model_cat_rate

        if self.user_item_matrix is None:
            self.create_adjacency_matrix(train_df)
        # optimizer
        # user and item embedding layers
        self.E0 = nn.Embedding(self.n_users + self.n_items, self.embed_size)
        self.embeddings_dropout = nn.Dropout(0.1)
        self.embeddings_norm = nn.BatchNorm1d(self.embed_size)

        # Side information layers
        self.image_feat = nn.Linear(image_embeddings_data.shape[1], self.embed_size)
        self.image_batch_norm = nn.BatchNorm1d(embed_size)
        self.image_feat_dropout = nn.Dropout(0.1)
        self.text_feat = nn.Linear(text_embeddings_data.shape[1], self.embed_size)
        self.text_batch_norm = nn.BatchNorm1d(embed_size)
        self.text_feat_dropout = nn.Dropout(0.1)

        # augmented feature linear layers and dropout
        self.user_profiles = nn.Linear(user_profiles_data.shape[1], self.embed_size)
        self.usuer_profiles_batch_norm = nn.BatchNorm1d(embed_size)
        self.user_profiles_dropout = nn.Dropout(0.2)
        self.book_attributes = nn.Linear(book_attributes_data.shape[1], self.embed_size)
        self.book_attributes_batch_norm = nn.BatchNorm1d(embed_size)
        self.book_attributes_dropout = nn.Dropout(0.2)
        # self.softmax = nn.Softmax(dim=-1)

        # weight initialization with xavier uniform
        init.kaiming_normal_(self.E0.weight)  # kaiming
        init.xavier_uniform_(self.text_feat.weight)
        init.xavier_uniform_(self.image_feat.weight)
        init.xavier_uniform_(self.user_profiles.weight)
        init.xavier_uniform_(self.book_attributes.weight)

        # Set model Parameter
        # self.model_parameters = nn.ParameterList([self.E0.weight, self.text_feat.weight, self.image_feat.weight])

        # map data to device (GPU if available)
        self.user_item_matrix = self.user_item_matrix.to(device)
        self.image_embeddings_data = torch.tensor(image_embeddings_data, dtype=torch.float32).to(device)
        self.text_embeddings_data = torch.tensor(text_embeddings_data, dtype=torch.float32).to(device)
        self.book_attributes_data = torch.tensor(book_attributes_data, dtype=torch.float32).to(device)
        self.user_profiles_data = torch.tensor(user_profiles_data, dtype=torch.float32).to(device)
        self.user_item_interactions = interactions.to(device)
        self.item_user_interactions = interactions_t.to(device)

    def propagate(self):
        """
        Propagate the embeddings of the users and items
        :return:  user_embeddings, item_embeddings
        :rtype: torch.Tensor, torch.Tensor
        """
        # normalize the weights of the embeddings
        e_layer_weight = self.embeddings_dropout(self.E0.weight)
        all_embeddings = [e_layer_weight]
        # normalize the embeddings
        # all_image_embeddings = [self.image_feat.weight.t()]
        # all_text_embeddings = [self.text_feat.weight.t()]
        user_image_feature, item_image_feature, user_text_feature, item_text_feature = None, None, None, None
        user_attributes, item_attributes, user_profile_feat, item_profile_feat = None, None, None, None

        image_layer_weight = self.image_feat_dropout(self.image_batch_norm(self.image_feat(self.image_embeddings_data)))
        text_layer_weight = self.text_feat_dropout(self.text_batch_norm(self.text_feat(self.text_embeddings_data)))
        user_profiles_weight = self.user_profiles_dropout(
            self.usuer_profiles_batch_norm(self.user_profiles(self.user_profiles_data)))
        items_attributes_weight = self.book_attributes_dropout(
            self.book_attributes_batch_norm(self.book_attributes(self.book_attributes_data)))
        item_image_feature = image_layer_weight
        item_text_feature = text_layer_weight
        user_profile_feat = user_profiles_weight
        item_attributes = items_attributes_weight
        for _ in range(self.n_layers):
            e_layer_weight = torch.mm(self.user_item_matrix, e_layer_weight)
            all_embeddings.append(e_layer_weight)

        # calculate embeddings for side information and augmented data
        for _ in range(1):
            user_image_feature = torch.sparse.mm(self.user_item_interactions, item_image_feature)
            item_image_feature = torch.sparse.mm(self.item_user_interactions, user_image_feature)

            user_text_feature = torch.sparse.mm(self.user_item_interactions, item_text_feature)
            item_text_feature = torch.sparse.mm(self.item_user_interactions, user_text_feature)

            user_attributes = torch.sparse.mm(self.user_item_interactions, item_attributes)
            item_attributes = torch.sparse.mm(self.item_user_interactions, user_attributes)

            item_profile_feat = torch.sparse.mm(self.item_user_interactions, user_profile_feat)
            user_profile_feat = torch.sparse.mm(self.user_item_interactions, item_profile_feat)

        # stack the embeddings
        all_embeddings = torch.stack(all_embeddings, dim=0)

        # get the mean of the embeddings
        all_embeddings_mean = torch.mean(all_embeddings, dim=0)

        # Split the embeddings of the users and items
        user_embeddings, item_embeddings = torch.split(all_embeddings_mean, [self.n_users, self.n_items], dim=0)
        self.user_E0, self.item_E0 = torch.split(self.E0.weight, [self.n_users, self.n_items], dim=0)

        # side information incorporation
        user_embeddings = user_embeddings + self.model_cat_rate * F.normalize(user_image_feature, p=2, dim=1)
        user_embeddings = user_embeddings + self.model_cat_rate * F.normalize(user_text_feature, p=2, dim=1)

        item_embeddings = item_embeddings + self.model_cat_rate * F.normalize(item_image_feature, p=2, dim=1)
        item_embeddings = item_embeddings + self.model_cat_rate * F.normalize(item_text_feature, p=2, dim=1)
        # augmented data incorporation
        user_embeddings += self.user_cat_rate * F.normalize(user_profile_feat, p=2, dim=1)
        user_embeddings += self.item_cat_rate * F.normalize(user_attributes, p=2, dim=1)

        item_embeddings += self.user_cat_rate * F.normalize(item_profile_feat, p=2, dim=1)
        item_embeddings += self.item_cat_rate * F.normalize(item_attributes, p=2, dim=1)

        return user_embeddings, item_embeddings, user_image_feature, item_image_feature, user_text_feature, item_text_feature, user_attributes, item_attributes, user_profile_feat, item_profile_feat

    def forward(self, user_indices, pos_item_indices, neg_item_indices):

        user_embeddings, item_embeddings, user_image_embeddings, item_image_embeddings, user_text_embeddings, item_text_embeddings, user_attributes_embeddings, item_attributes_embeddings, user_profile_feat_embd, item_profile_feat_embed = self.propagate()

        self.user_image_embeddings = user_image_embeddings
        self.item_image_embeddings = item_image_embeddings
        self.user_text_embeddings = user_text_embeddings
        self.item_text_embeddings = item_text_embeddings

        # get the embeddings of the users and items
        user_embeddings = user_embeddings[user_indices]
        pos_item_embeddings = item_embeddings[pos_item_indices]
        neg_item_embeddings = item_embeddings[neg_item_indices]

        # get image embeddings of the users and items 
        user_image_embeddings = user_image_embeddings[user_indices]
        pos_item_image_embeddings = item_image_embeddings[pos_item_indices]
        neg_item_image_embeddings = item_image_embeddings[neg_item_indices]

        # get text embeddings of the users and items 
        user_text_embeddings = user_text_embeddings[user_indices]
        pos_item_text_embeddings = item_text_embeddings[pos_item_indices]
        neg_item_text_embeddings = item_text_embeddings[neg_item_indices]

        # get user profile embeddings of the users and items
        user_profile_embeddings = user_profile_feat_embd[user_indices]
        item_profile_pos_embeddings = item_profile_feat_embed[pos_item_indices]
        item_profile_neg_embeddings = item_profile_feat_embed[neg_item_indices]

        # get user attributes embeddings of the users and items
        user_attributes_embeddings = user_attributes_embeddings[user_indices]
        item_attributes_pos_embeddings = item_attributes_embeddings[pos_item_indices]
        item_attributes_neg_embeddings = item_attributes_embeddings[neg_item_indices]

        self.E0_user = self.user_E0[user_indices]
        self.E0_item_pos = self.item_E0[pos_item_indices]
        self.E0_item_neg = self.item_E0[neg_item_indices]

        # create a dictionary of the embeddings
        results = {
            "embeddings": (user_embeddings, pos_item_embeddings, neg_item_embeddings),
            "image_embeddings": (user_image_embeddings, pos_item_image_embeddings, neg_item_image_embeddings),
            "text_embeddings": (user_text_embeddings, pos_item_text_embeddings, neg_item_text_embeddings),
            "profile_embeddings": (user_profile_embeddings, item_profile_pos_embeddings, item_profile_neg_embeddings),
            "attributes_embeddings": (
                user_attributes_embeddings, item_attributes_pos_embeddings, item_attributes_neg_embeddings)
        }

        return results

    def create_adjacency_matrix(self, train_df):
        """
        Create the adjacency matrix from the user-item interactions
        :param train_df:  training data frame containing the user-item interactions data frame
        :return:  None
        """
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

# %%

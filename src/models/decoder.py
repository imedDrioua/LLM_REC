"""
Script that defines the Decoder class
"""
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, feat_size, embed_size=64):
        super(Decoder, self).__init__()
        self.feat_size = feat_size
        self.user_net = nn.Sequential(
            nn.Linear(embed_size, int(self.feat_size)),
            nn.LeakyReLU(True),
        )

        self.item_net = nn.Sequential(
            nn.Linear(embed_size, int(self.feat_size)),
            nn.LeakyReLU(True),
        )

    def forward(self, user, item):
        user_output = self.user_net(user.float())
        tensor_list = []
        keys = item.keys()
        for _,value in enumerate(keys):
            tensor_list.append(item[value])
        item_tensor = torch.stack(tensor_list)
        item_output = self.item_net(item_tensor.float())
        return user_output, item_output
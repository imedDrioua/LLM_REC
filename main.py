# main script to run the program

from src.data_loader.data_loader import BooksDataset
from src.models.mm_model import MmModel
from src.train import Trainer
import torch
import numpy as np
import random

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# set the seed for reproducibility
def set_seed(seed):
    """
    Set the seed for reproducibility
    :param seed:  seed value
    :return:  None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# main script
if __name__ == "__main__":
    dataset_name = "netflix"
    set_seed(2022)
    # load the dataset
    dataset = BooksDataset(data_dir=f"./data/{dataset_name}")
    # load the model
    model = MmModel(n_users=dataset.n_users, n_items=dataset.n_items,
                    adjacency_matrix=dataset.get_dataset("adjacency_matrix"),
                    interactions=dataset.get_dataset("interactions"),
                    interactions_t=dataset.get_dataset("interactions_T"),
                    image_embeddings_data=dataset.get_dataset("images"),
                    text_embeddings_data=dataset.get_dataset("text"), embed_size=64, n_layers=1,
                    user_profiles_data=dataset.get_dataset("user_profiles"),
                    book_attributes_data=dataset.get_dataset("books_attributes"))
    model.to(device)

    # load the trainer
    trainer = Trainer(model=model, dataset=dataset, lr=0.005)

    # train the model
    trainer.train(epochs=10, batch_size=1024)

    # save the model
    torch.save(model.state_dict(), f'./model/{dataset_name}_model.pth')

# %%

# main script to run the program

from src.data_loader.data_loader import BooksDataset
from src.models.mm_model import MmModel
from src.train import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # load the dataset
    dataset = BooksDataset(data_dir="./data/books")
    # load the model
    model = MmModel(n_users=dataset.n_users, n_items=dataset.n_items,
                    adjacency_matrix=dataset.get_dataset("adjacency_matrix"),
                    interactions=dataset.get_dataset("interactions"),
                    image_embeddings_data=dataset.get_dataset("images"),
                    text_embeddings_data=dataset.get_dataset("text"), embed_size=128, n_layers=3,
                    user_profiles_data=dataset.get_dataset("user_profiles"),
                    book_attributes_data=dataset.get_dataset("books_attributes"))
    model.to(device)
    # load the trainer
    trainer = Trainer(model=model, dataset=dataset, lr=0.005)
    # train the model
    trainer.train(epochs=2, batch_size=1024)
    # save the model
    torch.save(model.state_dict(), '../model/model.pth')

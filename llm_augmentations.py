"""
This script is used to generate the data augmentation for the book recommendation system using the LLM model.
"""
# import libraries
import pandas as pd
from langchain_community.chat_models import ChatCohere, HumanMessage
import json
from src.data_augmentation.user_profile import llm_user_profile
from src.data_augmentation.user_item_interactions import llm_user_item_interaction
from src.data_augmentation.item_attributes import llm_book_profile
import secret


def llm_data_augmentation(n_users, n_books, users_history, users_candidates, api_key):
    """
    Function that generates the data augmentation for the book recommendation system using the LLM model.

    :param n_users: number of users
    :type n_users: int
    :param n_books: number of items
    :type n_books: int
    :param users_history: list of users interaction history
    :type users_history: dict
    :param users_candidates: list of items candidates
    :type users_candidates: dict
    :param api_key: API key
    :type api_key: str
    :return: None
    :rtype: None
    """
    # 1. Item attributes data augmentation
    # define the llm model
    llm = ChatCohere(cohere_api_key=api_key, model="chat", max_tokens=256, temperature=0.5,
                     connectors=[{"id": "web-search"}])
    # initial context
    initial_context = "You are now a books wiki. You are charged to provide the inquired information of the given book."
    llm.invoke([HumanMessage(content=initial_context)])
    # generate book profiles
    book_profiles = {}
    for i in range(n_books):
        book = books.iloc[i]
        llm = ChatCohere(cohere_api_key=api_key, model="chat", max_tokens=256, temperature=0.5,
                         # the temperature is set to 0.75
                         connectors=[{"id": "web-search"}])
        book_profile = llm_book_profile(book, llm)
        book_profiles[i] = book_profile
    # save book profiles
    with open("data/books/book_profiles.json", "w") as books_f:
        json.dump(book_profiles, books_f)

    # 2. User profile data augmentation
    # define the llm model
    llm = ChatCohere(cohere_api_key=api_key, model="chat", max_tokens=256, temperature=0.75,  # the temperature is set
                     # to 0.75, because we need more diversity in the output
                     connectors=[{"id": "web-search"}])
    # initial context
    initial_context = "You are now charged to generate user profiles that match the user history of read books."
    llm.invoke([HumanMessage(content=initial_context)])
    # generate user profiles
    user_profiles = {}
    for i in range(n_users):
        user_history = users_history[str(i)]
        user_item_interaction = llm_user_profile(user_history, llm)
        user_profiles[i] = user_item_interaction
    # save user profiles
    with open("data/books/user_profiles.json", "w") as user_f:
        json.dump(user_profiles, user_f)

    # 3. User-item interactions data augmentation
    # define the llm model
    llm = ChatCohere(cohere_api_key=api_key, model="chat", max_tokens=256, temperature=0.5,
                     connectors=[{"id": "web-search"}])
    # initial context
    initial_context = ("You are now a books recommender systems. Given user history of read books and  a list of "
                       "candidates The recommendation system needs to predict which books the user will like and which "
                       "books will dislike from the provided candidates based on the history and analysis.")

    llm.invoke([HumanMessage(content=initial_context)])
    # generate user-item interactions
    user_item_interactions = {}
    for i in range(n_users):
        user_history = users_history[str(i)]
        candidates = users_candidates[str(i)]
        user_item_interaction = llm_user_item_interaction(user_history, candidates, llm)
        user_item_interactions[i] = user_item_interaction
    # save user-item interactions
    with open("data/books/user_item_interactions.json", "w") as interactions_f:
        json.dump(user_item_interactions, interactions_f)


# define main function
if __name__ == "__main__":
    # load books data
    books = pd.read_csv("data/books/items_with_attributes.csv", sep=";")
    users = pd.read_csv("data/books/users.txt")

    # load users history
    with open("data/books/users_history.json", "r") as f:
        users_history_data = json.load(f)
    # load users candidates
    with open("data/books/users_candidates.json", "r") as f:
        users_candidates_data = json.load(f)
    API_KEY = secret.API_KEY
    # data dimensions
    number_books = len(books)
    number_users = len(users)
    # data augmentation
    llm_data_augmentation(number_users, number_books, users_history_data, users_candidates_data, API_KEY)

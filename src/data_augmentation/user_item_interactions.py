"""
Script that defines the prompt for the user-item interactions data augmentation.
"""
from langchain_core.messages import HumanMessage

def llm_user_item_interaction(user_history,candidates,llm):
    """
    function that defines the prompt for the user-item interactions data augmentation.

    :param user_history: history of books read by user
    :type user_history: str
    :param candidates: a list of candidates books to be recommended
    :type candidates: str
    :param llm: LLM model
    :type llm: LLM
    :return: LLM response
    :rtype: str
    """
    prompt = f"""The user has read the following books,that each book with [index] title, year, genres. :
            {user_history}
            Candidates:
            {candidates}
            Task:
            The recommendation system needs to predict which books the user will like and which books will dislike from the provided candidates based on the history and analysis.
            Please provide the output in the format of [index of one liked, index of one disliked] with no introductions or acknowledgments."""

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content
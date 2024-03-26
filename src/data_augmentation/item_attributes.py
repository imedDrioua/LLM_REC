"""
Script that defines the prompt for the item attributes data augmentation.
"""

from langchain_core.messages import HumanMessage

def llm_book_profile(book,llm):
    """
    function that defines the prompt for the item attributes data augmentation.

    :param book: book information (title, year, author)
    :param llm: LLm model
    :return: llm response
    """
    title, year,author = book
    prompt = f"""Provide the inquired information of the given book title : {title} ({year}) written by {author}. The inquired information is: genres, language. Please provide directly the output in the format of : {{"genres" : " | " , "language" :}} with no introductions or acknowledgments. Please consider only the specific genres"""

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    return response.content
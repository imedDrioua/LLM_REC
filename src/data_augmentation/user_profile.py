"""
Script that defines the prompt for the user profile data augmentation.
"""

from langchain_core.messages import HumanMessage


def llm_user_profile(history, llm):
    """
    function that defines the prompt for the user profile data augmentation.

    :param history: user history of books read
    :type history: str
    :param llm: LLm model
    :type llm: LLm
    :return: LLM response
    :rtype: str
    """
    u_profile = (f"Generate a user profile based on the provided user history and analysis, that each book with ["
                 f"index] title, year, author. History: {history}. Please output the following information of user "
                 f"without missing or None fields, output format: {{age: , gender: , liked genre: , disliked genre: , "
                 f"liked authors: , country: , language: }}. Give specific answers and no ranges, for gender give "
                 f"either male or female. Please don't give any 'None' answers it's important. please output only the "
                 f"content in format above, but no other thing else, no reasoning. Reiterating once again!! Please "
                 f"only output the content after \"output format: \", and do not include any other content such as "
                 f"introduction or acknowledgments.")
    messages = [HumanMessage(content=u_profile)]
    response = llm.invoke(messages)
    return response.content

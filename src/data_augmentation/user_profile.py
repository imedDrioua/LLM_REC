"""
Script that defines the prompt for the user profile data augmentation.
"""

from langchain_core.messages import HumanMessage

def llm_user_profile(history,llm):
    """
    function that defines the prompt for the user profile data augmentation.

    :param history: user history of books read
    :type history: str
    :param llm: LLm model
    :type llm: LLm
    :return: LLM response
    :rtype: str
    """
    u_profile = f"Generate a user profile based on the provided user history and analysis, that each book with [index] title, year, author. History: {history}. Please output the following infomation of user without missing or None fields, output format: {{age: , gender: , liked genre: , disliked genre: , liked authors: , country: , language: }}. Give specific answers and no ranges, for gender give either male or female. Please don't give any 'None' answers it's important. please output only the content in format above, but no other thing else, no reasoning. Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments. "
    messages = [HumanMessage(content=u_profile)]
    response = llm.invoke(messages)
    return response.content
class CustomPromptData:
    def __init__(self, customPromptData=None):
        if customPromptData:
            self.__custom_prompt_data = customPromptData
        else:
            self.__custom_prompt_data = f"""You are an Information Retrieval System capable of fetching data from scientific papers. Given the context below, answer the user's question. If you don't know the answer, politely tell that you are unware of that and do not try to make an answer of your own.

            Context: {context}
            Question: {question}

            Return the answer below.
            Answer:
            """

    def get_prompt_data(self):
        return self.__custom_prompt_data

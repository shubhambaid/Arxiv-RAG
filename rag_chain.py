from langchain.chains import RetrievalQA
from model import LLMModel
from prompt_structure import CustomPromptTemplate
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import (
    ConversationSummaryBufferMemory,
    ConversationBufferMemory,
)


class RetrievalQAChain:
    """
    Use when working with RAGs
    """

    def __init__(
        self,
        llm: LLMModel,
        prompt: CustomPromptTemplate,
        db: any,
        return_source_documents: bool,
    ) -> None:
        self.__llm = llm
        self.__prompt = prompt
        self.__db = db
        self.__return_source_documents = return_source_documents

    def getRetrievalQAChain(self) -> RetrievalQA:
        qna_chain = RetrievalQA.from_chain_type(
            llm=self.__llm,
            retriever=self.__db.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=self.__return_source_documents,
        )
        return qna_chain


if __name__ == "__main__":
    print("works")

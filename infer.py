from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from prompt import CustomPromptData
from prompt_structure import CustomPromptTemplate
from model import LLMModel
from rag_chain import RetrievalQAChain

if __name__ == "__main__":

    # Load the model
    llmModel = LLMModel()
    llm = llmModel.getLLM()

    combine_custom_data = """You are an Information Retrieval System capable of fetching data from scientific papers. Given the context below, answer the user's question. If you don't know the answer, politely tell that you are unware of that and do not try to make an answer of your own.

            Context: {context}
            Question: {question}

            Return the answer below.
            Answer:
            """
    # Create prompt template object
    combine_custom_prompt = CustomPromptTemplate(
        CustomPromptData(combine_custom_data).get_prompt_data(),
        input_variables=["context", "question"],
    ).getCustomPromptTemplate()

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    # Load FAISS vector store
    db = FAISS.load_local(
        "./tilda_vector_store", embeddings, allow_dangerous_deserialization=True
    )

    # Create Retrieval QA Chain object and fetch the chain
    convRetrievalQAChainObject = RetrievalQAChain(
        llm=llm,
        # condense_prompt=condense_prompt,
        prompt=combine_custom_prompt,
        db=db,
        return_source_documents=True,
    )
    chain = convRetrievalQAChainObject.getRetrievalQAChain()

    # Run inference
    while True:
        ques_ = input("What do you want help with ? \n")
        output = chain.invoke(ques_)
        # print(output)
        print(
            f"\nQuestion : {ques_}\nAnswer : {output['result']} \n Source: {output['source_documents']}"
        )

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


class FAISSVectorStores:

    def __init__(self):
        self.dataset_path = "./dataset"
        self.embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.faiss_db_path = "./tilda_vector_store"
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.device = "cpu"
        self.use_multithreading = True

    def create_vector_store(self):
        # Loading all the files in the directory
        loader = DirectoryLoader(
            self.dataset_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=self.use_multithreading,
        )

        documents = loader.load()

        # Splitting text in chunks recursively
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        texts = text_splitter.split_documents(documents)
        try:
            # Using the sentence-transformer model to extract embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embeddings_model,
                model_kwargs={"device": self.device},
            )
            print("Embeddings generated!")
        except Exception as e:
            print("Error while generating Embeddings", e)

        try:
            # Store the data in the db
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(self.faiss_db_path)
            print("Embeddings stored successfully!")
        except Exception as e:
            print("Errow while storing vectors", e)


if __name__ == "__main__":

    faiss_vector_store = FAISSVectorStores()
    # Call this function to create and store embeddings
    faiss_vector_store.create_vector_store()

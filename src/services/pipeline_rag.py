import os

from langchain.chains.retrieval_qa.base import RetrievalQA  # Or your chosen LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver

from src.config.groq import create_groq_embeddings, create_groq_llm
from src.config.milvus import create_milvus_connection
from src.config.redis import close_redis_connection, create_redis_connection
from src.data.csvLoaders import CustomCSVLoader
from src.services.retreivers import RetrieverFactory


class RAGPipeline:
    def __init__(self, csv_filepath, content_column):
        self.csv_filepath = csv_filepath
        self.content_column = content_column
        self.redis_client = None  # Initialize outside of load_data to avoid recreation
        self.milvus = None  # Initialize outside of load_data to avoid recreation
        self.llm = None
        self.retriever = None
        self.qa = None
        self.retriever_factory = None
        self.memory_saver = MemorySaver()

    def load_data(self):
        try:
            self.redis_client = create_redis_connection()  # Create connection here
            loader = CustomCSVLoader(self.csv_filepath, self.content_column)
            documents = loader.load()
            if not documents:
                raise ValueError("No documents loaded from CSV.")

            embeddings = (
                create_groq_embeddings()
            )  # For Groq, uncomment this line and comment out the next line
            # embeddings = OpenAIEmbeddings() #For OpenAI, uncomment this line and comment out the previous line

            self.milvus = create_milvus_connection()  # Create Milvus connection
            self.milvus.add_documents(documents, embeddings=embeddings)
            print("Data loaded and indexed into Milvus.")
            self.retriever_factory = RetrieverFactory(self.milvus)

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def initialize_llm(self, model_name: str = "llama-3.3-70b-versatile"):
        try:
            self.llm = create_groq_llm(model_name)  # For Groq
            # self.llm = OpenAI(temperature=0.8) #For OpenAI
            print("LLM initialized.")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise

    def create_qa_chain(self, retriever_type: str = "milvus"):
        try:
            self.retriever = self.retriever_factory.create_retriever(retriever_type)
            if self.retriever is None:
                raise ValueError("Retriever creation failed.")

            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
            )
            print("RetrievalQA chain created.")
        except Exception as e:
            print(f"Error creating RetrievalQA chain: {e}")
            raise

    def run_query(self, query: str):
        try:
            result = self.qa({"query": query})
            self.memory_saver.writes.values({"query": query, "result": result})
            return result
        except Exception as e:
            print(f"Error running query: {e}")
            return None

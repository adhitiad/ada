import os

from dotenv import load_dotenv
from langchain.chains import history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import (
    create_retrieval_chain as create_retrieval_chain_langchain,
)
from langchain.chains.retrieval_qa import base as retrieval_qa_base
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.vectorstores import Pinecone
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_nvidia import NVIDIAEmbeddings
from pinecone import Pinecone as pinecone
from pinecone import ServerlessSpec

# Load environment variables from .env
load_dotenv()

# Define the data directory and ensure it exists
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "ghg.csv")

# Ensure the CSV file exists before proceeding
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The required CSV file does not exist at: {file_path}")


# Buat CSVLoader dan TextLoader untuk memuat data
def load_data(file_path):
    if file_path.endswith(".csv"):
        return CSVLoader(
            file_path,
            encoding="utf-8",
            csv_args={"delimiter": ",", "quotechar": '"', "escapechar": "\\"},
        )
    else:
        return TextLoader(file_path)


print("\n--- Using Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(load_data(file_path).load())
print("Number of documents after splitting:", len(rec_char_docs))


# Define the embedding model
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-mistral-7b-v2",
    api_key=os.getenv("NVIDIA_API_KEY"),
    truncate="NONE",
)

# Initialize Pinecone
pc = pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "ragunsub"

# Check if index already exists
if index_name not in pc.list_indexes().names():
    # Create index if it doesn't exist
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists. Skipping index creation.")


# Create or load vector store
def create_vector_store(documents, index_name):
    # Check if documents already exist in the index
    index = pinecone.Index(index_name)
    stats = index.describe_index_stats()

    if stats["total_vector_count"] == 0:
        # If index is empty, create new vectors
        db = Pinecone.from_documents(documents, embeddings, index_name=index_name)
    else:
        # If documents exist, just connect to existing index
        db = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    return db


# Create a retriever for querying the vector store
retriever = create_vector_store(rec_char_docs, index_name).as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatOpenAI model
llm = ChatGroq(
    temperature=0,
    model="mixtral-8x7b-32768",
)

# Contextualize question prompt
contextualize_q_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of context to answer the "
    "question. If you don't know the answer, just say that you "
    "semua tentang Universitas Subang. Use three sentences "
    "maximum and keep the answer concise."
    "\n\n"
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


history_aware_retriever = history_aware_retriever.create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=contextualize_q_prompt
)
# Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt,
)
# Buat rantai pengambilan yang menggabungkan pengambil berbasis riwayat dan rantai pertanyaan jawaban
rag_chain = create_retrieval_chain_langchain(
    history_aware_retriever, question_answer_chain
)


# Function to simulate a continual chat
def continual_chat():
    """
    Simulate a continual chat with the AI.

    """
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval QA chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


def create_retrieval_chain(history_aware_retriever, question_answer_chain):
    """
    Create a chain that takes conversation history and returns documents.

    If there is no `chat_history`, then the `input` is just passed directly to the
    retriever. If there is `chat_history`, then the prompt and LLM will be used
    to generate a search query. That search query is then passed to the retriever.

    Args:
        history_aware_retriever: A retriever that takes a string as input and outputs
            a list of Documents.
        question_answer_chain: A chain that takes a list of Documents and returns
            an answer.
    Returns:
        A chain that takes a string as input and returns an answer.
    """
    return retrieval_qa_base.BaseRetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=history_aware_retriever,
        combine_documents_chain=question_answer_chain,
    )


# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()

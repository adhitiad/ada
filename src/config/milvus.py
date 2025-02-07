from langchain_milvus import Milvus

from src.config.groq import create_groq_embeddings


def create_milvus_connection():
    """
    Create a Milvus connection.
    """
    milvus_uri = "https://in03-6693ff551a6401a.serverless.gcp-us-west1.cloud.zilliz.com"
    milvus_api_key = "5a9e9d7665961e2a9330d4486de8b36578044979edcbcce6813f9cd459098de39a27e5938f618b21665c968520974796b389e111"

    milvus = Milvus(
        connection_args={"uri": milvus_uri, "token": milvus_api_key},
        embedding_function=create_groq_embeddings(),
        collection_name="unsub_RaG",
        consistency_level="Strong",
    )

    return milvus

from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


def create_groq_llm(model_name):
    """
    Create a Groq LLM.
    """
    return ChatGroq(
        temperature=0.8,
        model_name=model_name,
        api_key="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",
    )


def create_groq_embeddings():
    """
    Create Groq embeddings.
    """
    client = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key="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",
        truncate="NONE",
    )

    return client

from typing import Optional

from langchain_milvus import Milvus


class RetrieverFactory:
    """Factory class to create different types of retrievers."""

    def __init__(self, milvus_connection: Milvus):
        """
        Initializes the RetrieverFactory.

        Args:
            milvus_connection: A pre-initialized Milvus connection object.
        """
        self.milvus_connection = milvus_connection

    def create_retriever(self, retriever_type: str = "milvus") -> Optional[Milvus]:
        """
        Creates a retriever based on the specified type.

        Args:
            retriever_type: The type of retriever to create ("milvus" is currently supported).

        Returns:
            A retriever object or None if the type is not supported.
        """
        if retriever_type == "milvus":
            return self.milvus_connection.as_retriever()
        else:
            print(f"Unsupported retriever type: {retriever_type}")
            return None

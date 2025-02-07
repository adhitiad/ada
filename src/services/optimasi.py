import time

from langchain.embeddings.base import Embeddings
from langchain_milvus import Milvus
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.config.groq import create_groq_embeddings
from src.services.pipeline_rag import RAGPipeline


def benchmark_rag_pipeline(rag_pipeline: RAGPipeline, queries: list[str]):
    """Benchmarks the RAG pipeline's query execution time."""
    results = []
    for query in queries:
        start_time = time.time()
        result = rag_pipeline.run_query(query)
        end_time = time.time()
        results.append(
            {"query": query, "time": end_time - start_time, "result": result}
        )
    return results


def optimize_embeddings(embeddings: Embeddings, documents: list):
    """Optimizes the embeddings (if applicable).  This is a placeholder."""
    #  Add your specific embedding optimization here,
    #  e.g., dimensionality reduction, quantization.
    # Get embeddings for all documents

    all_embeddings = []
    for doc in documents:
        embedding = embeddings.embed_query(doc.page_content)
        all_embeddings.append(embedding)
    # Apply dimensionality reduction using PCA

    pca = PCA(n_components=min(len(all_embeddings[0]) // 2, 100))
    reduced_embeddings = pca.fit_transform(all_embeddings)

    # Apply quantization to reduce memory footprint
    scaler = MinMaxScaler()
    normalized_embeddings = scaler.fit_transform(reduced_embeddings)
    quantized_embeddings = (normalized_embeddings * 255).astype("uint8")

    print("Embeddings optimized.")
    return quantized_embeddings


def optimize_milvus(milvus: Milvus):
    """Optimizes the Milvus index (if applicable).  This is a placeholder."""
    # Add your Milvus index optimization here, e.g.,'
    milvus_connect = create_groq_embeddings()
    if milvus_connect:
        collection = milvus._conn.get_collection(milvus.collection_name)
        collection.flush()
        collection.release()
        collection.load()

        # Optimize index parameters for better search performance
        index_params = {
            "index_type": "IVF_FLAT",  # Using IVF_FLAT for better balance of speed/accuracy
            "metric_type": "IP",  # Inner Product distance metric
            "params": {"nlist": 1024},  # Number of clusters
        }

        # Drop existing index if any
        collection.drop_index()

        # Create new optimized index
        collection.create_index(
            field_name="vector", index_params=index_params  # Vector field name
        )

        # Load collection into memory for faster search
        collection.load()

        # Set search parameters for better performance
        milvus.search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 16},  # Number of clusters to search
        }

    print("Milvus optimization not implemented.")
    return milvus


def optimize_pipeline(rag_pipeline: RAGPipeline, queries: list[str] = None):
    """
    Optimizes the RAG pipeline. This is a placeholder function.
    You would put your specific optimization strategies here.
    """
    # 1. Embeddings Optimization
    # Get the embeddings object (you might need to refactor pipeline_rag.py)
    if hasattr(rag_pipeline, "milvus") and hasattr(
        rag_pipeline.milvus, "embedding_function"
    ):
        embeddings = rag_pipeline.milvus.embedding_function

        documents = (
            rag_pipeline.milvus.get_documents()
        )  # Use getter method instead of accessing _documents
        optimize_embeddings(embeddings, documents)

    # 2. Milvus Index Optimization:
    if hasattr(rag_pipeline, "milvus"):
        rag_pipeline.milvus = optimize_milvus(rag_pipeline.milvus)

    # 3. Benchmarking: (If queries is not None)
    if queries:
        print("\nBenchmarking RAG pipeline after optimization:")
        results = benchmark_rag_pipeline(rag_pipeline, queries)
        for result in results:
            print(f"Query: {result['query']}, Time: {result['time']:.4f} seconds")

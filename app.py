from flask import Flask, jsonify, request

from src.config.redis import close_redis_connection
from src.services.optimasi import benchmark_rag_pipeline, optimize_pipeline
from src.services.pipeline_rag import RAGPipeline

app = Flask(__name__)

#
# Configuration (replace with your values)
CSV_FILE_PATH = "ghg.csv"  # Path to your CSV data  # Column name for text content


CONTENT_COLUMN = "text"  # Changed from list to string


class PipelineManager:
    def __init__(self, csv_file_path, content_column):
        self.rag_pipeline = None
        self.csv_file_path = csv_file_path
        self.content_column = content_column

    def load_data(self):
        try:
            if self.rag_pipeline:
                return jsonify({"message": "Data already loaded"}), 200

            self.rag_pipeline = RAGPipeline(self.csv_file_path, self.content_column)
            self.rag_pipeline.load_data()
            self.rag_pipeline.initialize_llm()
            self.rag_pipeline.create_qa_chain()
            return jsonify({"message": "Data loaded successfully"}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if hasattr(self.rag_pipeline, "redis_client"):
                close_redis_connection(self.rag_pipeline.redis_client)


pipeline_manager = PipelineManager(CSV_FILE_PATH, CONTENT_COLUMN)


@app.route("/", methods=["GET"])
def home():
    return (
        jsonify(
            {
                "message": "Welcome to the RAG Pipeline API!",
                "data": {
                    "endpoints": {
                        "/load": "Load data into the RAG pipeline",
                        "/query": "Run a query through the RAG pipeline",
                        "/optimize": "Optimize the RAG pipeline",
                        "/benchmark": "Benchmark the RAG pipeline's query execution time",
                    },
                    "parameters": {
                        "query": "The query to run through the RAG pipeline",
                        "queries": "A list of queries to benchmark the RAG pipeline's query execution time",
                    },
                    "response": {
                        "load": "Returns a success message and status code",
                        "query": "Returns the result of the query and status code",
                        "optimize": "Returns a success message and status code",
                        "benchmark": "Returns the benchmark results and status code",
                    },
                },
            }
        ),
        200,
    )


@app.route("/load", methods=["GET"])
def load_data_route():
    try:
        if not pipeline_manager.rag_pipeline:
            return pipeline_manager.load_data()
        return jsonify({"message": "Data already loaded"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/load", methods=["POST"])
def load_data():
    try:
        if pipeline_manager.rag_pipeline:
            return jsonify({"message": "Data already loaded"}), 200

        return pipeline_manager.load_data()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def run_query():
    try:
        if not pipeline_manager.rag_pipeline:
            return (
                jsonify({"error": "Data not loaded yet. Use the /load endpoint."}),
                500,
            )
        data = request.get_json()
        query = data.get("query")
        if not query:
            return jsonify({"error": "Query parameter is missing"}), 400

        if not hasattr(pipeline_manager.rag_pipeline, "run_query") or not callable(
            getattr(pipeline_manager.rag_pipeline, "run_query", None)
        ):
            return jsonify({"error": "RAG pipeline not properly initialized"}), 500

        result = pipeline_manager.rag_pipeline.run_query(query)
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/optimize", methods=["POST"])
def optimize_pipeline_route():
    try:
        if not pipeline_manager.rag_pipeline:
            return (
                jsonify({"error": "Data not loaded yet. Use the /load endpoint."}),
                500,
            )

        queries = request.get_json().get("queries", [])
        optimize_pipeline(pipeline_manager.rag_pipeline, queries)
        return jsonify({"message": "Pipeline optimized"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/benchmark", methods=["POST"])
def benchmark_pipeline():
    try:
        if not pipeline_manager.rag_pipeline:
            return (
                jsonify({"error": "Data not loaded yet. Use the /load endpoint."}),
                500,
            )

        queries = request.get_json().get("queries", [])
        if not queries:
            return jsonify({"error": "Queries are required"}), 400
        results = benchmark_rag_pipeline(pipeline_manager.rag_pipeline, queries)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    try:
        if (
            hasattr(pipeline_manager, "rag_pipeline")
            and pipeline_manager.rag_pipeline
            and hasattr(pipeline_manager.rag_pipeline, "redis_client")
        ):
            close_redis_connection(pipeline_manager.rag_pipeline.redis_client)
        return jsonify({"message": "Server shutting down"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)

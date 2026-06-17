"""Local journal engine — ingestion, hybrid retrieval, RAG, and analytics.

Everything runs locally: LanceDB (embedded) for storage + full-text search,
Ollama (localhost) for embeddings and generation. No cloud, no telemetry.
"""

__version__ = "1.0.0"

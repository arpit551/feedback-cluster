import os


class Settings:
    db_path: str = os.getenv("CLUSTER_API_DB_PATH", "ideas.db")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))


settings = Settings()

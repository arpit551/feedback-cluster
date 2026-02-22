import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    db_path: str = os.getenv("CLUSTER_API_DB_PATH", "ideas.db")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))


settings = Settings()

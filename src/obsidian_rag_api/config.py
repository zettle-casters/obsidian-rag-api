from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()
class Settings(BaseSettings):
    # Neo4j connection - can be set via NEO4J_URL or constructed from NEO4J_URI/USER/PASSWORD
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "test1234"

    @property
    def neo4j_url(self) -> str:
        """Construct Neo4j URL from URI and credentials."""
        # Remove bolt:// prefix if present
        uri = self.neo4j_uri.replace("bolt://", "")
        return f"bolt://{self.neo4j_user}:{self.neo4j_password}@{uri}"

    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_prefer_grpc: bool = False

    embeddings_model: str = "openai/text-embedding-3-large"

    openai_api_key: str
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_cheap_model: str = "gpt-4o-mini"

    max_recursion_depth: int = 3
    search_top_k: int = 5

    host: str = "0.0.0.0"
    port: int = 8000

    # Путь для сохранения метаданных vault'ов
    vaults_metadata_path: str = "/app/data/.vaults_metadata.json"

    model_config = {"env_file": ".env"}


settings = Settings()

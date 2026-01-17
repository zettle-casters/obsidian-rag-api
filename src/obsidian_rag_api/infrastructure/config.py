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

    database_url: str
    web_origins: str = "http://localhost:3000"
    ui_base_url: str = "http://localhost:3000"

    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"

    session_cookie_name: str = "obsidian_session"
    session_ttl_days: int = 30
    cookie_secure: bool = False
    cookie_samesite: str = "lax"
    demo_user_email: str = "demo@local"
    default_model_avatar_url: Optional[str] = None

    model_config = {"env_file": ".env"}

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.web_origins.split(",") if origin.strip()]


settings = Settings()

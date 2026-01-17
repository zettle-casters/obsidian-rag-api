from typing import List, Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "openai/text-embedding-3-large",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
        )

        self.dim = len(self.model.embed_query("test"))

    def _sanitize(self, embeddings: np.ndarray) -> np.ndarray:
        if not np.isfinite(embeddings).all():
            # Replace non-finite values to avoid Qdrant validation errors.
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        return embeddings

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=float)
        embeddings = self.model.embed_documents(texts)
        return self._sanitize(np.array(embeddings, dtype=float))

    def encode_one(self, text: str) -> np.ndarray:
        embedding = self.model.embed_query(text)
        return self._sanitize(np.array([embedding], dtype=float))

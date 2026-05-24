"""ChromaDB-backed vector store for the UdaPlay game catalog."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions


def _flatten_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def game_to_document(game: dict[str, Any]) -> str:
    """Render a game record as the text we'll embed.

    We concatenate the semantically rich fields so a single embedding captures
    title, studio, platform, era, genre, and the prose description.
    """
    return (
        f"{game['name']} ({game['release_year']}) — "
        f"Genre: {game['genre']}. "
        f"Platforms: {_flatten_value(game['platform'])}. "
        f"Developer: {game['developer']}. "
        f"Publisher: {game['publisher']}. "
        f"{game['description']}"
    )


class GameVectorStore:
    """Thin wrapper around a Chroma persistent collection of games."""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "games",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: str | None = None,
    ) -> None:
        self.persist_dir = str(persist_dir)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self._embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embedding_model,
        )
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def add_games(self, games: list[dict[str, Any]]) -> None:
        """Upsert a list of game dicts. Idempotent on `id`."""
        ids = [g["id"] for g in games]
        documents = [game_to_document(g) for g in games]
        metadatas = [
            {
                "name": g["name"],
                "platform": _flatten_value(g["platform"]),
                "genre": g["genre"],
                "publisher": g["publisher"],
                "developer": g["developer"],
                "release_year": int(g["release_year"]),
            }
            for g in games
        ]
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def load_from_json(self, json_path: str | Path) -> int:
        with open(json_path, "r") as f:
            games = json.load(f)
        self.add_games(games)
        return len(games)

    def query(self, query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Run a semantic search. Returns a list of {id, document, metadata, distance}."""
        raw = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        results: list[dict[str, Any]] = []
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0] if raw.get("distances") else [None] * len(ids)
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            results.append(
                {
                    "id": i,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": (1 - dist) if dist is not None else None,
                }
            )
        return results

    def reset(self) -> None:
        """Drop and recreate the collection. Useful when re-indexing."""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

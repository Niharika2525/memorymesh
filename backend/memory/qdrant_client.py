"""Qdrant vector memory service.

Provides persistent vector storage, embedding management, collection lifecycle,
and semantic search retrieval for the MemoryMesh multi-agent system.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models

from backend.config import settings

logger = logging.getLogger(__name__)


def _mask_url(url: str) -> str:
    """Mask credentials or sensitive tokens in URL for safe logging and status."""
    if not url or url == ":memory:":
        return url or ":memory:"
    try:
        parsed = urlparse(url)
        if parsed.password:
            netloc = f"{parsed.username or ''}:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return parsed._replace(netloc=netloc).geturl()
        return url
    except Exception:
        return "[masked-url]"


def _normalize_point_id(raw_id: Optional[str]) -> str:
    """Ensure point ID is a valid RFC 4122 UUID string required by Qdrant."""
    if not raw_id:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(str(raw_id)))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))


class QdrantMemoryService:
    """Manages vector storage, indexing, and semantic retrieval with Qdrant."""

    DISTANCE_MAPPING = {
        "cosine": models.Distance.COSINE,
        "euclid": models.Distance.EUCLID,
        "dot": models.Distance.DOT,
        "manhattan": models.Distance.MANHATTAN,
    }

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ):
        """Initialize the Qdrant service without exposing secrets.

        Credentials and connection details are read from environment variables
        via backend.config.settings unless explicitly overridden.
        """
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_size = vector_size or settings.QDRANT_VECTOR_SIZE
        self.distance_name = distance or settings.QDRANT_DISTANCE

        # Determine target URL and API key from settings or overrides
        raw_url = url if url is not None else settings.QDRANT_URL
        raw_api_key = api_key if api_key is not None else settings.QDRANT_API_KEY

        # Fallback to host/port if URL is not provided
        if not raw_url and settings.QDRANT_HOST:
            raw_url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

        # Safeguard: detect accidental inversion between URL and JWT API key
        if raw_url and raw_url.startswith("eyJ") and raw_api_key and (raw_api_key.startswith("http://") or raw_api_key.startswith("https://")):
            logger.warning("Detected inverted QDRANT_URL and QDRANT_API_KEY; automatically swapping them.")
            raw_url, raw_api_key = raw_api_key, raw_url

        self.url = raw_url
        self._api_key = raw_api_key.strip() if raw_api_key else None

        if client is not None:
            self.client = client
        else:
            self.client = self._init_client()

    def _init_client(self) -> QdrantClient:
        """Create QdrantClient instance using configured parameters."""
        if not self.url or self.url == ":memory:":
            logger.info("Initializing QdrantClient with in-memory storage")
            return QdrantClient(":memory:")

        masked = _mask_url(self.url)
        logger.info("Initializing QdrantClient pointing to %s", masked)
        return QdrantClient(
            url=self.url,
            api_key=self._api_key,
            timeout=30.0,
        )

    def check_connection(self) -> Dict[str, Any]:
        """Verify connection to Qdrant cluster safely."""
        masked = _mask_url(self.url)
        try:
            collections_res = self.client.get_collections()
            collection_names = [col.name for col in collections_res.collections]
            has_required_col = self.collection_name in collection_names

            return {
                "connected": True,
                "status": "connected",
                "url": masked,
                "collection": self.collection_name,
                "collection_exists": has_required_col,
                "collections_count": len(collection_names),
                "error": None,
            }
        except Exception as e:
            # Never include raw API key in error messages
            err_msg = str(e)
            if self._api_key and self._api_key in err_msg:
                err_msg = err_msg.replace(self._api_key, "***")
            logger.warning("Qdrant connection check failed for %s: %s", masked, err_msg)
            return {
                "connected": False,
                "status": "disconnected",
                "url": masked,
                "collection": self.collection_name,
                "collection_exists": False,
                "collections_count": 0,
                "error": err_msg,
            }

    def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: Optional[str] = None,
    ) -> bool:
        """Create the collection if it does not exist."""
        col_name = collection_name or self.collection_name
        size = vector_size or self.vector_size
        dist_key = (distance or self.distance_name).lower()
        dist_metric = self.DISTANCE_MAPPING.get(dist_key, models.Distance.COSINE)

        try:
            if not self.client.collection_exists(col_name):
                logger.info("Creating Qdrant collection '%s' (size=%d, distance=%s)", col_name, size, dist_key)
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=models.VectorParams(size=size, distance=dist_metric),
                )
            return True
        except Exception as e:
            logger.error("Failed to ensure collection '%s': %s", col_name, str(e))
            raise

    def store_memory(
        self,
        content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a memory embedding with associated metadata.

        If vector is not provided, it is automatically generated from content.
        """
        if vector is None:
            from backend.services.llm import get_llm_service

            vector = get_llm_service().generate_embedding(content, dimensions=self.vector_size)

        col_name = collection_name or self.collection_name
        self.ensure_collection(col_name, vector_size=len(vector))

        point_id = _normalize_point_id(memory_id)
        original_id = memory_id or point_id
        created_at = datetime.now(timezone.utc).isoformat()

        payload = {
            "content": content,
            "created_at": created_at,
            "metadata": metadata or {},
            "custom_id": original_id,
        }
        # Also flatten metadata top-level keys for easy direct filtering
        if metadata:
            for k, v in metadata.items():
                if k not in payload:
                    payload[k] = v

        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )

        self.client.upsert(
            collection_name=col_name,
            points=[point],
        )

        return {
            "id": original_id,
            "status": "stored",
            "content": content,
            "metadata": metadata or {},
            "created_at": created_at,
        }

    def store_memories_batch(
        self,
        memories: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """Store multiple memory embeddings with metadata in a single batch operation."""
        if not memories:
            return []

        col_name = collection_name or self.collection_name
        first_vec_size = len(memories[0]["vector"])
        self.ensure_collection(col_name, vector_size=first_vec_size)

        points = []
        stored_ids = []
        created_at = datetime.now(timezone.utc).isoformat()

        for item in memories:
            raw_id = item.get("id")
            point_id = _normalize_point_id(raw_id)
            original_id = raw_id or point_id
            stored_ids.append(original_id)

            meta = item.get("metadata") or {}
            payload = {
                "content": item["content"],
                "created_at": item.get("created_at") or created_at,
                "metadata": meta,
                "custom_id": original_id,
            }
            for k, v in meta.items():
                if k not in payload:
                    payload[k] = v

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=item["vector"],
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=col_name,
            points=points,
        )
        return stored_ids

    def search_memories(
        self,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search and return structured memory objects."""
        col_name = collection_name or self.collection_name

        if query_vector is None:
            if not query_text:
                return []
            from backend.services.llm import get_llm_service

            query_vector = get_llm_service().generate_embedding(query_text, dimensions=self.vector_size)

        # Ensure collection exists before search
        if not self.client.collection_exists(col_name):
            return []

        # Build filter if criteria provided
        query_filter = None
        if filter_criteria:
            conditions = []
            for key, val in filter_criteria.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=val),
                    )
                )
            query_filter = models.Filter(must=conditions)

        # Support both modern query_points and legacy search API
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=col_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )
            raw_points = response.points
        else:
            raw_points = self.client.search(
                collection_name=col_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )

        results = []
        for pt in raw_points:
            payload = pt.payload or {}
            ret_id = payload.get("custom_id") or str(pt.id)
            results.append({
                "id": ret_id,
                "score": float(pt.score) if pt.score is not None else 0.0,
                "content": payload.get("content", ""),
                "metadata": payload.get("metadata", {}),
                "created_at": payload.get("created_at"),
            })

        return results

    def get_memory(self, memory_id: str, collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by its ID."""
        col_name = collection_name or self.collection_name
        if not self.client.collection_exists(col_name):
            return None

        point_id = _normalize_point_id(memory_id)
        points = self.client.retrieve(collection_name=col_name, ids=[point_id])
        if not points:
            return None

        pt = points[0]
        payload = pt.payload or {}
        ret_id = payload.get("custom_id") or str(pt.id)
        return {
            "id": ret_id,
            "content": payload.get("content", ""),
            "metadata": payload.get("metadata", {}),
            "created_at": payload.get("created_at"),
        }

    def delete_memory(self, memory_id: str, collection_name: Optional[str] = None) -> bool:
        """Delete a memory by its ID."""
        col_name = collection_name or self.collection_name
        if not self.client.collection_exists(col_name):
            return False

        point_id = _normalize_point_id(memory_id)
        self.client.delete(
            collection_name=col_name,
            points_selector=models.PointIdsList(points=[point_id]),
        )
        return True

    def count_memories(self, collection_name: Optional[str] = None) -> int:
        """Get total count of memories in the collection."""
        col_name = collection_name or self.collection_name
        if not self.client.collection_exists(col_name):
            return 0
        return self.client.count(collection_name=col_name).count

    @staticmethod
    def format_memories_for_agent(memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories into a clean context block for agent workflow."""
        if not memories:
            return "No previous relevant memories found."

        lines = ["### Relevant Retrieved Context & Memories:"]
        for idx, mem in enumerate(memories, 1):
            score = mem.get("score")
            score_str = f" [relevance: {score:.2f}]" if score is not None else ""
            content = mem.get("content", "").strip()
            meta = mem.get("metadata", {})
            source = meta.get("source", "system")
            created = mem.get("created_at") or "unknown date"
            lines.append(f"{idx}. ({source} | {created}){score_str}: {content}")

        return "\n".join(lines)


# Singleton instance helper
_memory_service_instance: Optional[QdrantMemoryService] = None


def get_qdrant_service() -> QdrantMemoryService:
    """Dependency helper returning active QdrantMemoryService instance."""
    global _memory_service_instance
    if _memory_service_instance is None:
        _memory_service_instance = QdrantMemoryService()
    return _memory_service_instance


if __name__ == "__main__":
    # Self-test script when run directly
    print("Testing Qdrant connection via environment...")
    svc = QdrantMemoryService()
    status = svc.check_connection()
    print("Connection status:", status["status"])
    print("Cluster URL:", status["url"])
    print("Target collection:", status["collection"])
    print("Collection exists:", status["collection_exists"])
    if status["error"]:
        print("Note:", status["error"])

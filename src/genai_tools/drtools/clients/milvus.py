# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Milvus client: vector and text search, collection creation, insert, and inspection.

All third-party API logic lives here; tools only validate, obtain config, and call this client.
"""

import logging
from typing import Any
from typing import Literal

from fastmcp.exceptions import ToolError
from pymilvus import DataType
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from genai_tools.auth.utils import get_access_configs

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 5
DEFAULT_DB = "default"

# Config names match headers (x-milvus-uri, etc.) and env vars (X_MILVUS_URI_ENV_VAR, etc.)
MILVUS_CONFIG_SPEC = {
    "uri": {"required": True},
    "token": {"required": False},
    "db": {"required": False, "default": "default"},
}

# Map string type names to pymilvus DataType for custom schema
_SCHEMA_TYPE_MAP = {
    "INT64": DataType.INT64,
    "INT32": DataType.INT32,
    "VARCHAR": DataType.VARCHAR,
    "STRING": DataType.VARCHAR,
    "FLOAT": DataType.FLOAT,
    "DOUBLE": DataType.DOUBLE,
    "BOOL": DataType.BOOL,
    "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
    "BINARY_VECTOR": DataType.BINARY_VECTOR,
}

_VECTOR_DTYPES = (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR)


def _field_to_dtype_and_kw(field: dict[str, Any]) -> tuple[DataType, bool, dict[str, Any]]:
    """Parse a field_schema entry into dtype, is_primary, and add_field kwargs."""
    type_str = (field.get("type") or "FLOAT_VECTOR").upper().replace(" ", "_")
    dtype = _SCHEMA_TYPE_MAP.get(type_str, DataType.FLOAT_VECTOR)
    is_primary = field.get("is_primary", False)
    add_kw: dict[str, Any] = {}
    if (dim := field.get("dim")) is not None and dtype in _VECTOR_DTYPES:
        add_kw["dim"] = int(dim)
    if (max_length := field.get("max_length")) is not None:
        add_kw["max_length"] = int(max_length)
    return dtype, is_primary, add_kw


def _build_schema_from_field_schema(
    field_schema: list[dict[str, Any]],
    *,
    auto_id: bool,
    dimension: int,
    primary_field_name: str,
    vector_field_name: str,
) -> tuple[bool, list[tuple[str, DataType, bool, dict[str, Any]]]]:
    """
    Merge field_schema with quick-setup params. field_schema takes precedence per field:
    - If a primary key is in field_schema (is_primary or name==primary_field_name), use it.
      The primary field may include "auto_id": True so Milvus generates IDs; else auto_id is False.
    - If a vector field is defined in field_schema (FLOAT_VECTOR/BINARY_VECTOR), use it.
    - Otherwise use primary_field_name/vector_field_name + dimension from params.
    Returns (schema_auto_id, list of (field_name, dtype, is_primary, add_kw)).
    """
    primary_from_schema: dict[str, Any] | None = None
    vector_from_schema: dict[str, Any] | None = None
    other_fields: list[dict[str, Any]] = []

    for f in field_schema:
        name = f.get("name") or ""
        if not name:
            continue
        is_primary = f.get("is_primary", False)
        type_str = (f.get("type") or "").upper().replace(" ", "_")
        is_vector = type_str in ("FLOAT_VECTOR", "BINARY_VECTOR")
        if is_primary or name == primary_field_name:
            primary_from_schema = f
        elif is_vector or name == vector_field_name:
            # Prefer field named vector_field_name; else first vector type
            if vector_from_schema is None or name == vector_field_name:
                vector_from_schema = f
        else:
            other_fields.append(f)

    effective: list[tuple[str, type[DataType], bool, dict[str, Any]]] = []
    schema_auto_id = auto_id

    if primary_from_schema:
        dtype, is_primary, add_kw = _field_to_dtype_and_kw(primary_from_schema)
        effective.append((primary_from_schema["name"], dtype, True, add_kw))
        # Respect "auto_id" on the primary field when set; else default False (caller supplies IDs)
        schema_auto_id = (
            bool(primary_from_schema["auto_id"]) if "auto_id" in primary_from_schema else False
        )
    else:
        effective.append((primary_field_name, DataType.INT64, True, {}))

    if vector_from_schema:
        dtype, _, add_kw = _field_to_dtype_and_kw(vector_from_schema)
        effective.append((vector_from_schema["name"], dtype, False, add_kw))
    else:
        effective.append((vector_field_name, DataType.FLOAT_VECTOR, False, {"dim": dimension}))

    for f in other_fields:
        dtype, is_primary, add_kw = _field_to_dtype_and_kw(f)
        effective.append((f["name"], dtype, is_primary, add_kw))

    return schema_auto_id, effective


def get_milvus_access_configs() -> dict[str, str]:
    """
    Get Milvus connection config (URI, token, db) from headers or env.

    Reads from x-milvus-uri, x-milvus-token, x-milvus-db or, when local deployment
    is enabled, from X_MILVUS_URI_ENV_VAR, X_MILVUS_TOKEN_ENV_VAR, X_MILVUS_DB_ENV_VAR.

    Returns
    -------
    dict
        Keys: uri, token (optional), db (default "default").

    Raises
    ------
    ToolError
        If required config (URI) is not found.
    """
    return get_access_configs("milvus", MILVUS_CONFIG_SPEC)


class MilvusError(Exception):
    """Raised when a Milvus client operation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _build_filter(text_field: str | None, query_text: str | None, filter_expr: str | None) -> str:
    """Combine optional text match with user filter expression."""
    parts: list[str] = []
    if text_field and query_text and query_text.strip():
        # Escape single quotes in query_text for Milvus expr (string literals use single quotes)
        safe = query_text.strip().replace("\\", "\\\\").replace("'", "\\'")
        parts.append(f"{text_field} like '%{safe}%'")
    if filter_expr and filter_expr.strip():
        parts.append(f"({filter_expr.strip()})")
    if not parts:
        return ""
    return " and ".join(parts)


def _normalize_search_results(raw: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Convert pymilvus search result (list of lists per query) to a flat list of hit dicts."""
    out: list[dict[str, Any]] = []
    for query_hits in raw:
        for hit in query_hits:
            entity = hit.get("entity") or {}
            row: dict[str, Any] = {"id": hit.get("id"), "distance": hit.get("distance")}
            row.update(entity)
            out.append(row)
    return out


class MilvusClientWrapper:
    """
    Wrapper around pymilvus MilvusClient for search operations.

    Uses server-side push-down filtering; no business logic in tools.
    """

    def __init__(self, config: dict[str, str]) -> None:
        uri = config.get("uri")
        if not uri:
            raise MilvusError("Milvus URI is required")
        token = config.get("token") or None
        self._db = config.get("db") or DEFAULT_DB
        self._client = MilvusClient(uri=uri, token=token)
        try:
            self._client.using_database(self._db)
        except MilvusException as e:
            if e.code == 800 or "database not found" in (e.message or str(e)).lower():
                raise ToolError(
                    f"Database '{self._db}' not found. Create it in Milvus first or set "
                    "x-milvus-db / X_MILVUS_DB_ENV_VAR to 'default'."
                ) from e
            raise
        except Exception as e:
            logger.warning("Milvus Database Selection failed: %s", e)
            logger.warning("Will continue without using a database: %s", e)


    def close(self) -> None:
        """Release the underlying client."""
        self._client.close()

    def __enter__(self) -> "MilvusClientWrapper":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        self.close()

    def search(
        self,
        collection_name: str,
        search_type: Literal["text", "vector", "hybrid"],
        query_text: str | None = None,
        vector: list[float] | None = None,
        vector_field: str = "vector",
        text_field: str | None = None,
        limit: int = DEFAULT_LIMIT,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
        metric_type: Literal["COSINE", "L2", "IP"] = "COSINE",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search a collection by text, vector, or hybrid (vector + text filter)."""
        # Default to all fields so collections without a "content" field (e.g. products) work
        out_fields = output_fields if output_fields else ["*"]
        combined_filter = _build_filter(text_field, query_text, filter_expr)
        if search_type == "text":
            return self._search_text(
                collection_name=collection_name,
                query_text=query_text or "",
                text_field=text_field,
                limit=limit,
                output_fields=out_fields,
                filter_expr=combined_filter,
            )
        if search_type == "vector":
            return self._search_vector(
                collection_name=collection_name,
                vector=vector or [],
                vector_field=vector_field,
                limit=limit,
                output_fields=out_fields,
                filter_expr=combined_filter,
                metric_type=metric_type,
            )
        # hybrid: vector search with optional text filter
        return self._search_vector(
            collection_name=collection_name,
            vector=vector or [],
            vector_field=vector_field,
            limit=limit,
            output_fields=out_fields,
            filter_expr=combined_filter,
            metric_type=metric_type,
        )

    def _search_text(
        self,
        collection_name: str,
        query_text: str,
        text_field: str | None,
        limit: int,
        output_fields: list[str],
        filter_expr: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run a query with filter on text field (no vector)."""
        if not filter_expr:
            return [], {"collection": collection_name, "type": "text", "applied_filter": ""}
        self._ensure_collection_loaded(collection_name)
        try:
            res = self._client.query(
                collection_name=collection_name,
                filter=filter_expr,
                limit=limit,
                output_fields=output_fields,
            )
        except Exception as e:
            logger.exception("Milvus query failed: %s", e)
            raise ToolError(f"Milvus text query failed: {e}") from e
        results = list(res) if res else []
        metadata = {"collection": collection_name, "type": "text", "applied_filter": filter_expr}
        return results, metadata

    def _search_vector(
        self,
        collection_name: str,
        vector: list[float],
        vector_field: str,
        limit: int,
        output_fields: list[str],
        filter_expr: str,
        metric_type: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run vector similarity search with optional filter."""
        if not vector:
            return [], {
                "collection": collection_name,
                "type": "vector",
                "applied_filter": filter_expr,
            }
        self._ensure_collection_loaded(collection_name)
        search_params: dict[str, Any] = {"metric_type": metric_type, "params": {}}
        try:
            raw = self._client.search(
                collection_name=collection_name,
                data=[vector],
                anns_field=vector_field,
                search_params=search_params,
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr or "",
            )
        except Exception as e:
            logger.exception("Milvus search failed: %s", e)
            raise ToolError(f"Milvus vector search failed: {e}") from e
        results = _normalize_search_results(raw)
        metadata = {"collection": collection_name, "type": "vector", "applied_filter": filter_expr}
        return results, metadata

    def _ensure_collection_loaded(self, collection_name: str) -> None:
        """Load collection into memory if not already loaded (required before query/search)."""
        try:
            self._client.load_collection(collection_name=collection_name)
        except MilvusException as e:
            msg = (e.message or str(e)).lower()
            if "already loaded" in msg:
                return
            # Code 700: index not found — collection must have an index before it can be loaded
            if getattr(e, "code", None) == 700 or "index not found" in msg:
                raise ToolError(
                    f"Collection '{collection_name}' cannot be loaded: no index found. "
                    "Create an index on the collection (e.g. vector field via Milvus API) first."
                ) from e
            raise

    def query(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Query a collection using a Boolean filter expression (server-side push-down)."""
        self._ensure_collection_loaded(collection_name)
        out_fields = output_fields if output_fields is not None else []
        try:
            res = self._client.query(
                collection_name=collection_name,
                filter=filter_expr,
                limit=limit,
                output_fields=out_fields if out_fields else None,
            )
        except Exception as e:
            logger.exception("Milvus query failed: %s", e)
            raise ToolError(f"Milvus query failed: {e}") from e
        return list(res) if res else []

    def create_collection(
        self,
        collection_name: str,
        auto_id: bool = True,
        dimension: int = 768,
        primary_field_name: str = "id",
        vector_field_name: str = "vector",
        metric_type: Literal["COSINE", "L2", "IP"] = "COSINE",
        field_schema: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Create a new collection via quick setup or custom schema.

        When field_schema is provided it is merged with quick-setup (field_schema wins per field):
          - Primary: field with is_primary=True or name==primary_field_name; "auto_id" for IDs.
          - Vector: FLOAT_VECTOR/BINARY_VECTOR or name==vector_field_name; else dimension.
          - Other fields (e.g. text, category) added as-is.
        Example full override: field_schema=[{"name": "id", "type": "INT64", "is_primary": True},
          {"name": "vector", "type": "FLOAT_VECTOR", "dim": 256}, ...]
        Example extra fields only: field_schema=[{"name": "text", "type": "VARCHAR",
          "max_length": 65535}, ...]
        """
        try:
            if field_schema:
                schema_auto_id, effective = _build_schema_from_field_schema(
                    field_schema,
                    auto_id=auto_id,
                    dimension=dimension,
                    primary_field_name=primary_field_name,
                    vector_field_name=vector_field_name,
                )
                schema = MilvusClient.create_schema(
                    auto_id=schema_auto_id,
                    enable_dynamic_field=kwargs.pop("enable_dynamic_field", True),
                )
                for name, dtype, is_primary, add_kw in effective:
                    schema.add_field(
                        field_name=name,
                        datatype=dtype,
                        is_primary=is_primary,
                        **add_kw,
                    )
                self._client.create_collection(
                    collection_name=collection_name, schema=schema, **kwargs
                )
            else:
                self._client.create_collection(
                    collection_name=collection_name,
                    dimension=dimension,
                    primary_field_name=primary_field_name,
                    vector_field_name=vector_field_name,
                    metric_type=metric_type,
                    auto_id=auto_id,
                    **kwargs,
                )
        except Exception as e:
            logger.exception("Milvus create_collection failed: %s", e)
            raise ToolError(f"Milvus create_collection failed: {e}") from e
        return {"status": "success", "collection": collection_name}

    def insert_data(
        self,
        collection_name: str,
        data: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """Insert data into a collection. data: field names -> lists of values (columnar)."""
        if not data:
            raise ToolError("Validation Error: 'data' dictionary cannot be empty.")
        keys = list(data.keys())
        lengths = [len(data[k]) for k in keys]
        if len(set(lengths)) != 1:
            raise ToolError("Validation Error: all fields in 'data' must have the same length.")
        n = lengths[0]
        rows = [{k: data[k][i] for k in keys} for i in range(n)]
        try:
            res = self._client.insert(collection_name=collection_name, data=rows)
        except Exception as e:
            logger.exception("Milvus insert failed: %s", e)
            raise ToolError(f"Milvus insert failed: {e}") from e
        insert_count = res.get("insert_count", n) if isinstance(res, dict) else n
        return {"inserted_count": insert_count}

    def inspect_collections(
        self,
        collection_name: str | None = None,
    ) -> dict[str, Any]:
        """List all collections or return detailed metadata (schema, stats) for one."""
        try:
            if collection_name:
                desc = self._client.describe_collection(collection_name=collection_name)
                stats = self._client.get_collection_stats(collection_name=collection_name)
                fields = desc.get("fields") or []
                field_names = [f["name"] for f in fields if f.get("name")]
                enable_dynamic = desc.get("enable_dynamic_field", False)
                dynamic_field_keys: list[str] = []
                if enable_dynamic and fields:
                    primary = next((f["name"] for f in fields if f.get("is_primary")), None)
                    if primary:
                        primary_type = next(
                            (f.get("type") for f in fields if f.get("name") == primary), 5
                        )
                        filter_expr = f"{primary} >= 0" if primary_type == 5 else f'{primary} != ""'
                        try:
                            self._ensure_collection_loaded(collection_name)
                            # output_fields="*" returns schema + dynamic $meta keys per row
                            sample = self._client.query(
                                collection_name=collection_name,
                                filter=filter_expr,
                                limit=10,
                                output_fields=["*"],
                            )
                            seen: set[str] = set()
                            schema_names = set(field_names)
                            for row in sample or []:
                                if not isinstance(row, dict):
                                    continue
                                meta = row.get("$meta") or row.get("meta")
                                if isinstance(meta, dict):
                                    for k in meta:
                                        if k and k not in seen:
                                            seen.add(k)
                                # Dynamic keys flattened on row when using output_fields="*"
                                for k in row:
                                    if k not in schema_names and k not in ("$meta", "meta") and k:
                                        seen.add(k)
                            dynamic_field_keys = sorted(seen)
                        except Exception as e:
                            logger.debug(
                                "Milvus inspect: could not sample dynamic field keys for %s: %s",
                                collection_name,
                                e,
                            )
                return {
                    "collection_details": {
                        "schema": desc,
                        "properties": stats,
                        "field_names": field_names,
                        "dynamic_field_keys": dynamic_field_keys,
                    }
                }
            names = self._client.list_collections()
            return {"collections": list(names) if names else []}
        except MilvusException as e:
            if e.code == 800 or "database not found" in (e.message or str(e)).lower():
                raise ToolError(
                    f"Database '{self._db}' not found. Create it in Milvus first or set "
                    f"x-milvus-db / X_MILVUS_DB_ENV_VAR to 'default'."
                ) from e
            logger.exception("Milvus inspect_collections failed: %s", e)
            raise ToolError(f"Milvus inspect_collections failed: {e}") from e
        except Exception as e:
            logger.exception("Milvus inspect_collections failed: %s", e)
            raise ToolError(f"Milvus inspect_collections failed: {e}") from e

"""Document storage — MinIO/S3-compatible object store integration.

Handles listing, downloading, and tracking documents from MinIO.
Used by the ingestion worker to detect new/changed files.

Usage:
    storage = DocumentStorage(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        bucket="documents",
    )
    new_files = storage.list_new_files(manifest)
    local_path = storage.download(object_key, "/tmp/")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StoredFile:
    """A file in the object store."""

    key: str  # e.g. "automotor/manuals/Allianz/AUTOS.pdf"
    etag: str  # MD5 hash from S3
    size: int
    last_modified: str


class DocumentStorage:
    """MinIO/S3-compatible document storage.

    Parameters
    ----------
    endpoint:
        MinIO server endpoint (host:port).
    access_key:
        Access key (username).
    secret_key:
        Secret key (password).
    bucket:
        Bucket name for documents.
    secure:
        Use HTTPS.

    """

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket: str = "documents",
        secure: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._secure = secure
        self._client = None

    def _get_client(self):
        """Lazy-init MinIO client."""
        if self._client is None:
            from minio import Minio  # noqa: PLC0415

            self._client = Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
            )
            # Ensure bucket exists
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
                logger.info("Created bucket '%s'", self._bucket)
        return self._client

    def list_files(self, prefix: str = "", extensions: set[str] | None = None) -> list[StoredFile]:
        """List all files in the bucket, optionally filtered by prefix and extension."""
        exts = extensions or {".pdf", ".docx", ".txt", ".md", ".json"}
        client = self._get_client()

        files: list[StoredFile] = []
        for obj in client.list_objects(self._bucket, prefix=prefix, recursive=True):
            if obj.is_dir:
                continue
            key = obj.object_name or ""
            suffix = Path(key).suffix.lower()
            if suffix in exts:
                files.append(
                    StoredFile(
                        key=key,
                        etag=obj.etag or "",
                        size=obj.size or 0,
                        last_modified=str(obj.last_modified or ""),
                    )
                )

        return files

    def list_new_files(
        self,
        manifest_etags: dict[str, str],
        prefix: str = "",
    ) -> list[StoredFile]:
        """List files that are new or changed since the last check.

        Parameters
        ----------
        manifest_etags:
            Dict of {object_key: etag} from the ingestion manifest.
            Files not in the dict or with a different etag are returned.

        """
        all_files = self.list_files(prefix=prefix)
        new_files = []
        for f in all_files:
            stored_etag = manifest_etags.get(f.key)
            if stored_etag is None or stored_etag != f.etag:
                new_files.append(f)
        return new_files

    def download(self, object_key: str, local_dir: str | Path) -> Path:
        """Download a file from MinIO to a local directory.

        Returns the local file path.
        """
        client = self._get_client()
        local_path = Path(local_dir) / Path(object_key).name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        client.fget_object(self._bucket, object_key, str(local_path))
        logger.info("Downloaded %s → %s", object_key, local_path)
        return local_path

    def upload(self, local_path: str | Path, object_key: str | None = None) -> str:
        """Upload a local file to MinIO.

        Returns the object key.
        """
        client = self._get_client()
        path = Path(local_path)
        key = object_key or path.name

        client.fput_object(self._bucket, key, str(path))
        logger.info("Uploaded %s → %s/%s", path, self._bucket, key)
        return key

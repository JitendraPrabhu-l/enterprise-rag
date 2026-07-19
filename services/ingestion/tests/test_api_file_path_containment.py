"""Tests for `_resolve_staged_file_path` — the containment check that fixes
an unauthenticated arbitrary-local-file-read: `file_path` must resolve
inside `IngestionSettings.upload_dir`, never anywhere else on the
filesystem. Without this, any network caller reaching `/ingest` (no service
in this stack requires per-request auth) could pass `file_path=/etc/passwd`
or any other container-readable path and have it parsed and indexed as a
searchable document.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException

from rag_ingestion.api import _resolve_staged_file_path
from rag_ingestion.config import IngestionSettings


@pytest.fixture
def staging_dir(tmp_path: Path) -> Path:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir


@pytest.fixture
def settings(staging_dir: Path) -> IngestionSettings:
    return IngestionSettings(upload_dir=str(staging_dir))


class TestResolveStagedFilePathAcceptsLegitimateFiles:
    def test_a_file_directly_inside_upload_dir_is_accepted(
        self, staging_dir: Path, settings: IngestionSettings
    ) -> None:
        staged_file = staging_dir / "document.pdf"
        staged_file.write_bytes(b"fake pdf content")

        resolved = _resolve_staged_file_path(str(staged_file), settings)

        assert resolved == staged_file.resolve()

    def test_a_file_in_a_subdirectory_of_upload_dir_is_accepted(
        self, staging_dir: Path, settings: IngestionSettings
    ) -> None:
        subdir = staging_dir / "batch-2026-07-15"
        subdir.mkdir()
        staged_file = subdir / "document.pdf"
        staged_file.write_bytes(b"fake pdf content")

        resolved = _resolve_staged_file_path(str(staged_file), settings)

        assert resolved == staged_file.resolve()


class TestResolveStagedFilePathRejectsEscapes:
    def test_a_path_outside_upload_dir_entirely_is_rejected(
        self, settings: IngestionSettings
    ) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as outside_file:
            outside_file.write(b"not staged, should never be readable")
            outside_path = outside_file.name

        try:
            with pytest.raises(HTTPException) as exc_info:
                _resolve_staged_file_path(outside_path, settings)
            assert exc_info.value.status_code == 403
        finally:
            Path(outside_path).unlink()

    def test_dot_dot_traversal_out_of_upload_dir_is_rejected(
        self, staging_dir: Path, settings: IngestionSettings
    ) -> None:
        """The core attack this fix closes: `upload_dir/../../etc/passwd`
        resolves OUTSIDE upload_dir even though the raw string starts with
        the staging path — `Path.resolve()` must normalize this before the
        containment check runs, which is exactly what makes the check
        actually safe rather than a string-prefix check that traversal
        segments would defeat."""
        escaping_path = str(staging_dir / ".." / ".." / "etc" / "passwd")

        with pytest.raises(HTTPException) as exc_info:
            _resolve_staged_file_path(escaping_path, settings)
        assert exc_info.value.status_code == 403

    def test_absolute_path_to_a_sensitive_file_is_rejected(
        self, settings: IngestionSettings
    ) -> None:
        """The literal originally-reported attack: an absolute path to a
        real system file must never be accepted regardless of whether it
        happens to exist."""
        sensitive_path = "/etc/passwd" if Path("/etc/passwd").exists() else str(Path.home())

        with pytest.raises(HTTPException) as exc_info:
            _resolve_staged_file_path(sensitive_path, settings)
        assert exc_info.value.status_code == 403

    def test_nonexistent_path_inside_upload_dir_is_404_not_403(
        self, staging_dir: Path, settings: IngestionSettings
    ) -> None:
        """A path that's properly contained but simply doesn't exist yet
        (e.g. a typo, or a batch job that hasn't finished staging) gets the
        ORIGINAL not-found error, not the containment error — the two
        failure modes are distinct and must not be conflated."""
        missing_path = staging_dir / "never-staged.pdf"

        with pytest.raises(HTTPException) as exc_info:
            _resolve_staged_file_path(str(missing_path), settings)
        assert exc_info.value.status_code == 404

    def test_upload_dir_itself_is_not_treated_as_a_valid_file(
        self, staging_dir: Path, settings: IngestionSettings
    ) -> None:
        """upload_dir passed as file_path directly is contained (trivially,
        it IS the root) but is a directory, not a file — exists() is True
        for directories too, so this documents that the containment check
        alone doesn't validate file-vs-directory; downstream parsing will
        fail on a directory path, which is acceptable since it's not the
        security property this check exists for."""
        resolved = _resolve_staged_file_path(str(staging_dir), settings)
        assert resolved == staging_dir.resolve()

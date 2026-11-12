#!/usr/bin/env python3
"""Verify a public source ZIP manifest and safe paths; do not execute its contents."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


MANIFEST_NAME = "MANIFEST.json"
CHUNK_BYTES = 1024 * 1024
MAX_MEMBERS = 10000
MAX_MEMBER_BYTES = 16 * 1024 ** 2
MAX_TOTAL_BYTES = 64 * 1024 ** 2
MAX_MANIFEST_BYTES = 16 * 1024 ** 2


class VerificationError(ValueError):
    """Archive safety, integrity or relocated acceptance failed."""


def _checked_path(path):
    absolute = Path(os.path.abspath(str(path)))
    for component in reversed((absolute,) + tuple(absolute.parents)):
        if component.is_symlink():
            raise VerificationError("Symlink paths are not accepted: %s" % component)
    return absolute


def _safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise VerificationError("Unsafe ZIP member name: %r" % (name,))
    parts = name.split("/")
    if (PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in parts)
            or any(":" in part or part.endswith((".", " ")) for part in parts)
            or any(ord(character) < 32 for character in name)):
        raise VerificationError("Unsafe ZIP member name: %r" % name)
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update("COM%d" % index for index in range(1, 10))
    reserved.update("LPT%d" % index for index in range(1, 10))
    if any(part.split(".", 1)[0].upper() in reserved for part in parts):
        raise VerificationError("Non-portable ZIP member name: %r" % name)
    return name


def _check_member(info):
    _safe_name(info.filename)
    mode = (info.external_attr >> 16) & 0xFFFF
    kind = stat.S_IFMT(mode)
    if info.is_dir() or kind not in (0, stat.S_IFREG):
        raise VerificationError("Only regular ZIP files are allowed; no links/directories/devices: %s" % info.filename)
    if info.flag_bits & 1:
        raise VerificationError("Encrypted ZIP members are not accepted")
    if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
        raise VerificationError("ZIP member exceeds the size limit: %s" % info.filename)


def _hash_stream(stream, destination=None):
    digest, size = hashlib.sha256(), 0
    while True:
        chunk = stream.read(CHUNK_BYTES)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    return size, digest.hexdigest()


def _file_hash(path):
    _checked_path(path)
    with Path(path).open("rb") as stream:
        return _hash_stream(stream)[1]


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError("Duplicate key in manifest JSON: %s" % key)
        result[key] = value
    return result


def verify_archive(archive_path):
    """Validate all members before extraction; return verified manifest data."""
    archive_path = _checked_path(archive_path)
    if not archive_path.is_file() or not stat.S_ISREG(archive_path.stat().st_mode):
        raise VerificationError("ZIP input must be an existing regular file")
    archive_hash = _file_hash(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_MEMBERS:
            raise VerificationError("ZIP member count is empty or above the limit")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or len(names) != len(set(name.casefold() for name in names)):
            raise VerificationError("Duplicate or case-colliding ZIP member names")
        for info in infos:
            _check_member(info)
        if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
            raise VerificationError("ZIP uncompressed size exceeds the total limit")
        if MANIFEST_NAME not in names:
            raise VerificationError("MANIFEST.json is required")
        manifest_info = archive.getinfo(MANIFEST_NAME)
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise VerificationError("Manifest exceeds the size limit")
        raw_manifest = archive.read(MANIFEST_NAME)
        try:
            manifest = json.loads(raw_manifest.decode("utf-8"), object_pairs_hook=_unique_json_object,
                                  parse_constant=lambda value: (_ for _ in ()).throw(VerificationError("Non-finite JSON number")))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise VerificationError("Manifest is not valid UTF-8 JSON") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise VerificationError("Unsupported manifest schema")
        entries, metadata = manifest.get("files"), manifest.get("release")
        if not isinstance(entries, list) or not isinstance(metadata, dict):
            raise VerificationError("Manifest files/release fields have invalid types")
        expected = {}
        for item in entries:
            if not isinstance(item, dict):
                raise VerificationError("Manifest file entries must be objects")
            name = _safe_name(item.get("path"))
            if name == MANIFEST_NAME or name in expected:
                raise VerificationError("Manifest contains itself or a duplicate file")
            size, digest = item.get("size_bytes"), item.get("sha256")
            if type(size) is not int or not 0 <= size <= MAX_MEMBER_BYTES:
                raise VerificationError("Invalid manifest file size: %s" % name)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise VerificationError("Invalid manifest SHA256: %s" % name)
            expected[name] = item
        if set(names) != set(expected) | {MANIFEST_NAME}:
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected) - {MANIFEST_NAME})
            raise VerificationError("ZIP/manifest members differ; missing=%r extra=%r" % (missing, extra))
        if metadata.get("payload_file_count") != len(expected):
            raise VerificationError("Manifest payload_file_count is inconsistent")
        payload_bytes = sum(item["size_bytes"] for item in expected.values())
        if metadata.get("payload_bytes") != payload_bytes:
            raise VerificationError("Manifest payload_bytes is inconsistent")
        if [item["path"] for item in entries] != sorted(expected):
            raise VerificationError("Manifest entries are not deterministically sorted")
        if names != sorted(names):
            raise VerificationError("ZIP members are not deterministically sorted")
        for name, item in sorted(expected.items()):
            info = archive.getinfo(name)
            if info.file_size != item["size_bytes"]:
                raise VerificationError("ZIP/manifest size mismatch: %s" % name)
            with archive.open(info) as stream:
                size, digest = _hash_stream(stream)
            if size != item["size_bytes"] or digest != item["sha256"]:
                raise VerificationError("File digest/size mismatch: %s" % name)
    if _file_hash(archive_path) != archive_hash:
        raise VerificationError("ZIP changed during integrity verification")
    return {"archive": str(archive_path), "archive_sha256": archive_hash,
            "archive_bytes": archive_path.stat().st_size, "manifest": manifest,
            "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "manifest_size_bytes": len(raw_manifest), "payload_file_count": len(expected),
            "payload_bytes": payload_bytes, "member_names": sorted(names),
            "path_and_link_checks_passed": True, "all_file_hashes_verified": True}


def extract_verified(archive_path, project_dir, verified):
    """Safely copy previously verified regular members, rechecking each digest."""
    project_dir = _checked_path(project_dir)
    if project_dir.exists():
        raise FileExistsError("Extraction directory must be new: %s" % project_dir)
    project_dir.mkdir(parents=True)
    expected = {item["path"]: item for item in verified["manifest"]["files"]}
    expected[MANIFEST_NAME] = {"size_bytes": verified["manifest_size_bytes"], "sha256": verified["manifest_sha256"]}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if names != verified["member_names"]:
            raise VerificationError("ZIP member list changed before extraction")
        for name in names:
            info = archive.getinfo(name)
            _check_member(info)
            destination = project_dir.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _checked_path(destination)
            with archive.open(info) as source, destination.open("xb") as target:
                size, digest = _hash_stream(source, target)
            if size != expected[name]["size_bytes"] or digest != expected[name]["sha256"]:
                raise VerificationError("Member changed during extraction: %s" % name)
            permissions = (info.external_attr >> 16) & 0o111
            destination.chmod(0o755 if permissions else 0o644)
    if _file_hash(archive_path) != verified["archive_sha256"]:
        raise VerificationError("ZIP changed between verification and extraction")
    return project_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    try:
        verified = verify_archive(args.archive)
    except (VerificationError, OSError, ValueError, zipfile.BadZipFile):
        print("Verification failed: invalid archive, unsafe path or content mismatch", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "payload_files": len(verified["manifest"]["files"]), "archive_sha256": verified["archive_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

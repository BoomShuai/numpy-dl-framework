#!/usr/bin/env python3
"""Build a reproducible public source ZIP. Models and private artifacts are excluded."""

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

if __package__:
    from .privacy_check import inspect as inspect_public_file
else:
    from privacy_check import inspect as inspect_public_file


VERSION = "1.0.0"
CHUNK_BYTES = 1024 * 1024
MANIFEST_NAME = "MANIFEST.json"
EXACT_FILES = frozenset()
ROOT_FILES = frozenset(['.github/ISSUE_TEMPLATE/bug.md', '.github/pull_request_template.md', '.github/workflows/ci.yml', '.gitignore', 'CHANGELOG.md', 'CONTRIBUTING.md', 'PRIVACY.md', 'README.md', 'SECURITY.md', 'pyproject.toml', 'requirements-core.txt'])
ARTIFACT_FILES = frozenset()
SOURCE_TREES = {'numpy_dl': frozenset({".py"}), "scripts": frozenset({".py"}), "tests": frozenset({".py"}), "docs": frozenset({".md"})}
EXCLUDED_COMPONENTS = frozenset({"__pycache__", ".git", ".venv", "venv", "node_modules", ".cache", "dist", "runs", "outputs", "raw", "checkpoints", "models", "private", "artifacts"})
EXCLUDED_SUFFIXES = frozenset({".pyc", ".log", ".tmp", ".bak"})
REQUIRED_FILES = frozenset(['PRIVACY.md', 'README.md', 'docs/VALIDATION.md', 'numpy_dl/__init__.py', 'numpy_dl/__main__.py', 'numpy_dl/cli.py', 'pyproject.toml', 'scripts/build_release.py', 'scripts/privacy_check.py', 'scripts/verify_release.py'])


class ReleaseError(ValueError):
    """A release cannot safely or completely be built from this source."""


def safe_archive_name(name):
    """Reject traversal and platform-ambiguous names before writing a ZIP."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise ReleaseError("Unsafe archive member name: %r" % (name,))
    parts = name.split("/")
    if (PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in parts)
            or any(":" in part or part.endswith((".", " ")) for part in parts)
            or any(ord(character) < 32 for character in name)):
        raise ReleaseError("Unsafe archive member name: %r" % name)
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update("COM%d" % index for index in range(1, 10))
    reserved.update("LPT%d" % index for index in range(1, 10))
    if any(part.split(".", 1)[0].upper() in reserved for part in parts):
        raise ReleaseError("Non-portable archive member name: %r" % name)
    return name


def _assert_no_symlinks(path):
    """Check all existing path components, without resolving symlinks first."""
    path = Path(os.path.abspath(str(path)))
    for component in reversed((path,) + tuple(path.parents)):
        if component.is_symlink():
            raise ReleaseError("Symlinks are not allowed in release paths: %s" % component)
    return path


def _is_regular(path):
    _assert_no_symlinks(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ReleaseError("Only regular files may be packaged: %s" % path)
    return info


def _walk_files(directory):
    """Walk one explicitly selected tree, refusing every encountered symlink."""
    _assert_no_symlinks(directory)
    if not directory.is_dir():
        raise ReleaseError("Approved source tree is not a directory: %s" % directory)
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        # A link is never followed, including links with an otherwise excluded
        # suffix. Excluded root-level runtimes are not traversed at all.
        if child.is_symlink():
            raise ReleaseError("Symlink found inside approved source tree: %s" % child)
        if child.name.startswith(".") or child.name in EXCLUDED_COMPONENTS:
            continue
        info = child.lstat()
        if stat.S_ISDIR(info.st_mode):
            yield from _walk_files(child)
        elif stat.S_ISREG(info.st_mode):
            if child.suffix.lower() not in EXCLUDED_SUFFIXES:
                yield child
        else:
            raise ReleaseError("Unsupported filesystem entry in source tree: %s" % child)


def collect_release_files(source_root):
    """Return sorted (ZIP-relative-name, local-path) pairs from the allowlist."""
    root = _assert_no_symlinks(source_root)
    if not root.is_dir():
        raise ReleaseError("Source root is not an existing directory: %s" % root)
    selected = {}

    def add(path):
        _is_regular(path)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ReleaseError("File escapes the approved source root") from error
        name = safe_archive_name(relative)
        if inspect_public_file(name, path.read_bytes()):
            raise ReleaseError("Public source privacy check failed for: " + name)
        selected[name] = path

    for name in sorted(ROOT_FILES):
        path = root / name
        if path.exists() or path.is_symlink():
            add(path)
    for name, suffixes in sorted(SOURCE_TREES.items()):
        directory = root / name
        if directory.exists() or directory.is_symlink():
            for path in _walk_files(directory):
                if suffixes is None or path.suffix.lower() in suffixes:
                    add(path)
    for name in sorted(EXACT_FILES):
        path = root / name
        if path.exists() or path.is_symlink():
            add(path)
    for name in sorted(ARTIFACT_FILES):
        path = root / "artifacts" / name
        if path.exists() or path.is_symlink():
            add(path)
    missing = sorted(REQUIRED_FILES - set(selected))
    if missing:
        raise ReleaseError("Required release files are missing: " + ", ".join(missing))
    if any((root / name).stat().st_size == 0 for name in REQUIRED_FILES):
        raise ReleaseError("Required release files must not be empty")
    names = sorted(selected)
    folded = [name.casefold() for name in names]
    if len(folded) != len(set(folded)):
        raise ReleaseError("Case-insensitive duplicate member names are unsafe to extract")
    return [(name, selected[name]) for name in names]


def _snapshot(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _read_chunks(path, consume):
    """Read without following a final symlink, and detect concurrent changes."""
    before = _is_regular(path)
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if _snapshot(before) != _snapshot(opened):
            raise ReleaseError("Source changed while opening: %s" % path)
        digest, size = hashlib.sha256(), 0
        while True:
            chunk = stream.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if consume is not None:
                consume(chunk)
        after = os.fstat(stream.fileno())
    current = _is_regular(path)
    if _snapshot(before) != _snapshot(after) or _snapshot(after) != _snapshot(current):
        raise ReleaseError("Source changed during release construction: %s" % path)
    return size, digest.hexdigest()


def _output_path(source_root, output):
    path = _assert_no_symlinks(output)
    if path.suffix.lower() != ".zip":
        raise ReleaseError("Release output must have a .zip extension")
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to overwrite existing output: %s" % path)
    # Refuse an output anywhere inside selected source/model/data/doc trees,
    # even if its extension would otherwise be ignored. Output under dist or
    # outside the project is safe and is never selected recursively.
    for top in set(SOURCE_TREES) | {"numpy_dl", "data", "artifacts", "models", "private"}:
        boundary = source_root / top
        if path == boundary or boundary in path.parents:
            raise ReleaseError("Output cannot be inside an included source tree: %s" % path)
    return path


def _release_epoch(source_date_epoch):
    if source_date_epoch is None:
        source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH", str(int(time.time())))
    try:
        epoch = int(source_date_epoch)
    except (TypeError, ValueError) as error:
        raise ReleaseError("SOURCE_DATE_EPOCH must be an integer timestamp") from error
    # These are the inclusive timestamp bounds representable by standard ZIP.
    if not 315532800 <= epoch <= 4354819199:
        raise ReleaseError("SOURCE_DATE_EPOCH must be between 1980-01-01 and 2107-12-31 UTC")
    return epoch


def _zip_info(name, timestamp):
    item = zipfile.ZipInfo(safe_archive_name(name), date_time=timestamp)
    item.create_system = 3
    executable = name.endswith((".sh", ".command"))
    item.external_attr = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
    item.compress_type = zipfile.ZIP_DEFLATED
    return item


def build_release(source_root, output, version=VERSION, source_date_epoch=None):
    """Build one ZIP exclusively; remove only our own partial output on error."""
    root = _assert_no_symlinks(source_root)
    target = _output_path(root, output)
    epoch = _release_epoch(source_date_epoch)
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        raise ReleaseError("Release version must contain only letters, numbers, dots, underscores or hyphens")
    files = collect_release_files(root)
    entries = []
    paths = dict(files)
    for name, path in files:
        size, digest = _read_chunks(path, None)
        entries.append({"path": name, "size_bytes": size, "sha256": digest})
    manifest = {
        "schema_version": 1,
        "release": {
            "name": "numpy-dl-framework", "version": version,
            "created_utc": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
            "source_date_epoch": epoch,
            "scope": "Public source-only teaching and research project; excludes private models, raw data and historical results.",
            "model_included": False,
            "payload_file_count": len(entries),
            "payload_bytes": sum(entry["size_bytes"] for entry in entries),
            "manifest_policy": "All payload files are listed. MANIFEST.json excludes itself to avoid recursive hashing.",
        },
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected = {entry["path"]: entry for entry in entries}
    timestamp = tuple(datetime.fromtimestamp(epoch, timezone.utc).timetuple()[:6])
    target.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlinks(target.parent)
    # Exclusive create is authoritative even if a second process creates this
    # file after the earlier existence check. Never use zipfile's overwrite mode
    # directly on a user-selected path.
    descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    created = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as archive:
                for name in sorted(list(paths) + [MANIFEST_NAME]):
                    info = _zip_info(name, timestamp)
                    if name == MANIFEST_NAME:
                        archive.writestr(info, manifest_bytes)
                    else:
                        with archive.open(info, "w", force_zip64=True) as member:
                            size, digest = _read_chunks(paths[name], member.write)
                        if size != expected[name]["size_bytes"] or digest != expected[name]["sha256"]:
                            raise ReleaseError("Source changed between hashing and packaging: %s" % name)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Only discard the partial file we just created, never a replacement.
        try:
            current = target.lstat()
            if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                target.unlink()
        except FileNotFoundError:
            pass
        raise
    _, archive_hash = _read_chunks(target, None)
    return {
        "output": str(target), "version": version,
        "payload_files": len(entries), "payload_bytes": manifest["release"]["payload_bytes"],
        "zip_bytes": target.stat().st_size, "sha256": archive_hash,
        "manifest": MANIFEST_NAME,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="New .zip path; existing files are never overwritten")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--source-date-epoch", type=int, help="Optional fixed UTC timestamp for reproducible ZIP bytes")
    arguments = parser.parse_args(argv)
    root = Path(__file__).absolute().parent.parent
    output = arguments.output or root / "dist" / ("numpy-dl-framework-v%s.zip" % arguments.version)
    try:
        result = build_release(root, output, arguments.version, arguments.source_date_epoch)
    except (ReleaseError, FileExistsError, OSError, zipfile.BadZipFile) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

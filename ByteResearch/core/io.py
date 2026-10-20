# File input/output helpers shared by every ByteResearch stage.
# All writers are atomic: they write a sibling temporary file and then os.replace it, so a killed
# process never leaves a truncated result behind.
# Readers treat a missing or unreadable file as absent and return a default instead of raising,
# which lets an interrupted stage simply redo the affected step.
# Gzipped JSONL is written with mtime=0 so identical content gives identical bytes and hashes.
# The module also provides hashing, read-only marking, a cross-process file lock, a UTF-8 console fix
# for Windows, tee_output, which copies console output into a per-unit log file, and publish_log, which
# appends a finished local log to its final location in one write.

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
from filelock import FileLock


def _tmp_path(path: Path) -> Path:
    # Input: a destination path. Output: a process-unique sibling temporary path.
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_bytes_atomic(path, data: bytes) -> Path:
    # Inputs: destination path and raw bytes. Output: the path, after an atomic tmp + replace write
    # (parent directories are created).
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _json_default(obj):
    # Input: an object json cannot serialise. Output: a JSON-compatible equivalent for numpy
    # scalars/arrays, paths and sets; anything else raises TypeError.
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path, set, frozenset)):
        return str(obj) if isinstance(obj, Path) else sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def dumps_json(obj, indent=1) -> str:
    # Inputs: a JSON-like object and indent. Output: its UTF-8-friendly JSON text (numpy-aware).
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=_json_default)


def write_json_atomic(path, obj, indent=1) -> Path:
    # Inputs: destination path, JSON-like object and indent. Output: the path, written atomically.
    return write_bytes_atomic(path, dumps_json(obj, indent).encode("utf-8"))


def read_json(path, default=None):
    # Inputs: a path and a default. Output: the parsed JSON, or the default when the file is
    # missing or unreadable (e.g. truncated by a crash).
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_jsonl_gz_atomic(path, rows) -> Path:
    # Inputs: destination path and an iterable of JSON-like rows. Output: the path, written as
    # gzipped JSON lines with mtime=0 (byte-reproducible) and atomically replaced.
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False, default=_json_default) + "\n").encode("utf-8"))
    return write_bytes_atomic(path, buf.getvalue())


def read_jsonl_gz(path) -> list:
    # Input: a gzipped JSONL path. Output: the list of parsed rows (blank lines skipped).
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_npy_atomic(path, array) -> Path:
    # Inputs: destination .npy path and an array. Output: the path, written atomically.
    buf = io.BytesIO()
    np.save(buf, np.asarray(array), allow_pickle=False)
    return write_bytes_atomic(path, buf.getvalue())


def write_npz_atomic(path, **arrays) -> Path:
    # Inputs: destination .npz path and named arrays. Output: the path, written atomically
    # (uncompressed, so loads can memory-map individual members cheaply).
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    return write_bytes_atomic(path, buf.getvalue())


def sha256_file(path, chunk: int = 1 << 20) -> str:
    # Inputs: a file path and read chunk size. Output: the hex SHA-256 of its contents.
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    # Input: a JSON-like object. Output: the hex SHA-256 of its canonical sorted-key JSON.
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def make_readonly(path) -> Path:
    # Input: a file path. Output: the path, with write permission bits removed (immutable caches).
    path = Path(path)
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return path


@contextlib.contextmanager
def locked(path, timeout: float = 600.0):
    # Inputs: the path to protect and a timeout in seconds. Yields while holding an exclusive
    # cross-process lock on "<path>.lock"; the lock is released on exit, even after an exception.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock", timeout=timeout)
    with lock:
        yield path


def utf8_stdout() -> None:
    # Reconfigures stdout/stderr to UTF-8 with replacement, so Windows consoles can print
    # non-Latin text and table glyphs. No inputs or outputs.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")


class Tee:
    # A text stream that forwards every write to the original console stream and to an open log file,
    # flushing the log at once so a killed process keeps its output; other attributes come from the
    # console stream.

    def __init__(self, stream, log):
        # Inputs: the original stream and the open log file handle.
        self.stream = stream
        self.log = log

    def write(self, text: str) -> int:
        # Input: text. Writes it to both targets. Output: the number of characters written.
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self) -> None:
        # Flushes both targets.
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        # Input: an attribute name. Output: that attribute of the console stream (encoding, isatty, ...).
        return getattr(self.stream, name)


@contextlib.contextmanager
def tee_output(path, header: str = ""):
    # Inputs: a log file path and an optional header line. Yields while sys.stdout and sys.stderr are
    # also appended to that file (parents created, header written first); the original streams are
    # restored and the file closed on exit, also after an exception.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = sys.stdout, sys.stderr
    with open(path, "a", encoding="utf-8", errors="replace") as log:
        if header:
            print(header, file=log)
        sys.stdout, sys.stderr = Tee(saved[0], log), Tee(saved[1], log)
        try:
            yield path
        finally:
            sys.stdout, sys.stderr = saved


def publish_log(local, final) -> None:
    # Inputs: the log file written while a unit ran and its final path. Appends the local log to the final
    # one in a single write and deletes the local file; does nothing when both are the same path or the
    # local log does not exist.
    local, final = Path(local), Path(final)
    if local == final or not local.exists():
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    with open(final, "a", encoding="utf-8", errors="replace") as out:
        out.write(local.read_text(encoding="utf-8", errors="replace"))
    local.unlink()

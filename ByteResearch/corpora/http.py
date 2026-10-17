# Network access for the data stages: plain downloads, Hugging Face downloads and gzip line streams.
# fetch_bytes downloads a small file with retries and returns its bytes plus provenance (final URL
# after redirects, selected headers and SHA-256).
# hf_fetch resolves a Hugging Face repository to an exact commit and downloads chosen files into a
# local directory, so manifests can record the revision that was actually used.
# GzLineStream streams a remote (possibly multi-member) gzip file line by line, follows redirects,
# resumes after network errors with an HTTP Range request and hashes the compressed bytes it consumed.
# Streams can be abandoned early, which is how the score-sorted NLLB files are read only partially.

from __future__ import annotations

import hashlib
import http.client
import time
import zlib

import requests
import urllib3

KEEP_HEADERS = ("content-length", "content-type", "content-encoding", "content-range", "etag", "last-modified",
                "accept-ranges", "x-goog-generation", "x-goog-hash", "x-goog-stored-content-length")
NETWORK_ERRORS = (requests.RequestException, urllib3.exceptions.HTTPError, http.client.HTTPException,
                  ConnectionError, TimeoutError)
USER_AGENT = "ByteResearch-data/1.0"


def header_subset(headers) -> dict:
    # Input: a response header mapping. Output: {lower-case name: value} for the provenance headers.
    lower = {k.lower(): v for k, v in headers.items()}
    return {k: lower[k] for k in KEEP_HEADERS if k in lower}


def retryable(exc) -> bool:
    # Input: a network exception. Output: False for definitive HTTP client errors (4xx other than 408
    # and 429), True for everything else (connection drops, timeouts, server errors).
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return not (status and 400 <= status < 500 and status not in (408, 429))


def fetch_bytes(url: str, retries: int = 5, timeout: float = 60.0, backoff: float = 2.0) -> tuple:
    # Inputs: URL, retry count, timeout and backoff base in seconds. Output: (bytes, info) where info
    # holds url, final_url, status, headers, sha256 and byte count; raises after the last retry.
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            data = resp.content
            return data, {"url": url, "final_url": resp.url, "status": resp.status_code,
                          "headers": header_subset(resp.headers), "sha256": hashlib.sha256(data).hexdigest(),
                          "bytes": len(data)}
        except NETWORK_ERRORS as exc:
            if attempt == retries - 1 or not retryable(exc):
                raise
            time.sleep(backoff ** attempt)
    raise RuntimeError("unreachable")


def hub_errors() -> tuple:
    # Output: the transient error types raised by huggingface_hub downloads (httpx transport errors and
    # OS-level connection errors).
    import httpx
    return (httpx.TransportError, OSError)


def retry(fn, errors: tuple, retries: int = 5, backoff: float = 2.0, label: str = ""):
    # Inputs: a zero-argument function, the transient error types, retry count, backoff base and a label
    # for log lines. Output: fn()'s result, retrying with exponential backoff; the last error is raised.
    for attempt in range(retries):
        try:
            return fn()
        except errors as exc:
            if attempt == retries - 1:
                raise
            print(f"[data] retrying {label} after {exc!r}", flush=True)
            time.sleep(backoff ** attempt)
    raise RuntimeError("unreachable")


def hf_fetch(repo_id: str, local_dir, files=None, patterns=None, repo_type: str = "model", revision=None) -> dict:
    # Inputs: repository id, destination directory, explicit file names or allow-patterns, repo type and an
    # optional pinned revision (resolved to the current commit when None). Downloads the files into
    # local_dir, retrying on transient errors (finished files are reused). Output: {repo, repo_type,
    # requested_revision, revision, files: [names], urls}.
    once = lambda: _hf_fetch_once(repo_id, local_dir, files, patterns, repo_type, revision)
    return retry(once, hub_errors(), label=repo_id)


def _hf_fetch_once(repo_id: str, local_dir, files, patterns, repo_type: str, revision) -> dict:
    # Inputs: as hf_fetch without the retry settings. Output: the hf_fetch result for one attempt.
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, snapshot_download
    api = HfApi()
    sha = api.repo_info(repo_id, repo_type=repo_type, revision=revision).sha
    if files:
        names = list(files)
        for name in names:
            hf_hub_download(repo_id, name, repo_type=repo_type, revision=sha, local_dir=str(local_dir))
    else:
        snapshot_download(repo_id, repo_type=repo_type, revision=sha, local_dir=str(local_dir),
                          allow_patterns=list(patterns or ["*"]))
        siblings = api.repo_info(repo_id, repo_type=repo_type, revision=sha).siblings
        names = sorted(s.rfilename for s in siblings if _matches(s.rfilename, patterns))
    return {"repo": repo_id, "repo_type": repo_type, "requested_revision": revision, "revision": sha,
            "files": names, "urls": {n: hf_hub_url(repo_id, n, repo_type=repo_type, revision=sha) for n in names}}


def _matches(name: str, patterns) -> bool:
    # Inputs: a repository file name and fnmatch patterns (None = everything). Output: whether it matches.
    import fnmatch
    return patterns is None or any(fnmatch.fnmatch(name, p) for p in patterns)


def inflate(chunks, counter: dict):
    # Inputs: an iterable of compressed byte chunks and a dict whose "members" entry counts the gzip
    # members seen. Yields decompressed bytes, starting a fresh decompressor whenever a member ends and
    # more data follows (multi-member gzip); raises ValueError when the input ends inside a member.
    dec = zlib.decompressobj(31)
    fed = False
    counter["members"] = 1
    for chunk in chunks:
        data = chunk
        while data:
            if dec.eof:
                dec = zlib.decompressobj(31)
                counter["members"] += 1
            yield dec.decompress(data)
            fed = True
            data = dec.unused_data if dec.eof else b""
    if fed and not dec.eof:
        raise ValueError("gzip stream ended inside a member (truncated file)")


class GzLineStream:
    # Iterates the lines (str, without the newline) of a remote gzip file. Construct with the URL and
    # optional chunk size, retry count, timeout, backoff and requests session; iterate once; call
    # stats() for provenance (final URL, headers, bytes and SHA-256 consumed, lines, members, resumes).

    def __init__(self, url: str, chunk: int = 1 << 16, retries: int = 8, timeout: float = 60.0,
                 backoff: float = 2.0, session=None):
        # Inputs: URL, read chunk size, retries per failure streak, timeout, backoff base and session.
        self.url = url
        self.chunk = chunk
        self.retries = retries
        self.timeout = timeout
        self.backoff = backoff
        self.session = session or requests.Session()
        self.final_url = url
        self.headers = {}
        self.status = None
        self.total = None
        self.consumed = 0
        self.sha = hashlib.sha256()
        self.lines = 0
        self.resumes = 0
        self.exhausted = False
        self.counter = {"members": 0}
        self._resp = None

    def _open(self, offset: int):
        # Input: the byte offset to start from. Output: an open streaming response positioned at offset
        # (a server that ignores Range is fast-forwarded); validates that a resumed file did not change.
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        resp = self.session.get(self.final_url, stream=True, timeout=self.timeout, headers=headers)
        resp.raise_for_status()
        if resp.headers.get("Content-Encoding", "identity") != "identity":
            raise IOError(f"{self.url} was served with Content-Encoding {resp.headers['Content-Encoding']}")
        if not offset:
            self.final_url, self.status, self.headers = resp.url, resp.status_code, header_subset(resp.headers)
            self.total = int(resp.headers["Content-Length"]) if "Content-Length" in resp.headers else None
            return resp
        self._check_same_file(resp)
        if resp.status_code != 206:
            self._skip(resp, offset)
        return resp

    def _check_same_file(self, resp) -> None:
        # Input: a resumed response. Raises if its ETag or Last-Modified differs from the first response.
        new = header_subset(resp.headers)
        for key in ("etag", "last-modified"):
            if key in self.headers and key in new and self.headers[key] != new[key]:
                raise IOError(f"{self.url} changed during streaming ({key} {self.headers[key]} -> {new[key]})")

    def _skip(self, resp, offset: int) -> None:
        # Inputs: a full (200) response and the number of bytes already consumed. Discards that prefix.
        left = offset
        while left:
            got = resp.raw.read(min(left, self.chunk), decode_content=False)
            if not got:
                raise IOError("server ignored Range and the body ended early")
            left -= len(got)

    def _read(self, resp) -> bytes:
        # Input: an open response. Output: the next raw compressed chunk (b"" at a genuine end); a short
        # body (fewer bytes than Content-Length) raises a connection error so the caller resumes.
        data = resp.raw.read(self.chunk, decode_content=False)
        if not data and self.total is not None and self.consumed < self.total:
            raise ConnectionError(f"connection closed at {self.consumed}/{self.total} bytes")
        return data

    def _chunks(self):
        # Yields raw compressed chunks from the current offset, reopening with Range after network
        # errors (up to `retries` consecutive failures with exponential backoff); updates the running
        # SHA-256 and byte count of consumed data.
        failures = 0
        while True:
            try:
                self._resp = self._resp or self._open(self.consumed)
                data = self._read(self._resp)
            except NETWORK_ERRORS as exc:
                failures = self._recover(failures, exc)
                continue
            if not data:
                return
            failures = 0
            self.consumed += len(data)
            self.sha.update(data)
            yield data

    def _recover(self, failures: int, exc) -> int:
        # Inputs: consecutive failure count and the exception. Closes the response, waits and returns the
        # incremented count; re-raises once the retry budget is spent.
        self._close_resp()
        if failures >= self.retries or not retryable(exc):
            raise exc
        self.resumes += 1
        time.sleep(min(self.backoff ** failures - 1, 60))
        return failures + 1

    def _close_resp(self) -> None:
        # Closes the current HTTP response, if any.
        if self._resp is not None:
            self._resp.close()
        self._resp = None

    def __iter__(self):
        # Yields decoded lines (UTF-8 with replacement, trailing "\r" removed); marks the stream exhausted
        # when the file ends and always closes the connection, also when the consumer stops early.
        tail = b""
        try:
            for block in inflate(self._chunks(), self.counter):
                parts = (tail + block).split(b"\n")
                tail = parts.pop()
                yield from self._emit(parts)
            yield from self._emit([tail] if tail else [])
            self.exhausted = True
        finally:
            self._close_resp()

    def _emit(self, parts):
        # Input: raw line byte strings. Yields them decoded and counts them.
        for raw in parts:
            self.lines += 1
            yield raw.decode("utf-8", errors="replace").rstrip("\r")

    def close(self) -> None:
        # Closes the underlying connection (safe to call more than once).
        self._close_resp()

    def stats(self) -> dict:
        # Output: provenance of what was consumed: url, final_url, status, headers, total size, compressed
        # bytes consumed and their SHA-256, lines yielded, gzip members, resumes and whether exhausted.
        return {"url": self.url, "final_url": self.final_url, "status": self.status, "headers": self.headers,
                "total_bytes": self.total, "bytes_consumed": self.consumed,
                "sha256_consumed": self.sha.hexdigest(), "lines": self.lines,
                "gzip_members": self.counter["members"], "resumes": self.resumes, "exhausted": self.exhausted}

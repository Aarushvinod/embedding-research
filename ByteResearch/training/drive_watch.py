# Upload watchdog for a Google-Drive-backed artifacts root (Colab), which writes to a local cache and uploads
# in the background, so a file can look saved long before Drive has it or after Drive stopped accepting it.
# It is on only when BYTERESEARCH_DRIVE_CHECK=1 and BYTERESEARCH_DRIVE_ROOT names the mounted "My Drive"
# folder; the notebook sets both after authenticating, and the Drive v3 API then answers with default
# credentials. Sweeps save no checkpoints, so only full training runs are watched.
# DriveChecker maps a path under that root to Drive folders and asks Drive's servers whether every file a
# checkpoint's meta records is there with the same size and md5, plus the meta and COMPLETE themselves.
# Training calls confirm on the previous checkpoint at each save and on best/ at the end, waiting up to the
# timeout (BYTERESEARCH_DRIVE_CHECK_TIMEOUT seconds, default 900) and then stopping with DriveUploadError, so
# work stops at the last checkpoint Drive really holds instead of piling up in a cache a dying session loses.

from __future__ import annotations

import os
import time
from pathlib import Path

from ..core.io import read_json
from ..core.paths import COMPLETE_MARKER, META_FILE

ENABLE_VAR = "BYTERESEARCH_DRIVE_CHECK"
ROOT_VAR = "BYTERESEARCH_DRIVE_ROOT"
TIMEOUT_VAR = "BYTERESEARCH_DRIVE_CHECK_TIMEOUT"
DEFAULT_TIMEOUT_S = 900
POLL_S = 30
FOLDER_MIME = "application/vnd.google-apps.folder"


class DriveUploadError(RuntimeError):
    # Raised when Drive's servers do not hold a checkpoint's files within the timeout (or cannot be queried).
    pass


def enabled() -> bool:
    # Output: whether the watchdog is switched on (BYTERESEARCH_DRIVE_CHECK=1).
    return os.environ.get(ENABLE_VAR) == "1"


def timeout_s() -> float:
    # Output: how long to wait for a checkpoint to reach Drive, in seconds.
    return float(os.environ.get(TIMEOUT_VAR, DEFAULT_TIMEOUT_S))


def _quote(value: str) -> str:
    # Input: a file or folder name. Output: the name escaped for a Drive API query string.
    return value.replace("\\", "\\\\").replace("'", "\\'")


class DriveApi:
    # Lists Drive files through the v3 API with default credentials (on Colab, those of authenticate_user).

    def __init__(self):
        # Builds the Drive v3 files resource; the client library is only imported when the watchdog is on.
        from googleapiclient.discovery import build
        self.files = build("drive", "v3", cache_discovery=False).files()

    def __call__(self, parent: str, name=None, folder: bool = False) -> list:
        # Inputs: a parent folder id, an optional exact name and whether to list folders only. Output: the
        # matching non-trashed items as {"id", "name", "size", "md5Checksum"} dicts, all pages.
        query = [f"'{_quote(parent)}' in parents", "trashed = false"]
        query += [f"name = '{_quote(name)}'"] if name is not None else []
        query += [f"mimeType = '{FOLDER_MIME}'"] if folder else []
        items, token = [], None
        while True:
            page = self.files.list(q=" and ".join(query), pageSize=1000, pageToken=token,
                                   fields="nextPageToken, files(id, name, size, md5Checksum)").execute()
            items += page.get("files", [])
            token = page.get("nextPageToken")
            if not token:
                return items


class DriveChecker:
    # Answers whether a local directory's files are on Drive's servers. Construct with a lister
    # (parent_id, name=None, folder=False) -> [item] such as DriveApi(), and the mounted "My Drive" path.

    def __init__(self, lister, drive_root):
        # Inputs: the lister and the local path of the mounted "My Drive" folder.
        self.lister = lister
        self.root = Path(drive_root)
        self.ids = {}

    def folder_id(self, directory):
        # Input: a local directory under the root. Output: its Drive folder id, or None while Drive does not
        # have it yet; raises DriveUploadError for a path outside the root. Found ids are cached.
        try:
            parts = Path(directory).relative_to(self.root).parts
        except ValueError as exc:
            raise DriveUploadError(f"{directory} is not under the Drive root {self.root}") from exc
        parent = "root"
        for i in range(len(parts)):
            key = parts[: i + 1]
            if key not in self.ids:
                found = self.lister(parent, name=parts[i], folder=True)
                if not found:
                    return None
                self.ids[key] = found[0]["id"]
            parent = self.ids[key]
        return parent

    def remote_files(self, directory) -> dict:
        # Input: a local directory. Output: {name: [Drive items with that name]} of the files Drive holds in
        # it (empty while the folder itself has not arrived).
        fid = self.folder_id(directory)
        out = {}
        for item in self.lister(fid) if fid is not None else []:
            out.setdefault(item["name"], []).append(item)
        return out

    def pending(self, directory, records: dict) -> list:
        # Inputs: a checkpoint directory and its recorded {relative path: digest}. Output: the names Drive does
        # not yet hold with the recorded size and md5, plus the meta and COMPLETE when they are absent.
        directory = Path(directory)
        wanted = {**records, META_FILE: None, COMPLETE_MARKER: None}
        listings = {}
        return [name for name, rec in wanted.items() if not self._arrived(directory / name, rec, listings)]

    def _arrived(self, path: Path, record, listings: dict) -> bool:
        # Inputs: a local file path, its digest (None: presence only) and a cache of folder listings. Output:
        # whether some Drive item of that name in that folder matches the digest.
        if path.parent not in listings:
            listings[path.parent] = self.remote_files(path.parent)
        items = listings[path.parent].get(path.name, [])
        if record is None:
            return bool(items)
        return any(int(i.get("size", -1)) == record["size"] and i.get("md5Checksum") == record["md5"] for i in items)

    def wait(self, directory, records: dict, timeout: float, poll: float = POLL_S, sleep=time.sleep,
             clock=time.monotonic) -> None:
        # Inputs: a checkpoint directory, its records, the timeout and poll interval in seconds, and the sleep
        # and clock functions. Returns once Drive holds every file; raises DriveUploadError naming the missing
        # ones when the timeout passes first.
        deadline = clock() + timeout
        missing = self.pending(directory, records)
        while missing and clock() < deadline:
            sleep(poll)
            missing = self.pending(directory, records)
        if missing:
            raise DriveUploadError(f"Drive has not received {missing} of {directory} after {timeout:.0f}s; it is "
                                   f"probably refusing uploads (quota); stopping at the last checkpoint it holds")


def checker_from_env(artifacts_root):
    # Input: the artifacts root. Output: a DriveChecker on the Drive API when the watchdog is on, else None;
    # raises DriveUploadError when the root is not under BYTERESEARCH_DRIVE_ROOT or Drive cannot see it.
    if not enabled():
        return None
    checker = DriveChecker(DriveApi(), os.environ.get(ROOT_VAR, "/content/drive/MyDrive"))
    if checker.folder_id(artifacts_root) is None:
        raise DriveUploadError(f"Drive does not show {artifacts_root}; check the mount, authentication and "
                               f"{ROOT_VAR}")
    return checker


def confirm(checker, ckpt, timeout: float):
    # Inputs: a DriveChecker (or None), a checkpoint directory (or None) and the timeout. Output: None once
    # Drive holds every file the checkpoint's meta records (or when there is nothing to check, e.g. no checker,
    # no checkpoint or one already rotated away), else the DriveUploadError message.
    meta = read_json(Path(ckpt) / META_FILE) if checker is not None and ckpt is not None else None
    if meta is None:
        return None
    try:
        checker.wait(ckpt, meta.get("files") or {}, timeout)
    except DriveUploadError as exc:
        return str(exc)
    return None

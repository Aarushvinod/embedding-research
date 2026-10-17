# Resumable, unit-keyed result files for long interpretability measurements.
# A measurement is split into units such as a layer, a grid point, a target or a neuron set.
# Each finished unit's fields are stored under a canonical key in the result's per_item block,
# and the key is listed in status.units_done, so a rerun skips finished units and continues.
# The file carries a fingerprint of the measurement config, and reopening it with a different
# fingerprint is refused unless --force, which starts the measurement afresh.
# Partial progress is flushed at most once per interval (60 s by default) and always when the
# context exits with an error.
# finalize() writes the aggregate and sets complete=true.

from __future__ import annotations

import dataclasses
import time

from ..core.config import config_hash
from ..core.results import ResultFile


class FingerprintMismatch(RuntimeError):
    # Raised when an existing partial result was produced under a different config fingerprint.
    pass


def fingerprint(cfg) -> str:
    # Input: a config dataclass or JSON-like dict. Output: its 16-hex-char config hash.
    return config_hash(cfg)


def unit_key(unit) -> str:
    # Input: a unit as a string, int, tuple/list, dict or object with as_dict() (e.g. registry Unit).
    # Output: its canonical string key ("a|b" for sequences, "k=v|..." sorted for mappings).
    if isinstance(unit, str):
        return unit
    if hasattr(unit, "as_dict"):
        unit = unit.as_dict()
    if isinstance(unit, dict):
        return "|".join(f"{k}={unit[k]}" for k in sorted(unit))
    if isinstance(unit, (tuple, list)):
        return "|".join(str(u) for u in unit)
    return str(unit)


def _as_result_file(target, envelope: dict) -> ResultFile:
    # Inputs: a ResultFile or path, and envelope meta. Output: a ResultFile whose meta includes the
    # envelope's experiment/run_id/measurement/variant/unit entries.
    rf = target if isinstance(target, ResultFile) else ResultFile(target)
    rf.meta.update({k: v for k, v in envelope.items() if k != "config"})
    return rf


class UnitFile:
    # A resumable unit-keyed measurement file. Create with UnitFile.open; use as a context manager
    # to guarantee a flush if the measurement crashes.

    def __init__(self, rf: ResultFile, config: dict, fp: str, per_item: dict, done: list, flush_seconds: float):
        # Inputs: the ResultFile, config dict, fingerprint, restored per_item and done keys, and the
        # minimum seconds between flushes.
        self.rf, self.config, self.fp = rf, config, fp
        self.per_item, self.done = per_item, list(done)
        self._done_set = set(self.done)
        self.flush_seconds = float(flush_seconds)
        self._last = time.monotonic()
        self.finalized = False

    @classmethod
    def open(cls, target, envelope: dict | None = None, fingerprint: str = "", force: bool = False,
             flush_seconds: float = 60.0) -> "UnitFile":
        # Inputs: a ResultFile or JSON path, envelope dict (meta fields and "config"), the config
        # fingerprint, the force flag and the flush interval. Output: a UnitFile resuming any matching
        # partial result; a fingerprint mismatch raises FingerprintMismatch unless force, and force
        # always starts fresh.
        envelope = dict(envelope or {})
        config = envelope.get("config", {})
        config = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)
        rf = _as_result_file(target, envelope)
        data = rf.load()
        if data is None or force:
            return cls(rf, config, fingerprint, {}, [], flush_seconds)
        stored = data.get("status", {}).get("fingerprint")
        if stored != fingerprint:
            raise FingerprintMismatch(f"{rf.path}: fingerprint {stored} != {fingerprint}; rerun with --force")
        return cls(rf, config, fingerprint, dict(data.get("per_item") or {}),
                   data.get("status", {}).get("units_done", []), flush_seconds)

    def has(self, unit) -> bool:
        # Input: a unit. Output: whether it is already finished.
        return unit_key(unit) in self._done_set

    def get(self, unit):
        # Input: a finished unit. Output: its stored fields.
        return self.per_item[unit_key(unit)]

    def put(self, unit, fields) -> None:
        # Inputs: a unit and its JSON-like fields. Stores them, marks the unit done and flushes if the
        # flush interval has elapsed.
        key = unit_key(unit)
        self.per_item[key] = fields
        if key not in self._done_set:
            self.done.append(key)
            self._done_set.add(key)
        self.flush()

    def _write(self, aggregate, complete: bool, per_item=None) -> dict:
        # Inputs: aggregate, complete flag and optional per_item override. Writes the envelope with
        # the done list and fingerprint in status. Output: the envelope.
        status = {"units_done": self.done, "fingerprint": self.fp}
        return self.rf.write(self.per_item if per_item is None else per_item, aggregate, config=self.config,
                             complete=complete, extra={"status": status})

    def flush(self, force: bool = False) -> bool:
        # Input: whether to ignore the interval. Writes partial progress (complete=false) when forced
        # or when flush_seconds have passed since the last write. Output: whether it wrote.
        if self.finalized or not (force or time.monotonic() - self._last >= self.flush_seconds):
            return False
        self._write({}, False)
        self._last = time.monotonic()
        return True

    def finalize(self, aggregate, per_item=None) -> dict:
        # Inputs: the aggregate block and an optional final per_item (e.g. a columnar layout).
        # Writes the complete result. Output: the envelope.
        env = self._write(aggregate, True, per_item)
        self.finalized = True
        return env

    def __enter__(self) -> "UnitFile":
        # Output: self, for use as a context manager.
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Flushes partial progress when the block raised before finalize. Output: False (never
        # swallows the exception).
        if exc_type is not None and not self.finalized:
            self.flush(force=True)
        return False


def measurement_done(target, fingerprint: str) -> bool:
    # Inputs: a ResultFile or path and the expected fingerprint. Output: whether the result is
    # complete and was produced under that fingerprint.
    rf = target if isinstance(target, ResultFile) else ResultFile(target)
    data = rf.load()
    status = (data or {}).get("status", {})
    return bool(status.get("complete")) and status.get("fingerprint") == fingerprint


def row_chunks(rows, size: int) -> list:
    # Inputs: row indices and a chunk size. Output: consecutive lists of at most size rows (the resume
    # units of chunked measurements).
    rows = list(rows)
    return [rows[i:i + int(size)] for i in range(0, len(rows), int(size))]

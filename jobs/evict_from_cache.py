#!/usr/bin/env python3
"""Evict your own files from the page cache, so a big GPU job starts with memory that is truly free.

On a GB10 the GPU allocates from system RAM and its driver needs FREE pages; file cache only
counts as "available". Before the Turing freeze of 2026-10-10 the box held 83 GiB of cache
(mostly our own checkpoints, just read by a per-epoch scoring pass) and 27 GiB free when a
training job started (see ``stop_on_low_free.py``). ``drop_caches`` needs root; this needs
none: ``posix_fadvise(DONTNEED)`` on every regular file under the given paths that you can
open drops that file's clean cached pages (dirty pages stay until written back). Directory
symlinks are not followed. Prints one JSON line: files, bytes and MemFree/Cached before and
after. Stdlib only.

    jobs/evict_from_cache.py PATH [PATH ...]
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "machine" / "memory"))
from keep_free import KIB_PER_GIB, MEMINFO, meminfo

USAGE = 2  # exit code: no paths given


def _files(root: str) -> Iterator[str]:
    if not os.path.isdir(root):
        yield root
        return
    for parent, _, names in os.walk(root):
        for name in names:
            yield os.path.join(parent, name)


def evict(paths: list[str]) -> dict:
    """fadvise(DONTNEED) every readable regular file under ``paths``; return counts.

    Symlinked files are skipped (``O_NOFOLLOW``), so a link cannot evict someone else's file.
    """
    files = size = skipped = 0
    for path in (f for root in paths for f in _files(root)):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            skipped += 1
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            files += 1
            size += os.fstat(fd).st_size
        finally:
            os.close(fd)
    return {"files": files, "bytes": size, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return USAGE
    fd = os.open(MEMINFO, os.O_RDONLY)
    gib = lambda kib: round(kib / KIB_PER_GIB, 2)
    free0, cached0 = meminfo(fd)
    record = evict(paths)
    free1, cached1 = meminfo(fd)
    print(json.dumps({"event": "evicted", **record,
                      "free_gib": [gib(free0), gib(free1)], "cached_gib": [gib(cached0), gib(cached1)]}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

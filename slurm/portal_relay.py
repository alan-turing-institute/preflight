#!/usr/bin/env python3
"""The portal's front door for `portal.auth = "ssh"` (D19): the ssh login each person already has is the sign-in.

Each person forwards a local port to this Unix socket through their own ssh session (`LocalForward 8080
/run/gpujobs-portal.sock`). sshd opens the socket as that person, so the kernel names their account for every
connection (SO_PEERCRED; measured on the Turing box, 2026-10-21: a forward from user1 arrives as uid 1002). For each
connection the relay reads one request head, drops any identity or key the request claims, adds the account's name and
the key Apache checks (portal.py `_ssh_block`), forces `Connection: close` so the connection carries that one request
only, and then copies bytes both ways. Nothing the person sends can choose the name.

A browser also sends requests other pages ask for, so a request must name `localhost` as its host (a page that
re-points its own domain at 127.0.0.1 still names its own domain) and must not be a cross-site one, other than a person
following a link (Sec-Fetch-Site, Sec-Fetch-Mode). Cross-reviewed 2026-10-21; its findings are the guards below.

Started by systemd (gpujobs-portal-relay.socket): the listening socket arrives as fd 3, the key as the credential `key`.
"""

from __future__ import annotations

import os
import pwd
import re
import socket
import struct
import sys
import threading
import time
from pathlib import Path

BACKEND = ("127.0.0.1", 80)  # Apache, plain http on the loopback: in this mode the ssh connection is the encryption
LISTEN_FD = 3  # sd_listen_fds(3): the first socket systemd passes
KEY_HEADER = b"X-Frontdoor-Key"  # portal.py PROXY_KEY_HEADER
USER_HEADER = b"X-Gpujobs-User"  # portal.py SSH_USER_HEADER
MAX_HEAD = 64 * 1024
HEAD_TIMEOUT_S = 30  # for the whole head, not each read: a slow sender cannot hold a connection open
IDLE_TIMEOUT_S = 300
CHUNK = 64 * 1024
MAX_CONNECTIONS = 256  # at once, everyone together; well under the unit's LimitNOFILE (two descriptors each)
MAX_PER_USER = 64
MIN_UID = 1000  # nginx_stage's own floor (configuration.rb min_uid): system accounts get no portal
NOBODY_UID = 65534
# Each pattern is used with fullmatch: `$` alone also matches before a final newline.
USER_RE = re.compile(r"[a-z_][a-z0-9_-]{0,31}")  # the only names Apache's check accepts (portal.py _ssh_block)
KEY_RE = re.compile(r"\S{16,}")
REQUEST_LINE_RE = re.compile(rb"[A-Z]+ /[\x21-\x7e]* HTTP/1\.[01]")  # origin-form only: an absolute URI names a host
HEADER_NAME_RE = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+")  # RFC 9110 token
HOST_RE = re.compile(rb"localhost(:[0-9]{1,5})?")  # portal.server_name, which ssh mode requires to be localhost
# What a request may not bring: the identity and key this relay sets, and the hop-by-hop headers that could keep the
# connection open for a second request. Compared with `_` read as `-`, so no spelling of them gets through.
DROPPED = {b"x-frontdoor-key", b"x-gpujobs-user", b"connection", b"keep-alive", b"proxy-connection", b"upgrade"}


class RefusedError(Exception):
    """The request is not passed on; the argument is the HTTP status line to answer with."""


def peer_user(conn: socket.socket) -> str:
    """The account that opened the connection, as the kernel records it; never root, never a name Apache would refuse."""
    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    _pid, uid, _gid = struct.unpack("3i", creds)
    try:
        name = pwd.getpwuid(uid).pw_name
    except KeyError:
        raise RefusedError("403 Forbidden") from None
    if uid < MIN_UID or uid == NOBODY_UID or not USER_RE.fullmatch(name):
        raise RefusedError("403 Forbidden")
    return name


def read_head(conn: socket.socket) -> tuple[bytes, bytes]:
    """(request head without its closing blank line, whatever of the body arrived with it), within HEAD_TIMEOUT_S."""
    data = b""
    deadline = time.monotonic() + HEAD_TIMEOUT_S
    while (end := data.find(b"\r\n\r\n")) < 0:
        if len(data) > MAX_HEAD:
            raise RefusedError("431 Request Header Fields Too Large")
        left = deadline - time.monotonic()
        if left <= 0:
            raise RefusedError("408 Request Timeout")
        conn.settimeout(left)
        try:
            chunk = conn.recv(CHUNK)
        except TimeoutError:
            raise RefusedError("408 Request Timeout") from None
        if not chunk:
            raise RefusedError("400 Bad Request")
        data += chunk
    return data[:end], data[end + 4:]


def rewrite(head: bytes, user: str, key: str) -> bytes:
    """The request head as Apache receives it: the client's own lines, less the dropped ones, plus identity and key.

    A line holding a lone CR, LF or NUL is refused, as is a folded continuation line: a server that took a bare LF as a
    line end would see a second header the drop list never saw (Apache 2.4.58 refuses them itself; the relay does not
    rely on that). The host must be localhost, once, and a cross-site request is refused unless it is a navigation."""
    request_line, *lines = head.split(b"\r\n")
    if not REQUEST_LINE_RE.fullmatch(request_line):
        raise RefusedError("400 Bad Request")
    kept = [request_line]
    seen: dict[bytes, list[bytes]] = {}
    for line in lines:
        name, sep, value = line.partition(b":")
        if not sep or not HEADER_NAME_RE.fullmatch(name) or any(c in line for c in b"\r\n\x00"):
            raise RefusedError("400 Bad Request")
        seen.setdefault(name.lower(), []).append(value.strip())
        if name.lower().replace(b"_", b"-") not in DROPPED:
            kept.append(line)
    hosts = seen.get(b"host", [])
    if len(hosts) != 1 or not HOST_RE.fullmatch(hosts[0]):
        raise RefusedError("421 Misdirected Request")
    navigation = request_line.startswith(b"GET ") and seen.get(b"sec-fetch-mode") == [b"navigate"]
    if seen.get(b"sec-fetch-site") == [b"cross-site"] and not navigation:
        raise RefusedError("403 Forbidden")
    kept += [KEY_HEADER + b": " + key.encode(), USER_HEADER + b": " + user.encode(), b"Connection: close"]
    return b"\r\n".join(kept) + b"\r\n\r\n"


def _reply(conn: socket.socket, status: str) -> None:
    try:
        conn.sendall(f"HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode())
    except OSError:
        pass


def _pump(src: socket.socket, dst: socket.socket) -> None:
    """Copy until `src` ends, then end `dst`'s sending side."""
    try:
        while data := src.recv(CHUNK):
            dst.sendall(data)
    except OSError:
        pass
    try:
        dst.shutdown(socket.SHUT_WR)
    except OSError:
        pass


class Limits:
    """Connections open now, per user, so no one account can take every slot."""

    def __init__(self, per_user: int = MAX_PER_USER) -> None:
        self.per_user, self.open, self.lock = per_user, {}, threading.Lock()

    def take(self, user: str) -> bool:
        with self.lock:
            if self.open.get(user, 0) >= self.per_user:
                return False
            self.open[user] = self.open.get(user, 0) + 1
            return True

    def give_back(self, user: str) -> None:
        with self.lock:
            self.open[user] -= 1


def handle(conn: socket.socket, key: str, backend: tuple[str, int] = BACKEND, limits: Limits | None = None) -> None:
    """One connection: one request to Apache as the account that opened it, and its answer back."""
    limits = limits or Limits()
    with conn:
        try:
            user = peer_user(conn)
        except RefusedError as refused:
            _reply(conn, str(refused))
            return
        except OSError:
            return
        if not limits.take(user):
            _reply(conn, "503 Service Unavailable")
            return
        try:
            _serve(conn, user, key, backend)
        finally:
            limits.give_back(user)


def _serve(conn: socket.socket, user: str, key: str, backend: tuple[str, int]) -> None:
    try:
        head, body = read_head(conn)
        request = rewrite(head, user, key)
    except RefusedError as refused:
        _reply(conn, str(refused))
        return
    except OSError:
        return
    try:
        upstream = socket.create_connection(backend, timeout=HEAD_TIMEOUT_S)
    except OSError:
        _reply(conn, "502 Bad Gateway")
        return
    with upstream:
        conn.settimeout(IDLE_TIMEOUT_S)
        upstream.settimeout(IDLE_TIMEOUT_S)
        try:
            upstream.sendall(request + body)
        except OSError:
            return
        sending = threading.Thread(target=_pump, args=(conn, upstream), daemon=True)
        sending.start()
        _pump(upstream, conn)  # the answer; Apache closes after it (Connection: close)
        try:
            conn.shutdown(socket.SHUT_RDWR)  # ends the other copy's wait for more from the client
        except OSError:
            pass
        sending.join()


def main() -> int:
    if os.environ.get("LISTEN_PID") != str(os.getpid()) or os.environ.get("LISTEN_FDS") != "1":
        sys.exit("portal_relay: start it through gpujobs-portal-relay.socket, which passes the listening socket")
    credentials = os.environ.get("CREDENTIALS_DIRECTORY", "")
    key = (Path(credentials) / "key").read_text().strip() if credentials else ""
    if not KEY_RE.fullmatch(key):
        sys.exit("portal_relay: no usable key (systemd LoadCredential=key:<the site's key_file>)")
    serve(socket.socket(fileno=LISTEN_FD), key)
    return 0


def serve(server: socket.socket, key: str, slots: threading.BoundedSemaphore | None = None) -> None:
    """Accept for ever. A failed accept (out of descriptors, say) is waited out, never fatal, and a connection beyond
    MAX_CONNECTIONS is closed at once: no local user can stop the relay for everyone else."""
    slots = slots or threading.BoundedSemaphore(MAX_CONNECTIONS)
    limits = Limits()
    while True:
        try:
            conn, _ = server.accept()
        except OSError as error:
            print(f"portal_relay: accept failed ({error}); retrying", file=sys.stderr, flush=True)
            time.sleep(0.5)
            continue
        if not slots.acquire(blocking=False):
            conn.close()
            continue

        def run(conn: socket.socket = conn) -> None:
            try:
                handle(conn, key, limits=limits)
            finally:
                slots.release()

        try:
            threading.Thread(target=run, daemon=True).start()
        except RuntimeError:  # no thread to be had: drop this one, keep serving
            conn.close()
            slots.release()


if __name__ == "__main__":
    sys.exit(main())

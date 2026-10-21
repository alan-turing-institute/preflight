"""portal_relay: the name comes from the kernel, never from the request, and one connection carries one request."""

from __future__ import annotations

import getpass
import socket
import struct
import threading
import time
from types import SimpleNamespace

import portal_relay
import pytest

KEY = "k" * 32
FORGED = (b"X-Gpujobs-User: root", b"x-gpujobs-user: root", b"X_Gpujobs_User: root", b"X-Frontdoor-Key: guessed",
          b"x_frontdoor_key: guessed", b"Connection: keep-alive", b"Keep-Alive: timeout=5", b"Upgrade: websocket")


def test_rewrite_drops_every_claimed_identity_and_adds_the_relays_own():
    head = b"\r\n".join([b"GET /pun/sys/gpujobs/ HTTP/1.1", b"Host: localhost:8080", *FORGED, b"Accept: */*"])
    lines = portal_relay.rewrite(head, "alice", KEY).split(b"\r\n")
    assert lines[:3] == [b"GET /pun/sys/gpujobs/ HTTP/1.1", b"Host: localhost:8080", b"Accept: */*"]
    assert lines[3:] == [b"X-Frontdoor-Key: " + KEY.encode(), b"X-Gpujobs-User: alice", b"Connection: close", b"", b""]


@pytest.mark.parametrize("head", [
    b"GET / HTTP/1.1\r\nX-Probe: a\nX-Gpujobs-User: root",  # Apache splits on a bare LF: a second header inside one line
    b"GET / HTTP/1.1\r\nX-Probe: a\rX-Gpujobs-User: root",
    b"GET / HTTP/1.1\r\nX-Probe: a\x00b",
    b"GET / HTTP/1.1\r\nX-Probe: a\r\n X-Gpujobs-User: root",  # obsolete line folding
    b"GET / HTTP/1.1\r\nX-Gpujobs-User : root",  # a space before the colon is not a token
    b"GET / HTTP/1.1\r\nno colon here",
    b"PRI * HTTP/2.0",
    b"GET /\nX-Gpujobs-User: root HTTP/1.1",
    b"GET / HTTP/1.1\n\r\nHost: localhost",  # `$` alone would match before that final LF (review, 2026-10-21)
    b"GET http://evil.example/ HTTP/1.1\r\nHost: localhost",  # Apache takes the host from an absolute URI
])
def test_rewrite_refuses_what_could_carry_a_header_past_the_drop_list(head):
    with pytest.raises(portal_relay.RefusedError, match="400"):
        portal_relay.rewrite(head, "alice", KEY)


@pytest.mark.parametrize(("lines", "status"), [
    ([], "421"),  # no host
    ([b"Host: evil.example:8080"], "421"),  # a page that re-pointed its own domain at 127.0.0.1 (DNS rebinding)
    ([b"Host: 127.0.0.1:8080"], "421"),
    ([b"Host: localhost", b"Host: evil.example"], "421"),
    ([b"Host: localhost", b"Sec-Fetch-Site: cross-site", b"Sec-Fetch-Mode: no-cors"], "403"),  # <img src=...> elsewhere
    ([b"Host: localhost", b"Sec-Fetch-Site: cross-site", b"Sec-Fetch-Mode: cors"], "403"),
])
def test_a_request_another_page_could_make_is_refused(lines, status):
    with pytest.raises(portal_relay.RefusedError, match=status):
        portal_relay.rewrite(b"\r\n".join([b"GET / HTTP/1.1", *lines]), "alice", KEY)


def test_a_person_following_a_link_from_elsewhere_gets_in():
    navigate = [b"Host: localhost:18080", b"Sec-Fetch-Site: cross-site", b"Sec-Fetch-Mode: navigate"]
    assert portal_relay.rewrite(b"\r\n".join([b"GET / HTTP/1.1", *navigate]), "alice", KEY).endswith(b"close\r\n\r\n")
    with pytest.raises(portal_relay.RefusedError, match="403"):  # a cross-site form post is not a link
        portal_relay.rewrite(b"\r\n".join([b"POST /x HTTP/1.1", *navigate]), "alice", KEY)


def test_root_system_and_unknown_accounts_are_refused():
    for uid in (0, 999, portal_relay.NOBODY_UID, 2**31 - 2):  # root; a system account; nobody; a uid with no account
        conn = SimpleNamespace(getsockopt=lambda *_a, uid=uid: struct.pack("3i", 1, uid, uid))
        with pytest.raises(portal_relay.RefusedError, match="403"):
            portal_relay.peer_user(conn)


def _apache_stand_in() -> tuple[tuple[str, int], dict[str, bytes], threading.Thread]:
    """A one-request server on the loopback that records what it got and answers 200 `ok`, then closes."""
    server = socket.create_server(("127.0.0.1", 0))
    got: dict[str, bytes] = {}

    def serve() -> None:
        with server, server.accept()[0] as conn:
            data = b""
            while b"\r\n\r\nbody" not in data:
                data += conn.recv(4096)
            got["request"] = data
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server.getsockname(), got, thread


def test_one_request_reaches_apache_as_the_connecting_account_and_its_answer_comes_back():
    backend, got, apache = _apache_stand_in()
    client, relay_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)  # the kernel names this test's own user
    relay = threading.Thread(target=portal_relay.handle, args=(relay_end, KEY, backend), daemon=True)
    relay.start()
    client.sendall(b"POST /x HTTP/1.1\r\nHost: localhost:8080\r\nX-Gpujobs-User: root\r\nContent-Length: 4\r\n\r\nbody")
    answer = b""
    while chunk := client.recv(4096):
        answer += chunk
    relay.join(5)
    apache.join(5)
    assert answer.endswith(b"\r\n\r\nok") and not relay.is_alive()
    head, _, body = got["request"].partition(b"\r\n\r\n")
    assert body == b"body" and f"X-Gpujobs-User: {getpass.getuser()}".encode() in head.split(b"\r\n")
    assert b"root" not in head and b"X-Frontdoor-Key: " + KEY.encode() in head.split(b"\r\n")


@pytest.mark.parametrize(("sent", "status"), [(b"x" * (portal_relay.MAX_HEAD + 2), b"431"), (b"BAD\r\n\r\n", b"400")])
def test_a_refused_request_is_answered_and_never_reaches_apache(sent, status):
    client, relay_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    relay = threading.Thread(target=portal_relay.handle, args=(relay_end, KEY, ("127.0.0.1", 9)), daemon=True)
    relay.start()
    client.sendall(sent)
    assert client.recv(4096).startswith(b"HTTP/1.1 " + status)
    relay.join(5)
    assert not relay.is_alive()


def test_a_slow_head_is_cut_off_at_one_deadline_not_per_read(monkeypatch):
    monkeypatch.setattr(portal_relay, "HEAD_TIMEOUT_S", 0.5)
    client, relay_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    relay = threading.Thread(target=portal_relay.handle, args=(relay_end, KEY, ("127.0.0.1", 9)), daemon=True)
    relay.start()
    started = time.monotonic()
    try:
        for _ in range(20):  # one byte every 0.1 s: each read is quick, the head never ends
            client.sendall(b"G")
            time.sleep(0.1)
    except OSError:
        pass
    assert client.recv(4096).startswith(b"HTTP/1.1 408") and time.monotonic() - started < 1.5


def test_no_account_can_take_every_connection():
    limits = portal_relay.Limits(per_user=1)
    assert limits.take("alice") and not limits.take("alice") and limits.take("bob")
    client, relay_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    portal_relay.handle(relay_end, KEY, ("127.0.0.1", 9), limits=portal_relay.Limits(per_user=0))
    assert client.recv(4096).startswith(b"HTTP/1.1 503")
    limits.give_back("alice")
    assert limits.take("alice")


class _Stop(Exception):
    pass


def test_serving_survives_a_failed_accept_and_closes_what_it_has_no_slot_for(monkeypatch):
    monkeypatch.setattr(portal_relay.time, "sleep", lambda _s: None)
    served, served_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    turned_away, turned_away_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    slots = threading.BoundedSemaphore(1)
    events = iter([OSError(24, "Too many open files"), (served_end, None), (turned_away_end, None), _Stop()])

    def accept():
        event = next(events)
        if isinstance(event, BaseException):
            raise event
        return event

    served.sendall(b"GET / HTTP/1.1\r\n\r\n")  # no host: answered 421 by the one slot's thread
    with pytest.raises(_Stop):
        portal_relay.serve(SimpleNamespace(accept=accept), KEY, slots)
    assert turned_away.recv(4096) == b""  # closed at once: the only slot was taken
    assert served.recv(4096).startswith(b"HTTP/1.1 421")

"""components/portal.py: each auth mode renders, secrets stay out of the repo, unsafe setups are refused."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import site_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "slurm" / "install"))  # `from render import ...` as the installer does
spec = importlib.util.spec_from_file_location("portal", ROOT / "slurm" / "install" / "components" / "portal.py")
portal = importlib.util.module_from_spec(spec)
sys.modules["portal"] = portal
spec.loader.exec_module(portal)

SECRET = "s3cr3t-client-value"
KEY = "k" * 32
HASH = "$2a$12$iKLecAIN9MrxOZ0UltRb.OQOms/bgQbs5F.qCehq15oc3CvGFYzLy"
USER = {"username": "alice", "email": "alice@localhost", "hash": HASH, "userID": "D642A38C-402F-47AA-879B-FEC95745F5BA"}
# Dex's built-in account as the generator hardcodes it (ood-portal-generator/lib/ood_portal_generator/dex.rb:257-261).
DEX_DEFAULT = {
    "email": "ood@localhost",
    "userID": "08a8684b-db88-4b73-90a9-3cd1661f5466",
    "hash": "$2a$10$2b2cU8CPhOTaGrs1HRQuAueS7JTT5ZHsHSzYiFPm1leZck7Mc8T4W",
}
DASHBOARD = {
    "orgId": 2,
    "dashboard": {"name": "jobs", "uid": "abc", "panels": {"cpu": 1}},
    "labels": {"jobid": "jobid"},
}


def secret_file(tmp_path: Path, name: str, text: str, mode: int = 0o600) -> str:
    path = tmp_path / name
    path.write_text(text)
    path.chmod(mode)
    return str(path)


def make_site(tmp_path: Path, mode: str = "install", auth: str = "dex-local", tls: bool = True,
              portal_extra: dict | None = None, **settings: str):
    """A site whose secret files exist (root-only) unless a setting overrides the path."""
    defaults = {"crypto_passphrase_file": secret_file(tmp_path, "pass", "p" * 32)}
    if auth == "dex-local":
        defaults["users_file"] = secret_file(tmp_path, "users.json", json.dumps([USER]))
    elif auth == "proxy":
        defaults = {"user": "alice", "key_file": secret_file(tmp_path, "key", KEY)}
        portal_extra = {"allow_from": ["192.0.2.8/32"], **(portal_extra or {})}
    elif auth == "oidc":
        defaults |= {
            "issuer": "https://idp.example.org",
            "client_id": "ood",
            "client_secret_file": secret_file(tmp_path, "client", SECRET),
        }
    else:
        defaults |= {"host": "ldap.example.org:636", "user_base_dn": "ou=People,dc=example,dc=org"}
    portal_table = {"mode": mode, "auth": auth, "server_name": "ood.example.org", "auth_settings": defaults | settings}
    portal_table |= portal_extra or {}
    if tls:
        portal_table |= {
            "tls_cert": secret_file(tmp_path, "c.pem", "x"),
            "tls_key": secret_file(tmp_path, "k.pem", "x"),
        }
    return site_config.parse(
        {
            "name": "spark-test",
            "nodes": [{"name": "n", "cpus": 2, "real_memory_mb": 1000}],
            "metrics": {"mode": "local", "grafana_port": 3001},
            "portal": portal_table,
        }
    )


@pytest.fixture(autouse=True)
def no_metrics_module(monkeypatch):
    monkeypatch.setattr(portal, "_metrics_module", lambda: None)


@pytest.fixture(autouse=True)
def empty_apt(tmp_path, monkeypatch):
    """No test reads the host's own /etc/apt (this box has NodeSource configured by hand)."""
    monkeypatch.setattr(portal, "APT_DIR", tmp_path / "no-apt")


def test_dex_local_renders_users_and_runs_update_with_insecure(tmp_path):
    site = make_site(tmp_path)
    spec_ = portal.files(site)[portal.PORTAL_YML]
    assert (spec_.mode, spec_.owner) == (portal.SECRET_MODE, "root")
    assert f'hash: "{HASH}"' in spec_.content
    assert DEX_DEFAULT["email"] not in spec_.content and DEX_DEFAULT["hash"] not in spec_.content
    assert 'servername: "ood.example.org"' in spec_.content and "security_strict_transport: true" in spec_.content
    assert 'root_uri: "/pun/sys/gpujobs/"' in spec_.content  # the portal opens on the jobs page
    assert portal.refusals(site) == []
    assert "ondemand-dex" in portal.packages(site) and "python-is-python3" in portal.packages(site)
    update = next(c for c in portal.post_install(site) if c[0] == portal.UPDATE_OOD_PORTAL)
    assert update == [portal.UPDATE_OOD_PORTAL, "--force", "--insecure"]
    assert ["systemctl", "restart", "ondemand-dex"] in portal.post_install(site)
    assert "oidc_session" not in spec_.content and not any(".service.d/" in f for f in portal.files(site))  # defaults
    # one sign-in for a year, in an encrypted cookie that outlives Apache restarts; only the listed networks get in
    long = make_site(tmp_path, portal_extra={"session_days": 365, "allow_from": ["100.64.0.0/10"]})
    text = portal.files(long)[portal.PORTAL_YML].content
    assert "oidc_session_max_duration: 31536000" in text and 'OIDCSessionType: "client-cookie:persistent"' in text
    for unit in ("apache2", "ondemand-dex"):
        dropin = portal.files(long)[f"/etc/systemd/system/{unit}.service.d/10-gpujobs.conf"].content
        assert "IPAddressDeny=any\nIPAddressAllow=localhost 100.64.0.0/10\n" in dropin
    steps = portal.post_install(long)
    assert steps.index(["systemctl", "daemon-reload"]) < steps.index(["systemctl", "restart", "apache2"])


def test_dex_ldap_renders_connector_and_never_uses_insecure(tmp_path):
    site = make_site(
        tmp_path,
        auth="dex-ldap",
        bind_dn="cn=admin,dc=example,dc=org",
        bind_pw_file=secret_file(tmp_path, "bind", "ldap-pw"),
    )
    text = portal.files(site)[portal.PORTAL_YML].content
    assert 'type: "ldap"' in text and 'host: "ldap.example.org:636"' in text and "insecureSkipVerify: false" in text
    assert 'bindPW: "ldap-pw"' in text and 'oidc_remote_user_claim: "preferred_username"' in text
    assert "static_passwords" not in text
    update = next(c for c in portal.post_install(site) if c[0] == portal.UPDATE_OOD_PORTAL)
    assert "--insecure" not in update


def test_oidc_renders_with_secret_only_in_the_root_only_file(tmp_path):
    site = make_site(tmp_path, auth="oidc", user_map_match="^([^@]+)@example.org$")
    out = portal.files(site)
    main = out[portal.PORTAL_YML]
    assert main.mode == portal.SECRET_MODE
    assert 'oidc_provider_metadata_url: "https://idp.example.org/.well-known/openid-configuration"' in main.content
    assert (
        f'oidc_client_secret: "{SECRET}"' in main.content and 'user_map_match: "^([^@]+)@example.org$"' in main.content
    )
    others = [s.content for target, s in out.items() if target != portal.PORTAL_YML]
    others += [s.content for s in portal.apt_sources(site).values()]
    assert not any(SECRET in c for c in others)
    assert "libapache2-mod-auth-openidc" in portal.packages(site) and "ondemand-dex" not in portal.packages(site)
    assert ["a2enmod", "auth_openidc"] in portal.post_install(site)
    # a secret written straight into the site file is not an accepted setting
    inline = make_site(tmp_path, auth="oidc", client_secret=SECRET)
    assert any("unknown key" in r and "client_secret" in r for r in portal.refusals(inline))
    loose = make_site(tmp_path, auth="oidc", client_secret_file=secret_file(tmp_path, "loose", SECRET, 0o644))
    assert any("must be root-only" in r for r in portal.refusals(loose))


def test_proxy_serves_everyone_as_one_user_behind_the_key_and_runs_no_login(tmp_path):
    site = make_site(tmp_path, auth="proxy")
    text = portal.files(site)[portal.PORTAL_YML].content
    digest = hashlib.sha1(KEY.encode()).hexdigest()
    assert portal.refusals(site) == []
    # the key is checked on every OOD location and on `/`, and the header never goes on to the app
    assert text.count(f"sha1(req('X-Frontdoor-Key')) == '{digest}'") == 2 and text.count("RequestHeader unset X-Frontdoor-Key") == 2
    assert 'user_env: "GPUJOBS_PORTAL_USER"' in text and '"SetEnv GPUJOBS_PORTAL_USER alice"' in text and "dex: false" in text
    assert KEY not in text and "openid-connect" not in text and "oidc_" not in text  # only the digest is written
    assert "ondemand-dex" not in portal.packages(site) and "libapache2-mod-auth-openidc" not in portal.packages(site)
    steps = portal.post_install(site)
    assert ["a2enmod", "auth_openidc"] not in steps and [portal.UPDATE_OOD_PORTAL, "--force"] in steps
    assert not any("ondemand-dex" in c and c[:2] != ["sh", "-c"] for c in steps)  # no Dex enabled or restarted
    stop = next(c for c in steps if c[:2] == ["sh", "-c"])  # a machine switched from Dex stops it, one without it skips
    assert "systemctl cat ondemand-dex" in stop[2] and "disable --now ondemand-dex" in stop[2]
    assert not any("ondemand-dex" in f for f in portal.files(site))  # no Dex drop-in either
    for setup, expect in (
        ({"portal_extra": {"allow_from": []}}, "allow_from"),
        ({"user": "Not A User"}, "Linux user"),
        ({"key_file": secret_file(tmp_path, "loose", KEY, 0o644)}, "must be root-only"),
        ({"key_file": secret_file(tmp_path, "short", "short")}, "unusable value"),
        ({"crypto_passphrase_file": "/x"}, "unknown key"),
    ):
        refused = make_site(tmp_path, auth="proxy", **setup)
        assert any(expect in r for r in portal.refusals(refused)), setup
        assert portal.PORTAL_YML not in portal.files(refused)


@pytest.mark.parametrize("field", ["email", "userID", "hash"])
def test_dex_default_account_is_refused(tmp_path, field):
    users = [USER, {**USER, "username": "ood", "email": "bob@localhost", field: DEX_DEFAULT[field]}]
    site = make_site(tmp_path, users_file=secret_file(tmp_path, "u2.json", json.dumps(users)))
    assert any("default account" in r for r in portal.refusals(site))
    assert portal.PORTAL_YML not in portal.files(site)  # a refused config is never offered for writing


@pytest.mark.parametrize(
    ("setup", "expect"),
    [
        (lambda t: {"users_file": str(t / "missing.json")}, "does not exist"),
        (lambda t: {"users_file": secret_file(t, "e.json", "[]")}, "non-empty JSON list"),
        (lambda t: {"users_file": secret_file(t, "o.json", json.dumps([USER]), 0o644)}, "must be root-only"),
        (lambda t: {"users_file": secret_file(t, "p.json", json.dumps([{**USER, "hash": "plaintext"}]))}, "bcrypt"),
        (
            lambda t: {"users_file": secret_file(t, "q.json", json.dumps([{**USER, "email": "../x@localhost"}]))},
            "Linux user",
        ),
        (lambda t: {"crypto_passphrase_file": str(t / "none")}, "cookie passphrase file"),
        (lambda t: {"crypto_passphrase_file": secret_file(t, "pp", "a" * 64, 0o644)}, "must be root-only"),
        (lambda t: {"typo_key": "x"}, "unknown key"),
    ],
)
def test_dex_local_refusals(tmp_path, setup, expect):
    site = make_site(tmp_path, **setup(tmp_path))
    assert any(expect in r for r in portal.refusals(site))
    assert portal.PORTAL_YML not in portal.files(site)


def test_oidc_http_issuer_and_missing_server_name_refused(tmp_path):
    http = make_site(tmp_path, auth="oidc", issuer="http://idp.example.org")
    assert any("https://" in r for r in portal.refusals(http))
    nameless = site_config.parse(
        {
            "name": "x",
            "nodes": [{"name": "n", "cpus": 1, "real_memory_mb": 1}],
            "portal": {"mode": "install", "auth_settings": {}},
        }
    )
    assert any("server_name" in r for r in portal.refusals(nameless))


def test_tls_site_pair_versus_self_signed_fallback(tmp_path):
    site = make_site(tmp_path)
    assert str(site.portal.tls_cert) in portal.files(site)[portal.PORTAL_YML].content
    assert "openssl" not in portal.packages(site)
    assert not any(c[:2] == ["sh", "-c"] for c in portal.post_install(site))

    selfsigned = make_site(tmp_path, tls=False)
    assert portal.refusals(selfsigned) == []  # missing TLS is not refused: a pair is generated and a warning printed
    text = portal.files(selfsigned)[portal.PORTAL_YML].content
    assert str(portal.SELF_SIGNED_KEY) in text and "security_strict_transport: false" in text
    steps = portal.post_install(selfsigned)
    gen = next(c for c in steps if c[:2] == ["sh", "-c"])
    assert gen[3:] == [
        "sh",
        str(portal.SELF_SIGNED_CERT),
        str(portal.SELF_SIGNED_KEY),
        "ood.example.org",
        "DNS:ood.example.org",
    ]
    assert "chmod 0600" in gen[2] and "SELF-SIGNED" in gen[2] and "http://" not in text
    assert steps.index(gen) < steps.index(["update-ca-certificates"]) < steps.index(["a2enmod", "ssl"])
    # one half of a pair is refused
    half = site_config.parse(
        {
            "name": "x",
            "nodes": [{"name": "n", "cpus": 1, "real_memory_mb": 1}],
            "portal": {
                "mode": "install",
                "server_name": "h.example.org",
                "tls_cert": str(tmp_path / "c.pem"),
                "auth_settings": {"users_file": "/x"},
            },
        }
    )
    assert any("both portal.tls_cert and portal.tls_key" in r for r in portal.refusals(half))


def test_cluster_file_with_and_without_grafana(tmp_path, monkeypatch):
    site = make_site(tmp_path)
    target = f"{portal.CLUSTER_DIR}/spark-test.yml"
    plain = portal.files(site)[target]
    assert (plain.mode, plain.owner) == (portal.FILE_MODE, "root")
    assert 'adapter: "slurm"' in plain.content and 'bin: "/usr/bin"' in plain.content
    assert 'conf: "/etc/slurm/slurm.conf"' in plain.content and "grafana" not in plain.content

    monkeypatch.setattr(portal, "_metrics_module", lambda: SimpleNamespace(GRAFANA_DASHBOARD=DASHBOARD))
    with_graf = portal.files(site)[target].content
    assert 'host: "http://ood.example.org:3001"' in with_graf and "orgId: 2" in with_graf and 'uid: "abc"' in with_graf
    off = make_site(tmp_path)
    off = site_config.parse({**_raw(off), "metrics": {"mode": "off"}})
    assert "grafana" not in portal.files(off)[target].content


def _raw(site) -> dict:
    return {
        "name": site.name,
        "nodes": [{"name": "n", "cpus": 2, "real_memory_mb": 1000}],
        "portal": {"mode": "install", "server_name": site.portal.server_name, "auth_settings": {}},
    }


def test_existing_mode_writes_only_cluster_file_and_app_link(tmp_path):
    site = make_site(tmp_path, mode="existing")
    assert list(portal.files(site)) == [f"{portal.CLUSTER_DIR}/spark-test.yml"]
    assert portal.packages(site) == [] and portal.apt_sources(site) == {} and portal.refusals(site) == []
    assert portal.post_install(site) == [["ln", "-sfT", "/opt/gpujobs/slurm/ood_app", "/var/www/ood/apps/sys/gpujobs"]]
    off = make_site(tmp_path, mode="off")
    assert not portal.enabled(off) and portal.files(off) == {} and portal.post_install(off) == []


def test_install_links_our_app_after_the_restart_and_ships_the_repo_keys(tmp_path, monkeypatch):
    site = make_site(tmp_path)
    steps = portal.post_install(site)
    assert steps[-1] == ["ln", "-sfT", "/opt/gpujobs/slurm/ood_app", "/var/www/ood/apps/sys/gpujobs"]
    assert steps.index([portal.UPDATE_OOD_PORTAL, "--force", "--insecure"]) < steps.index(
        ["systemctl", "restart", "apache2"]
    )
    # running per-user servers keep the old server name until stopped, after Apache has the new one
    assert steps.index(["systemctl", "restart", "apache2"]) < steps.index([portal.NGINX_STAGE, "nginx_clean", "--force"])
    sources = portal.apt_sources(site)
    assert sources[portal.OOD_LIST].content == (
        f"deb [signed-by={portal.OOD_KEYRING}] https://apt.osc.edu/ondemand/4.2/web/apt noble main\n"
    )
    assert sources[portal.NODE_LIST].content.endswith("https://deb.nodesource.com/node_22.x nodistro main\n")
    for keyring in (portal.OOD_KEYRING, portal.NODE_KEYRING):
        assert sources[keyring].content.startswith("-----BEGIN PGP PUBLIC KEY BLOCK-----")
    # a source the machine already has (NodeSource added by hand, deb822 format) is used as it is, never doubled:
    # apt refuses one source with two signing keys, and every apt-get update then fails (gpu-node-1, 2026-10-19)
    apt = tmp_path / "apt"
    (apt / "sources.list.d").mkdir(parents=True)
    (apt / "sources.list.d" / "nodesource.sources").write_text(
        "Types: deb\nURIs: https://deb.nodesource.com/node_22.x\nSuites: nodistro\nSigned-By: /usr/share/keyrings/nodesource.gpg\n"
    )
    (apt / "sources.list.d" / "gpujobs-ondemand-web.list").write_text("ours, from an earlier apply\n")
    monkeypatch.setattr(portal, "APT_DIR", apt)
    assert sorted(portal.apt_sources(site)) == sorted([portal.OOD_KEYRING, portal.OOD_LIST])

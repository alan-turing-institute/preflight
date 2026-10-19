"""The web portal component: Open OnDemand (OOD) with its login, the cluster file and our jobs app.

Component interface (see the installer): `NAME`, `enabled`, `files`, `packages`, `refusals`, `post_install`, plus
`apt_sources`. Modes (site `[portal] mode`): `install` sets up OOD end to end; `existing` writes only the cluster
file and links our app into a portal that is already there; `off` disables the component.

Every install detail below cites the OOD documentation (https://osc.github.io/ood-documentation/latest/, v4.2) or the
v4.2.5 source (github.com/OSC/ondemand, tag v4.2.5; paths are relative to the repository root). Secrets (the OIDC client
secret, the cookie passphrase, the Dex users and their password hashes, an LDAP bind password) never live in this
repository or the site file: the site file names root-only files and `files` reads them when it renders.
"""

from __future__ import annotations

import importlib
import ipaddress
import json
import re
import stat
import uuid
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

import render
from render import FILE_MODE, SECRET_MODE, TEMPLATES, FileSpec
from site_config import AuthMode, MetricsMode, PortalMode

if TYPE_CHECKING:
    from site_config import Site

NAME = "portal"

TEMPLATE_DIR = TEMPLATES / "portal"
SECRETS_DIR = Path("/etc/gpujobs/portal")  # default home of the root-only secret files
PORTAL_YML = "/etc/ood/config/ood_portal.yml"
CLUSTER_DIR = "/etc/ood/config/clusters.d"
SSL_DIR = Path("/etc/ood/ssl")
SELF_SIGNED_CERT = SSL_DIR / "portal.crt"
SELF_SIGNED_KEY = SSL_DIR / "portal.key"
CA_TRUST_COPY = "/usr/local/share/ca-certificates/gpujobs-portal.crt"
APP_NAME = "gpujobs"
APP_LINK = f"/var/www/ood/apps/sys/{APP_NAME}"
APP_SOURCE = Path("slurm") / "ood_app"  # under site.install_dir; the tools component installs it
UPDATE_OOD_PORTAL = "/opt/ood/ood-portal-generator/sbin/update_ood_portal"  # ondemand-dex postinst, line 3
NGINX_STAGE = "/opt/ood/nginx_stage/sbin/nginx_stage"
SECRET_FILE_MODE_MASK = stat.S_IRWXG | stat.S_IRWXO

# Package repository: docs "1. Install Software", Ubuntu 24.04 tab. Its release .deb
# (https://apt.osc.edu/ondemand/4.2/ondemand-release-web_4.2.0-noble_all.deb) carries exactly these two source lines
# and keys: ondemand-web.gpg (3D4C D7E7 D4BC 4526 3AC1 1388 C2AD B19A FD77 5498) and nodesource.gpg (6F71 F525 2828
# 41EE DAF8 51B4 2F59 B5F9 9B1B E0B4); the node source is needed because `ondemand` depends on nodejs 22.
# The .asc files in templates/portal are those keys, armored.
OOD_KEYRING = "/usr/share/keyrings/gpujobs-ondemand-web.asc"
NODE_KEYRING = "/usr/share/keyrings/gpujobs-nodesource.asc"
OOD_LIST = "/etc/apt/sources.list.d/gpujobs-ondemand-web.list"
NODE_LIST = "/etc/apt/sources.list.d/gpujobs-nodesource.list"

# Dex's built-in account, dex.rb:257-261 (generated only by `update_ood_portal --insecure` when no users are given).
DEFAULT_EMAIL = "ood@localhost"
DEFAULT_USER_ID = "08a8684b-db88-4b73-90a9-3cd1661f5466"
DEFAULT_HASH = "$2a$10$2b2cU8CPhOTaGrs1HRQuAueS7JTT5ZHsHSzYiFPm1leZck7Mc8T4W"

# Per mode: required and optional keys of [portal.auth_settings]; any other key is refused (typos must not pass).
_COMMON_OPTIONAL = {"crypto_passphrase_file"}
_AUTH_KEYS: dict[AuthMode, tuple[set[str], set[str]]] = {
    AuthMode.OIDC: (
        {"issuer", "client_id", "client_secret_file"},
        {"remote_user_claim", "scope", "user_map_match"} | _COMMON_OPTIONAL,
    ),
    AuthMode.DEX_LDAP: (
        {"host", "user_base_dn"},
        {"bind_dn", "bind_pw_file", "user_filter", "username_attr", "email_attr", "root_ca", "start_tls"}
        | _COMMON_OPTIONAL,
    ),
    AuthMode.DEX_LOCAL: ({"users_file"}, _COMMON_OPTIONAL),
}

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HOST_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$")
_PATH_RE = re.compile(r"^/[A-Za-z0-9._/@+-]+$")
_ID_RE = re.compile(r"^[A-Za-z0-9._:@-]+$")
_URL_RE = re.compile(r"^https://[A-Za-z0-9.-]+(:[0-9]+)?(/[A-Za-z0-9._~/-]*)?$")
_SCOPE_RE = re.compile(r"^[A-Za-z0-9._:-]+( [A-Za-z0-9._:-]+)*$")
_MATCH_RE = re.compile(r"^[^\"\\\r\n]+$")
_TOKEN_RE = re.compile(r"^\S{16,}$")
_SECRET_RE = re.compile(r"^[^\s\"']+$")
_UNIX_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
_BCRYPT_RE = re.compile(r"^\$2[aby]\$[0-9]{2}\$[./A-Za-z0-9]{53}$")
_LDAP_VALUE_RE = re.compile(r"^[^\x00-\x1f]+$")
_LDAP_HOST_RE = re.compile(r"^[A-Za-z0-9.-]+:[0-9]+$")
_USER_KEYS = {"username", "email", "hash", "userID"}


def enabled(site: Site) -> bool:
    return site.portal.mode in (PortalMode.INSTALL, PortalMode.EXISTING)


def _installing(site: Site) -> bool:
    return site.portal.mode is PortalMode.INSTALL


def _dex(site: Site) -> bool:
    return site.portal.auth in (AuthMode.DEX_LDAP, AuthMode.DEX_LOCAL)


def _self_signed(site: Site) -> bool:
    return site.portal.tls_cert is None and site.portal.tls_key is None


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _quote(value: Any) -> str:
    """A YAML scalar. JSON strings are valid YAML double-quoted scalars, so no YAML library is needed."""
    return json.dumps(value)


def _yaml(value: Any, indent: int) -> list[str]:
    """Block-style YAML lines for a nested dict/list of scalars, each line indented by `indent` spaces."""
    pad = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict | list):
                lines.append(f"{pad}{key}:")
                lines.extend(_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_quote(item)}")
    else:
        for item in value:
            if isinstance(item, dict):
                first, *rest = _yaml(item, indent + 2)
                lines.append(f"{pad}- {first.lstrip()}")
                lines.extend(rest)
            else:
                lines.append(f"{pad}- {_quote(item)}")
    return lines


def _check_mode(path: Path, what: str, problems: list[str]) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        problems.append(f"{what} {path} does not exist or cannot be read")
        return False
    if mode & SECRET_FILE_MODE_MASK:
        problems.append(f"{what} {path} must be root-only: run `sudo chmod 0600 {path}`")
        return False
    return True


def _secret(path_text: str, what: str, pattern: re.Pattern[str], problems: list[str]) -> str | None:
    """The stripped content of a root-only secret file, or None with the reason recorded."""
    path = Path(path_text)
    if not _PATH_RE.match(path_text):
        problems.append(f"{what}: {path_text!r} is not an absolute path of plain characters")
        return None
    if not _check_mode(path, what, problems):
        return None
    text = _read(path)
    if not text or not pattern.match(text):
        problems.append(f"{what} {path} is empty or has an unusable value (the value is not shown)")
        return None
    return text


def _passphrase(settings: dict[str, str], problems: list[str]) -> str | None:
    # update_ood_portal needs oidc_crypto_passphrase for every OIDC/Dex login (docs "OpenID Connect" and
    # "OpenID Connect with Dex": "now requires oidc_crypto_passphrase"; view.rb:128-134 raises without it).
    path = settings.get("crypto_passphrase_file", str(SECRETS_DIR / "crypto_passphrase"))
    value = _secret(path, "cookie passphrase file", _TOKEN_RE, problems)
    if value is None:
        problems.append(f"create it with: openssl rand -hex 32 | sudo install -m 0600 /dev/stdin {path}")
    return value


def _users(path_text: str, problems: list[str]) -> list[dict[str, str]]:
    """The Dex local users from the root-only JSON file: [{"username","email","hash","userID"}, ...]."""
    path = Path(path_text)
    if not _PATH_RE.match(path_text) or not _check_mode(path, "users file", problems):
        return []
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        problems.append(f"users file {path} is not valid JSON (a list of username/email/hash/userID objects)")
        return []
    if not isinstance(raw, list) or not raw:
        problems.append(
            f"users file {path} must hold a non-empty JSON list; with no users Dex would fall back to ood@localhost"
        )
        return []
    users: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        where = f"users file entry {i}"
        if (
            not isinstance(entry, dict)
            or set(entry) != _USER_KEYS
            or not all(isinstance(v, str) for v in entry.values())
        ):
            problems.append(f"{where} must have exactly the string keys {', '.join(sorted(_USER_KEYS))}")
            continue
        if (
            entry["email"].lower() == DEFAULT_EMAIL
            or entry["userID"].lower() == DEFAULT_USER_ID
            or entry["hash"] == DEFAULT_HASH
        ):
            problems.append(f"{where} is Dex's default account (ood@localhost / password); remove it")
            continue
        local, _, domain = entry["email"].partition("@")
        if not _UNIX_USER_RE.match(local) or not domain or not _HOST_RE.match(domain):
            problems.append(f"{where}: email must be <unix user>@<domain>; the part before @ becomes the Linux user")
            continue
        if not _BCRYPT_RE.match(entry["hash"]):
            problems.append(f"{where}: hash must be a bcrypt hash ($2a$/$2b$/$2y$, 60 characters), not a password")
            continue
        try:
            uuid.UUID(entry["userID"])
        except ValueError:
            problems.append(f"{where}: userID must be a UUID")
            continue
        if not _ID_RE.match(entry["username"]) or entry["email"].lower() in seen:
            problems.append(f"{where}: username has unusable characters, or the email is listed twice")
            continue
        seen.add(entry["email"].lower())
        users.append(entry)
    return users


def _check_keys(site: Site, problems: list[str]) -> None:
    required, optional = _AUTH_KEYS[site.portal.auth]
    given = set(site.portal.auth_settings)
    if unknown := sorted(given - required - optional):
        problems.append(f"[portal.auth_settings] has unknown key(s) for auth {site.portal.auth}: {', '.join(unknown)}")
    if missing := sorted(required - given):
        problems.append(f"[portal.auth_settings] needs {', '.join(missing)} for auth {site.portal.auth}")


def _oidc_block(site: Site, problems: list[str]) -> list[str]:
    s = site.portal.auth_settings
    issuer, client_id = s.get("issuer", ""), s.get("client_id", "")
    claim, scope = s.get("remote_user_claim", "preferred_username"), s.get("scope", "openid profile email")
    # Plain HTTP is never configured for logins: the issuer must be https (docs "OpenID Connect" example).
    if not _URL_RE.match(issuer):
        problems.append("auth_settings.issuer must be an https:// URL of plain characters")
    if not _ID_RE.match(client_id):
        problems.append("auth_settings.client_id has unusable characters")
    if not re.match(r"^[A-Za-z0-9_.:-]+$", claim) or not _SCOPE_RE.match(scope):
        problems.append("auth_settings.remote_user_claim or scope has unusable characters")
    secret = _secret(s.get("client_secret_file", ""), "client secret file", _SECRET_RE, problems)
    block = {
        "oidc_uri": "/oidc",
        "oidc_provider_metadata_url": f"{issuer.rstrip('/')}/.well-known/openid-configuration",
        "oidc_client_id": client_id,
        "oidc_client_secret": secret or "",
        "oidc_remote_user_claim": claim,
        "oidc_scope": scope,
        # docs "OpenID Connect" example: logout through the provider.
        "logout_redirect": f"/oidc?logout=https%3A%2F%2F{site.portal.server_name}",
    }
    # User mapping, docs "Setup User Mapping": REMOTE_USER (from the claim above) maps to the Linux user; the default
    # user_map_match '.*' takes it unchanged; a site whose claim is an email sets a lua pattern here.
    if match := s.get("user_map_match"):
        if _MATCH_RE.match(match):
            block["user_map_match"] = match
        else:
            problems.append("auth_settings.user_map_match has quotes, backslashes or newlines")
    return _yaml(block, 0)


def _ldap_block(site: Site, problems: list[str]) -> list[str]:
    s = site.portal.auth_settings
    host = s.get("host", "")
    if not _LDAP_HOST_RE.match(host):
        problems.append("auth_settings.host must be <host>:<port> (use the LDAPS port; plain LDAP is not offered)")
    for key in ("user_base_dn", "bind_dn", "user_filter", "username_attr", "email_attr"):
        if key in s and not _LDAP_VALUE_RE.match(s[key]):
            problems.append(f"auth_settings.{key} has control characters")
    config: dict[str, Any] = {"host": host, "insecureSkipVerify": False}
    if s.get("start_tls", "false").lower() == "true":
        config["startTLS"] = True
    if root_ca := s.get("root_ca"):
        if _PATH_RE.match(root_ca):
            config["rootCA"] = root_ca
        else:
            problems.append("auth_settings.root_ca is not an absolute path of plain characters")
    if bind_dn := s.get("bind_dn"):
        config["bindDN"] = bind_dn
        # docs "Configuring OnDemand Dex for LDAP": a bindPW makes ood_portal.yml root-only (0600), as it is here.
        if pw_file := s.get("bind_pw_file"):
            if pw := _secret(pw_file, "bind password file", re.compile(r"^.+$"), problems):
                config["bindPW"] = pw
        else:
            problems.append("auth_settings.bind_dn needs bind_pw_file")
    username = s.get("username_attr", "uid")
    config["userSearch"] = {
        "baseDN": s.get("user_base_dn", ""),
        "filter": s.get("user_filter", "(objectClass=posixAccount)"),
        "username": username,
        "idAttr": username,
        "emailAttr": s.get("email_attr", "mail"),
        "nameAttr": "gecos",
        "preferredUsernameAttr": username,
    }
    # User mapping: with a connector the claim defaults to preferred_username (view.rb:117), which the connector
    # fills from preferredUsernameAttr, so REMOTE_USER is the LDAP uid and maps to the Linux user of that name
    # (docs "Setup User Mapping", "Remote User"). dex.rb:311 sets the email claim only when there are no connectors.
    return _yaml(
        {
            "oidc_remote_user_claim": "preferred_username",
            "dex": {"connectors": [{"type": "ldap", "id": "ldap", "name": "LDAP", "config": config}]},
        },
        0,
    )


def _local_block(site: Site, problems: list[str]) -> list[str]:
    users = _users(site.portal.auth_settings.get("users_file", ""), problems)
    # `dex.static_passwords` is read by dex.rb:252-270 (entries with `hash` are kept as given, `password` would be
    # hashed) and used only under `update_ood_portal --insecure` (dex.rb:41-46); see post_install.
    # User mapping, docs "Setup User Mapping" > "Dex Automatic Configuration": with no connectors the email claim is
    # used and user_map_match becomes ^([^@]+)@.*$ (dex.rb:311-314), so alice@localhost is the Linux user alice.
    return _yaml({"dex": {"static_passwords": users}}, 0) if users else []


def _tls(site: Site, problems: list[str]) -> tuple[Path, Path]:
    cert, key = site.portal.tls_cert, site.portal.tls_key
    if _self_signed(site):
        return SELF_SIGNED_CERT, SELF_SIGNED_KEY
    if cert is None or key is None:
        problems.append(
            "set both portal.tls_cert and portal.tls_key, or neither (a self-signed pair is then generated)"
        )
        return SELF_SIGNED_CERT, SELF_SIGNED_KEY
    for path in (cert, key):
        if not _PATH_RE.match(str(path)) or not path.is_file():
            problems.append(f"TLS file {path} is not an existing file with a plain absolute path")
    return cert, key


def _server_name(site: Site, problems: list[str]) -> str:
    name = site.portal.server_name
    if not name or not _HOST_RE.match(name):
        problems.append("portal.server_name must be set to the host name (or IP) the portal answers to")
    return name


def _session_lines(site: Site) -> str:
    """One sign-in for `session_days`: mod_auth_openidc keeps the session in an encrypted cookie that survives browser
    restarts (`:persistent`, its expiry is the inactivity timeout), not in Apache's shared memory, which a restart or
    reboot empties (auth_openidc.conf, OIDCSessionType and OIDCCacheType). The cookie is encrypted with the
    site's stable crypto passphrase. 0 keeps OOD's own 8-hour browser session (ood_portal_example.yml)."""
    if not site.portal.session_days:
        return ""
    seconds = site.portal.session_days * 86400
    return (f"oidc_session_inactivity_timeout: {seconds}\noidc_session_max_duration: {seconds}\n"
            'oidc_settings:\n  OIDCSessionType: "client-cookie:persistent"\n')


def _build_portal_yml(site: Site) -> tuple[str, list[str]]:
    """The ood_portal.yml text and every reason it cannot be applied."""
    problems: list[str] = []
    servername = _server_name(site, problems)
    cert, key = _tls(site, problems)
    _check_keys(site, problems)
    passphrase = _passphrase(site.portal.auth_settings, problems)
    builders = {AuthMode.OIDC: _oidc_block, AuthMode.DEX_LDAP: _ldap_block, AuthMode.DEX_LOCAL: _local_block}
    auth_lines = builders[site.portal.auth](site, problems)
    # docs "4. Secure Apache httpd": servername plus ssl directives; the HTTP vhost then redirects to https.
    text = Template((TEMPLATE_DIR / "ood_portal.yml").read_text()).substitute(
        servername=_quote(servername),
        root_uri=_quote(f"/pun/sys/{APP_NAME}/"),
        ssl_lines="\n".join(_yaml([f'SSLCertificateFile "{cert}"', f'SSLCertificateKeyFile "{key}"'], 2)),
        hsts="false" if _self_signed(site) else "true",
        passphrase=_quote(passphrase or ""),
        session_lines=_session_lines(site),
        auth_block="\n".join(auth_lines),
    )
    return text, problems


def _metrics_module() -> Any | None:
    """The metrics component module if the installer has one, else None (the Grafana block is then left out)."""
    candidates = [f"{__package__}.metrics"] if __package__ else []
    for candidate in [*candidates, "metrics"]:
        try:
            return importlib.import_module(candidate)
        except ModuleNotFoundError as exc:
            if exc.name != candidate:
                raise
    return None


def _grafana(site: Site) -> dict[str, Any] | None:
    """OOD's `custom.grafana` block (docs "Customizations" > "Grafana support"), or None."""
    if site.metrics.mode is MetricsMode.OFF:
        return None
    dashboard = getattr(_metrics_module(), "GRAFANA_DASHBOARD", None)
    host = site.metrics.grafana_url or (
        f"http://{site.portal.server_name}:{site.metrics.grafana_port}" if site.portal.server_name else ""
    )
    if not dashboard or not host:
        return None
    return {"host": host, **dashboard}


def _cluster_file(site: Site) -> FileSpec:
    grafana = _grafana(site)
    custom = "\n".join(["  custom:", "    grafana:", *_yaml(grafana, 6)]) if grafana else ""
    template = Template((TEMPLATE_DIR / "cluster.yml").read_text())
    text = template.substitute(title=_quote(site.cluster_name), custom_block=custom)
    return FileSpec(text.rstrip("\n") + "\n", FILE_MODE, "root")


def refusals(site: Site) -> list[str]:
    if not enabled(site):
        return []
    problems: list[str] = []
    if not _NAME_RE.match(site.cluster_name):
        problems.append(f"cluster name {site.cluster_name!r} cannot name a cluster file")
    if _installing(site):
        problems.extend(_build_portal_yml(site)[1])
    return problems


def files(site: Site) -> dict[str, FileSpec]:
    if not enabled(site):
        return {}
    out: dict[str, FileSpec] = {}
    if _NAME_RE.match(site.cluster_name):  # OOD's Slurm docs: the cluster file is named after slurm's ClusterName
        out[f"{CLUSTER_DIR}/{site.cluster_name}.yml"] = _cluster_file(site)
    if _installing(site):
        text, problems = _build_portal_yml(site)
        if not problems:  # never write a config that is missing a secret, whatever the caller checked
            out[PORTAL_YML] = FileSpec(text, SECRET_MODE, "root")
        if filtered := render.ip_filter_lines(site.portal.allow_from):
            # Apache serves the portal; Dex (behind it) also has a metrics port on every address.
            dropin = f"# Rendered from the site file by slurm/install/components/portal.py: [portal] allow_from.\n[Service]\n{filtered}"
            for unit in ["apache2", *(["ondemand-dex"] if _dex(site) else [])]:
                out[f"/etc/systemd/system/{unit}.service.d/10-gpujobs.conf"] = FileSpec(dropin, FILE_MODE, "root")
    return out


#: Where the machine's apt sources are (patched in tests).
APT_DIR = Path("/etc/apt")


def _configured_elsewhere(uri: str, own: str) -> bool:
    """Whether an apt source file other than our `own` one already names `uri`.

    apt refuses one source defined twice with different signing keys ("Conflicting values set for option Signed-By"),
    and every later `apt-get update` on the machine fails: measured on gpu-node-1 on 2026-10-19, where NodeSource
    had been added by hand (`nodesource.sources`). Such a source is then used as it is.
    """
    sources = APT_DIR / "sources.list.d"
    for path in [APT_DIR / "sources.list", *sorted(sources.glob("*.list")), *sorted(sources.glob("*.sources"))]:
        if path.name == Path(own).name:
            continue
        try:
            if uri in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def apt_sources(site: Site) -> dict[str, FileSpec]:
    if not _installing(site):
        return {}
    out: dict[str, FileSpec] = {}
    for name, keyring, listing in (("ondemand-web", OOD_KEYRING, OOD_LIST), ("nodesource", NODE_KEYRING, NODE_LIST)):
        line = Template((TEMPLATE_DIR / f"{name}.list").read_text()).substitute(keyring=keyring)
        uri = line.split("] ", 1)[1].split()[0]
        if _configured_elsewhere(uri, listing):
            continue
        out[keyring] = FileSpec((TEMPLATE_DIR / f"{name}.asc").read_text(), FILE_MODE, "root")
        out[listing] = FileSpec(line, FILE_MODE, "root")
    return out


def packages(site: Site) -> list[str]:
    if not _installing(site):
        return []
    # `ondemand` pulls apache2, the per-user nginx, passenger and nodejs 22 (Packages index, apt.osc.edu 4.2 noble);
    # `ondemand-dex` depends on libapache2-mod-auth-openidc, which the plain OIDC mode needs directly.
    # python-is-python3: Passenger starts Python apps through nginx_stage/bin/python:11-14, which runs
    # `python` when the app has no bin/python, and Ubuntu 24.04 has no `python` without this package.
    names = ["apt-transport-https", "ca-certificates", "ondemand", "python-is-python3"]
    names += ["ondemand-dex"] if _dex(site) else []
    names += ["libapache2-mod-auth-openidc"] if site.portal.auth is AuthMode.OIDC else []
    names += ["openssl"] if _self_signed(site) else []
    return names


def post_install(site: Site) -> list[list[str]]:
    link = ["ln", "-sfT", str(site.install_dir / APP_SOURCE), APP_LINK]
    if not _installing(site):
        return [link] if enabled(site) else []
    steps: list[list[str]] = []
    if _self_signed(site):
        san = f"IP:{site.portal.server_name}" if _is_ip(site.portal.server_name) else f"DNS:{site.portal.server_name}"
        script = (TEMPLATE_DIR / "self_signed.sh").read_text()
        steps.append(
            ["sh", "-c", script, "sh", str(SELF_SIGNED_CERT), str(SELF_SIGNED_KEY), site.portal.server_name, san]
        )
        # mod_auth_openidc fetches the provider metadata from this very host over https (view.rb:137-139 gives the
        # Dex issuer an https scheme), so the self-signed certificate must be trusted by the host (docs
        # "4. Secure Apache httpd", step 2 "Update CA (Dex Users only)", Ubuntu & Debian tab).
        steps.append(["install", "-m", "0644", str(SELF_SIGNED_CERT), CA_TRUST_COPY])
        steps.append(["update-ca-certificates"])
    # packaging/deb/postinst:17-23 enables rewrite, lua, headers, proxy, proxy_http, proxy_wstunnel but not ssl; the
    # auth_openidc.load file ships with libapache2-mod-auth-openidc (packages.ubuntu.com, noble filelist).
    steps += [["a2enmod", "ssl"], ["a2enmod", "auth_openidc"]]
    # --force: replace ood-portal.conf even if its checksum differs (the old file is backed up first,
    # application.rb:213-218). --insecure is what makes Dex use the users in ood_portal.yml at all
    # (application.rb:354, dex.rb:41-46); without it Dex's password database stays off. It is passed only for the
    # local-users mode, and `files`/`refusals` have already guaranteed the default account is not in the list.
    update = [UPDATE_OOD_PORTAL, "--force"]
    steps.append([*update, "--insecure"] if site.portal.auth is AuthMode.DEX_LOCAL else update)
    steps.append(["systemctl", "daemon-reload"])  # the allow_from drop-ins
    if _dex(site):
        steps += [["systemctl", "enable", "ondemand-dex"], ["systemctl", "restart", "ondemand-dex"]]
    steps += [["systemctl", "enable", "apache2"], ["systemctl", "restart", "apache2"]]
    # Each user's running web server keeps the environment Apache gave it at start, including OOD_ALLOWED_HOSTS
    # (the server name): after a rename the dashboard refused every page with "Blocked hosts" (measured 2026-10-19).
    # Stopping them all makes the next request start one with the current settings; jobs are not touched.
    steps += [[NGINX_STAGE, "nginx_clean", "--force"], link]
    return steps


def _is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True

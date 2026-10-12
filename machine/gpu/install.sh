#!/usr/bin/env bash
# Install the recorder as the invoking user and the common root-owned job lock.
set -euo pipefail
if [[ $EUID -ne 0 || -z ${SUDO_USER:-} || $SUDO_USER == root ]]; then
    echo 'Run with sudo bash machine/gpu/install.sh from your own account.' >&2
    exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
account_home="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
runtime="$account_home/.local/bin/uv"
[[ -x "$runtime" ]] || { echo "uv not found at $runtime" >&2; exit 1; }
printf 'f /run/lock/preflight-gpu.lock 0644 root root -\n' > /etc/tmpfiles.d/preflight-gpu.conf
systemd-tmpfiles --create /etc/tmpfiles.d/preflight-gpu.conf
cat > /etc/systemd/system/preflight-gpu-usage.service <<EOF
[Unit]
Description=Preflight GPU usage and process ownership recorder
After=local-fs.target

[Service]
User=$SUDO_USER
ExecStart="$runtime" run --offline --no-project --python /usr/bin/python3 python "$HERE/record_gpu_usage.py" --log /var/log/preflight/gpu.jsonl
Environment=UV_CACHE_DIR=/var/cache/preflight
LogsDirectory=preflight
LogsDirectoryMode=0750
CacheDirectory=preflight
UMask=0027
Restart=on-failure
RestartSec=5
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=yes
MemoryMax=128M
CPUQuota=10%
OOMScoreAdjust=-250

[Install]
WantedBy=multi-user.target
EOF
systemd-analyze verify /etc/systemd/system/preflight-gpu-usage.service
systemctl daemon-reload
systemctl enable preflight-gpu-usage.service
systemctl restart preflight-gpu-usage.service
systemctl --no-pager status preflight-gpu-usage.service

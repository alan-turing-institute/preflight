#!/usr/bin/env bash
# Install GPU records and cache-first earlyoom, retaining its memory limits.
set -euo pipefail
if [[ $EUID -ne 0 || -z ${SUDO_USER:-} || $SUDO_USER == root ]]; then
    echo 'Run with sudo bash machine/gpu/install.sh from your own account.' >&2
    exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
account_home="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
runtime="$account_home/.local/bin/uv"
[[ -x "$runtime" ]] || { echo "uv not found at $runtime" >&2; exit 1; }
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
install -d -m 0755 /usr/local/lib/preflight /var/cache/preflight-oom
install -m 0644 "$HERE/../earlyoom/cache_first_oom.py" /usr/local/lib/preflight/cache_first_oom.py
install -m 0755 "$runtime" /usr/local/lib/preflight/uv
# Validate a bounded reclaim under the same permissions before replacing earlyoom.
# This check never starts an OOM killer or allocates GPU memory.
systemd-run --quiet --wait --pipe --collect \
    -p EnvironmentFile=/etc/default/earlyoom \
    -p Environment=UV_CACHE_DIR=/var/cache/preflight-oom \
    -p ProtectSystem=strict -p ProtectHome=yes -p NoNewPrivileges=yes \
    -p ReadWritePaths=/var/cache/preflight-oom -p MemoryMax=128M -p TimeoutStopSec=5s \
    -p 'CapabilityBoundingSet=CAP_KILL CAP_IPC_LOCK CAP_SYS_NICE CAP_SYS_RESOURCE CAP_SYS_ADMIN' \
    /usr/local/lib/preflight/uv run --offline --no-project --python /usr/bin/python3 \
    python /usr/local/lib/preflight/cache_first_oom.py --test-reclaim
install -d /etc/systemd/system/earlyoom.service.d
cat > /etc/systemd/system/earlyoom.service.d/20-cache-first.conf <<'EOF'
[Unit]
Wants=preflight-gpu-usage.service
After=preflight-gpu-usage.service

[Service]
User=root
DynamicUser=no
ExecStart=
ExecStart=/usr/local/lib/preflight/uv run --offline --no-project --python /usr/bin/python3 python /usr/local/lib/preflight/cache_first_oom.py
Environment=UV_CACHE_DIR=/var/cache/preflight-oom
CacheDirectory=preflight-oom
MemoryMax=128M
OOMScoreAdjust=-250
Nice=-20
NoNewPrivileges=yes
AmbientCapabilities=
CapabilityBoundingSet=CAP_KILL CAP_IPC_LOCK CAP_SYS_NICE CAP_SYS_RESOURCE CAP_SYS_ADMIN
TimeoutStopSec=5s
EOF
systemd-analyze verify /etc/systemd/system/preflight-gpu-usage.service /usr/lib/systemd/system/earlyoom.service
systemctl daemon-reload
systemctl enable preflight-gpu-usage.service
systemctl restart preflight-gpu-usage.service
systemctl restart earlyoom.service
systemctl --no-pager status preflight-gpu-usage.service earlyoom.service

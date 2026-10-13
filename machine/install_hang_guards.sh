#!/bin/bash
# Install, start and CHECK the two GB10 hang guards on this box (run with sudo):
#   preflight-keep-free.service  keeps >= 16 GiB truly free by dropping clean page cache
#   preflight-heartbeat.service  reboots the box if memory stays stalled >= 10 % for 300 s
#                                or the heartbeat cannot run at all (WatchdogSec=360)
# Each script's docstring holds the evidence and the validation. Idempotent: re-running
# replaces the files and restarts both services. Exits non-zero, naming the failed check,
# if either service is not demonstrably working at the end.
#   Undo: systemctl disable --now preflight-heartbeat preflight-keep-free
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'run with sudo' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB=/usr/local/lib/preflight
UNITS=/etc/systemd/system
install -d -m 0755 "$LIB"
install -m 0644 "$HERE/memory/keep_free.py" "$LIB/keep_free.py"
install -m 0644 "$HERE/watchdog/heartbeat.py" "$LIB/heartbeat.py"

cat > "$UNITS/preflight-keep-free.service" <<EOF
[Unit]
Description=preflight: keep truly free memory for the GPU driver (drops clean page cache)

[Service]
ExecStart=/usr/bin/python3 $LIB/keep_free.py
Restart=always
RestartSec=5
Nice=10
OOMScoreAdjust=-900

[Install]
WantedBy=multi-user.target
EOF

# FailureAction reboots on ANY failure, including at start, so ExecCondition skips the unit
# (skipped, not failed) if its interpreter or script is missing: no reboot loop at boot.
cat > "$UNITS/preflight-heartbeat.service" <<EOF
[Unit]
Description=preflight: reboot this box if it wedges (systemd service watchdog)

[Service]
Type=notify
NotifyAccess=main
ExecCondition=/usr/bin/test -x /usr/bin/python3 -a -r $LIB/heartbeat.py -a -r /proc/pressure/memory
ExecStart=/usr/bin/python3 $LIB/heartbeat.py
WatchdogSec=360
TimeoutAbortSec=30
FailureAction=reboot-immediate
StateDirectory=preflight-heartbeat
OOMScoreAdjust=-1000

[Install]
WantedBy=multi-user.target
EOF

systemd-analyze verify "$UNITS/preflight-keep-free.service" "$UNITS/preflight-heartbeat.service"
systemctl daemon-reload
systemctl enable preflight-keep-free.service preflight-heartbeat.service
systemctl restart preflight-keep-free.service preflight-heartbeat.service

# --- checks: each must PASS, or the install reports failure -------------------------
fail=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }
stamp() { systemctl show preflight-heartbeat -p WatchdogTimestampMonotonic --value; }
first=$(stamp)
sleep 12   # one heartbeat interval (10 s) plus margin
check "keep-free running" "systemctl is-active -q preflight-keep-free"
check "keep-free started its loop" \
    "journalctl -u preflight-keep-free -b --no-pager -o cat | grep -q '\"event\": \"started\"'"
check "keep-free renamed (out of earlyoom's python rule)" \
    "[[ \$(cat /proc/\$(systemctl show preflight-keep-free -p MainPID --value)/comm) == pf-keep-free ]]"
check "heartbeat running" "systemctl is-active -q preflight-heartbeat"
check "heartbeat reached systemd (watchdog timestamp advanced)" "[[ \$(stamp) -gt $first ]]"
check "heartbeat renamed (out of earlyoom's python rule)" \
    "[[ \$(cat /proc/\$(systemctl show preflight-heartbeat -p MainPID --value)/comm) == pf-heartbeat ]]"
check "heartbeat failure action is reboot" \
    "[[ \$(systemctl show preflight-heartbeat -p FailureAction --value) == reboot-immediate ]]"
exit $fail

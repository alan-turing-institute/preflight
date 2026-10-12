#!/bin/bash
# Run with sudo on each DGX: reboot when the kernel detects it is stuck, and let the invoking
# user read the system journal (kernel log) without sudo, so a freeze can be diagnosed.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install -m 644 "$D/98-reboot-when-stuck.conf" /etc/sysctl.d/98-reboot-when-stuck.conf
sysctl --system >/dev/null
if [[ -n "${SUDO_USER:-}" ]] && getent group systemd-journal >/dev/null; then
    usermod -aG systemd-journal "$SUDO_USER"
fi
for k in panic panic_on_rcu_stall softlockup_panic hung_task_panic hung_task_timeout_secs; do
    echo "kernel.$k = $(sysctl -n kernel.$k)"
done
echo "groups of ${SUDO_USER:-root}: $(id -nG "${SUDO_USER:-root}")"

#!/usr/bin/env bash
# Run on the GPU host, around the foreground job (inside its systemd ExecStart).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu_args=()
while [[ $# -gt 0 && "$1" != -- ]]; do gpu_args+=("$1"); shift; done
if [[ $# -lt 2 ]]; then
    echo 'usage: run_gpu_job.sh [GPU gate arguments] -- command [arguments...]' >&2
    exit 2
fi
shift
lock="${PREFLIGHT_GPU_LOCK:-/run/lock/preflight-gpu.lock}"
if [[ ! -f "$lock" || -L "$lock" ]]; then
    echo '[gpu] Shared lock missing; run machine/gpu/install.sh first.' >&2
    exit 1
fi
# flock's parent holds the lock even when uv/Python closes inherited descriptors.
# flock opens the existing root-owned file read-only; participants cannot edit it.
runtime="${PREFLIGHT_UV:-uv}"
if ! command -v "$runtime" >/dev/null; then runtime="$HOME/.local/bin/uv"; fi
exec flock -n -E 75 -o "$lock" bash -c '
    set -euo pipefail
    checker=$1; runtime=$2; shift 2
    gate_args=()
    while [[ "$1" != -- ]]; do gate_args+=("$1"); shift; done
    shift
    "$runtime" run --offline --no-project --python /usr/bin/python3 python "$checker" "${gate_args[@]}"
    exec "$@"
' preflight "$HERE/assert_gpu_free.py" "$runtime" "${gpu_args[@]}" -- "$@"

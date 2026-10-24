#!/usr/bin/env bash
# Integration test cluster driver: run.sh up | down | smoke | e2e [ARGS] | shell [-u USER] [CMD...]
# Every command that talks to slurm inside the container lives here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# GPUJOBS_TEST_PROJECT: the compose project, so several test clusters can run at once (one per stage group).
COMPOSE=(docker compose -p "${GPUJOBS_TEST_PROJECT:-gpujobs-test}" -f "$HERE/compose.yml")
SERVICE=cluster
READY_TIMEOUT=240

# Run a command in the container; $1 is the user, the rest the command. Starts in the user's home (batch output files
# land in the working directory, and / is not writable for a plain user).
in_cluster() {
    local user=$1; shift
    local home=/root
    [[ $user == root ]] || home=/home/$user
    "${COMPOSE[@]}" exec -T -u "$user" -w "$home" "$SERVICE" "$@"
}

wait_ready() {
    local waited=0 state=""
    while (( waited < READY_TIMEOUT )); do
        state=$(in_cluster root sinfo -h -N -o '%T' 2>/dev/null | head -n1 || true)
        case "$state" in
            idle|mixed|allocated) echo "cluster ready after ${waited}s: node $state"; return 0 ;;
        esac
        sleep 3; waited=$((waited + 3))
    done
    echo "cluster not ready after ${READY_TIMEOUT}s (node state: ${state:-none})" >&2
    in_cluster root systemctl --no-pager --failed >&2 || true
    return 1
}

# `docker exec` puts its process in the container's init cgroup. If one lands there before systemd has moved itself into
# init.scope and enabled the cpu/memory/cpuset controllers for its children, the kernel refuses ("device or resource
# busy": a cgroup with processes cannot enable controllers) and slurmd then dies at boot with "memory cgroup controller
# is not available". So nothing is exec'd until the container's init is in init.scope with the memory controller on
# (read from the host's /proc and cgroup files, which does not touch the container).
wait_booted() {
    local pid cgroup waited=0
    pid=$(docker inspect -f '{{.State.Pid}}' "$("${COMPOSE[@]}" ps -q "$SERVICE")")
    while (( waited < READY_TIMEOUT * 2 )); do
        cgroup=$(cut -d: -f3 "/proc/$pid/cgroup" 2>/dev/null || true)
        if [[ $cgroup == */init.scope ]] && grep -qw memory "/sys/fs/cgroup${cgroup%/init.scope}/cgroup.subtree_control" 2>/dev/null; then
            return 0
        fi
        sleep 0.5; waited=$((waited + 1))
    done
    echo "systemd did not enable the cgroup controllers (init cgroup: ${cgroup:-none})" >&2
    return 1
}

cmd_up() {
    "${COMPOSE[@]}" up -d --build
    wait_booted
    wait_ready
}

cmd_down() {
    "${COMPOSE[@]}" down --volumes --remove-orphans
}

# As alice: a trivial job shows in squeue, then a two-GPU job runs and is accounted.
cmd_smoke() {
    local alice=alice id gid state="" alloc gpu_used
    echo "== as alice: submit a trivial job, see it in squeue"
    id=$(in_cluster "$alice" sbatch --parsable --job-name=smoke-trivial --wrap 'sleep 30')
    echo "submitted job $id"
    in_cluster "$alice" squeue -h -j "$id" -o '%i %u %T %j' | tee /dev/stderr | grep -q "^$id alice " \
        || { echo "FAIL: job $id not in squeue" >&2; return 1; }
    in_cluster "$alice" scancel "$id"

    echo "== as alice: a job asking for gpu:2 runs"
    gid=$(in_cluster "$alice" sbatch --parsable --job-name=smoke-gpu2 --gres=gpu:2 --wrap 'sleep 5')
    echo "submitted job $gid"
    for _ in $(seq 1 30); do
        state=$(in_cluster "$alice" squeue -h -j "$gid" -o '%T' || true)
        [[ $state == RUNNING ]] && break
        sleep 1
    done
    [[ $state == RUNNING ]] || { echo "FAIL: gpu:2 job $gid is '${state:-gone}', not RUNNING" >&2; return 1; }
    alloc=$(in_cluster "$alice" scontrol show job "$gid" | grep -o 'AllocTRES=[^ ]*' || true)
    gpu_used=$(in_cluster "$alice" scontrol show node testnode | grep -o 'AllocTRES=[^ ]*' || true)
    echo "job: $alloc"; echo "node: $gpu_used"
    if [[ $alloc != *gres/gpu=2* || $gpu_used != *gres/gpu=2* ]]; then
        echo "FAIL: gpu:2 not allocated" >&2
        in_cluster "$alice" scontrol show node testnode >&2
        return 1
    fi

    echo "== accounting records the finished job"
    state=""
    for _ in $(seq 1 30); do
        state=$(in_cluster "$alice" sacct -n -X -j "$gid" -o State%12 | tr -d ' ' || true)
        [[ $state == COMPLETED ]] && break
        sleep 1
    done
    in_cluster "$alice" sacct -X -j "$gid" -o JobID,JobName,User,State,AllocTRES%30
    [[ $state == COMPLETED ]] || { echo "FAIL: sacct state '$state', not COMPLETED" >&2; return 1; }
    echo "SMOKE OK"
}

# The end-to-end run (README, "End-to-end run"): a fresh cluster, then every stage of e2e.py; exits 1 on any failed check.
# ARGS go to e2e.py (`--only install,jobs` for a subset, which reuses what is installed only with `python3 e2e.py`).
cmd_e2e() {
    cmd_down
    cmd_up
    python3 "$HERE/e2e.py" "$@"
}

# Interactive shell, or one command, in the container: run.sh shell [-u USER] [CMD...]
cmd_shell() {
    local user=root
    if [[ ${1:-} == -u ]]; then user=$2; shift 2; fi
    if (( $# == 0 )); then
        "${COMPOSE[@]}" exec -u "$user" -w "$([[ $user == root ]] && echo /root || echo "/home/$user")" "$SERVICE" bash -l
    else
        in_cluster "$user" "$@"
    fi
}

case "${1:-}" in
    up) cmd_up ;;
    down) cmd_down ;;
    smoke) cmd_smoke ;;
    e2e) shift; cmd_e2e "$@" ;;
    shell) shift; cmd_shell "$@" ;;
    *) echo "usage: run.sh up|down|smoke|e2e [ARGS]|shell [-u USER] [CMD...]" >&2; exit 2 ;;
esac

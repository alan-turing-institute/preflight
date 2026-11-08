#!/bin/sh
# slurmd epilog: remove the docker containers of the job whose allocation was just released (its end, a requeue, a
# pre-emption): every container labelled slurm-job=$SLURM_JOB_ID (README, "A job's docker containers"). They belong to
# the docker daemon, not to the job's cgroup, so slurm's own kill never reaches them.
# slurmd runs this as root. It must stay short, call no slurm command, and exit 0 whatever happens: a failing or
# timed-out epilog drains the node. Every decision is logged: journalctl -t gpujobs-job-containers
# slurm.conf names it on the first Epilog= line, so it runs before the others (slurm stops at the first epilog that
# fails), and one of them hung on the GPU driver cannot keep a container on the card.
# slurmd gives epilog scripts no PATH ("these programs do not have a search path set", slurm's prolog_epilog guide).
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin
export PATH
tag=gpujobs-job-containers
docker=${DOCKER:-docker} logger=${LOGGER:-logger} wait=${DOCKER_WAIT_S:-10} # the three are overridden by its test only
say() { timeout -k 1 2 "$logger" -t "$tag" -- "$*" 2>/dev/null || :; } # a journal not reading its socket must not hold the node
case $wait in '' | *[!0-9]*) wait=10 ;; esac # never an arithmetic error below: that would end the script non-zero
job=${SLURM_JOB_ID:-}
case $job in
'' | *[!0-9]*)
    say "SLURM_JOB_ID is not a job id ('$job'): no containers checked"
    exit 0
    ;;
esac
if ! command -v "$docker" >/dev/null 2>&1; then
    # a docker daemon without its CLI is a misconfiguration, not a node without docker: say so on every job
    for socket in /var/run/docker.sock /run/docker.sock; do
        if [ -S "$socket" ]; then
            say "job $job: $socket exists but no docker command was found: its containers were not checked"
            break
        fi
    done
    exit 0 # no docker on this node: nothing to remove (not logged: every job)
fi
# stdout only: an id list never holds docker's warnings; "set -f": a word in it is never a file name pattern
set -f
listed=$(timeout -k 2 "$wait" "$docker" ps -aq --no-trunc --filter "label=slurm-job=$job" 2>/dev/null)
answer=$?
if [ "$answer" -ne 0 ]; then
    say "job $job: '$docker ps --filter label=slurm-job=$job' did not answer (exit $answer); its containers were not checked"
    exit 0
fi
ids=
odd=0
for id in $listed; do
    case $id in
    *[!0-9a-f]*) odd=$((odd + 1)) ;;
    *) if [ ${#id} -eq 64 ]; then ids="$ids $id"; else odd=$((odd + 1)); fi ;;
    esac
done
if [ "$odd" -gt 0 ]; then
    say "job $job: skipped $odd word(s) of docker output that are not container ids"
fi
if [ -z "$ids" ]; then
    say "job $job: no containers"
    exit 0
fi
# shellcheck disable=SC2086 # one word per id, each checked above
if out=$(timeout -k 2 "$((wait * 3))" "$docker" rm -f $ids 2>&1); then
    say "job $job: removed container(s)$ids"
else
    say "job $job: '$docker rm -f$ids' failed ($out); the health check retries, status.py lists what is left"
fi
exit 0

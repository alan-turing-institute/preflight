#!/bin/sh
# Cluster, one account and the three users in slurm's accounting; waits for slurmdbd to answer first. Plain users have
# no admin level here; making carol an operator is the installer's `admin add` (F1).
set -eu
n=0
until sacctmgr -n show cluster >/dev/null 2>&1; do
    n=$((n + 1)); [ "$n" -lt 60 ] || { echo "slurmdbd did not answer" >&2; exit 1; }
    sleep 1
done
# `add` of something already there exits non-zero ("Nothing new added"); that is the second boot and is fine.
sacctmgr -i add cluster harness || true
sacctmgr -i add account harness Cluster=harness Description=harness Organization=harness || true
for u in alice bob carol; do
    sacctmgr -i add user "$u" Account=harness DefaultAccount=harness Cluster=harness || true
done

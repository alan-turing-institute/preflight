#!/bin/sh
# Start the Jobstats cgroup exporter on slurm's cgroup scope; $1 is the port to listen on.
# slurmd's scope is named <NodeName>_slurmstepd.scope under system.slice when the daemon runs under a node name
# (seen on Ubuntu 24.04, slurm 23.11.4: system.slice/gpu-node-1_slurmstepd.scope), but the exporter looks for
# system.slice/slurmstepd.scope unless told otherwise (--path.slurmstepd.scope).
root=${CGROUP_ROOT:-/sys/fs/cgroup}
scope=/system.slice/slurmstepd.scope
for dir in "$root"/system.slice/*slurmstepd.scope; do
    if [ -d "$dir" ]; then
        scope=${dir#"$root"}
        break
    fi
done
exec "${CGROUP_EXPORTER:-/usr/local/lib/gpujobs/cgroup_exporter}" --web.listen-address=":$1" \
    --path.cgroup.root="$root" --path.slurmstepd.scope="$scope" --collect.fullslurm

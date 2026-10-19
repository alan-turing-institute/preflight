#!/bin/sh
# slurmctld epilog (runs as SlurmUser when a job ends): store the job's Jobstats summary in its AdminComment as
# "JS1:<gzip+base64 summary>", which `jobstats` reads back for finished jobs (docs/setup/summaries.md:5-12).
# Never fails: the summary is a convenience and the job is already over.
jobstats=${JOBSTATS:-/usr/local/bin/jobstats}
sacctmgr=${SACCTMGR:-sacctmgr}
sleep "${SETTLE_SECONDS:-5}" # let slurmdbd receive the job's final record
# An array task is addressed as <array job>_<task>; its SLURM_JOB_ID alone would overwrite the whole array's record.
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ "$SLURM_ARRAY_JOB_ID" = "$SLURM_JOB_ID" ]; then
    id=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
else
    id=$SLURM_JOB_ID
fi
stats=$("$jobstats" -f -b "$SLURM_JOB_ID" 2>&1)
case $stats in
Short* | None* | H4s*) ;;
*)
    logger -t gpujobs-jobstats "job $id: no summary stored: $stats"
    exit 0
    ;;
esac
if ! out=$("$sacctmgr" -i update job where jobid="$id" set AdminComment="JS1:$stats" 2>&1); then
    logger -t gpujobs-jobstats "job $id: storing the summary failed: $out"
fi
exit 0

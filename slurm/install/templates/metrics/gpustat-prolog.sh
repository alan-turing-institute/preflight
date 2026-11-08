#!/bin/sh
# slurmd prolog: note which job holds each GPU, in the form the Jobstats GPU exporter reads (Jobstats docs,
# docs/setup/gpu_node_scripts.md:23-44): /run/gpustat/<GPU ordinal or MIG id> and /run/gpustat/<GPU UUID>, each holding
# "<job id> <uid>". Never fails, and never waits on a hung driver: a non-zero or hung prolog would drain the node.
dest=${GPUSTAT_DIR:-/run/gpustat}
smi=${NVIDIA_SMI:-/usr/bin/nvidia-smi}
mkdir -p -m 755 "$dest"
for gpu in $(echo "${GPU_DEVICE_ORDINAL:-} ${CUDA_VISIBLE_DEVICES:-}" | tr ',' ' '); do
    echo "$SLURM_JOB_ID $SLURM_JOB_UID" > "$dest/$gpu"
    case $gpu in
    MIG*) ;;
    *)
        uuid=$(timeout -k 2 10 "$smi" --query-gpu=uuid --format=noheader,csv -i "$gpu" 2>/dev/null)
        if [ -n "$uuid" ]; then
            echo "$SLURM_JOB_ID $SLURM_JOB_UID" > "$dest/$uuid"
        fi
        ;;
    esac
done
exit 0

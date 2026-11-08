#!/bin/sh
# slurmd epilog: remove what gpustat-prolog.sh wrote for this job's GPUs (docs/setup/gpu_node_scripts.md:23-44).
dest=${GPUSTAT_DIR:-/run/gpustat}
smi=${NVIDIA_SMI:-/usr/bin/nvidia-smi}
for gpu in $(echo "${GPU_DEVICE_ORDINAL:-} ${CUDA_VISIBLE_DEVICES:-}" | tr ',' ' '); do
    rm -f "$dest/$gpu"
    case $gpu in
    MIG*) ;;
    *)
        uuid=$(timeout -k 2 10 "$smi" --query-gpu=uuid --format=noheader,csv -i "$gpu" 2>/dev/null)
        if [ -n "$uuid" ]; then
            rm -f "$dest/$uuid"
        fi
        ;;
    esac
done
exit 0

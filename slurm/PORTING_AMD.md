# Porting the GPU jobs layer to an AMD machine

Written 2026-11-07 for a machine with an AMD GPU (first target: a Ryzen AI Max "Strix Halo" mini PC). Nothing here has
been run on AMD hardware. "Checked" below means checked against AMD's or SchedMD's documentation on that date; everything
else is an assumption to test.

## What carries over unchanged

The scheduling layer does not care who made the GPU: levels, pause and resume, urgent work, the job records, the web page,
the part-time machine logic and the installer's structure. It sees a GPU as a slurm `gres` called `gpu`.

## What is NVIDIA-only today, and what to do about it

Each row is a place the code calls an NVIDIA tool or names an NVIDIA device file.

| File and line | What it does with NVIDIA | AMD side |
|---|---|---|
| `slurm/install/detect.py:21`, `:50` | Lists GPUs with `nvidia-smi`; names the device `/dev/nvidia<N>` | Replace with `amd-smi` or `rocm-smi` output; device file is a render node under `/dev/dri/` (AMD's own gres.conf sample uses `/dev/dri/renderD128`) |
| `slurm/install/render.py:75`, site file `device = ...` | Writes `gres.conf` with `File=/dev/nvidia0` | Slurm can detect AMD GPUs itself: `AutoDetect=rsmi` (needs the ROCm SMI library on the node), with `Flags=amd_gpu_env` so jobs get `ROCR_VISIBLE_DEVICES`. Checked: gres.conf(5). If you list devices by hand, slurm checks them against what it detects and drains the node on a mismatch |
| `slurm/gpu_jobs.py:1003-1041` | GPU memory per process and GPU busy percentage, from `nvidia-smi` | `amd-smi` exposes per-process GTT memory (checked: AMD SMI docs). Not checked: whether `rocm-smi --showpidgpumem` reports it too, or whether the two tools agree on this chip |
| `slurm/job_wrapper.py:99-145` | Finds foreign compute processes on the job's GPU, with `nvidia-smi --query-compute-apps` | Same replacement as above |
| `slurm/agent.py:47`, `:1233` | Stops quietly when `nvidia-smi` does not answer | Follows from the replacement above |
| `slurm/install/components/metrics.py:165-176`, `slurm/install/templates/metrics/gpustat-*.sh` | Downloads an NVIDIA GPU exporter, and prolog/epilog scripts call `nvidia-smi` | Leave `[metrics]` off at first, or find an AMD exporter; the rest of metrics (node, cgroup) is vendor-neutral |
| `machine/gpu/record_gpu_usage.py`, `machine/flight_recorder/flight_recorder.py`, `dgx/assert_gpu_free.py` | Machine-level recorders and the DGX gate, all `nvidia-smi` | Skip them: they belong to the Turing DGX, not to the jobs layer |
| `hostguard/hostguard.py` | Caps PyTorch memory on a box where the GPU shares system memory | The idea applies to Strix Halo (checked: AMD says its memory is mapped, not partitioned, and the "VRAM" figure is a carve-out). Whether it works needs testing: it assumes PyTorch's CUDA calls, and a ROCm build of PyTorch is expected to answer the same calls (assumption, untested) |

## Things that behave differently on this class of AMD machine

- **Memory numbers.** The jobs layer sizes jobs from measured peaks (`--mem`, card sharing). On Strix Halo the GPU's "VRAM"
  is a carve-out and workloads mostly use the shared pool (GTT), so a reading of VRAM alone would under-report. Read GTT too.
  Source: AMD's Strix Halo system optimisation page.
- **Card sharing (`shards`)** rests on the memory sum of declared `--mem`, not on any GPU memory limit, so it needs no change.
  Do not turn it on until the per-process memory reading above works, because sharing's safety depends on honest peaks.
- **Slurm itself:** use the version pinned in the site file (see the README); slurm's AMD support is the `rsmi` detection
  above, nothing in this repository.

## The other repository: localmodels

`https://github.com/farhanferoz/localmodels` serves and downloads models and stores measured settings per model. Its
engine layer already separates device types (`src/localmodels/enums.py`, `backend.py`, `params.py`), but the only GPU kinds
it knows are CUDA and Apple's MPS, and `stack.py` reads the NVIDIA driver and CUDA versions. A ROCm or Vulkan llama.cpp
build would need a new device kind there, and its measured rules (`rules.md`) were measured on an NVIDIA GB10, so he
should re-measure rather than reuse them.

## Suggested order

1. Install slurm on the box with no GPU in the site file and run CPU jobs through the layer (everything works).
2. Add the GPU as a `gres` by `AutoDetect=rsmi` and check `gpujobs-status` shows it.
3. Replace the `nvidia-smi` calls in `gpu_jobs.py` and `job_wrapper.py` (one small function each, with the existing tests
   as the model), then `detect.py`.
4. Only then consider metrics, card sharing and hostguard.

---
name: dgx-spark
description: Work on the shared site-b NVIDIA DGX Spark. Use for preparing a checkout or Python environment on that host, or for previewing, submitting, monitoring, and troubleshooting its gpujobs workloads.
---

# Work on the site-b DGX Spark

## Prepare on the host

- Use the user's configured SSH alias for the site-b Spark (`spark` if configured). On Spark, read `/data/site/GPU_JOBS.md` for the current queue policy and portal instructions. Use `gpujobs-submit --help` to check installed flags.
- Locate the intended checkout on Spark and read its project instructions. Verify the commit, entrypoint, config, dataset and checkpoint paths, including files outside Git. The site-b site uses `dgx_gate = false` in `/etc/gpujobs/site.toml`. With that setting, `--entry` and `--repo` do not check code parity. Keep the queued job's inputs stable and use a small queued smoke when runtime validation is needed.
- Prepare the project with `uv` using its lockfile and platform instructions. Spark is Linux ARM64 with an NVIDIA GB10, so verify the CUDA build and compiled dependencies match. If noninteractive SSH cannot find `uv`, check `$HOME/.local/bin/uv`. With the environment already synced, use `uv run --no-sync` in the batch command.

## Preview and submit

- Use `gpujobs-submit` on Spark for GPU work, including CUDA smokes. The queue can coordinate GPU access only for work it knows about. Omit `--gpu` for CPU-only batch jobs.
- Use `--class restart` for work that starts over and `--class resume --resume-ok` only if the same command reliably continues from its latest checkpoint. Use `--after JOBID` when a job requires a successful predecessor.
- Set `--mem` for the first run, accounting for CPU and GPU use of the same RAM. Reuse `--name` for comparable runs so `gpujobs-footprint NAME` and automatic sizing use relevant history. Give smokes a distinct name. Without a measurement, automatic sizing can request the whole node for a GPU job.

From the Spark project checkout, adapt this preview to the workload.

```bash
gpujobs-submit --dry-run --project myproject --what 'train the baseline' \
  --class restart --gpu --mem 24G --name baseline \
  -- uv run --no-sync python scripts/train.py
```

Check the generated command, working directory, GPU and memory request, and log path. The preview renders the batch script. Verify imports, config and data separately. When asked to launch, remove `--dry-run` and record the job ID and log path. If SSH drops during submission, check for the job before retrying.

## Monitor and diagnose

- Use `gpujobs-status` for waiting reasons and GPU ownership. Read the log path reported by submit. Use `gpujobs-wait --once JOBID` for a snapshot or `gpujobs-wait JOBID` to follow a job and its dependent chain. Verify final states and expected outputs. Follow the host guide for pause, resume, cancellation and urgent work.
- For memory failures, compare the request to `gpujobs-footprint NAME` and inspect the job log. GB10 GPU allocations can be absent from Slurm's cgroup memory peak. Where readable, correlate the failure with `/var/log/preflight/gpu.jsonl` and its rotations (UTC), and with `journalctl -u earlyoom`. The Preflight PyTorch allocator hook skips capping under Slurm. Check for an automatic requeue before resubmitting a resumable job after a host OOM.

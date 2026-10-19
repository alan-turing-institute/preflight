# Jobstats configuration for ${site_name}, rendered from the site file by slurm/install/components/metrics.py.
# Only the settings jobstats reads (docs/setup/jobstats_command.md); no site-specific notes.

# prometheus server address, port, and retention period
PROM_SERVER = "${prometheus_server}"
PROM_RETENTION_DAYS = ${retention_days}

# number of seconds between measurements; equals scrape_interval in the Prometheus configuration
SAMPLING_PERIOD = ${scrape_interval}

# GPU statistics carry a jobid label (GPU exporter 0.2.2 or later)
GPU_EXPORTER_JOBID = True

# job summaries are kept in the slurm database (AdminComment), not in an external database
EXTERNAL_DB_CONFIG = {"enabled": False}

CLUSTER_TRANS = {}
CLUSTER_TRANS_INV = dict(zip(CLUSTER_TRANS.values(), CLUSTER_TRANS.keys()))

MAX_JOBNAME_LEN = 64

GPU_UTIL_RED = 15  # percentage
GPU_UTIL_BLACK = 25  # percentage
CPU_UTIL_RED = 65  # percentage
CPU_UTIL_BLACK = 80  # percentage
TIME_EFFICIENCY_RED = 10  # percentage
TIME_EFFICIENCY_BLACK = 60  # percentage
MIN_MEMORY_USAGE = 70  # percentage
MIN_RUNTIME_SECONDS = 10 * SAMPLING_PERIOD  # seconds

# detailed GPU metrics are off
GPU_METRICS_EXPORTER = "None"
GPU_METRICS_DOCS_URL = "https://princetonuniversity.github.io/jobstats/setup/detailed_gpu_metrics/"
GPU_METRICS = {}

NOTES = []

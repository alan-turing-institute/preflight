# Test cluster for the portable GPU jobs build: one Ubuntu 24.04 node running systemd as PID 1 with slurmctld, slurmd,
# slurmdbd, MariaDB and munge from the distro packages (the same packages and the 23.11 line this box runs), a
# bootstrap single-node config with four fake GPUs, and the users alice, bob (plain) and carol (admin).
# The project is mounted read-only at /src at run time; nothing of ours is baked in (installer test, F1).
# SLURM_VERSION / SLURM_SHA256 (build args, both or neither): empty, the distro's slurm packages (23.11); set, SchedMD's
# `slurm-smd*` debs built by slurm/install/slurm_build.py (passed as the `src` build context, the repository root).
FROM ubuntu:24.04 AS slurm-debs
ARG SLURM_VERSION=
ARG SLURM_SHA256=
ENV DEBIAN_FRONTEND=noninteractive
RUN install -d /debs
RUN --mount=type=bind,from=src,source=slurm/install/slurm_build.py,target=/build/slurm_build.py \
    if [ -n "$SLURM_VERSION" ]; then \
      apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates \
      && python3 -I /build/slurm_build.py --version "$SLURM_VERSION" --sha256 "$SLURM_SHA256" --cache /cache \
      && cp /cache/"$SLURM_VERSION"/deb-*/*.deb /debs/ ; \
    fi

FROM ubuntu:24.04
ARG SLURM_VERSION=

ENV DEBIAN_FRONTEND=noninteractive

# systemd + dbus: PID 1 and what `systemctl` talks to. libpam-systemd and dbus-user-session, with logind left unmasked:
# `systemctl --user --machine=USER@.host` (the lease component's way to reach a user's systemd) starts a PAM login
# session for USER and connects to that user's own D-Bus socket. sudo and python3: the installer's requirements. procps:
# ps/pgrep for tests that look at job processes.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      systemd systemd-sysv dbus libpam-systemd dbus-user-session \
      mariadb-server munge \
      sudo python3 procps ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# The slurm packages: the distro's, or the built debs (the daemons, client and accounting; no -dev or -dbgsym). The
# slurm-smd debs do not create the `slurm` user (the distro's do), so it is made first.
COPY --from=slurm-debs /debs /debs
RUN apt-get update \
 && if [ -n "$SLURM_VERSION" ]; then \
      useradd --system --home-dir /var/lib/slurm --shell /usr/sbin/nologin slurm \
      && apt-get install -y --no-install-recommends /debs/slurm-smd_*.deb /debs/slurm-smd-client_*.deb \
        /debs/slurm-smd-slurmctld_*.deb /debs/slurm-smd-slurmd_*.deb /debs/slurm-smd-slurmdbd_*.deb ; \
    else apt-get install -y --no-install-recommends slurm-wlm slurmdbd ; fi \
 && rm -rf /var/lib/apt/lists/*
# The built debs stay where slurm_build.py caches them, so the installer's `[slurm]` step finds them instead of building.
# The folder is slurm_build.build_dir(): family, os-release ID and VERSION_ID, machine.
RUN if [ -n "$SLURM_VERSION" ]; then \
      dir="/var/cache/gpujobs/slurm/$SLURM_VERSION/deb-$(. /etc/os-release && echo "$ID$VERSION_ID")-$(uname -m)" \
      && install -d "$dir" && cp /debs/*.deb "$dir/" ; \
    fi \
 && rm -rf /debs

# alice and bob: plain users. carol: admin (the Unix group the plan's admin model uses). Fixed ids so tests can name them.
RUN groupadd --gid 2000 gpujobs-admin \
 && useradd --uid 2001 --create-home --shell /bin/bash alice \
 && useradd --uid 2002 --create-home --shell /bin/bash bob \
 && useradd --uid 2003 --create-home --shell /bin/bash --groups gpujobs-admin carol

# Bootstrap slurm configuration (replaced by the installer's rendered files in F1) and first-boot units.
COPY bootstrap/slurm.conf bootstrap/gres.conf bootstrap/cgroup.conf /etc/slurm/
COPY bootstrap/slurmdbd.conf /etc/slurm/slurmdbd.conf
COPY bootstrap/90-slurm.cnf /etc/mysql/mariadb.conf.d/90-slurm.cnf
COPY bootstrap/gpujobs-db-init.service bootstrap/gpujobs-accounts-init.service /etc/systemd/system/
COPY bootstrap/slurmctld-order.conf /etc/systemd/system/slurmctld.service.d/10-harness-order.conf
COPY bootstrap/slurmdbd-order.conf /etc/systemd/system/slurmdbd.service.d/10-harness-order.conf
COPY bootstrap/db-init.sh bootstrap/accounts-init.sh /usr/local/libexec/gpujobs-harness/

# The fake GPUs' device files (see bootstrap/gres.conf for why they exist).
RUN install -d /opt/fakegpu \
 && for i in 0 1 2 3; do ln -s /dev/null /opt/fakegpu/gpu$i; done

RUN chown slurm:slurm /etc/slurm/slurmdbd.conf && chmod 600 /etc/slurm/slurmdbd.conf \
 && install -d -o slurm -g slurm /var/log/slurm /var/spool/slurmctld \
 && install -d /var/spool/slurmd \
 && chmod 755 /usr/local/libexec/gpujobs-harness/*.sh \
 && systemctl enable mariadb munge slurmdbd slurmctld slurmd gpujobs-db-init gpujobs-accounts-init \
 && systemctl mask getty.target console-getty.service systemd-udevd.service \
      systemd-udev-trigger.service systemd-remount-fs.service sys-kernel-debug.mount sys-kernel-tracing.mount

STOPSIGNAL SIGRTMIN+3
CMD ["/sbin/init"]

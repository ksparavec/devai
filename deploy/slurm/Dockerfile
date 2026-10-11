# devai-slurm -- all of Slurm (slurmctld, slurmdbd, slurmrestd, slurmd),
# MariaDB, nats-server and the bus log in ONE container (docs/slurm.md
# Sec. 3.1; platform-rebuild Phase 5, D10-D12, D16).
#
# Debian trixie packages only (D12): Slurm 24.11.5 from trixie-security
# (24.11.5-4+deb13u1, pinned below; the build fails when apt no longer
# offers that version), MariaDB 11.8, nats-server 2.10.27, supervisor, tini,
# and python3 for devai-gpu (python3-minimal has no ctypes; I3). No source
# build, no podman client.
#
# devai-bus and devai-jobs are static Go binaries from devai-tools, built on
# the host by `make slurm-tools` and bind-mounted read-only at
# /var/cache/devai-tools-dist for the build (the build-ollama convention):
# no Go toolchain in the image, and nothing pulled but the pinned Debian
# image.
#
# Build: make build-slurm      (context: deploy/slurm)

ARG BASE_IMAGE=docker.io/library/debian:trixie-slim@sha256:020c0d20b9880058cbe785a9db107156c3c75c2ac944a6aa7ab59f2add76a7bd
FROM ${BASE_IMAGE}

ARG APT_PROXY
ARG SLURM_VERSION=24.11.5-4+deb13u1

# policy-rc.d keeps mariadb-server's and munge's postinst from starting
# daemons during the build. trixie-security is enabled in the base image's
# debian.sources, and the pinned Slurm build comes from it (D12).
RUN set -eu; \
    if [ -n "${APT_PROXY:-}" ]; then echo "Acquire::http::Proxy \"$APT_PROXY\";" > /etc/apt/apt.conf.d/00cache; fi; \
    printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d; chmod 0755 /usr/sbin/policy-rc.d; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        slurmctld=$SLURM_VERSION slurmd=$SLURM_VERSION slurmdbd=$SLURM_VERSION \
        slurmrestd=$SLURM_VERSION slurm-client=$SLURM_VERSION \
        slurm-wlm-basic-plugins=$SLURM_VERSION slurm-wlm-jwt-plugin=$SLURM_VERSION \
        slurm-wlm-mysql-plugin=$SLURM_VERSION \
        mariadb-server nats-server supervisor tini python3 procps ca-certificates; \
    rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/00cache /usr/sbin/policy-rc.d

# The submit filter (JobSubmitPlugins=lua) needs Debian's job_submit_lua.so.
# Without it slurmctld would start with no filter at all, so the build stops.
RUN set -eu; \
    so=$(dpkg -L slurm-wlm-basic-plugins | grep '/job_submit_lua\.so$' || true); \
    test -n "$so" && test -f "$so" \
        || { echo "error: job_submit_lua.so (slurm-wlm-basic-plugins) is not installed" >&2; exit 1; }; \
    echo "job_submit_lua: $so"

# devai (uid 1000) is the one Slurm user every job runs as (I6); it is NOT in
# group 0 (I26). slurmrestd runs as its own user, since it refuses root.
RUN set -eu; \
    groupadd --gid 1000 devai; \
    useradd --uid 1000 --gid 1000 --no-create-home --home-dir /nonexistent --shell /bin/bash devai; \
    useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin slurmrest

# DIST_ID: a content hash of the mounted binaries, passed by the Makefile.
# Without it this layer is cached forever (podman does not look inside a bind
# mount), so rebuilt binaries would silently not be installed.
ARG DIST_ID=unset
RUN set -eu; \
    echo "devai-tools build $DIST_ID"; \
    for b in devai-bus devai-jobs; do \
        test -x /var/cache/devai-tools-dist/$b \
            || { echo "error: no $b in the mounted build -- run 'make slurm-tools' first" >&2; exit 1; }; \
        install -m 0755 /var/cache/devai-tools-dist/$b /usr/local/bin/$b; \
    done; \
    /usr/local/bin/devai-bus --help >/dev/null 2>&1 || true

COPY slurm.conf.in slurmdbd.conf.in nats-server.conf.in /usr/local/share/devai/slurm/
COPY job_submit.lua cgroup.conf /etc/slurm/
COPY supervisord.conf /etc/devai/supervisord.conf
COPY entrypoint.sh accounting-init.sh node-recover.sh prolog.sh epilog.sh healthcheck.sh devai-gpu /usr/local/lib/devai/
COPY jobs/ /usr/local/lib/devai/jobs/

RUN set -eu; \
    chmod 0755 /usr/local/lib/devai/*.sh /usr/local/lib/devai/devai-gpu /usr/local/lib/devai/jobs/*.sh; \
    chmod 0644 /etc/slurm/job_submit.lua /etc/slurm/cgroup.conf; \
    ln -s /usr/local/lib/devai/devai-gpu /usr/local/bin/devai-gpu; \
    python3 -c 'import ctypes' ; \
    test -x /usr/sbin/slurmctld && test -x /usr/sbin/slurmd && test -x /usr/sbin/slurmdbd \
        && test -x /usr/sbin/slurmrestd && test -x /usr/sbin/nats-server && test -x /usr/sbin/mariadbd

# tini -s: under --pid=host tini is not PID 1, so it registers as a
# subreaper (C6). A devai-bus hold whose job script was SIGKILLed is
# re-parented to it and notices (C3).
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "/usr/local/lib/devai/entrypoint.sh"]

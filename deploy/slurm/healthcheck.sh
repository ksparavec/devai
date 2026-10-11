#!/bin/bash
# Health check of devai-slurm (HealthCheckProgram, every 30 s; docs/slurm.md
# Sec. 6, I7). A node the guard or a failed epilog drained is resumed once
# the card is idle again (apart from a sleeping devai model); a drain anyone
# else set stays. Without this, a drained node would run nothing until
# someone resumed it by hand.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export DEVAI_BUS_URL=nats://127.0.0.1:4222
export DEVAI_BUS_USER=jobs
export DEVAI_BUS_PASSWORD_FILE=/run/devai/bus/jobs.pass

exec /usr/local/lib/devai/devai-gpu health

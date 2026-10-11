#!/bin/bash
# Entrypoint of devai-slurm (docs/slurm.md Sec. 3.1 and 10; platform-rebuild
# Phase 5, C6, C7, I4, I26). Started as `tini -s -- entrypoint.sh`.
#
# Renders the configuration from the templates and the 0600 files of C7
# (mounted read-only at /etc/devai/slurm: slurm.key, jwt_hs256.key,
# mariadb.pass, nats/<user>.pass), prepares the run and data directories,
# initialises MariaDB on an empty data directory, and execs supervisord,
# which runs everything else (the sacctmgr steps run as its program
# accounting-init).
set -euo pipefail

IN=/usr/local/share/devai/slurm
SECRETS=${DEVAI_SLURM_SECRETS:-/etc/devai/slurm}
JOBS=/var/cache/devai/jobs
BUS_USERS="router jobs operator control-engines control-laya control-operator buslog"

die() { echo "devai-slurm: $*" >&2; exit 1; }

# A secret is substituted into a configuration file with sed, so it must be
# plain: slurm-init.sh writes hex.
plain() { [[ "$1" =~ ^[A-Za-z0-9._~+-]{16,256}$ ]]; }

# --- secrets (C7) ---------------------------------------------------------
for f in slurm.key jwt_hs256.key mariadb.pass; do
    [ -s "$SECRETS/$f" ] || die "missing $SECRETS/$f -- run 'make slurm-init' on the host"
done
for u in $BUS_USERS; do
    [ -s "$SECRETS/nats/$u.pass" ] || die "missing $SECRETS/nats/$u.pass -- run 'make slurm-init' on the host"
    plain "$(tr -d '\n' < "$SECRETS/nats/$u.pass")" || die "$SECRETS/nats/$u.pass is not a plain token"
done
MARIADB_PASS=$(tr -d '\n' < "$SECRETS/mariadb.pass")
plain "$MARIADB_PASS" || die "$SECRETS/mariadb.pass is not a plain token"

install -d -m 0755 /etc/slurm
install -m 0600 -o root -g root "$SECRETS/slurm.key" /etc/slurm/slurm.key
install -m 0600 -o root -g root "$SECRETS/jwt_hs256.key" /etc/slurm/jwt_hs256.key

# --- the jobs directory (C4) ----------------------------------------------
[ -d "$JOBS" ] || die "$JOBS is missing -- create it on the host (a plain directory; plan E9)"
install -d -m 0755 "$JOBS/results" "$JOBS/bus" "$JOBS/slurmctld" "$JOBS/engines" "$JOBS/operator"
install -d -m 0750 "$JOBS/mariadb"
if [ "$(stat -c %U "$JOBS/mariadb")" != mysql ]; then
    chown -R mysql:mysql "$JOBS/mariadb"
fi

# --- run directories ------------------------------------------------------
install -d -m 0755 /run/slurm /var/spool/slurmd /run/devai
install -d -m 0755 -o mysql -g mysql /run/mysqld
install -d -m 0700 /run/devai/sampler
# The jobs bus password, the one secret a job may read (I26): a copy owned
# by devai, 0400, on the tmpfs.
install -d -m 0755 /run/devai/bus
install -m 0400 -o devai -g devai "$SECRETS/nats/jobs.pass" /run/devai/bus/jobs.pass

# --- slurm.conf: the node from the host (I4) ------------------------------
CPUS=$(nproc --all)
MEMORY_MB=$(awk '/^MemTotal:/ { printf "%d", $2 / 1024 }' /proc/meminfo)
[[ "$CPUS" =~ ^[0-9]+$ && "$MEMORY_MB" =~ ^[0-9]+$ ]] || die "cannot read the host's CPUs ($CPUS) or memory ($MEMORY_MB)"
sed -e "s/@CPUS@/$CPUS/g" -e "s/@MEMORY_MB@/$MEMORY_MB/g" "$IN/slurm.conf.in" > /etc/slurm/slurm.conf
chmod 0644 /etc/slurm/slurm.conf
echo "devai-slurm: node devai with CPUs=$CPUS RealMemory=$MEMORY_MB"

# --- slurmdbd.conf (0600; slurmdbd refuses anything wider) ---------------
umask 077
sed -e "s/@MARIADB_PASS@/$MARIADB_PASS/g" "$IN/slurmdbd.conf.in" > /etc/slurm/slurmdbd.conf
chmod 0600 /etc/slurm/slurmdbd.conf

# --- nats-server.conf (0600, on the tmpfs) --------------------------------
args=()
for u in $BUS_USERS; do
    args+=(-e "s/@PASS:$u@/$(tr -d '\n' < "$SECRETS/nats/$u.pass")/g")
done
sed "${args[@]}" "$IN/nats-server.conf.in" > /run/devai/nats-server.conf
chmod 0600 /run/devai/nats-server.conf
if grep -qE '@PASS:[a-z0-9-]+@' /run/devai/nats-server.conf; then
    die "nats-server.conf.in names a bus user with no password file"
fi

# --- MariaDB --------------------------------------------------------------
# slurmdbd's database user, (re)created with the current password at every
# start through mariadbd's --init-file.
cat > /run/devai/mariadb-init.sql <<SQL
CREATE DATABASE IF NOT EXISTS slurm_acct_db;
CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY '$MARIADB_PASS';
ALTER USER 'slurm'@'localhost' IDENTIFIED BY '$MARIADB_PASS';
GRANT ALL PRIVILEGES ON slurm_acct_db.* TO 'slurm'@'localhost';
FLUSH PRIVILEGES;
SQL
chown mysql:mysql /run/devai/mariadb-init.sql
chmod 0600 /run/devai/mariadb-init.sql
umask 022

if [ -z "$(ls -A "$JOBS/mariadb")" ]; then
    echo "devai-slurm: initialising MariaDB in $JOBS/mariadb"
    mariadb-install-db --no-defaults --user=mysql --datadir="$JOBS/mariadb" \
        --auth-root-authentication-method=socket --skip-test-db > /dev/null \
        || die "mariadb-install-db failed"
fi

exec /usr/bin/supervisord -n -c /etc/devai/supervisord.conf

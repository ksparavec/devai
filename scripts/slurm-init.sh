#!/bin/bash
# slurm-init.sh -- create devai-slurm's secrets, once (`make slurm-init`;
# docs/plans/platform-rebuild.md C7; docs/slurm.md Sec. 10).
#
#   ~/.config/devai/slurm/slurm.key        auth/slurm key, 1024 random bytes
#   ~/.config/devai/slurm/jwt_hs256.key    the HS256 key for slurmrestd's tokens, 32 random bytes
#   ~/.config/devai/slurm/mariadb.pass     slurmdbd's MariaDB password
#   ~/.config/devai/slurm/nats/<user>.pass one per bus user (C1)
#
# Mode 0600 (directories 0700), plain files, NEVER overwritten: a file that
# exists is kept as it is, so re-running is safe and the keys survive. They
# are backed up by devai-backup (archives are written 0600 for that reason).
# devai-slurm mounts the directory read-only; each other client mounts only
# its own file.
#
# DEVAI_SLURM_CONFIG_DIR overrides the directory (tests).
set -euo pipefail

DIR=${DEVAI_SLURM_CONFIG_DIR:-$HOME/.config/devai/slurm}
BUS_USERS="router jobs operator control-engines control-laya control-operator buslog"

umask 077
mkdir -p "$DIR/nats"
chmod 0700 "$DIR" "$DIR/nats"

random_bytes() { head -c "$1" /dev/urandom; }
# Hex, so entrypoint.sh can substitute it into a configuration file as is.
random_token() { od -An -v -tx1 -N32 /dev/urandom | tr -d ' \n'; echo; }

created=0
kept=0
make_secret() {
    local path=$1; shift
    if [ -e "$path" ]; then
        local mode
        mode=$(stat -c %a "$path")
        if [ "$mode" != 600 ]; then
            echo "WARNING: $path has mode $mode, not 600 (kept as it is)" >&2
        fi
        kept=$((kept + 1))
        return
    fi
    # noclobber: the redirection fails rather than overwrite a file that
    # appeared meanwhile.
    ( set -o noclobber; "$@" > "$path" )
    chmod 0600 "$path"
    echo "created $path"
    created=$((created + 1))
}

make_secret "$DIR/slurm.key" random_bytes 1024
make_secret "$DIR/jwt_hs256.key" random_bytes 32
make_secret "$DIR/mariadb.pass" random_token
for u in $BUS_USERS; do
    make_secret "$DIR/nats/$u.pass" random_token
done

echo "slurm-init: $created created, $kept kept, in $DIR"

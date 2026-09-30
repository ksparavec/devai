# devai-operator

_A web service, in its own image and container, that offers everything the Makefile offers today, by running devai's scripts; the host shell stays for development and one-time host setup._

## Status

Draft (2026-09-30). Decisions O1-O9 below are the operator's. Nothing is built.

## Dependencies

- [Plan: slurm-gatekeeper](./slurm-gatekeeper.md) -- operator actions run as Slurm jobs (O2) and land in its job history; until its Phase 1 exists, actions run directly (Phase 1 below).
- [Plan: minimal-external-images](./minimal-external-images.md) -- the pinned `debian:trixie-slim` base; the images the build actions build.

## Enables / Unblocks

- Running devai from a browser, without a host shell.
- Every operator action -- build, pull, probe, bench, backup -- in the same job history as the engines, with its log and its outcome.
- The host needs only podman and the one-time root setup.

## Out of scope

- Several users, roles or permissions: one operator account.
- Operations that need root on the host (O3).
- A browser terminal (O4).
- Replacing JupyterLab or the agents' own UIs.

## Operator decisions (2026-09-30)

- **O1 -- Apache with mod_wsgi.** Debian's `apache2` (2.4.68) and `libapache2-mod-wsgi-py3` (5.0.2): the Python app runs in Apache's own daemon processes, so TLS, login and the app are one server.
- **O2 -- Every action runs as a Slurm job.** It is queued, can be cancelled and suspended, has a time limit, and lands in the job history with its log; actions that use the GPU directly (probes) take the engine license, while a bench uses it through the router and keeps its engine with a hold. The exception is the actions that start or stop the stack itself (bring the services up or down, rebuild or restart `devai-slurm`): Slurm may be down for those, so they run directly and are logged by the operator.
- **O3 -- Host root stays host-shell setup.** `setup-logs` (LVM, mkfs, `/etc/fstab`) and `secrets-tmpfs` (a tmpfs mount) need `sudo`; `install` / `uninstall` write `~/.local/bin`. They stay one-time host setup: the web shows whether each is done and prints the exact command.
- **O4 -- Interactive targets become links.** The web starts and stops JupyterLab lab containers and shows their links; a shell is a JupyterLab terminal.
- **O5 -- The operator runs the scripts, not `make`.** Each action is a script with declared parameters. The Makefile stays as a development convenience that calls the same scripts.
- **O6 -- On the LAN, behind a login.** HTTPS on a published LAN port (as Open WebUI's proxy was on :8443), TLS with mkcert certificates or a self-signed fallback.
- **O7 -- One operator account,** an Apache htpasswd entry created by the init action.
- **O8 -- Go and Rust as pinned upstream downloads** in the image, each checked against its published checksum: Go >= 1.26 (Debian has 1.24) and the rustup toolchain vLLM's `rust-toolchain.toml` names (Debian has rustc 1.85).
- **O9 -- Compose from Docker, pinned.** Docker's `docker-compose-plugin` from Docker's own apt repository (`https://download.docker.com/linux/debian`, suite `trixie`, component `stable`), pinned at `5.1.0-1~debian.13~trixie`, the version this host runs today. The repository key is accepted only with Docker's fingerprint `9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88`; apt then checks the package against the signed index. The package installs `/usr/libexec/docker/cli-plugins/docker-compose`, and the image's `containers.conf` names that path in `compose_providers`, so `podman compose` does not depend on a search order.

## Open questions

1. ~~Exposure?~~ LAN, HTTPS, login (O6).
2. ~~How many accounts?~~ One (O7).
3. ~~Compose provider?~~ Docker's compose plugin, pinned (O9).
4. ~~Go and Rust?~~ Pinned upstream downloads with checksums (O8).

## Context

Everything devai can do is a Makefile target run in a host shell: 124 documented targets in 2,163 lines. That needs a shell on the host, leaves no record of what ran, and mixes development conveniences with operations. The Makefile's recipes are often inline shell -- long `podman run` lines, compose calls, variable defaults -- rather than calls to scripts, so there is no single entry point per operation that another program could call.

## Approach

**One image, `devai-operator`,** on the pinned `debian:trixie-slim`, holding what the scripts call today (measured by grepping the Makefile's recipes, 2026-09-30):

- Apache, mod_wsgi, Flask (Debian `python3-flask` 3.1.1) -- the web service;
- `python3`, `uv`, `git`, `curl`, `make`, podman (client) and Docker's compose plugin (O9) -- the tools the scripts use;
- for the from-source builds: `cmake`, `ninja`, CUDA 13.1 `nvcc` and headers from NVIDIA's debian13 apt repository, Go and Rust as pinned downloads (O8).

**One container,** on `devai-net`:

- the host's podman socket, read-write (`CONTAINER_HOST`), so the scripts' `podman` calls, builds and compose runs reach the host's podman;
- **path identity:** the repository, `/var/cache/devai` and the home volume are mounted at the same paths as on the host, because the scripts pass host paths to podman (`-v $(CURDIR)/scripts:/scripts`, `-v $(CACHE_DIR)/pip:...`), and podman resolves them on the host;
- the GPU device when the host has one (probes read `nvidia-smi`; the VRAM ballast allocates);
- the Slurm JWT key (read-only), to call slurmrestd, and its bus password (read-only), for `devai-control` and the job pages;
- TLS certificate and key, the htpasswd file;
- `tini` as PID 1, then supervisord, which runs Apache and `devai-control`.

**Actions.** A registry, `deploy/operator/actions.yaml`, lists every action: an id and group, the script and its fixed arguments, its parameters (name, type, allowed values or pattern, default, help, and how each maps to a flag or an environment variable), its kind (`job`, `direct`, `host`, `lab`), whether it takes the engine license (probes), its time limit, whether it asks for confirmation (destructive actions such as clean, restore, stack down), and the Makefile targets it covers. A test fails when a documented Makefile target is covered by no action and not listed as not offered with a reason, so "everything the Makefile offers" stays true as the Makefile changes.

**Running an action.** The app validates each parameter against the registry and builds an argument list; nothing passes through a shell. A `job` action is submitted to slurmrestd with a script that sends `devai.operator.actions.run {job, action, params}` over the bus and holds a lease while it runs, the same pattern as an engine job ([docs/slurm.md](../slurm.md) Sec. 4). `devai-control` in this container checks the action and its parameters against the registry again, runs the script, and reports its exit; `stop`, or a lease that runs out, stops it. Containers a script starts are labelled with the job id, so `devai-control` can remove them when the action stops. A `direct` action is started by the app with `setsid`, outside Apache's processes (an Apache restart must not kill it), and logs to `/var/cache/devai/jobs/operator/`.

**Pages:** the stack at a glance (containers, the GPU holder, the running engine, the queue); actions by group, each with its form; jobs (queue and history, from Slurm's REST API) and a job page with its log (polled), cancel, hold, suspend and resume; the lab (start, stop, link); host setup (what is done, and the command for what is not).

**Security.** The container can do anything the host's devai user can (the podman socket), so its login is as sensitive as that user's login: TLS only, the password in an htpasswd file created once, a CSRF token on every form, and parameters checked against the registry, never interpolated into a shell.

**The Makefile** stays for development in the host shell. Recipes that are inline shell today move into scripts, and both the Makefile and the registry call those scripts, so there is one implementation of each operation.

---

## Phase 1 -- Image, container and direct actions

`deploy/Dockerfile.operator`, Apache and mod_wsgi config, the Flask app, the action registry with its coverage test, login and TLS. Every action runs `direct` in this phase (no Slurm yet). Exit: from a browser, an operator can bring the stack up and down, pull a model, run a probe and a bench, and see each log; the coverage test passes.

## Phase 2 -- Recipes into scripts

Move each inline recipe into a script under `scripts/ops/` and make the Makefile call it; fill the registry. Exit: every documented Makefile target is an action or listed as not offered with a reason, and the Makefile's recipes are one-line script calls.

## Phase 3 -- Actions as Slurm jobs (O2)

After slurm-gatekeeper Phase 1: `job` actions submitted through slurmrestd, the job pages, cancel, hold, suspend and resume; probe actions take the engine license; only the stack-lifecycle actions stay `direct`. Exit: an action started from the browser appears in `devai-jobs list` with its log, and cancelling it removes the containers it started.

## Phase 4 -- Lab and host setup (O3, O4)

Start and stop JupyterLab lab containers and link them; the host-setup page. Exit: a lab session is one click away; every host-only item shows its state.

## Phase 5 -- Documentation

`docs/operator.md` (the reference), README and CLAUDE.md. Exit: a new user can go from the one-time host setup to a running stack through the browser, following the docs alone.

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| the web login controls the host user (podman socket), and it is on the LAN (O6) | 1 | TLS only, one strong htpasswd password (O7), CSRF tokens, parameters never passed through a shell; login attempts logged |
| the toolbox image is large (CUDA 13.1 dev packages, Go, Rust) | 1 | one image, shared layers |
| a path mounted at a different place breaks every podman call that passes it | 1 | path identity for the repo, `/var/cache/devai` and the home volume; a start-up check that refuses otherwise |
| a direct action dies with an Apache restart | 1 | started with `setsid`, outside Apache's processes |
| the Makefile and the registry drift apart | 2 | one script per operation, the coverage test |

## Migration / rollback story

The host shell and the Makefile keep working throughout; the operator is an addition until Phase 2 moves recipes into scripts, and the Makefile then calls the same scripts. Rollback is not starting `devai-operator`.

## Estimated effort

| Phase | Wall-clock |
| ----- | ---------- |
| 1 | 3-4 days |
| 2 | 3-5 days (124 targets) |
| 3 | 2 days |
| 4 | 1-2 days |
| 5 | 1 day |

Estimates, not measurements.

## References

- Apache mod_wsgi (Debian `libapache2-mod-wsgi-py3`), Flask (Debian `python3-flask`)
- [docs/slurm.md](../slurm.md) -- the bus, the jobs and the job history the actions use

# devai-operator

_A web service, in its own image and container, that offers everything the Makefile offers today, by running devai's scripts; the host shell stays for development and one-time host setup._

## Status

Draft (2026-09-30). Decisions O1-O5 below are the operator's. Nothing is built.

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
- **O2 -- Every action runs as a Slurm job.** It is queued, can be cancelled and suspended, has a time limit, and lands in the job history with its log; GPU actions take the engine license. The exception is the actions that start or stop the stack itself (bring the services up or down, rebuild or restart `devai-slurm`): Slurm may be down for those, so they run directly and are logged by the operator.
- **O3 -- Host root stays host-shell setup.** `setup-logs` (LVM, mkfs, `/etc/fstab`) and `secrets-tmpfs` (a tmpfs mount) need `sudo`; `install` / `uninstall` write `~/.local/bin`. They stay one-time host setup: the web shows whether each is done and prints the exact command.
- **O4 -- Interactive targets become links.** The web starts and stops JupyterLab lab containers and shows their links; a shell is a JupyterLab terminal.
- **O5 -- The operator runs the scripts, not `make`.** Each action is a script with declared parameters. The Makefile stays as a development convenience that calls the same scripts.

## Open questions

1. Exposure: HTTPS on the LAN with a login, as Open WebUI's proxy was on :8443 -- recommendation -- or `127.0.0.1` only?
2. Is one operator account (an Apache htpasswd entry made by the init action) enough?
3. Compose provider: `podman compose` hands off to an external provider, on this host `/usr/local/bin/docker-compose` v5.1.0, which is not a Debian package. Debian's `docker-compose` or `podman-compose` package, or a pinned binary?
4. Go and Rust: the from-source builds need Go >= 1.26 (Debian has 1.24; the host runs 1.27) and a rustup toolchain matching vLLM's `rust-toolchain.toml` (Debian has rustc 1.85). Pinned upstream downloads into the image, with their checksums -- recommendation -- or keep those two builds host-shell (development only)?

## Context

Everything devai can do is a Makefile target run in a host shell: 124 documented targets in 2,163 lines. That needs a shell on the host, leaves no record of what ran, and mixes development conveniences with operations. The Makefile's recipes are often inline shell -- long `podman run` lines, compose calls, variable defaults -- rather than calls to scripts, so there is no single entry point per operation that another program could call.

## Approach

**One image, `devai-operator`,** on the pinned `debian:trixie-slim`, holding what the scripts call today (measured by grepping the Makefile's recipes, 2026-09-30):

- Apache, mod_wsgi, Flask (Debian `python3-flask` 3.1.1) -- the web service;
- `python3`, `uv`, `git`, `curl`, `make`, podman (client) and a compose provider -- the tools the scripts use;
- for the from-source builds: `cmake`, `ninja`, CUDA 13.1 `nvcc` and headers from NVIDIA's debian13 apt repository, Go and Rust (Open question 4).

**One container,** on `devai-net`:

- the host's podman socket, read-write (`CONTAINER_HOST`), so the scripts' `podman` calls, builds and compose runs reach the host's podman;
- **path identity:** the repository, `/var/cache/devai` and the home volume are mounted at the same paths as on the host, because the scripts pass host paths to podman (`-v $(CURDIR)/scripts:/scripts`, `-v $(CACHE_DIR)/pip:...`), and podman resolves them on the host;
- the GPU device when the host has one (probes read `nvidia-smi`; the VRAM ballast allocates);
- the Slurm JWT key (read-only), to call slurmrestd;
- TLS certificate and key, the htpasswd file;
- `tini` as PID 1, then Apache.

**Actions.** A registry, `deploy/operator/actions.yaml`, lists every action: an id and group, the script and its fixed arguments, its parameters (name, type, allowed values or pattern, default, help, and how each maps to a flag or an environment variable), its kind (`job`, `direct`, `host`, `lab`), whether it needs the GPU (engine license), its time limit, whether it asks for confirmation (destructive actions such as clean, restore, stack down), and the Makefile targets it covers. A test fails when a documented Makefile target is covered by no action and not listed as not offered with a reason, so "everything the Makefile offers" stays true as the Makefile changes.

**Running an action.** The app validates each parameter against the registry and builds an argument list; nothing passes through a shell. A `job` action is submitted to slurmrestd with a script that runs `podman exec devai-operator devai-run <jobid> <script> <args...>` -- the same exec path as the engines ([docs/slurm.md](../slurm.md) Sec. 4). Containers a script starts are labelled with the job id, so a cancel or the epilog can remove them. A `direct` action is started by the app with `setsid`, outside Apache's processes (an Apache restart must not kill it), and logs to `/var/cache/devai/jobs/operator/`.

**Pages:** the stack at a glance (containers, the GPU holder, the running engine, the queue); actions by group, each with its form; jobs (queue and history, from Slurm's REST API) and a job page with its log (polled), cancel, hold, suspend and resume; the lab (start, stop, link); host setup (what is done, and the command for what is not).

**Security.** The container can do anything the host's devai user can (the podman socket), so its login is as sensitive as that user's login: TLS only, the password in an htpasswd file created once, a CSRF token on every form, and parameters checked against the registry, never interpolated into a shell.

**The Makefile** stays for development in the host shell. Recipes that are inline shell today move into scripts, and both the Makefile and the registry call those scripts, so there is one implementation of each operation.

---

## Phase 1 -- Image, container and direct actions

`deploy/Dockerfile.operator`, Apache and mod_wsgi config, the Flask app, the action registry with its coverage test, login and TLS. Every action runs `direct` in this phase (no Slurm yet). Exit: from a browser, an operator can bring the stack up and down, pull a model, run a probe and a bench, and see each log; the coverage test passes.

## Phase 2 -- Recipes into scripts

Move each inline recipe into a script under `scripts/ops/` and make the Makefile call it; fill the registry. Exit: every documented Makefile target is an action or listed as not offered with a reason, and the Makefile's recipes are one-line script calls.

## Phase 3 -- Actions as Slurm jobs (O2)

After slurm-gatekeeper Phase 1: `job` actions submitted through slurmrestd, the job pages, cancel, hold, suspend and resume; GPU actions take the engine license; only the stack-lifecycle actions stay `direct`. Exit: an action started from the browser appears in `devai-jobs list` with its log, and cancelling it removes the containers it started.

## Phase 4 -- Lab and host setup (O3, O4)

Start and stop JupyterLab lab containers and link them; the host-setup page. Exit: a lab session is one click away; every host-only item shows its state.

## Phase 5 -- Documentation

`docs/operator.md` (the reference), README and CLAUDE.md. Exit: a new user can go from the one-time host setup to a running stack through the browser, following the docs alone.

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| the web login controls the host user (podman socket) | 1 | TLS, htpasswd, CSRF, no shell interpolation; Open question 1 decides whether the LAN sees it at all |
| the toolbox image is large (CUDA 13.1 dev packages, Go, Rust) | 1 | one image, shared layers; or keep the two from-source builds host-shell (Open question 4) |
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
- [docs/slurm.md](../slurm.md) -- the exec path and the job history the actions use

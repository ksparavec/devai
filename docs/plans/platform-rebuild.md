# Platform rebuild

_Pull exactly one image from upstream; compile every engine here into one container that runs all the time; let Slurm, in its own container, decide what is on the GPU and record every job; connect the containers with a NATS bus; make the router a Slurm client; and run everything the Makefile does from a web service, as jobs. Alongside: decide SGLang's future, answer Claude Code's throwaway requests in the router, rank models before probing, diagnose failed launches, offer lanes on one port, and build the library service aiagent needs._

## Status

Draft (2026-10-05). Nothing is built or tested.

- **devai's only plan.** On 2026-10-05 it replaced every earlier plan; all were deleted, and git history keeps them. `docs/plans/README.md` lists each and what happened to it.
  - Its core: `slurm-gatekeeper`, `minimal-external-images` and `devai-operator`. Their decisions are kept below under their original IDs: M1-M15 (images), D1-D16 (Slurm), O1-O9 (operator), quoted as recorded. In them, "the Slurm plan", "the image-reduction plan" and "the devai-operator plan" now mean the matching phases of this plan.
  - The other plans: their parts that conflict with the rebuild or that it makes moot were dropped; the parts it absorbs are listed under "Absorbed from earlier plans"; their compatible or independent parts became Phases 15-21. The library-service plan's decisions are kept as LS1-LS28.
- **The architecture is [docs/slurm.md](../slurm.md).** This plan follows it section by section (see Approach).
  - Where the code needs more detail than the architecture gives, the detail is fixed under Contracts (C1-C8).
  - Where that detail adds to or differs from slurm.md, it is recorded as an implementation decision (I1-I26), taken by the operator on 2026-10-05; those that shape the architecture are folded into slurm.md.
- **21 phases.** Phases 1-14 build the new architecture; Phases 15-21 are the independent work folded in from earlier plans. Each has its files, steps, tests and exit criteria; the order table under Approach says what can run beside what.

## Dependencies

None inside devai. Outside it:

- **aiagent** (devitops-com/aiagent): the library service's field-level contract is aiagent's `docs/design/library-service-needs.md` (Phase 20). The laya trainer's API toward aiagent does not change (D14; reference: docs/laya-trainer.md).
- **Owner actions** that block parts of Phase 20 -- a contact address, API keys, contacting arXiv -- are listed there.

## Enables / Unblocks

- **The GPU.**
  - A guarantee that no engine starts on a GPU that another process still holds (the 2026-09-28 failure, see Context).
  - Queueing, suspend/resume, cancel and queue-clearing for GPU work, done by a scheduler instead of by hand.
  - One place that says what ran yesterday: every job's state, exit, times, GPU memory, utilisation and energy, plus its results.
- **Images.**
  - One distribution everywhere: every devai image is Debian trixie.
  - No floating upstream tags in the running stack. Today `latest` or `main` is what runs for the apt cache, the egress firewall, Open WebUI and 12 MCP servers.
  - Five containers fewer in the default stack, twelve today and seven after: registry-cache, open-webui, webui-proxy and the logger go (the logger into `devai-operator`, I20), the four per-engine containers (ollama, vllm, vllm-devai, sglang) become one, `devai-engines`, and `devai-slurm` and `devai-operator` are new. The opt-in mcp-gateway goes too, with up to 12 MCP containers it started per call. Two images fewer (`devai-base-cpu`, `devai-lab-cpu`).
  - About 117 GB of upstream images out of the local store.
  - One set of images for every host: a host without an NVIDIA GPU runs the same images on the CPU (M14, M15).
- **A re-bench that runs once, on final images.** The re-bench of 2026-09-27/28 is stopped; it runs again in Phase 10.
- **Operations.**
  - devai is run from a browser, without a host shell.
  - Every operator action -- build, pull, probe, bench, backup -- is in the same job history as the engines, with its log and its outcome.
  - The host needs only podman and the one-time root setup.
- **Independent work folded in (Phases 15-21).**
  - A measured decision on SGLang before it is compiled here.
  - Claude Code's throwaway requests answered without an engine.
  - Every catalog row ranked before probing; vendor launch recipes; failed launches diagnosed.
  - The owner's three library use cases: a ranked list of important recent articles with critical opinions and verifiable implementations; explanations built only from the library, every claim cited with a confidence level; software architectures built from published, tested algorithms, with their sources.
- **Later:** several local GPUs, one engine per GPU, through Slurm's GPU allocation; sandboxed execution of a paper's code (the library's verification level (b)). Not in scope here.

## Out of scope

- **GPU and cluster.**
  - Multi-host clustering. Slurm could do it later; devai does not use it.
  - Several GPUs on one host.
  - GPUs other than this host's: the engines are compiled for compute capability 12.0 (sm120) only, so a host with another NVIDIA GPU needs a rebuild for its architecture.
  - Shielding the GPU from the host user or from software outside devai. Application containers stop using the GPU by convention (D1), not because of a security boundary. A production setup would add one (driver device-file mode plus a service account); this plan does not.
- **Engines and builds.**
  - Building PyTorch, FlashInfer, Triton or other generic dependencies from source (M3).
  - Upgrading any engine version. Ollama, vLLM 0.28.0 and SGLang 0.5.16 keep their versions (M5); only how they are built changes.
  - CPU builds of vLLM or SGLang. Without a GPU, only Ollama and the laya trainer run (M15).
  - Developer tooling run ad hoc from images (gitleaks, hadolint, golangci-lint, actionlint in docs/security-ci.md). These mirror CI and are not part of the stack.
  - Bootstrapping Debian itself from the archive (mmdebstrap) to reach zero pulled images. The trust root would be the same Debian archive keys; it can be added later if an air-gapped host needs it.
- **Behaviour that stays as it is.**
  - In Phases 1-14, what the router does to requests (rewrites, reasoning policy, tool stripping, SSE keepalive): only its launch layer changes. Phases 16 and 19 add a short-circuit and a lanes port, both off by default.
  - Slurm's own job suspend. A suspended job script would stop renewing its lease, so bench tasks are re-run (D13) and engine jobs end with sleep or unload.
  - The laya trainer's API toward aiagent (D14).
- **Operator.**
  - Several users, roles or permissions in `devai-operator`: one operator account (O7).
  - Operations that need root on the host (O3).
  - A browser terminal (O4).
  - Replacing JupyterLab or the agents' own UIs.
- **Repository.** A separate project or repository (D5).
- **Library service** (Phase 20):
  - judgement of any kind: the service ranks nothing against a profile or topic, calls no LLM and runs no classifier;
  - translation (aiagent translates);
  - the storage medium, backups and exports (LS15);
  - repository clones and datasets (LS10);
  - verification level (b), sandboxed execution: a separate plan, later;
  - paid sources (LS7) and web-search APIs;
  - automated access to sources whose terms forbid it (LS8): no scraping, no captcha or Cloudflare workarounds;
  - an MCP server over the library (LS3), for now; with M10 it would need MCP reconsidered.
- **Routing by message content.** The lanes of Phase 19 are explicit only.

## Operator decisions

Quoted as recorded in the three plans this one replaces.

### Images (2026-09-30)

- **M1 -- Only Debian's official image comes from upstream,** pinned by digest. Everything else is built here from Debian packages, from hash-locked PyPI packages, or from source at a pinned tag -- the pattern `devai-ollama` and `vllm-devai` already follow.
- **M2 -- CUDA 13.1 only.** `cuda-nvcc-13-1` is the only nvcc the host's apt sources offer (`apt-cache policy`, 2026-09-30: no candidate for 12.9 or 13.0). A backend that cannot be built and run with it is dropped or upgraded. No stock builds of anything.
- **M3 -- What is compiled here, and where the rest comes from.** Compiled here: the engines and their own CUDA kernel packages -- Ollama, vLLM, and SGLang with its kernel packages. Every other dependency comes from the Debian package when its version satisfies the engine's pin, otherwise from the pinned, hash-locked PyPI wheel (as `scripts/build-vllm.sh` does for `torch==2.13.0` today). In practice nearly all of the ML stack comes from PyPI: Debian main carries torch 2.6.0 CPU-only (SGLang pins `torch==2.11.0` with CUDA) and does not package transformers; FlashInfer and Triton are not in the host's apt sources.
- **M4 -- One vLLM backend:** vLLM 0.28.0 + HyperQwen (today's `vllm-devai`), named `vllm`, on port 11435. Stock vLLM 0.22.1 (CUDA 12.9) is dropped; the `vllm-devai` backend name and port 11437 are retired.
- **M5 -- SGLang stays at 0.5.16 and is compiled here.** Only the build method changes, so a regression points at our build, not at a new release. `deploy/backend-flags.yaml` and the SGLang recovery flags stay valid. A version bump is a separate step.
- **M6 -- All backends in one image, `devai-engines`:** Ollama, vLLM, SGLang. The router and the laya trainer keep their own images.
- **M7 -- All Slurm components in one image, `devai-slurm`, run as one container** (slurmctld, slurmdbd, slurmrestd, slurmd, MariaDB), built by the Slurm plan. **No Slurm in the backends image:** jobs start engines inside `devai-engines` (the operator's answer to the Slurm session, 2026-09-30); the details are that plan's. *(Refined by D15: engines run all the time, and jobs only load and unload models.)*
- **M8 -- No per-model engine images.** The `image` field in `deploy/recovery-flags.json` (vLLM 0.25.1 for NVIDIA-Nemotron-Nano-9B-v2-NVFP4, the vLLM "gemma" build for diffusiongemma-26B-A4B-it-NVFP4) has no place in one image. A model that does not run on its backend's single version is dropped.
- **M9 -- Open WebUI is dropped,** with its nginx TLS proxy. Browser chat is the dsh web UI, which reaches every backend; Open WebUI only saw the Ollama port.
- **M10 -- MCP is dropped:** the gateway, its per-call servers and devai's own `devai-model-status` server. Deleted, not moved to `attic/`; git history keeps it.
- **M11 -- The AMD/ROCm overlay moves to `attic/`.** It never ran on ROCm hardware, and the rocm6.4 index has no torch for Python 3.14. devai is NVIDIA-only until it is restored.
- **M12 -- The sops/age secrets scaffold moves to `attic/`.** Its only remaining consumer was the MCP gateway's two servers that needed secrets. The Slurm plan can restore it if it wants its JWT key and MariaDB password encrypted.
- **M13 -- One Debian tag, `trixie-slim`,** for every image. The two tags differ by one file: `trixie-slim` carries `/etc/dpkg/dpkg.cfg.d/docker`, which keeps apt from installing docs, man pages, info pages and translations. Man pages inside containers do not matter.
- **M14 -- One lab image, no CPU variants.** `devai-base-cpu` and `devai-lab-cpu` and their targets are dropped. The remaining images lose the `-gpu` suffix (`devai-base`, `devai-lab`), since nothing distinguishes them any more. Only GPU-capable images are built (CUDA torch; the engines compiled for CUDA 13.1).
- **M15 -- GPU if the host has one, otherwise CPU.** One host check decides at launch:
  - at least one NVIDIA GPU usable through CDI -> the containers get the GPU;
  - none -> the same images run on the CPU, which works reasonably on unified-memory hosts.

  What runs on the CPU follows from the builds: Ollama (its build includes llama.cpp's CPU backend, preset `cpu` in `scripts/build-ollama.sh`) and the laya trainer (torch runs on the CPU). vLLM and SGLang are compiled for CUDA only and are not offered without a GPU. On macOS nothing changes in kind: Ollama runs natively on the host with Metal (INSTALL_macOS.md), and the lab runs in `podman machine`, now from the one lab image built for arm64.

### Slurm (2026-09-29)

- **D1 -- Workload control.** Both queueing and preemption:
  - queue a job for later;
  - suspend the running job, run a quick job (another backend and/or model), then resume the suspended job;
  - remove one job from the queue, or clear the queue;
  - when the interrupting job needs the same backend and model, the interruption must be fast (no engine restart).
  - devai application containers always go through the router and never use the GPU directly.
- **D2 -- Slurm runs in containers**, so devai stays independent of the host's Linux distribution. The containers get the privileges they need.
- **D3 -- Job results live in `/var/cache/devai/jobs`** (its own volume, per the mount-point convention in CLAUDE.md).
- **D4 -- Use a published image if one exists.** *(Superseded by D7 on 2026-09-30.)* Docker Hub has none that is maintained (2026-09-29: the `slurm/` namespace holds two repositories last updated in 2017 and 2018; the rest are personal images). SchedMD, Slurm's maintainer, publishes official images on GHCR (`ghcr.io/slinkyproject/{slurmctld,slurmd,slurmdbd,slurmrestd}:26.05-ubuntu26.04`, Slurm 26.05.4, recipe github.com/SlinkyProject/containers). Its `slurmd` was built without the `gpu_nvml` plugin.
- **D5 -- Not a separate project.** One consumer, contracts that change together with the router and the workloads, no product without devai. A second repository would make every change a cross-repo release (the aiagent experience). It stays extractable: own directory, own tests, config under `deploy/slurm/`, a written job/result format.
- **D6 -- No containers inside containers.** Ruled out after spike round 1, which ran engines as podman containers nested in the slurmd container. (It also meant, until D10, that an engine ran as the job's own process. D10 replaced that with `podman exec` into a sibling container, which is still not nesting. Since D15 no job starts a process in another container at all.)

### Slurm (2026-09-30, second round)

The first architecture (one Slurm node per backend image) was judged far too complicated; a second one (Slurm and every backend in one image) mixed Slurm with the backends.

- **D7 -- Fewer moving parts, above all fewer images.** Follow the image-reduction plan: `debian:trixie` is the only upstream image, everything else is built here. For this plan: Slurm's daemons come from devai's own Slurm build and MariaDB from Debian, which **supersedes D4**. *(Its Slurm-build part is superseded by D12: Slurm comes from Debian's packages.)*
- **D8 -- One version per backend.** One vLLM (0.28.0 + HyperQwen), one SGLang, one Ollama, one laya trainer; a model that does not run on its backend's version is dropped. One vLLM backend, `vllm` on 11435; `vllm-devai` and 11437 are retired.
- **D9 -- The backends live in one Debian trixie image, `devai-engines`, with no Slurm in it.** The laya trainer keeps its own image, also without Slurm.
- **D10 -- All of Slurm -- slurmctld, slurmdbd, slurmrestd, MariaDB and slurmd -- in one image, `devai-slurm`, run as one container.** A job starts its process inside the backend's container with `podman exec`. Chosen over putting slurmd into the backends. *(Its `podman exec` part is superseded by D15 on 2026-10-01; all of Slurm in one container stands.)*
- **D11 -- CUDA 13.1 only, everything compiled here** (the engines and their own CUDA kernels; torch, FlashInfer, Triton stay pinned, hash-locked PyPI packages). SGLang 0.5.16 is compiled here, with `sglang-kernel` built for sm120.
- **D12 -- Slurm from Debian trixie's own packages** (2026-09-30): `slurmctld`, `slurmd`, `slurmdbd`, `slurmrestd`, `slurm-wlm-jwt-plugin`, `slurm-wlm-mysql-plugin`, version 24.11.5; no source build. Slurm no longer accounts GPUs itself (D10), so it needs no NVML build, and nothing in the design needs 26.05. The current build is `24.11.5-4+deb13u1` from `trixie-security`, so the image keeps that suite enabled.
- **D13 -- A suspended bench task is re-run, not continued** (2026-09-30). inspect's per-sample clock would keep running through a suspend and score the samples in flight as wrong, so a bench job is not frozen. Suspending it cancels the running task and submits the same task again, held. Resuming releases it, and the task starts again from its first sample. The cancelled attempt stays in the history with its log and is not scored. Tasks already finished are kept (one job per model and task).
- **D14 -- The laya trainer's jobs become Slurm trainer jobs** (2026-09-30, agreed by the owner for devai and aiagent). Its fine-tuning API toward aiagent on :11438 does not change.

### Slurm (2026-10-01, third round)

- **D15 -- Engines run all the time; jobs only load and unload models.** Each engine container keeps its services running under its own supervisord; nothing in `devai-slurm` starts or stops an engine.
  - A Slurm job tells an engine to load a model onto the GPU and, when the job ends, to unload it. The engine never loads or drops a model on its own.
  - vLLM and SGLang serve one model per process, with its settings fixed at start (vLLM 0.28 source, SGLang 0.5.16 image). For them, the part that always runs is a devai control service, which starts and stops the model process when the job says so (option A).
  - The fast path: a vLLM or SGLang model that a pending job will need again is put to sleep instead of unloaded (vLLM `/sleep` level 1, SGLang `/release_memory_occupation`). At most one model sleeps at a time, and the next load of the same model wakes it.
  - The operator can start, stop, restart and reload any engine service.
  - Supersedes D10's `podman exec`.
- **D16 -- A message bus, NATS, inside `devai-slurm`** (Debian's `nats-server` 2.10.27, under devai-slurm's supervisord). Every devai container runs `devai-control`, which takes commands from the bus and publishes its state. The commands: start, stop, restart or reload a service; load, unload or sleep a model; train; run an operator action. Chosen over supervisord's built-in remote interface, which has no events and needs every client to know every container's address and password.

### Operator (2026-09-30)

- **O1 -- Apache with mod_wsgi.** Debian's `apache2` (2.4.68) and `libapache2-mod-wsgi-py3` (5.0.2): the Python app runs in Apache's own daemon processes, so TLS, login and the app are one server.
- **O2 -- Every action runs as a Slurm job.** It is queued, can be cancelled and suspended, has a time limit, and lands in the job history with its log. Actions that use the GPU directly (probes) take the engine license, while a bench uses it through the router and keeps its engine with a hold. The exception is the actions that start or stop the stack itself (bring the services up or down, rebuild or restart `devai-slurm`): Slurm may be down for those, so they run directly and are logged by the operator.
- **O3 -- Host root stays host-shell setup.** `setup-logs` (LVM, mkfs, `/etc/fstab`) and `secrets-tmpfs` (a tmpfs mount) need `sudo`; `install` / `uninstall` write `~/.local/bin`. They stay one-time host setup: the web shows whether each is done and prints the exact command.
- **O4 -- Interactive targets become links.** The web starts and stops JupyterLab lab containers and shows their links; a shell is a JupyterLab terminal.
- **O5 -- The operator runs the scripts, not `make`.** Each action is a script with declared parameters. The Makefile stays as a development convenience that calls the same scripts.
- **O6 -- On the LAN, behind a login.** HTTPS on a published LAN port (as Open WebUI's proxy was on :8443), TLS with mkcert certificates or a self-signed fallback.
- **O7 -- One operator account,** an Apache htpasswd entry created by the init action.
- **O8 -- Go and Rust as pinned upstream downloads** in the image, each checked against its published checksum: Go >= 1.26 (Debian has 1.24) and the rustup toolchain vLLM's `rust-toolchain.toml` names (Debian has rustc 1.85).
- **O9 -- Compose from Docker, pinned.** Docker's `docker-compose-plugin` comes from Docker's own apt repository (`https://download.docker.com/linux/debian`, suite `trixie`, component `stable`), pinned at `5.1.0-1~debian.13~trixie`, the version this host runs today.
  - The repository key is accepted only with Docker's fingerprint `9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88`; apt then checks the package against the signed index.
  - The package installs `/usr/libexec/docker/cli-plugins/docker-compose`, and the image's `containers.conf` names that path in `compose_providers`, so `podman compose` does not depend on a search order.

### Library service (2026-09-25/26)

Given in devai's session unless marked otherwise. LS1-LS28 were numbered D1-D28 in the library-service plan, and aiagent's documents may cite them that way; aiagent's own questions O1-O15 appear here as A-O1-A-O15. devai's answers to aiagent keep their names, O-D1-O-D13, and the library plan's Phases 1-5 are 20.1-20.5 here.

| # | Decision |
| --- | --- |
| LS1 | **The split.** aiagent owns everything that thinks: search planning and queries, classification (laya students), the multi-round loop between encoder and LLM, translation, verdicts, answers with citations, and the versioned research profile. devai owns the library service: all external network I/O, the store, the index, credentials, and model serving. |
| LS2 | **A devai service with a small API**, like the laya trainer, not aiagent's own files. |
| LS3 | **Consumers:** aiagent only, for now. An MCP server may be added later. |
| LS4 | **arXiv: the PDF plus arXiv's HTML rendering, no LaTeX source.** Chosen after the research showed that `arxiv.org/robots.txt` disallows `/e-print` and `/src` for robots. It replaces the first answer ("PDF plus LaTeX source"). |
| LS5 | **robots.txt and documented APIs.** A documented API follows its provider's API terms and limits. robots.txt governs every page and file fetch. (`export.arxiv.org/robots.txt` says `Disallow: /`, while arXiv's API terms sanction `/api/query` there; Wikipedia's robots.txt disallows `/w/` and `/api/`, while its bot policy documents those APIs.) |
| LS6 | **Sources, in this order:** arXiv; free, independent sources (open-access journals, author pages; SSRN see LS8); GitHub repositories that papers link to; carefully selected blogs, forums and Wikipedia. "Independent" means not corporate or government owned. aiagent's A-O4, answered by the owner in aiagent's session, reads this as who publishes the fetched document: company and government publications are out, and the authors' affiliations are information only. |
| LS7 | **Free sources only for now.** Paid sources may be added later. "Well-known" is a metric (for example the h-index), not a curated list. |
| LS8 | **Sources whose terms forbid automation or storage** (SSRN, Substack, Kaggle's web pages, Stack Exchange for AI use, Reddit, Drive-hosted Colab notebooks) are used only through an official API whose terms allow it, or as a link. The service first looks for the same work at another open location. |
| LS9 | **Logins** (Kaggle, GitHub, Medium, Substack, Reddit, Stack Exchange, Colab) are all wanted eventually. Honour robots.txt and every limit; never get banned. |
| LS10 | **Code: no clones, no datasets.** Keep short, relevant excerpts only, each with a permanent link to its exact commit (repository, file, lines) and the repository's licence. Verification level (a), metadata and content checks, is required now but not sufficient. Level (b), sandboxed execution compared against the paper's results, comes later and is the project's edge. |
| LS11 | **Relevance has two bounds:** the research profile is the outer bound (CS/AI, computational finance, analysis of PDEs, numerical analysis, probability); each query is the inner bound. |
| LS12 | **Rejected items keep a verdict record and no artifacts.** "A negative answer is as valuable as a positive one." |
| LS13 | **Verdict stamps.** Every verdict records the profile version and the model or student that decided. Nothing is re-judged automatically; an item is re-judged on request, or when it comes up again. Accepted items are never deleted silently. Some verdicts will go stale, or turn wrong, over time. How "comes up again" works is narrowed by aiagent's A-O1 (below): a resurfaced item is judged again without an owner override only when its current verdict is an `off_topic` reject for another topic; any other re-judge is an owner-requested override (A-O8). |
| LS14 | **Latest version only**, for papers and for code: a stored repository's excerpts stay pinned to the commit they were taken from; a refresh to a newer commit happens on request, and afterwards only the latest accepted commit's excerpts are kept. |
| LS15 | **Storage: one configurable directory**, configured once and used everywhere. Its type, backups and exports are out of this project's scope. Plain files, organised in a hierarchy like git's object store, fast for standard Linux tools, planned for millions of files. The directory may be any local path, not only under `/var/cache/devai/`, and exactly three things are required of it (2026-09-26): it is an absolute path, it already exists, and the owner can read and write it. Nothing else is a requirement: not whether it is its own volume, not its filesystem, not its size. |
| LS16 | **English only**, for everything stored. A non-English source is machine-translated first (by aiagent), marked as such with the model named; the original is kept as a link, and citations point to the original. |
| LS17 | **Search** is keyword plus vector, with an English embedding model on the CPU. Indexes are derived and rebuildable. |
| LS18 | **Citations:** section, paragraph and equation, plus the PDF page. PDF pages alone are good enough. |
| LS19 | **The research profile** lives in aiagent as a versioned file; the library records which version each verdict used. |
| LS20 | **Acquisition runs as background jobs**, and aiagent must show progress and a time estimate. A prompt that stays silent for minutes is unacceptable. |
| LS21 | **Contact identity:** a dedicated address, which the owner will create; a placeholder until then. |
| LS22 | **Blogs are selected, not crawled.** A blog qualifies when it strictly covers the topics and its author also publishes cited articles or implements algorithms on the platforms above (ideally both). |
| LS23 | **The first acceptance test:** the topic "Automatic Adjoint Differentiation method in computational finance". Not arXiv only: other sources are used where feasible. |
| LS24 | **The embedding model gets its own volume**: a new store, `/var/cache/devai/embed`, on its own LV (per the mount-point convention), filled only through `scripts/select-models.py` from a pinned catalog row in a new `deploy/embed-models.yaml`. |
| LS25 | **Other paths to the sources stay open.** The lab can reach arXiv and other sources directly (through pipelock, and trusted harnesses on devai-net; the MCP gateway's servers went with the gateway, platform-rebuild M10); the shared per-operator budget is accepted and documented, and nothing is closed. |
| LS26 | **GROBID** is used for PDFs without arXiv HTML (about 4 GB of RAM, its own container). |
| LS27 | **The anonymous arXiv PDF mirror** (Cornell's public GCS bucket) is checked in 20.1 and used only if its terms and robots.txt allow automated downloads. |
| LS28 | **Revoked items keep their artifacts for 30 days** in `trash/`, then lose them: a stated exception to LS12, as a safety net against a faulty client. |

**Relayed from aiagent's session.** The owner answered aiagent's A-O1-A-O14 there on 2026-09-26 with "go with recommendations" (aiagent's session reported it; the owner has not said it in devai's session). Those that shape devai's rules: an `off_topic` reject is scoped to its topic, so the same item may be judged again for another topic, while `out_of_bounds`, `low_quality` and `duplicate` rejects stay final (A-O1); a new arXiv version is reported as `latest_seen` and refreshed only on request, and the old version's text units are kept while its PDF and HTML are dropped, which is a small, owner-approved exception to LS14 so that old citations still resolve (A-O2); only the owner overrides a verdict (A-O8); every metadata-only rejection keeps a record (A-O9); the staging TTL for fetched items that were never judged is 30 days (A-O11). A-O13's original recommendation (arXiv only first) is superseded by LS23. A-O15 -- a grace period for the artifacts of revoked items, devai's O-D10 moved to the owner -- was answered the same day in devai's session: LS28.

**devai's own safety rules** (not owner decisions): the service makes no external request while the contact address is unset; it refuses to start when `LIBRARY_DIR` fails one of LS15's three requirements, and it never creates that directory.

**Decided by the operator on 2026-10-05, when the plan was folded in here:**

| # | Decision |
| --- | --- |
| LS29 | **GROBID is built here.** From a pinned release (the newest stable one when 20.1 starts; its tag and sha256 recorded then), in a two-stage `deploy/Dockerfile.grobid` on the pinned trixie-slim with Debian's JDK, CRF models only; no upstream image (M1). |
| LS30 | **The library's API keys are plain 0600 files,** as the Slurm keys are (C7): `~/.config/devai/library/<name>`, mounted read-only into `devai-library` only, backed up by `devai-backup`. sops/age stays in the attic (M12). |

## Implementation decisions (2026-10-05)

The plan proposed each of these as the smallest choice that lets the code be written, and the operator decided them on 2026-10-05. I1-I14 were decided first, three of them differently from the proposal (I4, I9, I10); I15-I24 the same day, after a readiness review; I25 and I26 after a second review, which also changed I24's model and defined I16's measurement. After a third review, on 2026-10-06, the operator revisited I26 and completed I16's measurement. Each is folded into slurm.md where it shapes the architecture, and each names those sections and the phase that needs it.

- **I1 -- The bus carries data, never command lines** (slurm.md Sec. 4; Phase 8).
  - The `load` request carries the router's launch configuration as typed fields (C2), and `devai-control` builds the argument list itself, with the router's builders moved into a shared package.
  - `devai-control` reads recovery flags and parser plugins from the same files as the router: `deploy/recovery-flags.json`, `deploy/recovery-flags.generated.json` (I25) and `deploy/vllm-plugins.json`.
  - So a job, or anyone holding the `jobs` bus password, cannot run arbitrary commands in `devai-engines` (slurm.md Sec. 12).
  - Probes and serving get identical command lines; today the probers build their own (`vllm_command_args`, `sglang_command_args`).
- **I2 -- The shared Go code lives in the router's module** (slurm.md Sec. 3; Phases 5 and 8): `gpu-arbiter/pkg/engineargs` (argument builders, Phase 8), `gpu-arbiter/pkg/bus` (subjects, messages, client), `gpu-arbiter/pkg/slurmrest` (JWT and the REST client).
  - `devai-tools` requires `github.com/sparavec/gpu-arbiter` with `replace => ../gpu-arbiter`; the router depends on nothing in `devai-tools`.
  - Both modules are built with the host's Go from the repository, so the relative replace costs nothing.
- **I3 -- The GPU guard reads NVML through `ctypes`** (slurm.md Sec. 3.1; Phase 5).
  - It is a stdlib Python script, `devai-gpu`, following the pattern of `scripts/vram-ballast.py`; `devai-slurm` adds Debian's `python3` for it (`python3-minimal` has no `ctypes`, which ships in `libpython3.13-stdlib`).
  - NVML, because `nvidia-smi` on this driver does not expose the energy counter (spike, 2026-09-29).
- **I4 -- The Slurm node is rendered per host** (slurm.md Sec. 3.1; Phase 5). The proposal was one static `CPUs=32`; the operator chose this instead.
  - At start, the entrypoint writes the node's `CPUs=` and `RealMemory=` into `slurm.conf`:
    - `CPUs=` is the host's CPU count from `nproc --all` (`slurmd -C` counted only 8 of 24 CPUs in the spike);
    - `RealMemory=` is `MemTotal` from `/proc/meminfo`, in MiB.
  - `SlurmdParameters=config_overrides` makes slurmd take those values as given.
  - `select/cons_tres` with `CR_CPU`, one CPU per job, so a host runs as many jobs at once as it has CPUs (24 here). With `proctrack/linuxproc` and no task plugin, nothing is pinned or confined.
  - `PrologFlags=Alloc`, so the prolog creates a job's results directory before the job script opens its log.
- **I5 -- Bench tasks are serialised by a second license, `bench:1`** (slurm.md Sec. 5 and 8; Phase 10).
  - The license keeps bench tasks one at a time; it does not touch the engine license.
  - D13's suspend: hold the run's pending tasks, cancel the running one, and submit it again, held.
  - Resume: release the re-run, wait until it runs, then release the rest.
  - Only holds and releases are used, no priorities, which would need Slurm administrator rights.
- **I6 -- Every job runs as one Slurm user, `devai` (uid 1000)** (slurm.md Sec. 3.1; Phase 5).
  - The router's and the operator's JWTs name this user (`sun`).
  - `devai` gets Slurm's `AdminLevel=Operator`, for hold, release and node resume.
  - slurmrestd runs as its own user, `slurmrest`: it will not run as root (the spike ran it as `nobody`).
- **I7 -- A node the guard drained comes back by itself** (slurm.md Sec. 6; Phase 5).
  - The prolog publishes its verdict, with the holders, on the bus (`devai.events.slurm.guard`); that is how the router names the holder in its 503.
  - Slurm's `HealthCheckProgram` runs every 30 s and resumes a node that the guard (or a failed epilog) drained, once the card is idle again.
  - Without this, a drained node runs nothing until someone resumes it by hand.
- **I8 -- A change in any of Ollama's server settings restarts `ollama serve`** (slurm.md Sec. 4; Phase 8).
  - Context length, KV cache type and flash attention are server environment in Ollama (today's `ollamaDynamicEnv`). A `load` whose settings differ from the running server's restarts it first. (The architecture first said this for the KV type only.)
  - The context length matters as much as the KV type: a `/v1` request uses the server's default context, and a mismatch with the loaded context makes Ollama load the model again.
- **I9 -- The router unloads a sleeper that no job needs any more** (slurm.md Sec. 4 and 7; Phases 8 and 9). The proposal was a lifetime (`ttl_s`) in `devai-control`; the operator chose this instead.
  - A sleeper is still needed while any of these names its launch: a PENDING job; an active hold; a RUNNING engine job (it is about to wake it); a RUNNING bench, probe or action job still within 60 s of its start without having taken its hold (a released re-run is RUNNING a moment before its hold). Without the hold and engine-job cases, a resumed bench would lose its sleeper in the gaps of a switch: between its hold and its engine job turning PENDING, and between that job turning RUNNING and its `load` arriving.
  - Once none of these holds -- typically its job was cancelled, or it ran with another launch -- the router sends `unload` with `target: sleeper` and `force`.
  - The hold and engine-job cases were added after the second readiness review.
  - The router checks this on every refresh of its pending list (every 2 s) and at boot.
  - So the `router` bus user may also publish `unload` on `devai.engines.*` (C1). `sleep` carries no lifetime.
  - While the router is down a sleeper stays; the router unloads it when it is back, if no job needs it.
- **I10 -- The router cuts over to Slurm in one step** (slurm.md Sec. 7; Phase 9). The proposal was a `DEVAI_LAUNCHER=podman|slurm` switch kept until the re-bench; the operator chose this instead.
  - Phase 9 removes the podman launch layer and the per-engine containers in the same change that adds the Slurm launcher. There is no switch.
  - Rollback is reverting Phase 9's commits, which restores the old router and compose.
- **I11 -- The laya trainer splits into its API and its run** (slurm.md Sec. 4, 5 and 7; Phases 8 and 9).
  - The controller serves the :11438 API unchanged (the 409 on a second job included) and runs permanently.
  - `devai-control` starts the run, `laya_trainer.run_job` (which the controller spawns itself today), on `train`.
  - The router submits the trainer job once the controller has accepted a create.
  - The time limit becomes Slurm's: a job is stopped at `LAYA_MAX_HOLD_S` whether or not another backend is waiting. Today the router evicts the trainer only when another backend needs the GPU.
- **I12 -- Bench and probes run from `devai-operator`** (slurm.md Sec. 8; Phase 10).
  - A bench task runs `bench_runner.py` in a lab-image container that its action starts, as `make bench-*` does today. The container gets the GPU device for NVML reads only, with `CUDA_VISIBLE_DEVICES=` empty.
  - The probers run in `devai-operator`'s own Python; that container has the GPU (Sec. 3).
- **I13 -- `devai-jobs` runs inside `devai-slurm` and `devai-operator`, never on the host** (slurm.md Sec. 9 and 12; Phase 5).
  - The host calls it through a small wrapper, `bin/devai-jobs`, which runs `podman exec devai-slurm devai-jobs "$@"`.
  - So neither slurmrestd nor NATS is published to the host, and the JWT key is never read on the host.
  - The operator web app calls the same binary, so there is one REST client (C8).
- **I14 -- Running without a GPU comes after the router moves to Slurm** (Phase 11).
  - The CPU mode of the router is written once, in the Slurm launcher, and never in the podman path that Phase 9 removes.
  - The earlier phases keep only the cheap no-GPU checks of the pieces they build (devai-slurm, devai-control).
- **I15 -- Phase 15's baseline is Qwen3.5-9B-NVFP4 on vLLM 0.28** (Phase 15; operator, 2026-10-05). No model fits both SGLang and the vLLM that survives (M4): the two models that fit vLLM 0.28 do not fit SGLang. Qwen3.5-9B-NVFP4 fits SGLang and its weights are on disk, so Phase 15 first probes it on vLLM 0.28 and then measures both engines on it. If it does not load on vLLM 0.28, gpt-oss-20b and then Gemma-4-26B-A4B-it-NVFP4 are tried; if none loads, SGLang is frozen by default (Phase 15, step 1).
- **I16 -- The SGLang keep thresholds** (Phase 15; operator, 2026-10-05). SGLang is kept only if both hold:
  - (i) its steady TTFT is within 3x of vLLM's on the same checkpoint;
  - (ii) its turn-2+ TTFT on an identical multi-turn workload is at least 25% better than vLLM's, with a non-zero engine-reported cache hit rate and a measurable gain over `--disable-radix-cache` (conditions (b)-(d) below).

  How it is measured (operator, 2026-10-05 and 2026-10-06, after the second and third readiness reviews; docs/statistics-primer.md conventions):
  - Both engines are launched by the check itself, outside the router (Phase 15), at one context (the smaller of the two probed maxima), with MTP off (the checkpoint declares `mtp:`, and the picker would launch it with `::mtp`).
  - Each engine runs with the KV-cache dtype its probe cell is stamped with, as the router would serve it: vLLM 0.28 at fp8 (step 1's probe); SGLang without `--kv-cache-dtype`, since the checkpoint has no KV quantization (its default, bf16). The check passes the dtype explicitly, never a prober's default; the context is the smaller of the two maxima probed under those dtypes; the report records both. So the ratios compare each engine as served.
  - Every request carries the same body:
    - reasoning off, as `chat_template_kwargs: {"enable_thinking": false}` at the top level, which both engines read (for the gpt-oss-20b fallback, whose template ignores it, `reasoning_effort: "low"`);
    - the same sampling;
    - no tools (the two engines use different Qwen3.5 tool parsers).
  - Steady TTFT: the bench's latency prompts (`scripts/bench/data/latency_prompts.jsonl`), prompt 0 as the warm-up, then the median over prompts 1-30. They are distinct and short (about 4-55 tokens, sharing at most 13 characters), so this measures scheduling, streaming and a short prefill, not long-prompt prefill.
  - Turn-2+ TTFT: 20 sessions per engine and configuration (vLLM, SGLang radix-on, SGLang radix-off), each of five turns with run_multiturn's sizes: a 4,096-token system prefix, 4,096 tokens added per turn, 64 tokens out (its defaults, fixed here before the runs).
    - Each session has its own salt, made of a per-run nonce, the engine and the session number. So every session starts on a new prefix: sessions are independent, and no cache is flushed between them.
    - A session's value is the median TTFT of its turns 2-5; the engine's is the median of its 20 session values.
  - Ratios are SGLang over vLLM, and radix-on over radix-off for the gain. Their 95% interval is a percentile bootstrap over session values (`boot_ci_two_sample` in scripts/stats/statlib.py, 10,000 resamples): sessions, not turns, because the turns of one session are not independent.
  - SGLang is kept only when all four hold; any other outcome is a freeze:
    - (a) the steady ratio is at most 3.0 (point estimate);
    - (b) the turn-2+ ratio, SGLang over vLLM, is at most 0.75 (point estimate), and its interval lies wholly below 1.0;
    - (c) the turn-2+ ratio of SGLang radix-on over radix-off has an interval lying wholly below 1.0 (radix-on faster): the "measurable gain over `--disable-radix-cache`" of (ii);
    - (d) the pooled hit rate of the radix-on sessions is above zero. It is the sum over sessions of the change in `sglang:cached_tokens_total` (summed over `cache_source`) over the sum of the change in `sglang:prompt_tokens_total`, each read before turn 2 and after the session's last turn (`--enable-metrics`); the per-session rates are reported too.
  - The measured run is not repeated: a failed condition is a freeze, not a reason to change the scorer. The scorer is checked before it, in an uncounted dry run (Phase 15, step 2).
- **I17 -- No downloads for the stale SGLang cells** (Phase 4; operator, 2026-10-05). The 7 SGLang cells that still carry a GPTQ "unsupported quantization" verdict -- 142 GB of weights, none on disk, Gemma-4-31B-IT-NVFP4 among them -- are removed from the SGLang probe cache. Their models stay out of the picker, each with a `manual` ledger verdict saying why. The rebuild re-probes only models whose weights are on disk. This also drops model-lifecycle-ledger's Gemma-4-31B re-check.
  - The other rows whose weights are not in their store -- 16 of the 22 vLLM-cache rows and 9 more SGLang rows on 2026-10-05 -- are not re-probed either and get no verdict. Their cells keep the old engine stamp: `make probe-check` lists them as not re-probed without failing, and the picker hides them (Phase 4, 4f-4g). A later `make model-pull NAME=<row>` and a probe bring one back.
- **I18 -- Phase 18 applies only safe flags** (Phase 18; operator, 2026-10-05).
  - An automatic fix is a flag that is safe to append, scoped with `backends` (for example `--enforce-eager` after a CUDA-graph OOM): applied once, logged and attributed, in the file of machine-made flags (I25). Every reader re-reads both recovery files at each launch: `devai-control` from Phase 8, the router and the probers from Phase 17.
  - Context, parser and memory-fraction suggestions are reported with the diagnosis and never applied: they are launch settings (C2), and a recovery flag would override every later launch.
- **I19 -- devai-operator runs as `devai` with `--userns=keep-id`** (C6; Phase 6; operator, 2026-10-05), as the lab containers do. Apache (on 8443), supervisord and every action run as uid 1000, mapped to the host user, so the files it writes on host paths belong to the host user.
- **I20 -- The log follower moves into devai-operator** (slurm.md Sec. 1, 3 and 12; Phase 6; operator, 2026-10-05). `deploy/logging.sh` becomes a supervisord program in `devai-operator`, which holds the podman socket anyway, and the `devai-logger` container goes. From Phase 6 on, `devai-operator` and the router hold the socket; from Phase 9, when the router gives it up, `devai-operator` alone, as slurm.md says.
- **I21 -- Phase 16 follows the original short-circuit design** (Phase 16; operator, 2026-10-05):
  - patterns in `deploy/shortcircuit-probes.json`, re-tunable without a rebuild;
  - streaming (SSE) handled from the start, since Claude Code streams most calls;
  - `/v1/messages` only;
  - count_tokens (16d) only if Claude Code calls it; the engine's own count (its `/v1/messages/count_tokens`, as revised after the third readiness review), never approximated.
- **I22 -- Phase 19 follows the original lanes design** (Phase 19; operator, 2026-10-05): an opt-in port, off by default; a configurable directive prefix, default `@@`; an unknown directive answers 400 listing the lanes; text content on `/v1/chat/completions` and `/v1/messages` first; broadcast refused.
- **I23 -- The laya trainer's Phase 4 is closed** (operator, 2026-10-05; it was listed under Phase 21 before). Its quality caveats stay recorded in docs/laya-trainer.md as known limits ("Quality of the installed student"). Phase 9 re-runs `make laya-check` on the new path anyway (Phase 8 runs trainer jobs by hand).
- **I24 -- The card-derived parser path is exercised without a download** (Phase 4; operator, 2026-10-05).
  - Every on-disk vLLM row has a curated `parsers:` block. Two on-disk SGLang rows lack a part of one -- Gemma-4-26B-A4B-it-NVFP4 has none, NVIDIA-Nemotron-Nano-9B-v2-NVFP4 no tool parser -- and both derive nothing on SGLang (`scripts/_card_hints.py` maps gemma4 and nemotron_json to no SGLang parser). So no re-probe exercises a positive derivation on its own.
  - Phase 4 therefore re-probes gpt-oss-20b, whose weights are in both stores, with a prober switch that ignores its curated block, its recipe seed and the name heuristic (Phase 17), so the card-derived parsers alone drive the launch. Its Harmony markup derives `openai` / `openai_gptoss` on vLLM and `gpt-oss` / `gpt-oss` on SGLang.
  - That cell goes to a scratch cache, with a scratch exclusion ledger (`--ledger`, so the run leaves `deploy/.model-status.json` alone), and is compared with the curated one.
  - gpt-oss-20b has not run on vLLM 0.28 yet. If it does not load there in 4g, the check runs on SGLang only (if kept) and records why.
  - The first choice, Qwen3.5-9B-NVFP4, rested on a wrong premise (that every vLLM/SGLang family had a curated block; 15 SGLang catalog rows have none). `_card_hints.py` refuses that checkpoint by design -- ambiguous Qwen3 XML tool markup, a bare `<think>` -- so it could only have exercised the refusal branch. The operator changed the model after the second readiness review.
- **I25 -- Machine-made recovery flags live in their own file** (C2, C6; Phases 8, 17 and 18; operator, 2026-10-05, after the second readiness review).
  - `deploy/recovery-flags.generated.json` holds every flag a program added -- recipe flags (Phase 17's `recipe-apply`) and automatic fixes (Phase 18) -- and nothing else. `deploy/recovery-flags.json` stays hand-curated, and its contract does not change.
  - Each model has a list of groups, `{backends, engine_flags, engine_env, source, reason, added}`. `backends` is required (there is no "every backend"); `source` is `recipe` or `auto-fix`; there is no `image` (M8). A missing file reads as empty.
  - Both readers -- the router's recovery registry (`pkg/engineargs` from Phase 8, used by the router and `devai-control`) and `scripts/_probe_hf_common.py` -- append, for a launch on backend B, the curated entry's flags and environment when it applies to B, then every generated group whose `backends` names B or B's engine (the curated file's rule: `appliesTo`, `_entry_applies`), in file order. The last flag still wins. Phase 17a adds the file and both readers; Phase 8 needs 17a and carries the Go reader into `pkg/engineargs`.
  - Removing a group, or the whole file, reverts what it added; the curated file is never written by a program.
  - Writers write the whole file to a temporary name and rename it (the repository's convention), and every reader re-reads both files at each launch. Today the router reads its registry only once, at boot.
  - So `devai-engines`, and the router while it still builds the engines' argument lists, mount the repository's `deploy/` directory, not single files (C6).
    - A single-file bind mount keeps the old inode after a rename, and the container goes on reading the old content (checked with podman in the second readiness review). A `git checkout` replaces files by rename too.
    - The mount uses the absolute path `${DEVAI_REPO:?}/deploy`, so an unset value stops compose instead of mounting `/deploy`. The Makefile exports `DEVAI_REPO := $(CURDIR)` (Phase 6 or 17a, whichever lands first); actions get it from C8's base environment, and the systemd unit sets it (Phase 7). A relative `./` mount is avoided because `make install-systemd` used to run compose from a copy in `~/.config/devai`, which also holds the keys (C7).
  - Tracked in git, like the curated file, so every recipe flag and automatic fix shows in `git diff` and is kept by a commit.
- **I26 -- devai-slurm stays in podman's default user namespace; each job's results directory is shared through a setgid group** (C3, C4, C6; Phases 5 and 8; operator, 2026-10-06).
  - **History.** First decided on 2026-10-05 as `--userns=keep-id`. The third readiness review found three problems with keep-id, and the operator revisited it:
    - devai-slurm's root becomes a subordinate uid, which may not signal your processes;
    - uid 1000 becomes the default user;
    - every job, which then runs as you, can read devai-slurm's keys.

    Checked with rootless podman 5.4.2 on this host on 2026-10-06, with `sleep` processes and no GPU.
  - **The guard.** Container root is the host user. With `--pid=host` it read `/proc/<pid>/cgroup` and killed three processes: one on the host, one in another default-namespace container and one in a keep-id container.
    - Killing a non-root process in another default-namespace container needs CAP_KILL, so devai-slurm keeps its default capabilities.
    - The uid `/proc` shows is translated (you appear as 0). So the guard identifies a holder by its cgroup (the `libpod-<id>.scope` segment).
    - A holder whose uid devai-slurm cannot map (shown as 65534) belongs to another host user: exit 1.
  - **Jobs.** They run as `devai`: uid 1000 inside, a subordinate uid on the host (100999 here).
    - They could not signal your processes, nor read the 0600 key files mounted into devai-slurm, which appear as root's.
    - The entrypoint copies only the jobs bus password to `/run/devai/bus/jobs.pass` (a tmpfs), owned by devai, mode 0400.
    - With `--pid=host` a job still sees every process's command line, and may signal processes that run as its own subordinate uid. So no devai service runs as uid 1000 in a default-namespace container, or as uid 999 in a keep-id one (the same host uid), and devai is not in group 0 inside devai-slurm. devai-engines and devai-laya-trainer run as root.
  - **Results.** For every job, the prolog creates `results/<jobid>/` and `results/<jobid>/artifacts/`, owned by devai, group 0 (in the default namespace, gid 0 is your own group), mode 2775. Job scripts run with umask 0002.
    - Slurm writes `job.log` as devai. devai-operator (keep-id: your uid and group) writes `action.log` and `result.json` through the group bit. The prolog and the epilog (root, that is, you) write `guard.json` and `gpu.json`.
    - Every file carries your group, but the owners are mixed: `job.log` has the subordinate uid. You can read every file and delete the directory.
    - The guard's files belong to your uid, as devai-operator's do, so devai-operator could rewrite them. Accepted: it holds the podman socket anyway.
  - **Untested until Phase 5:**
    - Slurm's own mode for `job.log`;
    - whether devai-slurm's daemons run in this layout, with or without the privileged fallback (Open question 2).

## Open questions

1. Does SGLang 0.5.16 build and serve with nvcc 13.1 on sm120? -- Phase 4 step 4a answers it. The rule for a no is already decided: drop or upgrade (M2).
2. Does slurmd run in an unprivileged container with `proctrack/linuxproc`, now that job scripts only send bus commands? -- Phase 5 step 1 answers it; the measured privileged setup is the fallback.
3. The sleep fast path: how long sleep and wake take, how much VRAM a sleeping vLLM or SGLang process keeps, and whether sleep support costs speed or memory while serving. -- Phase 8 step 1 measures it; `sleep_mode=false` by default is the fallback.

4. ~~What are the SGLang keep thresholds (Phase 15)?~~ Decided: I16.
5. ~~Short-circuit design (Phase 16)?~~ Decided: I21.
6. ~~Lanes design (Phase 19)?~~ Decided: I22.

None of them blocks coding: questions 1-3 are the first steps of their phases, each with a fallback, and the rest are decided.

## Context

Four problems, one rebuild.

**1. The GPU.** On 2026-09-28 the re-bench lost a day:

- Two bench runs overlapped (a watcher read a truncated `podman ps`).
- The router flipped the GPU between Ollama and vLLM and relaunched stock vLLM every ~10 s.
- vllm-devai then refused to start three times ("Free memory on device cuda:0 (3.25/23.43 GiB) on startup is less than desired GPU memory utilization (0.96, 22.49 GiB)").
- That tripped the router's launch circuit breaker, which refused the model for the rest of the day: every later bench task on it got a 503.

Reading the router showed why nothing stopped this (`gpu-arbiter/main.go`, `ensureBackendRunning` / `stopOtherBackends`):

- it never looks at the GPU: no free-VRAM check, no process list, between "stop the others" and "launch";
- it stops only backends its own flags call running, so anything it lost track of, and anything it did not start, is invisible;
- a failed stop is a warning and the launch goes ahead; Ollama's unload is asynchronous and followed by a fixed 2 s sleep;
- at 0.96 utilisation any holder above ~0.94 GiB makes vLLM refuse to start;
- the breaker counts such a refusal against the model, and only a router restart or another model resets it;
- the router has no GPU device at all. The device is handed to five compose services, every router-created engine, every lab and bench container, and the probers.

**2. The images.** On 2026-09-30 the operator asked to reduce the external containers to a minimum. The inventory that day (default NVIDIA setup, `attic/` excluded; sizes from `podman images`):

| Upstream image | Used for | Size |
| -------------- | -------- | ---- |
| `sameersbn/apt-cacher-ng:latest` | apt-cache | 87 MB, a personal image built 22 months ago |
| `registry:2` | registry-cache (Docker Hub pull-through) | 26 MB, 2 years old |
| `ghcr.io/open-webui/open-webui:main` | open-webui | 5.06 GB, floating `main` |
| `nginx:alpine` | webui-proxy | 64 MB |
| `quay.io/podman/stable` | logger (`podman --remote logs` only) | 641 MB |
| `ghcr.io/luckypipewrench/pipelock:latest` | pipelock, `make pipelock-ca-init` | 32 MB, floating `latest`; the binary is v3.0.0 (commit 9d2e37d) |
| `vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404` | vllm backend, `QUANT_IMAGE` | 30.3 GB |
| `vllm/vllm-openai:v0.25.1-x86_64-cu129-ubuntu2404` | one model (recovery-flags `image`) | 24.5 GB |
| `vllm/vllm-openai:gemma-x86_64-cu130` | one model (recovery-flags `image`) | 20.2 GB |
| `lmsysorg/sglang:v0.5.16-cu130` | sglang backend | 30.1 GB |
| `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04` | base of devai-base-gpu -> lab-gpu, laya trainer | 4.91 GB |
| `golang:1.27-bookworm` | build stage of router, mcp-modelstatus | 871 MB |
| `gcr.io/distroless/static-debian12` | runtime of router, mcp-modelstatus | 3 MB |
| `debian:trixie`, `debian:trixie-slim` | base-cpu; devai-ollama, vllm-devai | `trixie-slim` kept (M1, M13) |

Also: `docker/mcp-gateway:v0.43.3` plus up to 12 `docker.io/mcp/*` images pulled per call, all `latest` (opt-in profile); and three AMD images (`rocm/dev-ubuntu-24.04`, `vllm-openai-rocm`, `sglang:latest-rocm`).

Three things made this more than tidying:

- The component that decrypts all of the lab's egress (pipelock) ran a floating `latest` from a personal namespace.
- The registry mirror serves every docker.io pull over plain HTTP with no signature check (its own comment in `deploy/registries.conf`).
- The engines sat on two distributions and two CUDA majors, measured inside the images:

| Engine | Distribution | torch | FlashInfer | Triton | CUDA |
| ------ | ------------ | ----- | ---------- | ------ | ---- |
| stock vLLM 0.22.1 | Ubuntu 24.04 | 2.11.0+cu129 | 0.6.11.post2 | 3.6.0 | 12.9 |
| vllm-devai 0.28.0 | Debian 13 | 2.13.0 | 0.6.16.post3 | 3.7.1 | 13.0 runtime, 13.1 nvcc |
| SGLang 0.5.16 | Ubuntu 24.04 | 2.11.0+cu130 | 0.6.14 | 3.6.0 | 13.0 |

These engines cannot share one Python environment. And FlashInfer compiles its kernels with nvcc at first use (measured for vllm-devai on 2026-09-21), so stock vLLM would have needed a CUDA 12.9 toolkit that the host's apt sources do not carry. M2 and M4 settle that.

**3. Operations.** Everything devai can do is a Makefile target run in a host shell: 124 documented targets in 2,163 lines.

- That needs a shell on the host, leaves no record of what ran, and mixes development conveniences with operations.
- The recipes are often inline shell -- long `podman run` lines, compose calls, variable defaults -- rather than calls to scripts. So there is no single entry point per operation that another program could call.

**4. Results are scattered:** markdown, JSON, often outside the tree, with no single place that says what ran and how it ended.

**Why one plan.** Writing a process manager into the router was the first idea for problem 1. Slurm already does queueing, priorities, preemption, time limits, signals, process-tree cleanup, accounting and GPU allocation, and it is maintained. The Slurm design started as "add Slurm around today's engines". By its fourth layout (D15, D16) it needed one engines image, a control service in every container, and the operator's container as the place where bench and probes run. Image reduction and the operator therefore became parts of the same rebuild, so this plan holds all three.

**The earlier spike.** A spike on 2026-09-29 (commits 2e68b45 to e194309) tested two layouts that were then discarded: engines as containers nested in the Slurm node, and one Slurm node per backend image with the engine as the job's own process. It is obsolete. The facts it established that do not depend on the layout are the ones docs/slurm.md marks as measured:

- Slurm runs under rootless podman with a cgroup nesting step;
- slurmrestd accepts a JWT the client signs itself;
- the queue commands work;
- a second job waits on a license;
- a suspended job keeps its resources;
- the guard kills an outside GPU holder;
- MariaDB 11.8 needs its snapshot-isolation setting turned off.

## Approach

[docs/slurm.md](../slurm.md) is the architecture. This section follows it and says which phase builds each part.

### slurm.md, section by section

| slurm.md | What it describes | Built in |
|---|---|---|
| Sec. 1 What changes | the target state | all phases |
| Sec. 3 Containers and images | `devai-slurm`, `devai-engines`, `devai-laya-trainer`, `devai-operator`, `devai-router` | 1-4 (images), 5 (`devai-slurm`), 6 (`devai-operator`), 8 (supervisord and `devai-control` in engines and trainer) |
| Sec. 3.1 `devai-slurm` | image, privileges, processes, Slurm and NATS configuration | 5 |
| Sec. 4 How a job loads a model | bus verbs, what load / unload / sleep mean per engine, the engine job script | 5 (bus, `devai-control` core), 8 (engines, trainer) |
| Sec. 5 Jobs | kinds, licenses, time limits, comments | 5 (submission, scripts), 9 (engine, trainer jobs), 10 (probe, bench, action jobs) |
| Sec. 6 GPU guard | prolog, epilog, sampler, energy | 5 |
| Sec. 7 Router | Slurm client, blocked control routes, holds | 9 |
| Sec. 8 Flows | switch, bench, suspend, interrupt, restarts | 9 (switch, restarts), 10 (bench, suspend, interrupt) |
| Sec. 9 History | slurmdbd, `jobs/results/`, the bus log, `devai-jobs` | 5 (the records), 13 (`devai-jobs` joins them) |
| Sec. 10 Configuration | `deploy/slurm/`, `make slurm-init`, keys | 5, 8 |
| Sec. 11 Failure modes | each failure and its handling | tests in 5, 8, 9 |
| Sec. 12 Security | the podman socket, the bus users, the JWT key | 5 (bus users, JWT), 6 (operator login), 9 (router blocks control routes) |
| Sec. 13 From today to this | the changes | table below |
| Sec. 14 Open points | items 5-8 | Open questions 2-3 here; Phase 5 step 6 |

### slurm.md Sec. 2, interaction by interaction

| # | From -> to | Built in |
|---|---|---|
| 1 | lab agents -> router | unchanged |
| 2 | router -> engine | 4 (`devai-engines` image, still one container per engine), 9 (the permanent `devai-engines` container) |
| 3 | router -> slurmrestd | 9 |
| 4, 5, 6 | slurmrestd -> slurmctld, slurmctld <-> slurmd, slurmctld -> slurmdbd -> MariaDB | 5 |
| 7 | job scripts, guard -> bus -> `devai-control` | 5 (with a fake engine), 8 (engines, trainer), 10 (operator actions) |
| 8 | engine, trainer -> GPU | 4 (engines built), 8 (loaded by jobs) |
| 9 | operator jobs -> router (bench requests, holds) | 9 (the holds API), 10 (bench jobs) |
| 10 | `devai-operator` -> slurmrestd | 10 |
| 11 | GPU guard -> stray process | 5 |
| 12 | host volumes -> containers | 4, 5 (the jobs volume), 8 |
| 13 | `devai-slurm` -> `jobs/` | 5 |
| 14 | browser -> `devai-operator` | 6 |
| 15 | `devai-operator` -> podman | 6 |
| 16 | `devai-control` -> bus -> router, operator, bus log | 5 (bus log), 9 (router), 10 (operator) |
| 17 | `devai-operator` -> bus -> `devai-control` | 10 |

### From today to this (slurm.md Sec. 13, with phases)

| Today | Proposed | Phase |
|---|---|---|
| Open WebUI, MCP, the AMD overlay, sops/age | dropped (Open WebUI, MCP) or in `attic/` (AMD, sops/age) | 1 |
| upstream infrastructure images, floating tags, a plain-HTTP registry mirror | built here from Debian packages or source; one pinned Debian image | 2 |
| two lab and base images per host kind, on NVIDIA's Ubuntu image | one `devai-base` and one `devai-lab`, on Debian | 3 |
| 8 engine images, 3 distributions, stock builds | one engines image, trixie, CUDA 13.1, compiled here | 4 |
| vllm and vllm-devai on 11435 / 11437 | one `vllm` backend on 11435 | 4 |
| 5 engine containers recreated per model by the router | `devai-engines` + `devai-laya-trainer`, always running under supervisord; jobs load models through the bus | 4 (image), 8 (supervisor, `devai-control`), 9 (router) |
| no messaging between containers | NATS in `devai-slurm`; `devai-control` in every container | 5, 8, 10 |
| no check that the card is free | the prolog guard | 5 |
| the router holds the podman socket and recreates containers | only `devai-operator` holds the socket; the router is a Slurm client and follows engine state on the bus | 9 |
| laya busy hold in the router | trainer jobs | 8, 9 |
| bench clean slate and deadline in `bench-sync.py` | bench runs as jobs, with a time limit and a hold | 10, 14 |
| probers start engine containers | probe jobs that load through the bus | 10, 14 |
| results scattered | slurmdbd + `jobs/results/` + the bus log + `devai-jobs` | 5, 13 |
| operations are Makefile targets in a host shell | `devai-operator`, a web service running devai's scripts as jobs; the Makefile stays for development | 6, 7, 10, 12 |
| the stack assumes an NVIDIA GPU | the same images with or without one | 11 |
| SGLang's future undecided, with stale claims in the docs | a measured keep-or-freeze decision; the docs corrected | 15 |
| Claude Code's throwaway requests load models | answered in the router, off by default | 16 |
| an unprobed model has no estimate and no parser | vendor recipes and a labelled estimator | 17 |
| a failed launch needs a manual post-mortem | a diagnosis, and one recorded auto-fix retry in probes | 18 |
| one port per backend | an optional lanes port | 19 |
| no library service | the library service for aiagent | 20 |
| lab containers have the GPU | no GPU device in the lab | 14 |

### End state

| Image | Built from | Containers |
| ----- | ---------- | ---------- |
| `debian:trixie-slim` | pulled, digest-pinned | -- |
| `devai-engines` | trixie-slim + host builds | `devai-engines`: one permanent container whose supervisord keeps the engines running |
| `devai-slurm` | trixie-slim + Debian's Slurm 24.11.5, MariaDB, NATS | `devai-slurm` |
| `devai-operator` | trixie-slim + Debian's Apache, mod_wsgi, Flask; Docker's compose plugin; CUDA 13.1 compiler; Go, Rust, uv | `devai-operator`, which also runs the log follower (I20) |
| `devai-router` | empty image + host-built static binary | `devai-router` |
| `devai-laya-trainer` | `devai-base` | `devai-laya-trainer` |
| `devai-base`, `devai-lab` | trixie-slim | the lab, with or without the GPU until Phase 14, then without |
| `devai-apt-cache`, `devai-pipelock` | trixie-slim + Debian package or source | one each |
| `devai-library`, `devai-library-extract` | `devai-base` | one each (Phase 20) |
| `devai-grobid` | trixie-slim + Debian's JDK + GROBID built from source | `devai-grobid` (Phase 20) |

### Phase order

| Phase | Name | Needs | Can run beside |
|---|---|---|---|
| 1 | Remove and move to attic | -- | 5 |
| 2 | Infrastructure built here | 1 | 5 |
| 3 | One lab image on Debian | 2 | 5 |
| 4 | `devai-engines` | 3 | 5, 6, 7 |
| 5 | `devai-slurm` and the bus | -- (it uses Phase 2's Debian pin once that exists) | 1-4, 6, 7 |
| 6 | `devai-operator`: image, container, direct actions | 2, 3 (`scripts/host-gpu.sh`), and Phase 5's `make setup-jobs` (step 8, which can run on its own first: its runs write under `/var/cache/devai/jobs`) | 4, 5 |
| 7 | Recipes into scripts | 6 | 4, 5, 8 |
| 8 | Engines and trainer under `devai-control` | 4, 5, 17a (the generated recovery file and its readers, I25); its step 1 needs nothing, and runs before 4e retires `vllm-devai` | 7 |
| 9 | Router as Slurm client | 8 | 7 |
| 10 | Workloads and operator actions as jobs; the re-bench | 7, 9 | -- |
| 11 | Running without a GPU | 10 | 12, 13 |
| 12 | Lab and host setup in the web | 6 | 11, 13 |
| 13 | History | 10 | 11, 12 |
| 14 | Cleanup and documentation | 10, 11, 12, 13 | -- |
| 15 | SGLang: keep or freeze | -- (must finish before 4a) | 1-3, 5-7 |
| 16 | Router short-circuit | -- (16a early) | any |
| 17 | Catalog and fit tooling | -- (17a best before 4g; 17b's calibration after 4 and 10) | 17a beside 1-3, 5-7; 17b any time |
| 18 | Launch-failure diagnosis | 8, 10 | 11-13, 19 |
| 19 | Lanes on one port | 9 | 10-13, 18 |
| 20 | Library service | 2, 3, 6 (and 7 for its registry entries, 12 for the lab action's mount) | 4, 5, 7-19 |
| 21 | Loose ends | -- | any |

The critical path is 1 -> 2 -> 3 -> 4 -> 8 -> 9 -> 10 -> 13 -> 14. Phase 5 and Phases 6-7 run beside it and must be done before 8 and 10 respectively; Phase 15 runs beside 1-3 and must be done before 4, and 17a before 8. Phases 15-21 are outside slurm.md's scope.

## Absorbed from earlier plans (2026-10-05)

Open items of earlier plans that a phase of this plan does anyway, as part of
its own work. They were dropped from those plans and are not separate work.

| Earlier plan | Open item | Done in |
|---|---|---|
| sglang-backend-remediation | Phase 4, steps 4-5: re-probe the SGLang cells that still carry a GPTQ "unsupported quantization" verdict (7 on 2026-10-05) and the cells without a launch fingerprint, each with its load probe | Phase 4, step 4g: the cells without a launch fingerprint are re-probed with everything else; the 7 GPTQ cells are removed, not re-probed (I17: no downloads) |
| sglang-backend-remediation | Phase 1's 24-hour soak check: an SGLang model must not crash-loop through the launch breaker | Phase 10: an exit criterion of the re-bench |
| model-lifecycle-ledger | re-check Gemma-4-31B-IT-NVFP4 with its `--enforce-eager` recovery flag | dropped (I17): its weights are not on disk and are not downloaded; its vLLM `oom` verdict stands |
| model-lifecycle-ledger | a live `make model-sync` run | Phase 10: an exit criterion, once the probers load through the bus |
| review-fixes-2026-07 | the live `make test-vllm` and router run its Phase 3 required | Phases 4 and 9: their exit criteria run `make test-vllm`, `make test-sglang` and `make test-e2e` |
| bench-rewrite | bench coverage: 6 rows on 2026-10-05, none for SGLang | Phase 10: the one re-bench on the final images |
| odysseus-borrowed-ideas, RT-4 | foreground priority: background work yields to interactive requests | Phases 9-10: Slurm's queue, holds, the router's yielding, D1's interrupt |
| odysseus-borrowed-ideas, RT-5 | one aggregate health readout | Phase 9 (`/devai/v1/gpu`), Phase 13 (`devai-jobs engines`), Phase 6 (the operator's stack page) |

---

## Contracts

The interfaces the code is written against. They refine docs/slurm.md, which includes the I-decisions above.

### C1 -- Bus

**Server.** `nats-server` runs in `devai-slurm`, at `nats://devai-slurm:4222` on `devai-net`; job scripts and the guard use `127.0.0.1:4222`. No TLS (it is on `devai-net` only) and no JetStream. Messages are JSON, at most 64 KiB; log tails are cut to fit.

**Commands** are request/reply on `devai.<container>.<service>.<verb>`. **Events** are published on `devai.events.<container>.<service>` by `devai-control`, on every state change and every 30 s as a heartbeat. The guard publishes `devai.events.slurm.guard`. A subscriber treats a state older than 90 s as unknown.

| Container | Services | `devai-control` kinds |
|---|---|---|
| `engines` | `ollama`, `vllm`, `sglang`; `fake` in Phase 5's engines test container | engine; `fake` |
| `laya-trainer` | `trainer`; `controller` | trainer; program |
| `operator` | `actions`; `apache` | actions; program |

**Users and permissions.** Each user's password is a 0600 file (C7). Every client sets its own inbox prefix (`nats.CustomInboxPrefix`), so one user cannot read another's replies.

| User | Used by | May publish | May subscribe |
|---|---|---|---|
| `router` | the router | `devai.*.*.state`, `devai.engines.*.unload` (I9) | `devai.events.>`, `_INBOX.router.>` |
| `jobs` | job scripts, prolog, epilog, health check | `devai.engines.*.load`, `devai.engines.*.unload`, `devai.engines.*.sleep`, `devai.engines.*.lease`, `devai.engines.*.state`, `devai.laya-trainer.trainer.train`, `devai.laya-trainer.trainer.cancel`, `devai.laya-trainer.trainer.lease`, `devai.laya-trainer.trainer.state`, `devai.operator.actions.run`, `devai.operator.actions.stop`, `devai.operator.actions.lease`, `devai.operator.actions.state`, `devai.events.slurm.guard` | `devai.events.>`, `_INBOX.jobs.>` |
| `operator` | devai-operator's app, its actions (probers), `devai-jobs` | `devai.>` | `devai.events.>`, `_INBOX.operator.>` |
| `control-engines`, `control-laya`, `control-operator` | `devai-control` in that container | `devai.events.<container>.>` | `devai.<container>.>`, with `allow_responses` |
| `buslog` | the bus log | nothing | `devai.events.>` |

**Envelope.**

- A request is `{"v":1, "job":<Slurm job id, 0 for none>, ...}`, with the verb's own fields in place of `...`.
- A reply is `{"v":1, "ok":true, "state":{...}}`, or `{"v":1, "ok":false, "error":{"code", "message", "log_tail"}}`.
- Error codes:
  - `bad_request`;
  - `unknown_model` (not in the store);
  - `busy` (another job's model is loaded or loading in this container);
  - `not_owner`;
  - `load_failed`;
  - `timeout`;
  - `unsupported` (for example `sleep` on Ollama);
  - `no_gpu`.

**Verbs** (slurm.md Sec. 4, made concrete):

| Verb | Fields | Effect and reply |
|---|---|---|
| `state` | `job` (for `actions` only) | the service's state; for `actions`, that job's run |
| `start`, `stop`, `restart` | -- | the same on the supervisord program; then its state |
| `reload` | -- | re-read the program's configuration; restart it if it changed |
| `load` | `launch` (C2), `timeout_s` (default 600) | replies once serving, or `load_failed` with the log tail. The same launch already serving for the same job is a no-op; the same launch sleeping is woken |
| `unload` | `target` (`active`, the default, or `sleeper`), `force` (default false), `timeout_s` (default 60) | unloads the loaded model, or the sleeping one; replies once the program has exited and NVML's used memory is back to its level before that load (+256 MiB) |
| `sleep` | -- | vLLM `/sleep?level=1`, SGLang `/release_memory_occupation`; `unsupported` on Ollama. The sleeper stays until a load wakes or replaces it, or until an `unload` with `target: sleeper` (I9) |
| `lease` | -- (published, no reply) | renews the job's lease on its model, its trainer run or its action run; after 60 s without one, the model is unloaded, the trainer run cancelled, or the action's process group stopped as `stop` with `TERM` and a 30 s grace would |
| `train`, `cancel` | `ft_job`, `reason`; `cancel` also `force` | start or stop one laya run. `cancel` without `ft_job` stops the run of the sending job, or with `force` whatever run is going; the trainer's state names its run's `job` and `ft_job` |
| `run`, `stop` | `run`: `action`, `params`, `kind` (the kind of the job sending it); `stop`: `signal` (`INT` or `TERM`), `grace_s`, and `force` with `run_job` (the job whose run to stop) | `run` starts a registered action (C8) for the job and replies once it has started (`running`); it is refused (`bad_request`) when the action's `job_kind` is not `kind`, or when a `stop` for that job came first. The run's end is an event: `exited` with `exit_code`, or `lost`. `stop` signals the run's process group and replies at once; SIGKILL follows after the grace |

A loaded model belongs to the job that loaded it, and a trainer or action run to the job that started it. `unload`, `sleep`, `stop`, `cancel` and `lease` from another job are refused with `not_owner`. The exception is `unload`, `stop` and `cancel` with `force`, which the guard and the operator send, and the router for a sleeper's `unload`.

**State.** `{"container", "service", "status", "job", "launch", "since", "build", "sleeper", "error"}`:

- `status` is one of `idle`, `loading`, `serving`, `sleeping`, `waking`, `unloading`, `failed`, `stopped`. An `actions` service keeps one state per job instead -- `running`, `exited` with `exit_code`, or `lost` (a restarted `devai-control` adopted the run but cannot collect its exit status) -- which `state` with that `job` returns; an ended run's state is kept 10 minutes.
- `sleeper` is `{"job", "launch", "since", "used_mib"}` or null. `used_mib` is the card's used memory that `devai-control` read right after the model went to sleep, minus its reading before that model's load. The guard tolerates that much as the sleeper's (Phase 8).
- `error` is `{"code", "message", "log_tail"}` or null.
- `build` is the engine build id, read from `/etc/devai/engine-builds.json`, which `Dockerfile.engines` writes with the same content hashes as its `devai.engine.<backend>` labels (Phase 4, step 4c). The router compares it with the probe cache's stamp, without podman.

**Memory readings.** `devai-control` reads the card with `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`; CDI puts `nvidia-smi` into every container that gets the GPU. Without a GPU it skips every memory check.

### C2 -- Launch settings (the `launch` object of a `load`)

| Field | Engines | Type and check | Today's source |
|---|---|---|---|
| `backend` | all | `ollama`, `vllm` or `sglang` (`fake` for Phase 5's fake engine); must match the service | -- |
| `model` | all | a catalog name passing the router's `isSafeModelName`; for vLLM and SGLang, `<store>/<model>` must exist | the request |
| `ctx` | all | int, 1024 to `MAX_CONTEXT_LEN` | `launchConfig.MaxContext` |
| `kv_cache_dtype` | all | vLLM/SGLang: `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`; Ollama: `f16`, `q8_0`, `q4_0` | `KVCacheType` |
| `flash_attention` | ollama | bool or null | `FlashAttention` |
| `num_gpu` | ollama | 999 (GPU) or 0 (a host without a GPU) | `warmLoadOllama` |
| `mem_fraction` | vllm, sglang | 0.10 to 0.98 | `MemFraction` |
| `max_num_seqs` | vllm, sglang | int >= 0; 0 omits the flag | `MaxNumSeqs` |
| `tool_parser`, `reasoning_parser` | vllm, sglang | `^[a-z0-9_]{1,64}$` or empty | `ToolParser`, `ReasoningParser` |
| `mtp` | vllm, sglang | null, or the catalog's `mtp:` block (`configSpeculative`) | `Speculative` |
| `sleep_mode` | vllm, sglang | bool; default true (D15) | -- |

Some settings are not in the request: `devai-control` resolves them from files mounted read-only (I1):

- recovery flags and environment, by backend and model: first `deploy/recovery-flags.json` (hand-curated; that file's own contract, also named C2), then `deploy/recovery-flags.generated.json` (machine-made, I25); both re-read at every load (I18, I25; `devai-control` from Phase 8);
- parser plugin paths, from `deploy/vllm-plugins.json`.

All three come from the `deploy/` directory, mounted read-only (C6). A recovery flag that sets a launch setting overrides the C2 field, since the last flag wins: 9 of the 14 curated entries set `--gpu-memory-utilization` or `--mem-fraction-static`, the Qwen3.8 rows among them. Machine-made flags never set one (I18; Phase 17).

The router and the probers compute the fields as they do today (`computeLaunchConfig`; `host_scaled_fraction` for a probe band). `gpu-arbiter -print-load <backend> <name>[::mtp][@ctx]` prints the request the router would send, for tests.

Two launches are the same when backend, model, ctx, `kv_cache_dtype` and `mtp` are equal -- today's `currentModel` / `currentContext` / `currentSpec` comparison. For Ollama, a bare `<name>` matches whatever context is loaded; only `<name>@<ctx>` asks for a context (today's `ctxPinned` rule).

### C3 -- Jobs

**Submission.** `POST /slurm/v0.0.42/job/submit`, with the headers `X-SLURM-USER-NAME: devai` and `X-SLURM-USER-TOKEN: <JWT>`. The JWT is HS256, `{"sun":"devai", "iat":<now>, "exp":<now+300>}`, signed with `jwt_hs256.key`. The field names below are checked against 24.11's own spec in Phase 5, step 2.

Every job sets:

- `script`: two lines, `#!/bin/bash` and `exec /usr/local/lib/devai/jobs/<kind>.sh`. The job scripts ship in the `devai-slurm` image, so they change with it, not with the router.
- `name`: `engine-<backend>`, `trainer`, `probe-<backend>`, `bench-<task>` or `action-<id>`.
- `licenses`, `time_limit` and the warning signal, per kind (table below).
- `current_working_directory`: `/var/cache/devai/jobs/results`.
- `standard_output` and `standard_error`: `/var/cache/devai/jobs/results/%j/job.log`.
- `environment`: `PATH=/usr/local/bin:/usr/bin:/bin`, `DEVAI_BUS_URL=nats://127.0.0.1:4222`, `DEVAI_BUS_USER=jobs`, `DEVAI_BUS_PASSWORD_FILE=/run/devai/bus/jobs.pass` (the copy devai can read, I26), plus the kind's own (`DEVAI_BACKEND`, `DEVAI_LOAD`, `DEVAI_FT_JOB`, `DEVAI_ACTION`, `DEVAI_PARAMS`). A job gets only the environment its submitter passes (measured).
- `comment`: JSON, under 1 KiB: `{"v":1, "kind", "backend", "model", "ctx", "mtp", "requester", "git", ...}`. Per kind:
  - trainer: `ft_job`;
  - probe: `action`;
  - bench: `run`, `task`, `attempt`, `rerun_of`;
  - action: `action`.

| Kind | Licenses | Time limit and warning signal | Its script |
|---|---|---|---|
| engine | `engine` | none | `load` `$DEVAI_LOAD`; hold. TERM -> `unload`; USR1 -> `sleep` |
| trainer | `engine` | `LAYA_MAX_HOLD_S`, rounded up to whole minutes, Slurm's unit (0 = unlimited); `B:USR2@30` | `train`; hold. TERM -> `cancel`; USR2 -> `cancel` with reason `time_limit` |
| probe | `engine` | 60 min | `probe.sh`, one model (one model and band for Ollama): lease the model the prober loads with this job id (`devai-bus hold devai.engines.$DEVAI_BACKEND --job $J --lease-only`); `run` the per-model probe action `probe-<backend>-model` (C8) in `devai-operator`, and hold the run. TERM -> `stop` the action and wait for it, then `unload` the model (the job owns it) |
| bench | `bench` (I5) | 32 min; `B:INT@120` | `bench.sh`: `run` the bench-task action and hold the run. INT -> `stop {signal: INT, grace_s: 110}` and wait for the run to end; TERM -> `stop {signal: TERM}` and wait |
| action | none | per action, from the registry (C8) | `action.sh`: `run` and hold the run. TERM -> `stop` and wait |

These time limits are defaults; a C8 registry entry's `time_limit_min`, when set, wins for the jobs its action submits.

`devai-bus hold <subject> --job $J --parent $$` holds a model, a trainer run or an action run:
- It subscribes to the service's events first and then asks for its state, so a run that ended before it started is not missed. On `devai.operator.actions` it exits 0 when the action exits 0, 1 when it exits non-zero, and 3 when the run is `lost`; when the job has no run, it exits 0 at once.
- It renews the lease every 10 s.
- It ignores SIGTERM and SIGINT. On `scancel` and on the time limit, Slurm signals every process of the batch step, not only the script (24.11.5: slurmd's `_kill_all_active_steps` with SIGTERM; `proctrack/linuxproc` signals the whole process tree). The hold must outlive that signal to see its run end. Only `--batch` / `B:` signals reach the script alone. Slurm's SIGKILL after `KillWait` still ends it.
- It dies with its script: it sets `PR_SET_PDEATHSIG(SIGKILL)`, and exits at once if its parent is no longer the `--parent` pid. A hold whose script was SIGKILLed is re-parented to `tini -s` (C6) and so leaves the step's process tree; it would otherwise renew its lease for ever.
- With `--lease-only` it only renews a lease, and ends on SIGHUP from its script.
- `devai-bus request` keeps the default handling of SIGTERM, so a trap is never held up behind a long `load`.

If the script dies without its trap (SIGKILL, `devai-slurm` restarting), `devai-control` stops the run 60 s after the last renewal (C1). So no action outlives its job, and the prober of a dead probe job cannot go on loading models outside the engine license.

The action job script, `deploy/slurm/jobs/action.sh`:

```bash
#!/bin/bash
umask 0002
A=devai.operator.actions J=$SLURM_JOB_ID H=
stop() {   # stop replies at once (C1); the job ends only once the run has ended
  devai-bus request $A.stop "{\"job\":$J,\"signal\":\"$1\",\"grace_s\":$2}" --timeout 5s
  [ -n "$H" ] || { devai-bus hold $A --job "$J" --parent $$ & H=$!; }   # TERM came during `run`
  wait "$H"; rc=$?
  [ "$rc" = 1 ] && rc=0     # the run ended because this script stopped it; its own status is in result.json
  exit $rc
}
trap 'stop TERM 20' TERM          # scancel or the time limit; Slurm sends SIGKILL after KillWait (30 s)
devai-bus request $A.run "{\"kind\":\"action\",\"action\":\"$DEVAI_ACTION\",\"params\":$DEVAI_PARAMS}" --job "$J" --timeout 30s || exit $?
devai-bus hold $A --job "$J" --parent $$ &   # ignores TERM and INT; exits with the run: 0, 1 or 3
H=$!
wait $H
```

- `bench.sh` sends kind `bench` and adds `trap 'stop INT 110' INT` for Slurm's `B:INT@120` warning. The task scores its unbroken prefix and exits; only then does the job end and release the `bench` license.
- `probe.sh` sends kind `probe`. Before `run` it starts `devai-bus hold devai.engines.$DEVAI_BACKEND --job "$J" --lease-only --parent $$ &` for the model the prober loads with this job id. Once the run's hold has returned, on either path (its `stop()` and its normal end), it sends that model's `unload` (the job owns it: no `force`) and ends the lease with `kill -HUP`.

Exit codes, the same for `devai-bus` and for every job script, which passes them through:

- 0: done; a job ended normally, including after a cancel or a stop its own script sent;
- 1: refused or failed (an error reply such as `load_failed` or `busy`, or an action that exited non-zero on its own);
- 2: the bus could not be reached;
- 3: the service left its active state on its own (an engine `failed`, or an action run `lost`).

The engine job script, `deploy/slurm/jobs/engine.sh` (slurm.md Sec. 4 shows the same script):

```bash
#!/bin/bash
umask 0002
S=devai.engines.$DEVAI_BACKEND J=$SLURM_JOB_ID
trap 'devai-bus request $S.unload "{\"job\":$J}" --timeout 25s; exit 0' TERM INT
trap 'devai-bus request $S.sleep "{\"job\":$J}" --timeout 120s; exit 0' USR1
devai-bus request $S.load "$DEVAI_LOAD" --job "$J" --timeout 660s || exit $?
devai-bus hold $S --job "$J" --parent $$ &   # a lease every 10 s; ignores TERM and INT; exits 0 when the service stops serving this job, 3 on failed
wait $!
```

How the router ends an engine job:

- To unload it, the router cancels it: Slurm sends SIGTERM, waits `KillWait` (30 s), then sends SIGKILL.
- To make it sleep, the router signals the batch script alone, `DELETE /slurm/v0.0.42/job/<id>?signal=SIGUSR1&flags=BATCH_JOB` (`scancel --signal=USR1 --batch`).
- The trap's `unload` waits at most 25 s. The epilog sends `unload` again and waits up to 60 s, so a slow shutdown past `KillWait` is still cleaned up. The epilog skips that `unload` when the service reports `sleeping` for this job. For a trainer job the epilog sends `cancel` instead (no `ft_job`: the run of that job; reason `epilog`; a no-op once the run has ended).

### C4 -- Files under `/var/cache/devai/jobs`

This is its own volume (D3), created by `make setup-jobs`. A job's directory is shared through a setgid group (I26). The prolog creates `results/<jobid>/` and its `artifacts/` owned by devai, group 0 (the host user's group), mode 2775. `job.log` belongs to devai's subordinate uid and everything else to the host user; every file carries the host user's group.

```
results/<jobid>/job.log          the job script's output (Slurm)
results/<jobid>/result.json      written by the job's work (bench, probe, action)
results/<jobid>/action.log       an action's output (devai-control in devai-operator)
results/<jobid>/gpu.json         written by the epilog of engine, trainer and probe jobs
results/<jobid>/guard.json       the prolog's verdict: idle, unloaded, killed or busy, with the holders
results/<jobid>/artifacts/       anything else the job keeps
engines/<backend>/<jobid>.log    a vLLM or SGLang model process's output (devai-control)
engines/ollama/serve.log         ollama serve's output, rotated by supervisord (100 MB x 5)
bus/<YYYY-MM-DD>.jsonl           every bus event, one per line, with its receive time (the bus log)
operator/<run>/                  direct actions: run.json and log (devai-operator)
mariadb/                         MariaDB's data directory
slurmctld/                       slurmctld's StateSaveLocation
```

- `gpu.json`: `{"v":1, "job", "device", "interval_s":5, "started_at", "ended_at", "samples":[[t, used_mib, util_pct], ...], "peak_mib", "mean_mib", "mean_util_pct", "energy_j"}`. The epilog also puts `gpu peak=<MiB> mean=<MiB> util=<pct> energy=<J>` into the job's `AdminComment` (Phase 5 step 6 checks that 24.11 accepts this from an epilog).
- `result.json`: `{"v":1, "job", "kind", "status", "started_at", "ended_at", "summary", "outputs"}`.
  - `status` is one of `ok`, `failed`, `truncated`, `cancelled`.
  - `outputs` depends on the kind. Bench: `{model, backend, ctx, task, n_planned, n_scored, score, cache_key}`. Probe: `{model, backend, cells_written}`. Action: `{exit_code}`.

### C5 -- The router's own API

The router serves `/devai/v1/` on every backend port itself and never proxies that path.

| Method and path | Behaviour |
|---|---|
| `POST /devai/v1/holds` `{job_id, model}` | `model` is written as a client sends it (`<name>[::mtp][@ctx]`). The hold is checked against Slurm: `job_id` must be RUNNING, of kind bench, probe or action, and its comment must name this backend and model; otherwise 403. Starts or keeps the engine and answers 200 with the GPU record once it serves (long-poll up to `HEALTH_TIMEOUT_SECONDS`). 409 when another active hold needs a different launch; 503 + `Retry-After` when a trainer or probe job holds the GPU |
| `DELETE /devai/v1/holds/{job_id}` | 204. A hold is also dropped when its job leaves RUNNING, checked on every decision and every 30 s |
| `GET /devai/v1/gpu` | the GPU record: the holder's job, kind, backend, launch and state; the holds; the sleeper; the pending jobs and the launches they name |
| `GET /devai/v1/ready` | 200 only when slurmrestd and the bus answer, the probe caches are loaded and the model stores are mounted; 503 naming what is missing. `/health` stays a liveness answer (Phase 9; from odysseus RT-6) |

The router also blocks the engines' control routes on every listener, before the proxy, as `makeMutationGuard` blocks Ollama's store-changing paths today. It answers 403 with a body naming `devai-control`.

- vLLM: every route that `VLLM_SERVER_DEV_MODE` turns on (`/sleep`, `/wake_up`, `/is_sleeping`, `/collective_rpc`, weight updates and the rest).
- SGLang: its memory and weight routes (`/release_memory_occupation`, `/resume_memory_occupation`, `/update_weights_from_disk` and the rest).

The exact lists are read from the 0.28 and 0.5.16 sources in Phase 9 step 1 and pinned by a test.

### C6 -- Ports and mounts of the permanent containers

| Container | Ports | Mounts |
|---|---|---|
| `devai-slurm` | 6820 (slurmrestd) and 4222 (NATS) on `devai-net`; nothing published | podman's default user namespace, so its root is the host user, with its default capabilities (I26); `tini -s` as its entrypoint, since under `--pid=host` tini is not PID 1; a tmpfs at `/run/devai`; `/var/cache/devai/jobs` read-write; its C7 files; the bus passwords of every user, to render the server configuration. Inside, these appear as root's 0600 files; only the jobs password is copied for devai (I26) |
| `devai-engines` | Ollama 11434, vLLM 11435, SGLang 11436 on `devai-net` -- one port per engine, since they share a network namespace | the Ollama store at `/root/.ollama` read-write; the vLLM and SGLang stores at their host paths, read-only; the FlashInfer cache volumes; `jobs/engines/` read-write; `${DEVAI_REPO}/deploy` at `/etc/devai/deploy` and the plugin directory, read-only -- directories, so a file replaced by rename is seen, by absolute path (I25); its bus password; its processes run as root (I26) |
| `devai-laya-trainer` | the controller on 11434, on `devai-net` | `/var/cache/devai/laya` at `/laya`; its bus password |
| `devai-operator` | `${OPERATOR_PORT:-8443}` published on all interfaces (O6) | runs as `devai` with `--userns=keep-id` (I19); the host's podman socket read-write; the repository, `/var/cache/devai` and the home volume at their host paths (path identity), and `LIBRARY_DIR` the same way through `deploy/compose.operator-library.yaml`, which `make operator-up` adds only when it is set (Phase 20); TLS files, htpasswd and the HF token read-only (the token at `/etc/devai/operator/hf-token`, named by `HF_TOKEN_PATH`); from Phase 10, the JWT key and the `operator` and `control-operator` bus passwords; the GPU when present |
| `devai-router` | 11434, 11435, 11436, 11438 (11437 retired in Phase 4); `DEVAI_FANOUT_PORT` when Phase 19's lanes are on | the JWT key; its bus password; no podman socket (from Phase 9) |

Each engine keeps a cache directory of its own, as with today's per-backend volumes. The `devai-engine` launcher sets `HOME=/var/lib/devai-engines/<backend>` (Phase 4, step 4d), and the `devai-engine-cache-<backend>-*` volumes mount under it. Before every vLLM start, `devai-control` deletes `$HOME/.cache/vllm` and `/tmp/torchinductor_*` (docs/router.md: with that cache, vLLM under-measured activation memory and OOM-killed engines).

### C7 -- Configuration and secrets

`make slurm-init` (`scripts/slurm-init.sh`) and `make operator-init` (`scripts/operator-init.sh`) create these files once, mode 0600, never overwriting. `devai-backup` includes `~/.config/devai/slurm/`, `~/.config/devai/operator/` and `~/.config/devai/library/`.

| File | Content | Mounted read-only into |
|---|---|---|
| `~/.config/devai/slurm/slurm.key` | auth/slurm key, 1024 random bytes | `devai-slurm` |
| `~/.config/devai/slurm/jwt_hs256.key` | 32 random bytes | `devai-slurm`, the router, `devai-operator` |
| `~/.config/devai/slurm/mariadb.pass` | random | `devai-slurm` |
| `~/.config/devai/slurm/nats/<user>.pass` | one per bus user (C1) | `devai-slurm` (all of them); each client its own |
| `~/.config/devai/operator/htpasswd` | the one operator account (O7), bcrypt | `devai-operator` |
| `~/.config/devai/operator/tls/{cert,key}.pem` | mkcert certificate if present, else self-signed (O6) | `devai-operator` |
| `~/.config/devai/library/<name>` | the library's API keys (LS30): `openalex.key`, `github.token`, `hf.token`, `s2.key`, `openreview.credentials`, written by the owner; the directory is created by `make library-init` | `devai-library` |
| `HF_TOKEN_FILE` (default `~/.cache/huggingface/token`) | the Hugging Face token model pulls use today | `devai-operator`, at `/etc/devai/operator/hf-token`; `HF_TOKEN_PATH` points the `hf` CLI there |

The router's environment, set in compose:

- `SLURM_REST_URL=http://devai-slurm:6820`, `SLURM_JWT_KEY_FILE`, `SLURM_USER=devai`;
- `DEVAI_BUS_URL=nats://devai-slurm:4222`, `DEVAI_BUS_USER=router`, `DEVAI_BUS_PASSWORD_FILE`.

`devai-control` reads `/etc/devai/control.yaml`, one per container, and is started as `devai-control --container <name> --config /etc/devai/control.yaml`; the flag must equal the file's `container:`. The guard finds each container's `devai-control` by that command line (Phase 5). The argument builders take each engine's port from this file (Phase 8). The engines' copy:

```yaml
container: engines
bus: {url: "nats://devai-slurm:4222", user: control-engines, password_file: /etc/devai/bus/control-engines.pass}
supervisor: {socket: /run/supervisor.sock, program_dir: /run/devai/supervisor}
state_file: /run/devai/control.json
lease_timeout_s: 60
services:
  ollama: {kind: ollama, program: ollama, port: 11434, log: /var/cache/devai/jobs/engines/ollama/serve.log}
  vllm:   {kind: vllm,   port: 11435, store: /var/cache/devai/vllm,   log_dir: /var/cache/devai/jobs/engines/vllm}
  sglang: {kind: sglang, port: 11436, store: /var/cache/devai/sglang, log_dir: /var/cache/devai/jobs/engines/sglang}
```

### C8 -- The operator's action registry (`deploy/operator/actions.yaml`)

Both the web app and `devai-control` (kind `actions`) read it and check every parameter against it. Nothing runs through a shell:

- the argument list is `[script] + args + the mapped parameters`;
- the environment is a fixed base plus the parameters mapped to variables. The base: `PATH`, `HOME`, `DEVAI_REPO`; for a `job` action also `DEVAI_JOB_ID` and `DEVAI_RESULTS_DIR` (`/var/cache/devai/jobs/results/<jobid>`);
- a `host` action is never run, only shown with its command.

```yaml
version: 1
actions:
  - id: model-pull                    # [a-z0-9-]+, unique
    group: models                     # stack, build, models, probe, bench, test, backup, logs, lab, host, library, maintenance
    title: Pull a catalog model
    kind: job                         # job (a Slurm job) | direct (stack lifecycle, O2) | host (shown only, O3) | lab (O4)
    script: scripts/ops/model-pull.sh # repository-relative, run directly
    args: []                          # fixed arguments, before the parameters
    params:
      - name: name
        type: string                  # string | int | bool | enum (with `values`)
        pattern: '^[A-Za-z0-9._:@-]{1,128}$'
        required: true
        to: {flag: --name}            # or {env: NAME}; a bool flag is present or absent
        help: Catalog row name
    job_kind: action                  # action | probe | bench: the Slurm job (C3) a run is submitted as,
                                      # which sets its license; devai-control refuses a run from another kind
    time_limit_min: 240
    confirm: false                    # true for destructive actions: the page asks again
    covers: [model-pull]              # the Makefile targets this action replaces
not_offered:
  - target: shell
    reason: interactive; a JupyterLab terminal replaces it (O4)
pending: []                           # targets not yet moved to scripts; Phase 7 empties it
```

A `job` action is always submitted as the job kind its entry names, from the web and from `devai-jobs submit` alike, so a per-model probe started from the page runs as a probe job, with the engine license and the guard. `devai-control` refuses a `run` whose `kind` differs. This guards against mistakes, not attacks: whoever can submit holds the JWT key anyway (slurm.md Sec. 12).

The web app talks to Slurm only through `devai-jobs ... --json` (I13), and to the bus only through `devai-bus`. So there is one REST client and one bus client in Go, and none in Python.

---

## Phase 1 -- Remove and move to attic

### Goal

Open WebUI and MCP are gone; the AMD overlay and the sops/age scaffold sit in `attic/` with restore notes. No GPU, probe or bench data is touched.

### Deliverables

```
deploy/docker-compose.yaml     modify -- open-webui, webui-proxy, mcp-gateway (+ mcp-state volume) removed
deploy/webui-proxy/            remove
deploy/mcp-catalog-devai.yaml, deploy/mcp-gateway.env, deploy/mcp-secrets.sops.env.example,
deploy/Dockerfile.mcp-modelstatus, scripts/mcp-health.sh
                               remove
devai-tools/cmd/devai-mcp-modelstatus, devai-tools/internal/modelcache, devai-tools/internal/routerclient
                               remove -- routerclient and modelcache have no other importer
tests/test-mcp.sh, tests/test-mcp-modelstatus.sh, tests/python/test_mcp_gateway_phase1.py,
tests/python/test_mcp_gateway_phase2.py, tests/fixtures/modelstatus/
                               remove
docs/mcp.md, docs/mcp-model-status.md
                               remove
attic/amd-rocm/                new    -- devai-tools/cmd/devai-gpu-vendor, devai-tools/internal/envfile
                                         (its only user), tests/test-gpu-vendor.sh,
                                         tests/python/test_devai_agent_gpu_vendor.py, docs/gpu-vendors.md,
                                         the removed Makefile targets and AMD branches, RESTORE.md
attic/sops-age/                new    -- .sops.yaml, scripts/age-keygen-host.sh, scripts/render-secret.sh,
                                         deploy/setup-secrets-tmpfs.sh, docs/secrets.md,
                                         tests/python/test_sops_age_scaffold.py, the removed Makefile targets, RESTORE.md
attic/README.md                modify -- index both
Makefile                       modify -- mcp-*, build-mcp-modelstatus-image, test-mcp-modelstatus, gpu-vendor,
                                         test-gpu-vendor, secrets-*, age-keygen-host removed; AMD branch of
                                         GPU_BASE_IMAGE removed; MCP names out of PIPELOCK_NO_PROXY
deploy/Dockerfile.lab          modify -- ROCm branch of the torch step removed
bin/devai-agent                modify -- --gpu-vendor removed; MCP names out of NO_PROXY_HOSTS
devai-tools/internal/backup    modify -- the age-key entry (manifest.go, restore.go) removed, or it warns on every backup
```

### Detailed steps

1. Open WebUI: remove both services. Its data under `/var/cache/devai/open-webui` stays until the operator removes it (it may be its own volume, per the mount-point convention). The mkcert certificates JupyterLab uses are unaffected. Port 8443 is then free for `devai-operator` (O6).
2. MCP: delete the listed files, and remove MCP mentions from the pipelock allowlist if any exist only for MCP servers.
3. Move the AMD overlay to `attic/amd-rocm/`, the way `attic/cluster-mode/` was done: files moved, not deleted.
   - `DEVAI_GPU_DEVICE` stays: it is the CDI device string for NVIDIA too.
   - Tests that pin the AMD branches elsewhere (`gpu-arbiter/main_test.go`, `test_lab_python_and_torch.py`, `test_hf_store_linking.py`, `test_ollama_image_cutover.py`) keep only their NVIDIA cases.
4. Move sops/age to `attic/sops-age/`. (Its plan was removed on 2026-10-05.)

### Exit criteria

- `make cache-up` brings the stack up without open-webui, webui-proxy and mcp-gateway.
- `make test-router`, `make test-devai-tools`, `make test-python` and `make test-backup-restore` pass.
- No reference to the removed files outside `attic/` and history docs.

### Phase 1 risks

| Risk | Mitigation |
| ---- | ---------- |
| someone still uses Open WebUI or an MCP client config | operator decisions M9, M10; the dsh web UI covers browser chat |
| devai-backup warns about a missing age key forever | its age-key entry is removed in this phase |

---

## Phase 2 -- Infrastructure built here

### Goal

Every always-on service except the engines runs from an image built here; `debian:trixie-slim` is the only pulled image.

### Deliverables

```
deploy/Dockerfile.apt-cache         new    -- trixie-slim + Debian apt-cacher-ng (3.7.5)
scripts/build-pipelock.sh           new    -- host Go build of pipelock at v3.0.0 (commit 9d2e37d), retried fetch, tag checked
deploy/Dockerfile.pipelock          new    -- trixie-slim (CA bundle) + the binary at /pipelock
deploy/Dockerfile.router            modify -- FROM scratch + host-built static binary
deploy/Dockerfile.base              modify -- default BASE_IMAGE = the pinned trixie-slim
deploy/Dockerfile.vllm, deploy/Dockerfile.ollama
                                    modify -- default BASE_IMAGE = the pinned trixie-slim (today the unpinned tag)
deploy/docker-compose.yaml          modify -- localhost/devai-* images; registry-cache removed
deploy/registries.conf              modify -- mirror stanza removed
Makefile                            modify -- DEBIAN_IMAGE digest pin passed to every build; build-apt-cache,
                                              build-pipelock; router built with host Go;
                                              pull-images = Debian only; pipelock-ca-init uses the local image;
                                              registry-cache out of CACHE_SERVICES and PIPELOCK_NO_PROXY
tests/python/test_upstream_images.py new   -- the guard (below), with its transitional allowlist
```

### Detailed steps

1. Pin Debian: one Makefile variable, `docker.io/library/debian:trixie-slim@sha256:<digest>`, passed as `BASE_IMAGE` to every build, including `Dockerfile.base` (which moves from `trixie` to `trixie-slim`, M13). Bumping it is a deliberate edit followed by a rebuild of every image.
2. The apt-cache image. Check the apt-cacher-ng user id against the existing `/var/cache/devai/apt` volume, and `chown` once if it differs. The logger is not rebuilt: it moves into `devai-operator` in Phase 6 (I20) and keeps its current image until then.
3. pipelock from source, at the version already running, so behaviour does not change; the bump to a newer release (v3.5.0 is current) is a separate step. The image keeps the paths compose and the healthcheck use (`/pipelock`, `/config/...`).
4. The router: a `CGO_ENABLED=0` build with the host's Go (already required by `build-ollama` and `make test-devai-tools`), copied into `FROM scratch`. The router makes no HTTPS calls (no `https://` URL in `gpu-arbiter/*.go` outside tests) and sets no `USER`.
5. Remove registry-cache. Remove the mirror stanza from `deploy/registries.conf`, and tell existing hosts to do the same in `~/.config/containers/registries.conf` (INSTALL.md).
6. The guard test reads `deploy/docker-compose.yaml`, every `Dockerfile*` under `deploy/` and `tests/`, subdirectories included (so `deploy/slurm/Dockerfile` and `tests/slurm/Dockerfile.control-test` count), and the Makefile's image variables. Allowed: `localhost/devai-*`, the pinned Debian reference, and `scratch` (the router's base). Anything else fails with the file and line.
   - A transitional allowlist in the test names what later phases replace: the CUDA base (`GPU_BASE_IMAGE`) and `Dockerfile.lab`'s default `BASE_IMAGE=devai-base-cpu` (Phase 3); the engine images (`VLLM_IMAGE`, `VLLM_DEVAI_IMAGE`, `VLLM_DEVAI_IMAGE_TAG`, `SGLANG_IMAGE`, `QUANT_IMAGE`; Phase 4); the logger's `quay.io/podman/stable` (Phase 6).
   - Each of those phases removes its entries, and Phase 14 asserts the list is empty. Later phases add their own images under the same rule.

### Exit criteria

- `make pull-images` pulls only the pinned Debian image and the images still on the transitional allowlist.
- `make cache-up` brings the stack up; the logger writes a file for every devai container.
- From the lab, the pipelock checks in docs/pipelock.md behave as documented (allowed host 200, fake secret POST 403).
- An image build's apt traffic is served by apt-cache.
- `make test-router`, `make test-devai-tools` and `make test-python` pass.

### Phase 2 risks

| Risk | Mitigation |
| ---- | ---------- |
| apt-cacher-ng runs as a different uid than the old image | one `chown` of the cache volume, in the step itself |
| the router needs something distroless provided (CA bundle, tzdata, passwd) | the exit criteria run it; fall back to trixie-slim as the runtime base |
| pipelock's upstream image carries files the binary expects | compare the old image's filesystem before switching |

---

## Phase 3 -- One lab image on Debian

### Goal

One base image and one lab image, built on `debian:trixie-slim` instead of `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04`, run with the GPU when the host has one and on the CPU otherwise (M14, M15). The laya trainer follows the base.

### Deliverables

```
scripts/host-gpu.sh             new    -- the host check: prints the CDI device string when at least one
                                          NVIDIA GPU is usable (`nvidia-smi -L` lists one and the CDI spec
                                          names nvidia.com/gpu=all), nothing otherwise; an explicit
                                          DEVAI_GPU_DEVICE overrides it (empty = force CPU)
Makefile                        modify -- devai-base / devai-lab only; build-base-cpu, build-cpu, lab-cpu,
                                          shell-cpu removed; `lab` and `shell` attach the GPU per the host check;
                                          base = the pinned trixie-slim
deploy/Dockerfile.lab           modify -- CPU torch index branch removed; the workaround for the Ubuntu
                                          base's `ubuntu` user at uid 1000 removed; default BASE_IMAGE
                                          = localhost/devai-base
deploy/Dockerfile.laya-trainer  modify -- FROM devai-base; tini (Debian package) as PID 1; default command
                                          an idle loop (Phase 8 runs supervisord)
tests/python/test_upstream_images.py
                                modify -- GPU_BASE_IMAGE and Dockerfile.lab's old default leave the transitional
                                          allowlist
bin/devai-agent                 modify -- --cpu removed; image devai-lab; GPU per the host check
scripts/model-picker.py, scripts/dsh-web-launcher.sh, packages/jupyter-ai-launchers
                                modify -- messages naming lab-cpu|lab-gpu
INSTALL.md, INSTALL_macOS.md, README.md, AGENTS.md, docs/pipelock.md, docs/skypilot-user-guide.md,
scripts/sky-setup.sh            modify -- one lab image; macOS builds it for arm64
```

### Detailed steps

1. Build `devai-base` and `devai-lab` on trixie-slim with the CUDA torch. torch 2.14.0 from PyPI brings the CUDA 13.0 runtime libraries itself (`laya-trainer/requirements.lock` pins `nvidia-cuda-runtime==13.0.96`), so the old base's system CUDA 12.9 may be unused. If something does need system CUDA, it comes from NVIDIA's debian13 apt repository (the host's CUDA 13.1 packages), never from an image.
2. The host check decides the device for `make lab`, `make shell` and devai-agent. The same image runs in both cases; CUDA torch runs on the CPU when no device is attached.
3. The laya trainer image gets `tini` and an idle default command, so it can run as a permanent container. The router's current launch overrides the command, as today; Phase 8 makes supervisord and `devai-control` the default command.
4. macOS: `INSTALL_macOS.md` builds the one lab image natively for arm64 in `podman machine`, instead of the CPU image.

### Exit criteria

- On this host, in `devai-lab`: `torch.cuda.is_available()`, a matmul and a cuDNN convolution run on the card, and `nvtop` shows it.
- The same image with `DEVAI_GPU_DEVICE=` (forced CPU) starts, torch runs on the CPU, and the picker and an agent work against the router.
- `make test-laya-trainer` passes; `make laya-check` passes (a GPU window; it evicts the teacher for about 3 min).
- The arm64 build on a Mac succeeds, or is recorded as not yet verified.

### Phase 3 risks

| Risk | Mitigation |
| ---- | ---------- |
| a lab package links against the base's system CUDA or cuDNN | install the CUDA 13.1 packages from NVIDIA's debian13 repository |
| the hash locks lack aarch64 wheels, so the arm64 build fails | regenerate the locks for both platforms (`make lab-lock`) |
| the CUDA torch makes the lab image larger on hosts that never have a GPU | accepted (M14) |

---

## Phase 4 -- devai-engines

### Goal

One image holds every inference engine, each compiled here for CUDA 13.1 on sm120. The router serves from it, still recreating one container per engine as today, and every model is re-probed on it.

### Deliverables

```
scripts/build-sglang.sh              new    -- host build of SGLang 0.5.16 and its kernel packages (M3, M5), like build-vllm.sh
deploy/Dockerfile.engines            new    -- trixie-slim; tini; Ollama, vLLM env, SGLang env, one CUDA 13.1 toolkit,
                                               the launcher, per-engine labels, /etc/devai/engine-builds.json;
                                               default command an idle loop (Phase 8 runs supervisord)
scripts/devai-engine                 new    -- the launcher: `devai-engine <backend> <args>`
gpu-arbiter/*.go                     modify -- entrypoints via the launcher; one engines image; vllm-devai removed;
                                               per-engine drift; engine cache volumes under the engine's HOME
scripts/_probe_core.py, _probe_hf_common.py, _probe_load.py, probe-check.py,
scripts/bench/bench_runner.py, _bench_core.py, scripts/bench-sync.py
                                     modify -- per-engine identity instead of image digest; cache volumes under HOME
scripts/probe-vllm-devai-reasoning.py remove -- merged into the vllm prober
scripts/prepare-checkpoint.py        modify -- its image is devai-engines (vLLM environment), not docker.io/devai/vllm-devai
scripts/_probe_hf_common.py          modify -- PROBE_IGNORE_CURATED_PARSERS=1 with a scratch cache, and --ledger PATH so a
                                               scratch run leaves deploy/.model-status.json alone (I24)
scripts/_model_status.py             modify -- --exclude NAME::BACKEND --reason TEXT: a `manual` verdict, TEXT its detail, as --retire takes it (I17)
scripts/probe-cache-drop.py          new    -- removes named rows from a probe cache, atomically, as the probers save (I17)
tests/python/test_upstream_images.py modify -- the engine images, VLLM_DEVAI_IMAGE_TAG and QUANT_IMAGE leave the
                                               transitional allowlist
scripts/model-picker.py, scripts/probe-check.py
                                     modify -- a row whose weights are not in its store and whose cell carries
                                               another engine build is listed by probe-check as not re-probed,
                                               without failing, and hidden by the picker (I17)
scripts/model-picker.py, bin/devai-agent, config/*, scripts/generate-catalog.py, deploy/models.yaml,
deploy/recovery-flags.json, deploy/recovery-flags.generated.json, deploy/docker-compose.yaml, Makefile, tests/
                                     modify -- vllm-devai -> vllm (M4); `image` field removed (M8)
```

### Detailed steps

1. **4a -- SGLang build spike** (Open question 1).
   - List SGLang 0.5.16's dependencies that contain compiled code and sort them by M3. Its pins include `sglang-kernel==0.4.5`, `sgl-deep-gemm==0.1.4.post1`, `humming-kernels[cu13]==0.1.10`, `flash-attn-4>=4.0.0b18`, `quack-kernels>=0.6.1` and `flashinfer_python[cu13]==0.6.14` (PyPI metadata). The ones that are SGLang's own kernel packages are compiled; the rest come from Debian if its version satisfies the pin, else from PyPI.
   - Build for sm120 with nvcc 13.1 against the pinned `torch==2.11.0`, install in a trixie-slim environment with the host toolkit, and serve one model.
   - The spike also settles how a Debian package satisfies a pin inside the engine's environment: Debian's python3 with system site-packages, or none used.
   - If SGLang cannot be built, and no upgrade builds either, it is dropped (M2) and the rest of this phase proceeds without it.
2. **4b -- `scripts/build-sglang.sh`**, on the pattern of `build-vllm.sh`:
   - sources sha256-pinned; external repositories fetched once, at the revisions the source names; every network step tried 3 times; sm120 only;
   - output: a wheelhouse plus a content hash, and a record of where each dependency came from (compiled, Debian, PyPI).
3. **4c -- `Dockerfile.engines`.**
   - One stage per engine, copied into the final stage least-changed first.
   - The CUDA 13.1 compiler pieces are copied once from the host toolkit, as `Dockerfile.vllm` does today (FlashInfer needs nvcc at run time).
   - Each engine gets an image label, `devai.engine.<backend>`, set to its build's content hash (the `DIST_ID` idea already used for vLLM). The same hashes are written to `/etc/devai/engine-builds.json` (C1 `build`), so code inside the container can read them without podman.
   - `tini` is PID 1 and the default command is an idle loop, so Phase 8 can run the image as one permanent container; the router's current launch overrides the command.
4. **4d -- The launcher.** `devai-engine <backend>` sets `PATH`, `CUDA_HOME`, `LD_LIBRARY_PATH` and `HOME=/var/lib/devai-engines/<backend>` for that engine and `exec`s it.
   - For Ollama it also sets `OLLAMA_MODELS=/root/.ollama/models`, so the store stays where it is mounted (C6) and where `scripts/select-models.py` expects it.
   - Each engine thus has its own `~/.cache`. The router's `engineCacheVolumes` and the probers' `engine_cache_volumes` mount the existing `devai-engine-cache-<backend>-*` volumes under that `HOME` (C6).
   - The router's `vllmEntrypoint` / `sglangEntrypoint` and Ollama's launch call the launcher instead of a bare `python3`; so do the probers.
   - One `ENGINES_IMAGE` replaces `OLLAMA_IMAGE`, `VLLM_IMAGE`, `VLLM_DEVAI_IMAGE` and `SGLANG_IMAGE`.
5. **4e -- `vllm-devai` becomes `vllm`.** Run Phase 8 step 1 first: it needs `vllm-devai`, which this step retires. The rename covers the router's backend table, compose, the picker's backend lists and agent provider sets (`router-vllm-devai`), the probers, the derived catalog rows (`backend: [vllm]`), and the tests that pin the vllm-devai registration points. Port 11437 is retired. Every `vllm-devai` in a `backends` list becomes `vllm`, in `deploy/recovery-flags.json` and in `deploy/recovery-flags.generated.json` (groups recipe-apply wrote before 4e; the file is rewritten whole and renamed, I25); a test shows a pre-4e `["vllm-devai"]` group applying to a `vllm` launch.
6. **4f -- Per-engine staleness.** Today the HF probe caches stamp `_meta.current_image_digest`, and three places compare it: `make probe-check`, the router's drift check (`probeCachePathByBackend`, `readProbedImageDigest`) and `make bench-plan` (`stale_image`).
   - With one image for every backend, rebuilding it for an Ollama change would mark every vLLM and SGLang cell and bench row stale.
   - So stamp and compare the engine's `devai.engine.<backend>` label instead. Ollama's cache gets the same stamp (it has none today).
7. **4g -- Re-probe everything** on the new image: `make probe`, `probe-vllm`, `probe-sglang`, `probe-load-vllm`, `probe-load-sglang`.
   - Run `make backup-create` first.
   - Failures go to the exclusion ledger and the model is dropped (M8), including the two models that had their own images.
   - It also closes items absorbed from earlier plans. The SGLang cells without a launch fingerprint are re-probed with their load probe. The 7 SGLang cells that still carry a GPTQ "unsupported quantization" verdict (Gemma-4-31B-IT-NVFP4 among them) have no weights on disk and are not downloaded (I17): they are removed from the SGLang probe cache, and each model gets a `manual` ledger verdict saying why (`scripts/probe-cache-drop.py` and `scripts/_model_status.py --exclude`, after `make backup-create`).
   - Rows whose weights are not on disk are not re-probed and get no verdict (I17).
   - The card-derived parser path (`scripts/_card_hints.py`) gets its first real run (I24): gpt-oss-20b is re-probed on vLLM, and on SGLang if kept, with `PROBE_IGNORE_CURATED_PARSERS=1`, which makes the prober skip its curated `parsers:` block, its recipe seed and the name heuristic (Phase 17). The cells go to a scratch cache, with a scratch ledger (`--ledger`), and are compared with the curated cells: the same parsers, or a recorded disagreement. If gpt-oss-20b does not load on vLLM 0.28, the check runs on SGLang only (if kept) and records why.
8. **4h -- `QUANT_IMAGE`** points at `devai-engines` (its vLLM environment); `scripts/quant/quant_smoke.py` already adds llm-compressor on top of the image's torch.

### Exit criteria

- `make probe-check` is clean against `devai-engines` for every row whose weights are on disk, and lists the others as not re-probed (I17); the picker lists re-probed rows only.
- A picker-launched agent completes a turn on Ollama, vLLM and SGLang (if kept).
- `make test-router`, `make test-python`, `make test-vllm`, `make test-sglang` and `make test-e2e` pass.
- Rebuilding only the Ollama part leaves `make probe-check` clean for vLLM and SGLang.
- No file in the repository asserts an SGLang capability that the re-probed caches contradict (Phase 15's documentation check, run again on the new caches).
- Done before Phase 8 step 2 starts. Phase 8 step 1 is the exception: it runs on today's images before 4e.

### Phase 4 risks

| Risk | Mitigation |
| ---- | ---------- |
| SGLang does not build or run with nvcc 13.1 on sm120 | 4a answers it first; drop or upgrade (M2) |
| the upstream images carried components the source builds lack (SGLang's recovery flags for Gemma-4 rely on `flashinfer_cutlass`) | the re-probe finds them; the model is dropped or the component is added to the build |
| models lost to the single vLLM version (Nemotron-Nano-9B-v2, diffusiongemma) | accepted (M8); the re-probe lists them |
| a rebuild of one engine rewrites the layers after it | order stages least-changed first; vLLM (patched most often) last |
| the image grows past what the layer store handles well | measure; vllm-devai alone is 9.76 GB and SGLang's size built this way is unknown |
| a full re-probe takes GPU time the operator wants elsewhere | schedule it in the window the stopped re-bench would have used |

---

## Phase 5 -- devai-slurm and the bus

### Goal

`devai-slurm` runs Slurm, MariaDB, NATS and the bus log. The guard and the sampler work on the real card. `devai-bus`, `devai-control` (the `program`, `fake` and `actions` kinds) and a basic `devai-jobs` exist, and are tested against two test containers whose "engine" is the VRAM ballast. No engine, router or bench code changes. This phase can start at once, beside Phases 1-4.

### Deliverables

```
deploy/slurm/Dockerfile           new    -- FROM the pinned trixie-slim (Phase 2; if this phase comes first,
                                            it pins the digest and Phase 2 adopts it); trixie + trixie-security; slurmctld
                                            slurmd slurmdbd slurmrestd slurm-wlm-jwt-plugin slurm-wlm-mysql-plugin
                                            (24.11.5-4+deb13u1), mariadb-server, nats-server, supervisor, tini,
                                            python3 (I3), procps; users devai (1000) and slurmrest;
                                            devai-bus, devai-jobs; the files below
deploy/slurm/slurm.conf.in        new    -- ClusterName=devai; SlurmctldHost=localhost; NodeName=devai
                                            NodeAddr=127.0.0.1 CPUs=@CPUS@ RealMemory=@MEMORY_MB@ (I4); PartitionName=devai
                                            Default=YES MaxTime=INFINITE; SelectType=select/cons_tres,
                                            SelectTypeParameters=CR_CPU; Licenses=engine:1,bench:1 (I5);
                                            AuthType=auth/slurm, CredType=cred/slurm, AuthAltTypes=auth/jwt,
                                            AuthAltParameters=jwt_key=/etc/slurm/jwt_hs256.key;
                                            ProctrackType=proctrack/linuxproc, TaskPlugin=task/none,
                                            JobAcctGatherType=jobacct_gather/none;
                                            AccountingStorageType=accounting_storage/slurmdbd,
                                            AccountingStoreFlags=job_comment,job_script; PrologFlags=Alloc;
                                            Prolog/Epilog; HealthCheckProgram, HealthCheckInterval=30 (I7);
                                            PrologEpilogTimeout=180; BatchStartTimeout=240; KillWait=30; SlurmdParameters=config_overrides;
                                            StateSaveLocation=/var/cache/devai/jobs/slurmctld
deploy/slurm/slurmdbd.conf.in     new    -- MariaDB on localhost, the password from mariadb.pass, auth/jwt as alt type
deploy/slurm/nats-server.conf.in  new    -- C1 users and permissions; passwords filled in at start into /run/devai (0600)
deploy/slurm/supervisord.conf     new    -- mariadb, slurmdbd, slurmctld, slurmrestd (user slurmrest, -a rest_auth/jwt,
                                            :6820), slurmd (-N devai), nats-server, the bus log
deploy/slurm/entrypoint.sh        new    -- render the configs (the node's CPUs and memory from the host, I4); copy the jobs bus
                                            password to /run/devai/bus/jobs.pass (devai, 0400; I26); run directories for slurm, slurmrest, mysql; initialise
                                            MariaDB on an empty data directory (--innodb-snapshot-isolation=OFF);
                                            exec supervisord; once slurmdbd answers, idempotent sacctmgr: cluster devai,
                                            account devai, users devai and root, devai AdminLevel=Operator (I6)
deploy/slurm/devai-gpu            new    -- stdlib Python, NVML via ctypes (I3): `processes` (pid, used MiB),
                                            `used`, `sample --interval 5 --out F`, `energy`;
                                            DEVAI_GPU_FAKE=<json file> for tests
deploy/slurm/prolog.sh, epilog.sh, healthcheck.sh
                                  new    -- slurm.md Sec. 6, I7: the guard, the sampler and the idle check for
                                            engine, trainer and probe jobs only; for every job the prolog creates
                                            results/<jobid>/ and its artifacts/ as devai:0, mode 2775 (I26)
deploy/slurm/jobs/engine.sh, trainer.sh, probe.sh, bench.sh, action.sh
                                  new    -- C3
deploy/slurm/openapi-v0.0.42.json new    -- the spec dumped from this slurmrestd (step 2); fixture for pkg/slurmrest
gpu-arbiter/pkg/bus/              new    -- C1: subjects, envelope, verbs, states, error codes; a client (user and
                                            password file, inbox prefix, Request, Publish, Subscribe)
gpu-arbiter/pkg/slurmrest/        new    -- JWT signer; Submit, Job, Jobs, Signal, Cancel, Hold, Release, Node,
                                            ResumeNode, DBJobs (v0.0.42)
gpu-arbiter/go.mod, devai-tools/go.mod
                                  modify -- nats.go (nats-server for tests only); devai-tools requires gpu-arbiter
                                            with replace => ../gpu-arbiter (I2)
devai-tools/cmd/devai-bus/        new    -- request, publish, hold (C3: ignores TERM and INT, dies with its --parent;
                                            --lease-only ends on HUP; on devai.operator.actions it exits with the
                                            action), watch, log; the exit codes of C3
devai-tools/cmd/devai-control/    new    -- main
devai-tools/internal/control/     new    -- config, supervisord XML-RPC client (getProcessInfo, startProcess,
                                            stopProcess, signalProcess, reloadConfig, addProcessGroup,
                                            removeProcessGroup), the state machine, verbs, ownership, the lease
                                            timer, the state file, events; kinds `program`, `fake` and `actions`
                                            (registry C8, run with setsid, reply on start and publish `exited`,
                                            signal the group, stop a run whose lease lapsed 60 s ago, on exit
                                            remove containers labelled devai.job=<id> when a podman CLI is
                                            configured)
devai-tools/cmd/devai-jobs/       new    -- list, show, queue, cancel, node, submit; every command with --json
                                            (basic; Phase 13 completes it)
bin/devai-jobs                    new    -- `podman exec devai-slurm devai-jobs "$@"` (I13); installed by `make install`
tests/slurm/Dockerfile.control-test
                                  new    -- trixie-slim + supervisor + devai-control. test-slurm.sh runs it as two
                                            containers, so jobs use the production subjects, users and namespaces:
                                            devai-control-test-engines (default user namespace, root, the GPU;
                                            container `engines`, service `fake`: scripts/vram-ballast.py as the
                                            engine, whose `load` holds N MiB and is "serving" once it prints
                                            ready) and devai-control-test-operator (keep-id, user devai, no GPU;
                                            container `operator`, service `actions`, with a test registry); each
                                            with its own control.yaml
tests/slurm/test-slurm.sh         new    -- the live checks in the exit criteria; refuses to run while a production
                                            devai-control runs (devai-engines or devai-operator), since both would
                                            answer the same subjects and match the guard's command-line lookup; starts both
                                            test containers and removes them afterwards
tests/python/test_slurm_config.py new    -- static: the C1 table equals nats-server.conf.in; the slurm.conf.in keys and their rendering
                                            above; the compose service has no podman socket and publishes nothing
tests/python/test_devai_gpu.py    new    -- devai-gpu and the guard's decisions with DEVAI_GPU_FAKE
deploy/docker-compose.yaml        modify -- service `slurm` (container devai-slurm, hostname devai-slurm, devai-net
                                            only, pid: host, the default user namespace (I26), tmpfs /run/devai,
                                            the GPU when present, C6/C7 mounts); profile `slurm`
                                            until Phase 9
Makefile                          modify -- build-slurm, slurm-init, setup-jobs, slurm-up, slurm-down, test-slurm
scripts/slurm-init.sh             new    -- C7 (the Slurm part)
devai-tools/internal/backup/manifest.go
                                  modify -- ~/.config/devai/slurm/
```

### Detailed steps

1. **Open question 2 first.** Build the image and start it unprivileged (`--pid=host`, the GPU, podman's default user namespace: I26), with slurmd on `proctrack/linuxproc`.
   - Submit `sleep 600` with `-L engine` and cancel it; check that its process tree is gone.
   - If slurmd refuses to run this way, fall back to the measured setup: `--privileged --cgroupns=private`, the cgroup step in the entrypoint, `cgroup.conf` with `IgnoreSystemd=yes`, `proctrack/cgroup`.
   - Record which setup runs.
2. **The REST spec.** Dump it from this slurmrestd (`slurmrestd -d v0.0.42 -s slurmctld,slurmdbd --generate-openapi-spec`) into `deploy/slurm/openapi-v0.0.42.json`.
   - Check the names C3 and C5 use against it: `job.script`, `comment`, `licenses`, `signal` with `flags=BATCH_JOB`, hold and release, node `RESUME`.
   - Where 24.11 differs, correct the contract. The spike measured v0.0.44 on 26.05.
3. **Accounting.** MariaDB with `--innodb-snapshot-isolation=OFF`, slurmdbd, the `sacctmgr` steps. A REST-submitted job's JSON comment must arrive intact in `sacct` (measured on 26.05; re-checked here on 24.11).
4. **`pkg/slurmrest`.** Test it against an `httptest` server that checks every request against the spec dump. Test the JWT live in `test-slurm.sh`.
5. **`pkg/bus`, `devai-bus`, `devai-control`.** Go tests with an embedded nats-server, a fake supervisord XML-RPC server and a 2 s lease. They cover:
   - every verb, and `not_owner`;
   - lease expiry;
   - adopting a running program from the state file after a `devai-control` restart;
   - the permissions (a `jobs` client cannot publish `devai.engines.ollama.restart`);
   - an `actions` run refused when its parameters break the registry; a run whose process group is signalled and then killed after its grace; a run stopped once its lease lapses; `run` replying on start and `exited` arriving as an event; a `run` refused after a `stop` for its job; a forced `stop` with `run_job` stopping another job's run, and `not_owner` without `force`; a hold whose parent shell is SIGKILLed exiting within 1 s, and one started with a dead `--parent` exiting at once.
6. **The guard.** `devai-gpu`, the prolog, the epilog and the health check.
   - Unit tests with `DEVAI_GPU_FAKE`: idle; a devai holder that unloads; a foreign holder that is killed; a foreign holder of another user (exit 1); the energy delta; a bench or action job, whose prolog and epilog leave the card alone even when it is busy; a holder in the operator container whose job is gone (a fake environ carrying `DEVAI_JOB_ID`): a forced `stop` with `run_job`, then a kill; one without `DEVAI_JOB_ID`, treated as foreign; two `devai-control` processes in one cgroup, a configuration error (exit 1); a trainer job, whose epilog sends `cancel`, not `unload`.
   - Live tests on the card, in `test-slurm.sh`.
   - Mapping an NVML PID to a devai container: compare `/proc/<pid>/cgroup` with that of the `devai-control` process whose command line names the container. Both are read through `--pid=host`. Each container holds exactly one `devai-control`. A holder in devai-operator is matched to its run by the `DEVAI_JOB_ID` in its `/proc/<pid>/environ` (C8 sets it for every `job` action, and children inherit it). devai-slurm's root and devai-operator's devai are the same host uid, so it is readable (untested so far).
   - Check whether 24.11 accepts `scontrol update JobId=<id> AdminComment=...` from the epilog of a completing job. If it does not, the summary stays in `gpu.json`, and Phase 13 reads it from there.
   - Check whether a node drained by a failed prolog keeps a reason that the health check can recognise; otherwise the health check also reads `guard.json`.
7. **The bus log.** `devai-bus log --dir /var/cache/devai/jobs/bus` writes one JSON line per event, with the receive time.
8. **Compose, Makefile, init, backup.** `slurm-init`; `setup-jobs` (`sudo LV=cache_jobs MOUNTPOINT=/var/cache/devai/jobs SIZE=50G deploy/setup-logs-volume.sh`); the backup entry.

### Exit criteria

- After `make setup-jobs slurm-init build-slurm slurm-up`, `sinfo` shows `devai` idle, `sacctmgr show cluster` lists `devai`, and `podman port devai-slurm` lists nothing.
- A job submitted through slurmrestd from another container on `devai-net`, with a JWT signed by `pkg/slurmrest`, runs, and its JSON comment arrives intact in `sacct`. A second `-L engine` job waits as `PENDING (Licenses)`.
- Against the test containers:
  - an engine job loads the fake engine (2 GiB held), and `sacct` or `gpu.json` shows the GPU summary;
  - `scancel` unloads it and the card is back at its baseline;
  - when the job script is killed with SIGKILL, the epilog unloads;
  - when the job script is killed and the epilog is switched off (a test switch), the lease unloads within 70 s;
  - when an action job's script is killed with SIGKILL, the action's process group is stopped within 70 s;
  - `scancel` of an action job whose action needs 15 s to stop: the job stays RUNNING or COMPLETING until the action's `exited` event, and a queued `-L bench` job starts only after that; the same for a `scancel` during the `run` request.
- Ownership and isolation (I26):
  - In `results/<jobid>/`, the operator test container's devai-control, running keep-id as the host user, writes `action.log` and `result.json`. Every file carries the host user's group, and the host user can read every file and delete the directory.
  - A job cannot read `slurm.key`, nor signal a process of the host user.
  - The guard kills a GPU holder that runs as a non-root user in another rootless container.
- The next engine job's prolog starts only after the previous engine job's epilog has ended. Slurm returns the license only after the epilog, and starts nothing on a node with a completing job (read in the 24.11.5 source; checked here).
- A foreign VRAM holder (the ballast in a plain container) is killed by the next GPU job's prolog. A foreign holder the guard cannot kill drains the node and the job is held; the health check resumes the node within 60 s of the holder exiting.
- `devai-bus request devai.engines.fake.restart` restarts the program, and every event lands in `jobs/bus/<date>.jsonl`.
- Started without the GPU device (`test-slurm.sh` starts it so; the compose switch comes in Phase 11), `devai-slurm` runs, the prolog and the epilog skip NVML, and jobs run.
- `make test-router` (the `pkg/` tests), `make test-devai-tools`, `make test-python` and `make test-slurm` pass.

### Phase 5 risks

| Risk | Mitigation |
| ---- | ---------- |
| slurmd needs privileges after all | the measured privileged setup (step 1) |
| Debian's 24.11 REST differs from what the spike saw (v0.0.44 on 26.05) | the spec dump; the client is tested against it |
| a fake engine hides the timing of real engines | Phase 8 repeats the checks with real engines |
| `AdminComment` cannot be written from the epilog | `gpu.json` is the record; Phase 13 reads it |

---

## Phase 6 -- devai-operator: image, container and direct actions

### Goal

From a browser on the LAN, behind a login, the operator can:

- bring the stack up and down;
- pull a model;
- run a probe and a bench;
- read each log.

Every action runs `direct` in this phase: there is no Slurm yet.

### Deliverables

```
deploy/Dockerfile.operator        new    -- FROM the pinned trixie-slim. Debian: apache2, libapache2-mod-wsgi-py3,
                                            python3-flask, python3-yaml, python3, git, curl, make, podman (client),
                                            cmake, ninja-build, gcc, g++, patch, python3-dev, supervisor, tini,
                                            ca-certificates; the `hf` CLI (huggingface_hub, in its own venv from
                                            deploy/operator/requirements-hf.lock). Docker's
                                            docker-compose-plugin 5.1.0-1~debian.13~trixie from download.docker.com,
                                            the key accepted only with fingerprint 9DC8 5822 9FC7 DD38 854A E2D8
                                            8D81 803C 0EBF CD88 (O9); containers.conf with compose_providers. The
                                            CUDA 13.1 nvcc, headers and the dev packages build-vllm.sh and
                                            build-ollama.sh list, from NVIDIA's debian13 repository. Go (>= 1.26),
                                            the rustup toolchain rust-toolchain.toml names, and uv, as pinned
                                            downloads checked against their published checksums (O8; uv follows
                                            the same rule)
deploy/operator/requirements-hf.lock
                                  new    -- huggingface_hub with hashes; `make operator-lock` regenerates it
deploy/operator/apache.conf       new    -- one TLS virtual host on 8443; AuthType Basic with AuthUserFile
                                            /etc/devai/operator/htpasswd on every path (O7); WSGIDaemonProcess
                                            devai-operator, user devai; HSTS
deploy/operator/supervisord.conf  new    -- apache2 in the foreground; the log follower (deploy/logging.sh, I20);
                                            Phase 10 adds devai-control
deploy/operator/actions.yaml      new    -- C8; this phase's entries: stack-up, stack-down (direct), model-pull,
                                            probe-ollama, probe-vllm, bench, backup-create; everything else in
                                            `pending` or `not_offered`
devai-operator/devai_operator/    new    -- the Flask app: app.py (routes), registry.py (load and validate C8, build
                                            argv and env), runner.py (direct runs: setsid, outside Apache's
                                            processes, run.json and log under jobs/operator/<run>/, stop),
                                            csrf.py, startup.py (refuses to start without path identity or without
                                            the podman socket), templates/, static/ (no external CDN)
devai-operator/wsgi.py            new
devai-operator/tests/             new    -- registry validation (a value outside its pattern never reaches argv),
                                            argv and env building, CSRF, the start-up checks, and the coverage test:
                                            every documented Makefile target is in some action's `covers`, in
                                            `not_offered` with a reason, or in `pending`
scripts/ops/stack-up.sh, stack-down.sh, model-pull.sh, probe-ollama.sh, probe-vllm.sh, bench.sh, backup-create.sh
                                  new    -- the recipes of this phase's actions, moved out of the Makefile (O5)
scripts/operator-init.sh          new    -- C7 (the operator part): htpasswd -B for the one account, prompting for
                                            the password; TLS from mkcert files if present, else self-signed
deploy/docker-compose.yaml        modify -- service `operator` (container devai-operator, devai-net, userns keep-id and
                                            user devai (I19), the C6 mounts except the JWT key and the bus passwords
                                            (Phase 10), CONTAINER_HOST, DEVAI_REPO=$(CURDIR) for path identity, the
                                            host check's DEVAI_GPU_DEVICE); the `logger` service removed (I20)
deploy/logging.sh                 modify -- runs in devai-operator, against its podman socket (I20)
devai-tools/internal/backup/manifest.go
                                  modify -- ~/.config/devai/operator/
tests/python/test_upstream_images.py
                                  modify -- the logger's image leaves the transitional allowlist
Makefile                          modify -- build-operator, operator-init, operator-up, operator-lock, test-operator; the moved
                                            recipes call their scripts; `export DEVAI_REPO := $(CURDIR)` unless Phase 17a
                                            added it first (I25)
.env.example                      modify -- OPERATOR_PORT=8443 (WEBUI_PORT left with Open WebUI in Phase 1)
```

**Pages:**

- the stack at a glance: containers, the GPU (nvidia-smi), the router's `/health` per port;
- actions by group, each with its form;
- runs: list, a log that is polled, stop;
- host setup: a placeholder until Phase 12.

### Detailed steps

1. **The image.** The Docker repository key's fingerprint and the Go, Rust and uv checksums are checked in the Dockerfile; a mismatch fails the build.
2. **The app.** Registry, runner, CSRF and the pages. Parameters are checked against the registry and passed as an argument list; nothing goes through a shell.
3. **The container.** Path identity: the repository, `/var/cache/devai` and the home volume are mounted at their host paths, because the scripts pass host paths to podman. The start-up check refuses to start otherwise.
4. **This phase's actions.** Their scripts under `scripts/ops/`, and the Makefile targets calling them.
5. **The coverage test,** with the `pending` list that Phase 7 empties.
6. **The probers' imports** are checked against the image's Python (stdlib and PyYAML): this phase already runs probes from it.
7. **`stack-down` leaves `devai-operator` running**, with its direct runs; it stops every other service. `stack-up` likewise names every service but `devai-operator`, so neither action ever recreates the container it runs in. Rebuilding or restarting `devai-operator` itself is a host command (`make operator-up`), which also runs the host check (`scripts/host-gpu.sh`) and passes its result in as `DEVAI_GPU_DEVICE`.

### Exit criteria

- After `make build-operator operator-init operator-up`, `https://<host>:8443` asks for the login. A wrong password is refused and appears in Apache's log; the right one shows the stack page.
- From the browser:
  - the stack goes down and up;
  - a small catalog model is pulled;
  - `probe-ollama` runs for one model;
  - a bench of one task runs;
  - each run's log is visible while it runs and afterwards, and stop works.
- A parameter outside its pattern is refused, and no process starts. A POST without the CSRF token is refused.
- Restarting Apache (`supervisorctl restart apache2`) does not kill a running direct action.
- The log follower writes a file for every devai container from inside `devai-operator`, and no container other than `devai-operator` and the router mounts the podman socket (the router keeps it until Phase 9).
- `make test-operator` passes, including the coverage test.

### Phase 6 risks

| Risk | Mitigation |
| ---- | ---------- |
| the web login controls the host user (podman socket), and it is on the LAN (O6) | TLS only; one strong htpasswd password (O7); CSRF tokens; parameters never passed through a shell; login attempts logged |
| the toolbox image is large (CUDA 13.1 dev packages, Go, Rust) | one image, shared layers |
| a path mounted at a different place breaks every podman call that passes it | path identity for the repository, `/var/cache/devai` and the home volume; the start-up check |
| a direct action dies with an Apache restart | started with `setsid`, outside Apache's processes |
| Docker's repository key changes | the fingerprint check fails the build loudly; the key is updated deliberately |

---

## Phase 7 -- Recipes into scripts

### Goal

Every documented Makefile target is either an action in the registry, or listed as not offered with a reason. Every offered target's recipe is a one-line call to its script (O5).

### Deliverables

```
scripts/ops/*.sh                  new    -- one per operation: `set -euo pipefail`; its defaults declared at the top,
                                            taken from the Makefile's; parameters as the registry maps them
Makefile                          modify -- recipes call scripts/ops/; variables passed explicitly
deploy/operator/actions.yaml      modify -- filled; `pending` empty
tests/python/test_makefile_ops.py new    -- every offered target's recipe is one call to scripts/ops/ (or a dependency
                                            list); the registry's `covers` and `not_offered` together equal the
                                            documented targets
```

### Detailed steps

1. Inventory the 124 documented targets by group.
   - Interactive targets (`lab`, `shell`) become `lab` actions (Phase 12) or are not offered.
   - Host-root targets (`setup-logs`, `setup-jobs`, the volume set-ups) become `host` actions (O3).
   - Targets Phase 1 removed are gone already. `secrets-tmpfs` in O3 went to the attic with sops/age (M12).
   - `install-systemd` is already broken: its unit runs compose from a copy in `~/.config/devai`, which stages none of six current mounts. Its unit now runs compose in the repository (`WorkingDirectory=<repo>/deploy`), with `--env-file <repo>/.env`, `Environment=DEVAI_REPO=<repo>` (I25) and the override files `make cache-up` would add (`compose.gpu.yaml`, `compose.library.yaml`), since it reads neither the Makefile nor its checks. That also keeps compose's `./` paths away from the key directories under `~/.config/devai` (C7). It becomes a `host` action, since it writes the host user's systemd directory.
2. Move the recipes group by group: stack and build first (the most used), then models, probe and bench, then tests, logs and maintenance.
3. For each moved target, run the target before and after and compare what it does: its output, and the files it writes.

### Exit criteria

- The coverage test passes with `pending` empty.
- `make test-python` and `make test-operator` pass.
- `make cache-up`, `make probe PROBE_MODELS=<one>` and `make bench-plan` behave as before.

### Phase 7 risks

| Risk | Mitigation |
| ---- | ---------- |
| a recipe's Make variables (defaults, `?=`, `$(CURDIR)`) behave differently in a script | each script declares the Makefile's defaults; the Makefile passes its variables explicitly; step 3 compares behaviour |
| the Makefile and the registry drift apart | the coverage test and `test_makefile_ops.py` |

---

## Phase 8 -- Engines and trainer under devai-control

### Goal

`devai-engines` and `devai-laya-trainer` run permanently under supervisord with `devai-control`. Slurm jobs load, unload, sleep and wake real models through the bus, and the laya run is started by `train`. The router still serves the old way: the permanent containers run in the `slurm` compose profile while the router's backends are stopped.

### Deliverables

```
gpu-arbiter/pkg/engineargs/       new    -- moved out of main.go with no change in behaviour: LaunchConfig,
                                            VLLMArgs / SGLangArgs (were vllmEntrypoint / sglangEntrypoint), the
                                            speculative helpers, OllamaEnv (was ollamaDynamicEnv), the recovery and
                                            plugin registries (recovery_flags.go, with 17a's reader of the generated file, I25; vllm_plugins.go); FromLoad(C2,
                                            registries, store, port) -> argv + env; the router passes 11434
                                            (unchanged), devai-control the port from control.yaml
gpu-arbiter/main.go, recovery_flags.go, vllm_plugins.go
                                  modify -- use pkg/engineargs; argv_golden_test.go and flags_allowlist_test.go move
                                            with the builders, their corpus unchanged; `-print-load` (C2)
devai-tools/internal/control/engine_ollama.go
                                  new    -- load: restart serve on changed settings (I8), /api/chat warm load with
                                            num_gpu and keep_alive -1, check /api/ps; unload: keep_alive 0, then wait
                                            for /api/ps empty and the NVML baseline
devai-tools/internal/control/engine_hf.go
                                  new    -- vllm and sglang: write the program file (autostart=false,
                                            autorestart=false, stopasgroup, killasgroup, stopwaitsecs=30, the log
                                            file of C4), delete vLLM's compile cache (C6), start, wait for /health
                                            (fail fast if the program exits; 600 s); sleep and wake; unload of the
                                            loaded model or the sleeper (I9); at most one sleeper; the sleeper's
                                            VRAM (below); a vLLM or SGLang load on a host without a GPU is refused
                                            with no_gpu
devai-tools/internal/control/trainer.go
                                  new    -- train and cancel: program `python -m laya_trainer.run_job --job <ft_job>`;
                                            after it exits, finalize a record that is not terminal (`trainer_stopped`)
deploy/engines/supervisord.conf, deploy/engines/control.yaml
                                  new    -- tini -> supervisord: devai-control and ollama serve (its program file
                                            written by devai-control, I8); [include] /run/devai/supervisor/*.conf
deploy/laya-trainer/supervisord.conf, deploy/laya-trainer/control.yaml
                                  new    -- devai-control and the controller
deploy/Dockerfile.engines         modify -- supervisor, devai-control, devai-bus, the files above; CMD supervisord
deploy/Dockerfile.laya-trainer    modify -- supervisor, devai-control, devai-bus, its files; CMD supervisord
laya-trainer/laya_trainer/controller.py
                                  modify -- behind `LAYA_RUN_BY=control` (default `controller` until Phase 9 flips
                                            it, so the old router path keeps working): create records the job and
                                            spawns nothing; cancelling a job that has
                                            not started finalizes it; the start-up reconcile leaves queued jobs alone;
                                            POST /devai/v1/jobs/{id}/fail (called by the router only)
laya-trainer/laya_trainer/run_job.py
                                  modify -- on SIGTERM, finalize as cancelled, or as failed `time_limit` when
                                            devai-control passes that reason
laya-trainer/laya_trainer/jobs.py modify -- `finalize` entry point for devai-control (changes only a non-terminal record)
deploy/docker-compose.yaml        modify -- profile `slurm`: service `engines` (devai-engines, the C6 mounts, the GPU
                                            when present); laya-trainer runs supervisord; both run as root (I26)
tests/slurm/test-engines.sh       new    -- the live checks below; starts the trainer with LAYA_RUN_BY=control, so
                                            the hand-submitted trainer jobs exercise the new path before Phase 9
                                            flips the default
deploy/slurm/prolog.sh, epilog.sh, healthcheck.sh, devai-gpu
                                  modify -- tolerate a sleeper: devai-engines' processes count as the sleeper's while
                                            that container reports a sleeper and no loaded model, up to its `used_mib`
                                            + 256 MiB (C1). A probe job's prolog unloads the sleeper instead
                                            (`unload`, `target: sleeper`, `force`): a probe measures an empty card,
                                            and its ballast is sized for the full one. Unit tests for both
devai-tools/internal/control/nvml.go
                                  new    -- memory readings through nvidia-smi (C1, "Memory readings")
devai-tools/internal/control/recovery.go
                                  new    -- the recovery files and the plugin registry re-read at every load (I18, I25)
```

**The sleeper's VRAM** (`engine_hf.go`, `engine_ollama.go`). This covers a load on another service while a model sleeps. The two engines size memory in opposite ways (read in the vLLM 0.28 and SGLang 0.5.16 sources, 2026-10-06):

- **vLLM** takes `--gpu-memory-utilization` of the card's total memory, and refuses to start when less than that is free.
  - `devai-control` lowers it by the sleeper's `used_mib` over the card's total, and records the change in its reply.
  - Not for a model whose recovery flags set it (C2: 9 of the 14 curated entries do, the Qwen3.8 rows among them). For those only the retry below applies.
- **SGLang** takes `--mem-fraction-static` of the memory free when it starts, so it starts with a sleeper on the card, and nothing is lowered.
  - But its KV pool and runtime headroom shrink by about that fraction of the sleeper's `used_mib`.
  - A pool smaller than the context does not fail the load: SGLang caps the request length instead (`max_req_len = min(context_len - 1, pool - 1)`, 0.5.16 `managers/tp_worker.py`) and refuses longer prompts only when they arrive.
  - So after an SGLang load while a sleeper exists, `devai-control` reads `max_req_input_len` from the engine's `GET /server_info`. An unconstrained pool reports `ctx - 6`. Anything lower counts as a memory failure (`kv_pool_below_ctx`, both numbers in the reply).
- **If the load then fails on memory, or an SGLang pool comes up below its context**, `devai-control` unloads the sleeper and loads once more, and says so in its reply. Ollama, whose fit llama.cpp decides, gets only this retry.
  - A memory failure is recognised by these substrings of the engine's log:
    - vLLM: `less than desired GPU memory utilization`, `No available memory for the cache blocks`, `larger than the available KV cache memory`;
    - SGLang: `Loaded weights leave no GPU memory for the KV cache`;
    - any engine: `CUDA out of memory`, `torch.cuda.OutOfMemoryError`;
    - Ollama: `cudaMalloc failed: out of memory`.
- A load on the sleeper's own service never meets this: the new model process replaces the sleeper first.

### Detailed steps

1. **Open question 3 first, on today's images, before Phase 4's 4e retires `vllm-devai`.**
   - On `vllm-devai`, run Qwen3.8-27B-W4A16-devai-AutoRound with `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`, and measure:
     - how long `/sleep?level=1` and `/wake_up` take (5 times each);
     - VRAM after sleep (NVML);
     - host RAM (RSS);
     - decode tok/s and TTFT with and without sleep mode, on paired prompts (docs/statistics-primer.md conventions).
   - Do the same for one SGLang model with `--enable-memory-saver`.
   - If sleep mode costs speed or memory, `sleep_mode` defaults to false, and the router sets it only for a launch that a pending job names.
2. **Move the builders.** A pure refactor: `make test-router` passes unchanged and the golden corpus stays byte-identical.
3. **`devai-control`'s engine and trainer kinds.** Go tests with fake engines (an `httptest` server answering `/health`, `/api/ps`, `/sleep`, `/wake_up`) and the fake supervisor.
4. **The images.** supervisord and the configurations.
5. **The laya split.** Its tests go into `make test-laya-trainer`.
6. **The live checks.** Stop the router's backends (`make cache-down`) and start the `slurm` profile.

### Exit criteria

- For Ollama, vLLM and SGLang (if Phase 4 kept it), in `devai-engines`:
  - an engine job submitted with the request `gpu-arbiter -print-load` prints loads the model;
  - a chat completion sent straight to `devai-engines:<port>` answers;
  - `scancel` frees the card;
  - the model process's command line equals the router's golden argv for the same request, apart from the store path and the port.
- vLLM sleep and wake:
  - an engine job ended with USR1 leaves the model `sleeping`;
  - the next job with the same launch wakes it;
  - a job with another launch on the same service unloads the sleeper first (each engine serves one model process on its port);
  - an `unload` with `target: sleeper` unloads it.
- Ollama:
  - a load with another context length restarts `ollama serve`, and `/api/ps` then shows that context;
  - a `/v1/chat/completions` request without `num_ctx` does not make Ollama load the model again (its log shows one load).
- A sleeping model does not make the next engine job's prolog unload it, and the epilog's idle check tolerates it; a probe job's prolog unloads it.
- The sleeper's VRAM:
  - with an SGLang model sleeping (if SGLang is kept), a vLLM load starts with its utilization lowered by the sleeper's `used_mib`;
  - with a vLLM model sleeping (if SGLang is kept), an SGLang load at its probed context reports a `max_req_input_len` of at least ctx - 6, after a retry if the pool came up short, and a prompt filling 99% of that context succeeds;
  - with a model sleeping, a load forced to fail on memory (a fraction too large, set in the test) unloads the sleeper and succeeds on the second attempt.
- A recovery file replaced by rename is used by the next load, without restarting `devai-engines`.
- When `devai-control` is killed while vLLM serves, supervisord restarts it and it adopts the model: `serving`, the same job, the same process.
- Trainer jobs on the reference dataset `ds-a6c9c8248242`, submitted by hand after the controller created the job:
  - one runs to the end and finishes `succeeded`, with a record shaped as `make laya-check` expects;
  - `scancel` during training ends a job `cancelled`;
  - the time-limit signal ends a job `failed` with `time_limit`.
- When a trainer job's script is killed with SIGKILL, the epilog's `cancel` ends the run `cancelled`, and the card is idle.
- With `devai-engines` started without the GPU device (`test-engines.sh` starts it so; the compose switch comes in Phase 11), Ollama loads with `num_gpu: 0`, and a vLLM load is refused with `no_gpu`.
- `make test-router`, `make test-devai-tools`, `make test-laya-trainer` and `make test-slurm` pass.

### Phase 8 risks

| Risk | Mitigation |
| ---- | ---------- |
| sleep mode costs speed or memory while serving | measured in step 1; if so, it is used only for a launch that a pending job names |
| a sleeping model leaves too little VRAM for the next load | for a vLLM load, `devai-control` lowers the utilization by the sleeper's `used_mib`, unless recovery flags set it (SGLang sizes from free memory; an SGLang pool below its context counts as a failure); if a load still fails, it unloads the sleeper and loads once more |
| the refactor changes an argument | the golden corpus, compared byte for byte |
| one engine process crashing takes the others down | each is its own supervisord program; only a container restart affects all of them, and the router then submits new jobs |

---

## Phase 9 -- Router as Slurm client

### Goal

The router starts and ends engines only through Slurm jobs, follows their state on the bus, and has no podman socket. Today's launch layer and the per-engine containers are gone in the same step (I10). The 2026-09-28 failure cannot be reproduced.

### Deliverables

```
gpu-arbiter/slurm_launch.go       new    -- the Slurm launcher (below); replaces ensureBackendRunning and
                                            ensureOllamaRunning
gpu-arbiter/gpu_record.go         new    -- the GPU record
gpu-arbiter/holds.go              new    -- C5 holds and /devai/v1/gpu
gpu-arbiter/ready.go              new    -- C5 /devai/v1/ready (from odysseus RT-6)
gpu-arbiter/control_routes.go     new    -- C5 blocked engine routes, registered by newBackendMux on every listener
gpu-arbiter/trainer_jobs.go       new    -- :11438 create and cancel -> trainer jobs; the boot reconcile with the controller
gpu-arbiter/main.go               modify -- the podman launch layer removed (containerRecreate, createAndStart,
                                            buildContainerSpec, containerStop / Remove / State, stopOtherBackends,
                                            unloadOllama, warmLoadOllama, backendVanished, detectLaunchFailure and
                                            its log readers, reconcileBackendState, the podman client); backend URLs
                                            devai-engines:<port>; the breaker charged only on load_failed; /health
                                            from the record; the drift check from `build`; the idle sweep cancels
                                            the engine job
gpu-arbiter/job_runner.go, ollama_state.go, engine_cache.go
                                  remove -- the job-runner hold, Ollama's /api/ps reconcile, the per-recreate cache
                                            volumes (compose mounts them into devai-engines since Phase 8)
gpu-arbiter/*_test.go             modify -- tests of the podman layer (backend_vanished_test.go, lifecycle_test.go,
                                            reconcile_test.go, job_runner_test.go, the launch parts of others)
                                            removed or rewritten; a behaviour the Slurm launcher keeps (the breaker,
                                            Ollama's bare-name rule, request coalescing, drain) is ported, not dropped
deploy/docker-compose.yaml        modify -- router: C7 environment and mounts, no podman socket; the per-engine
                                            services (the ollama, vllm and sglang placeholders) removed; the `slurm`
                                            profile becomes the default stack
docs/router.md                    modify -- the launch sections rewritten for the Slurm launcher
scripts/select-models.py, Makefile (OLLAMA_CONTAINER), scripts/ollama-list.py
                                  modify -- Ollama commands run in devai-engines (`podman exec devai-engines
                                            devai-engine ollama ...`)
tests/test-router-vllm.sh, tests/test-e2e-picker.sh
                                  modify -- no per-engine containers to remove or recreate
scripts/bench/clean_slate.py, scripts/_probe_hf_common.py (MUTEX_CONTAINERS)
                                  modify -- the legacy paths name devai-engines and need the stack stopped;
                                            removed in Phase 14
laya-trainer (LAYA_RUN_BY)        modify -- flipped to `control` (Phase 8)
scripts/ops/probe-ollama.sh, Makefile (cache-down)
                                  modify -- the legacy Ollama probe path (in scripts/ops/probe-ollama.sh since
                                            Phase 6) starts a throwaway `devai-ollama` container from
                                            devai-engines (`devai-engine ollama serve`, the band's environment)
                                            with the stack stopped, as the HF probers' legacy path does, instead
                                            of compose's removed `ollama` service (removed in Phase 14);
                                            cache-down's list of router-recreated containers goes
tests/test-model-matrix.sh        modify -- the forced fresh Ollama load cancels the engine-ollama job
                                            (`bin/devai-jobs cancel`) instead of removing devai-ollama
tests/test-router-idle.sh         modify -- the idle sweep is checked by the engine job's end and /health, not by
                                            the devai-vllm container's status
```

### What the Slurm launcher does

**The record** (under `arbiter.mu`):

- `holder`: job id, kind, backend, launch, and status from the bus;
- `sleeper`;
- `holds`: job id -> backend and launch;
- `pending`: PENDING jobs and the launches their comments name, refreshed from `GET /jobs` at most every 2 s.

**Boot.** The router rebuilds the record from Slurm's RUNNING and PENDING jobs of devai's kinds and from `state` of every engine service. A serving engine whose job is RUNNING is adopted with its launch -- unlike today, the model is known, so the next request for it is proxied without a load. A loaded engine with no RUNNING job is left to its lease.

**Ensure** replaces `ensureBackendRunning` and `ensureOllamaRunning`:

1. The holder is this launch and `serving`: proxy.
2. The holder is this launch and pending, loading or waking: release `a.mu` and wait for the bus event, Slurm's terminal state or `HEALTH_TIMEOUT_SECONDS`, with SSE keepalive as today. The recreate-coalescing fields (`recreating`, `pendingModel`) keep a second request from submitting a second job.
3. A trainer or probe job holds the GPU: 503 `gpu_held_by_job`, `Retry-After: 30`. An engine bound to an active hold for another launch: 503 `gpu_held_by_hold`.
4. Otherwise, switch:
   - drain, as `drainBackend` does today;
   - end the holder's job -- `SIGUSR1` when its backend sleeps, a pending job names its launch, and the new launch is on another backend (on the same backend the new model process replaces it anyway); a cancel otherwise;
   - wait until Slurm reports the job ended (polling every 500 ms, at most 150 s: longer than the USR1 trap's 120 s `sleep` and `KillWait`'s 30 s). If it has not ended by then, cancel it (harmless after USR1) and submit anyway: the new job waits PENDING (Licenses), since Slurm returns `engine` only after the old job's epilog (Phase 5), and case 5's `gpu_queue` ends the request if the license never comes;
   - submit the new engine job, with `DEVAI_LOAD` = C2 built from `computeLaunchConfig` and the rest of today's resolution;
   - continue as in case 2.
5. How a launch can end:
   - The guard refuses it (the `devai.events.slurm.guard` verdict for the job is `busy`, or the job is held): cancel the job and answer 503 `gpu_held_by_foreign_process` naming the holders. The breaker is not charged. The next request submits a new job; a held job is never released (Slurm delays a requeued job -- one the prolog refused -- by about 2 minutes; measured).
   - `load_failed`: 502 with the log tail. The breaker is charged (`noteLaunchAttempt`, and repaid by a served response, as today).
   - The job is still waiting on the license when the request gives up: 503 `gpu_queue`, `Retry-After: 30`. When the last request waiting on that pending engine job gives up, the router cancels the job.

**Yielding.** A PENDING trainer or probe job makes the router end its engine job, unless that engine is bound to an active hold: drain, then cancel. Until the waiting job has run, new engine launches get case 3. Without this, a keep-warm engine job would hold the license forever, and the trainer or probe would never start.

**The sleeper (I9).** When no PENDING job, no active hold and no RUNNING engine job names the sleeper's launch any more, and no RUNNING bench, probe or action job naming it is within 60 s of its start without a hold, the router sends `unload` with `target: sleeper` and `force`. It checks this on every refresh of the pending list and at boot.

**Idle sweep.** With `IDLE_TIMEOUT` > 0, the idle sweep cancels the engine job.

### Trainer on :11438 (I11)

- **`POST /v1/fine_tuning/jobs`** is forwarded to the controller. On a 2xx answer, the router ends the engine job as for any switch and submits a trainer job carrying the returned `id`. If the submit fails, it tells the controller (`POST /devai/v1/jobs/{id}/fail`) and answers 503.
- **`POST /v1/fine_tuning/jobs/{id}/cancel`** is forwarded to the controller, which finalizes a job that has not started, and the trainer job is cancelled. The answer is the record once it is terminal, waiting at most 15 s, as today.
- **Reads** are proxied to the controller, which now always runs. The "status reads never launch the trainer" rule disappears together with the launch.
- **Boot:** a controller job that is not terminal and has no PENDING or RUNNING trainer job is failed with `trainer_stopped`.
- **The hold:** a RUNNING trainer job holds the GPU (Ensure case 3). `/health` on 11438 keeps its `job_runner` shape, with `hold_until` = the job's start + its time limit.

### Detailed steps

1. Read the control routes from vLLM 0.28's and SGLang 0.5.16's sources in `devai-engines`. Pin them (C5) with a test that fails when the installed source registers a route under the same guard that the list does not have.
2. Remove today's launch layer (Deliverables). Port the tests of the behaviour the Slurm launcher keeps before deleting the rest.
3. The record, Ensure, yielding, holds, the trainer and boot. Unit tests run against an `httptest` slurmrestd (requests checked against the spec dump) and an embedded nats-server with fake `devai-control` responders. They cover:
   - every Ensure case;
   - a guard refusal not charging the breaker;
   - boot adoption;
   - a hold refusing a switch, and a hold dropped when its job ends;
   - a sleeper chosen only when a pending job names the launch;
   - a sleeper unloaded once nothing names it, including at boot, but not while an active hold or a RUNNING engine job names it, nor while a just-released re-run is about to take its hold;
   - yielding to a pending probe.
4. Compose and the live checks.

### Exit criteria

- The router container has no podman socket, the per-engine containers are gone, and a picker-launched agent completes a turn on Ollama, vLLM and SGLang.
- The 2026-09-28 failure: a foreign process holds 4 GiB, and a request for a vLLM model arrives. Either the holder is killed and the model loads, or the client gets 503 naming the holder and the node drains. In both cases the model's breaker count does not change.
- Requests alternating between an Ollama model and a vLLM model give exactly one engine job per switch, and never two engine processes on the card at once (the samples in `gpu.json`; no overlapping engine jobs in `sacct`).
- After a router restart while vLLM serves, the next request for the same model is proxied without a load.
- `make laya-check` passes through the router on Slurm: swap, 503 hold, job, parity, teacher restored.
- `/devai/v1/ready` reports not ready with slurmrestd or NATS stopped, while `/health` still answers.
- `make test-router`, `make test-e2e`, `make test-vllm` and `make test-sglang` pass.

### Phase 9 risks

| Risk | Mitigation |
| ---- | ---------- |
| a model relied on something only the podman path provides (environment, mounts) | Phase 8's argv and environment parity check; `make test-vllm` |
| waits under `a.mu` stall other backends | the release-and-wait pattern `waitForHealthy` already uses |
| a REST call on every request | the record is the cache; Slurm is asked only on a switch, a hold, and every 2 s for pending jobs |
| a lab client calls `/devai/v1/holds` | holds are checked against Slurm (C5), and a lab client cannot create a job |
| no fallback inside the router: a defect in the Slurm path stops serving | Phase 8's live checks before the cut-over; this phase's exit criteria; rollback by reverting Phase 9's commits (I10) |

---

## Phase 10 -- Workloads and operator actions as jobs; the re-bench

### Goal

- A bench is one job per (model, task), with Slurm's time limit and a hold.
- Probes are probe jobs that load through the bus.
- Every `job` action in the registry runs as a Slurm job in `devai-operator` (O2).
- The web shows the queue and the history, and can cancel, hold, release, suspend and resume.
- D1's queue control and D13's suspend work, from the web and from `devai-jobs`.
- The stopped re-bench then runs once, on the final images, through this path.

### Deliverables

```
deploy/Dockerfile.operator        modify -- devai-control, devai-bus, devai-jobs, deploy/operator/control.yaml;
                                            supervisord gains devai-control beside Apache
deploy/operator/control.yaml      new    -- container `operator`, services `actions` (kind actions, registry C8,
                                            podman CLI for the devai.job label cleanup) and `apache` (kind program);
                                            user control-operator
deploy/docker-compose.yaml        modify -- devai-operator: the control-operator and operator bus passwords, the JWT key
devai-operator/devai_operator/jobs.py
                                  new    -- `job` actions submitted through `devai-jobs submit ... --json` as the job
                                            kind their entry names (C8);
                                            the queue, history and job pages (job.log, action.log and result.json,
                                            polled); cancel, hold, release, suspend, resume
devai-operator/devai_operator/registry.py
                                  modify -- `job` and `direct` kinds per O2: stack lifecycle (stack-up, stack-down,
                                            build and restart of devai-slurm) stays direct
deploy/operator/actions.yaml      modify -- `kind: job` for every non-lifecycle action. Entries: bench-task
                                            (job_kind bench); the submitters probe-ollama, probe-vllm,
                                            probe-sglang, probe-load-vllm, probe-load-sglang (job_kind action: they
                                            only submit probe jobs); and per model probe-ollama-model,
                                            probe-vllm-model, probe-sglang-model, probe-load-vllm-model,
                                            probe-load-sglang-model (job_kind probe), each run inside a probe job
                                            (C3). Like bench and bench-task, a submitter never runs a prober itself
scripts/ops/bench-task.sh         new    -- the bench-task action: `podman run --label devai.job=$DEVAI_JOB_ID ...
                                            devai-lab /scripts/bench/bench_task.py ...` with the GPU device for NVML
                                            reads and CUDA_VISIBLE_DEVICES= (I12), and $DEVAI_RESULTS_DIR mounted at
                                            the same path (C8); forwards INT and TERM to it
scripts/bench/bench_task.py       new    -- take the hold (C5); run one task with bench_runner; on SIGINT stop inspect
                                            and score the unbroken prefix (harvest_truncated's functions, imported);
                                            write result.json and the bench cache row with the job id; release the hold.
                                            On SIGTERM: stop inspect, result.json `cancelled`, no cache row. A 503 or
                                            409 on the hold: wait (`Retry-After`, else 30 s) and ask again until the
                                            job is stopped. Reads DEVAI_JOB_ID and DEVAI_RESULTS_DIR (C8)
scripts/bench/bench_runner.py, scripts/bench/_bench_core.py
                                  modify -- engine /metrics from devai-engines:<port> instead of the per-engine containers
scripts/bench-sync.py             modify -- `--submit`: the same plan, submitted as one bench job per (model, task) in
                                            plan order, with the run id in the comments; the clean-slate and deadline
                                            path stays behind `--legacy` until Phase 14
scripts/_probe_hf_common.py, scripts/_probe_load.py
                                  modify -- a bus driver beside the container driver: load and unload through
                                            devai-control with the job id; readiness from the load reply; the engine log
                                            from jobs/engines/<backend>/<job>.log for classify_failure_logs, the
                                            serving-OOM check and the --ctx-exact hint; container_state from `state`;
                                            assert_no_active_backends replaced by the license
scripts/probe-ollama-reasoning.py, scripts/ops/probe-ollama.sh
                                  modify -- per band: `load` the server settings (I8); the ballast inside the probe
                                            job; the chats go straight to devai-engines:11434 as today
scripts/probe-*-reasoning.py      modify -- `--list-targets` (the prober's own selection: one line per model, per
                                            model and band for Ollama) and one target per run (the -model actions)
scripts/ops/probe-*.sh, scripts/ops/bench.sh
                                  modify -- submit jobs; no cache-down needed
scripts/model-sync.py             modify -- no cache-down / cache-up bracket around the probes
devai-tools/cmd/devai-jobs/       modify -- submit (action, probe, bench run), hold, release, suspend, resume,
                                            interrupt, clear; `service <container> <service> start|stop|restart|reload`
                                            (D15's service control, over the bus as user operator)
devai-operator/devai_operator/engines.py
                                  new    -- the engines page: each service's state from the bus, and start, stop,
                                            restart, reload (D15)
```

### Detailed steps

1. **`devai-control` in the operator image.** As soon as Phase 6's image exists, `Dockerfile.operator` gains `devai-control` (C8's `actions` kind, from Phase 5) and the app submits `job` actions through `devai-jobs`. An action started from the browser then appears in `devai-jobs list` with its log.
2. **The probe job.**
   - One probe job per model (per model and band for Ollama). `scripts/ops/probe-*.sh` asks the prober for its targets (`--list-targets`, the prober's own selection) and submits one job each through `devai-jobs submit probe`, in order.
   - Each job runs `probe.sh` as C3 specifies: the model's lease first, then `run` for `probe-<backend>-model` with its model and the run's hold; on TERM, `stop` and wait for the run; then a plain `unload` (the job owns the model) and the lease's end.
   - The prober loads, measures and unloads inside the job.
   - The prober's request for a cell and `gpu-arbiter -print-load`'s request for the same cell are compared in a test (I1).
3. **The bench task** (C3, license `bench`).
   - The hold is taken before the first sample, so inspect's clock never runs during a load.
   - At 30 min Slurm sends SIGINT, 120 s before the limit. The task scores the unbroken prefix, as `harvest_truncated.py` does today.
4. **`devai-jobs suspend <run>`** (D13, I5) holds the run's pending tasks and cancels the running one; its `result.json` gets status `cancelled` and it is not scored. It then submits the cancelled task again, held, with `attempt` + 1 and `rerun_of`.
   - **`resume`** releases the re-run, waits until it is RUNNING, then releases the rest.
   - **`interrupt <run>`** suspends the run and prints the `resume` command; it does not prompt, since `bin/devai-jobs` runs without a terminal and the web calls it with `--json`. In between, the operator runs the quick job (an agent turn, another job), then resumes. If the quick job's model is on another backend, the router's Ensure finds the re-run pending and ends the bench's engine job with `sleep` (Phase 9, case 4). On the same backend the bench's model is unloaded instead: each engine serves one model process on its port.
   - **`clear`** cancels every PENDING job (D1); `hold`, `release` and `cancel` act on one job.
   - The web's buttons call the same commands.
5. **The re-bench.** With every exit criterion above met, run the stopped re-bench through `bench-sync.py --submit`, once, on the final images.

### Exit criteria

- The engines page and `devai-jobs service` restart an engine service, and its state events show it.
- An action started from the browser appears in `devai-jobs list` with its log. Cancelling it kills its process group and removes the containers it started. A stack-lifecycle action still runs direct while `devai-slurm` is down.
- A bench of one model and two tasks runs as two jobs. A task given `--time=3 --signal=B:INT@60` is cut off and scores its prefix, as the deadline path does today. Its `result.json` and cache row carry the job id.
- A bench and an interactive agent session run side by side on the same model without failures.
- `devai-jobs interrupt`:
  - with the same model, nothing is loaded (the bus log shows no load);
  - with a model on another backend, the bench's model sleeps and wakes, and the cancelled attempt is in the history, not scored.
- Probing one vLLM model and one Ollama model through probe jobs gives the same fit verdict and context as the container prober for the same cell. Cells are compared field by field; VRAM may differ within run-to-run noise.
- The re-bench has run on the Slurm path. Its rows carry job ids, and `bench-report` renders them.
  - It covers every pickable row at the context the picker offers, SGLang's included if SGLang is kept.
  - No model trips the launch breaker in a loop while it runs (sglang-backend-remediation's 24-hour soak check).
- A live `make model-sync` run onboards one new catalog row: download, probe jobs, ledger verdict.
- A prober left running after its probe job was killed with SIGKILL is stopped by the next GPU job's prolog.
- `make test-python` (the bench, probe and model-sync tests updated), `make test-devai-tools` and `make test-operator` pass.

### Phase 10 risks

| Risk | Mitigation |
| ---- | ---------- |
| the probers need a Python package that `devai-operator` lacks | checked in Phase 6; the probers use the stdlib and PyYAML |
| a bench container with a GPU device computes on it | `CUDA_VISIBLE_DEVICES=` empty; the device is there for NVML reads only (I12) |
| a held re-run starts late after its release | untested; it is a fresh submission, not a requeue, which the spike saw start in about 1 s |
| submitting a job can run any registered action, destructive ones included, without the page's confirmation | the JWT key only in `devai-slurm`, the router and `devai-operator`; nothing published (I13); bus users limited to their subjects (C1) |

---

## Phase 11 -- Running without a GPU

### Goal

On a host without an NVIDIA GPU, the stack comes up from the same images:

- Ollama serves on the CPU;
- the laya trainer trains on the CPU;
- vLLM and SGLang say that they need a GPU (M15).

### Deliverables

```
deploy/docker-compose.yaml        modify -- the `devices:` entries move out
deploy/compose.gpu.yaml           new    -- the GPU device for devai-slurm, devai-engines, devai-laya-trainer and
                                            devai-operator; `make cache-up` and `make operator-up` add this file when
                                            scripts/host-gpu.sh prints a device (operator-up recreates
                                            devai-operator, which would otherwise lose the GPU its probers need)
Makefile (operator-up), scripts/ops/stack-up.sh
                                  modify -- the host check runs on the host at `make operator-up` (inside
                                            devai-operator the CDI spec is not visible); its result reaches the
                                            stack-up action as DEVAI_GPU_DEVICE
deploy/docker-compose.yaml (router)
                                  modify -- `${DEVAI_GPU_DEVICE-nvidia.com/gpu=all}` without the colon, so an empty
                                            value is kept and means CPU
gpu-arbiter/slurm_launch.go       modify -- empty DEVAI_GPU_DEVICE = CPU mode: vLLM and SGLang requests refused
                                            before anything is submitted, with an error naming the reason; Ollama
                                            launches with num_gpu 0 (I14)
scripts/probe-ollama-reasoning.py, scripts/ops/probe-ollama.sh
                                  modify -- a `cpu` band: models loaded with no GPU layers, resident memory measured
                                            against the host's RAM, cells stamped with the band
scripts/model-picker.py           modify -- without a GPU: reads the `cpu` band, hides vLLM and SGLang rows
tests/                            modify -- CPU mode of the router, picker and probe band
```

### Detailed steps

1. **Devices.** Compose cannot make a device conditional, so the device list moves to `deploy/compose.gpu.yaml`, which `make cache-up` includes only when the host check prints a device.
2. **The router.** The device is exactly what compose hands over, and an empty one means CPU mode. (The old podman path fell back to `nvidia.com/gpu=all`; it is gone since Phase 9.)
3. **The CPU probe band.** Ollama on the CPU keeps weights and KV cache in RAM.
   - The band's budget is the host's RAM minus a named reserve for everything else on the host (a unified-memory host shares that RAM with any GPU it may have).
   - The probe decides each model's context as on the GPU; the compose default `OLLAMA_CONTEXT_LENGTH=262144` is not assumed to fit.
4. **The picker.** Without a GPU it reads the `cpu` band. The bench stays GPU-only, so CPU rows show no bench columns.

### Exit criteria

- On this host, with the GPU forced off (`DEVAI_GPU_DEVICE=`):
  - `make cache-up` works;
  - `make probe` fills the `cpu` band;
  - the picker lists Ollama rows only;
  - an agent completes a turn on the CPU;
  - a request on port 11435 or 11436 returns the no-GPU error.
- `make test-laya-trainer` passes without a GPU (its tests already run on the CPU).
- With the GPU present nothing changes: `make test-router`, `make test-python` and `make test-e2e` pass.

### Phase 11 risks

| Risk | Mitigation |
| ---- | ---------- |
| CPU inference is slow on hosts without unified memory | accepted; M15 targets unified-memory hosts |
| a forced-CPU run on this host is not the same as a real unified-memory host | recorded as such; verified on real hardware when one is available |

---

## Phase 12 -- Lab and host setup in the web

### Goal

A lab session is one click away (O4). Every host-only item shows its state and its exact command (O3).

### Deliverables

```
scripts/ops/lab-start.sh, lab-stop.sh
                                  new    -- what `make lab` does, as `lab` actions: a devai-lab container with
                                            JupyterLab (and dsh when DSH_PORT is set), labelled devai.lab=<name>;
                                            the read-only /library mount once Phase 20.1 has landed
deploy/operator/actions.yaml      modify -- lab-start, lab-stop (kind lab); the host items (kind host): setup-logs,
                                            setup-jobs, the model-store volumes, install
devai-operator/devai_operator/lab.py, host.py
                                  new    -- the lab page (running labs, their JupyterLab and dsh links with the token,
                                            start and stop); the host-setup page (each item's state and command)
```

### Detailed steps

1. **The lab page** lists running `devai-lab` containers by their label and shows their links. Start and stop are `lab` actions; they run direct, because they hold no GPU and must not wait in Slurm's queue.
2. **The host-setup page.** For each item it checks what the container can see. The mounts under `/var/cache/devai` are visible through the bind mount, so `setup-logs`, `setup-jobs` and the store volumes can be checked. The page then prints the command. `install` writes the host's `~/.local/bin`, which the container does not see, so it is shown with its command and "state not checked".

### Exit criteria

- From the browser, a lab starts, its JupyterLab link opens with the token, and it stops.
- Each host item shows done, not done, or not checked, with its command.
- `make test-operator` passes.

---

## Phase 13 -- History

### Goal

"What happened yesterday" is one command, and one page.

### Deliverables

```
devai-tools/cmd/devai-jobs/       modify -- list --since, show <id>, queue, engines: slurmdbd (REST /slurmdb/v0.0.42)
                                            joined with results/<id>/ and bus/<date>.jsonl
devai-operator/devai_operator/jobs.py
                                  modify -- the history page from `devai-jobs list --since ... --json`
scripts/bench/bench_report.py     modify -- a job id column
```

### Exit criteria

- `devai-jobs list --since yesterday` (on the host through `bin/devai-jobs`, and as the history page) shows every job with its kind, model, state, exit, times, GPU peak, mean and energy (engine, trainer and probe jobs; the others have no GPU record), and result status.
- `devai-jobs show <id>` adds the job's log tail, its `gpu.json` summary and its bus events.
- `make test-devai-tools` passes. There is no MCP tool (M10).

---

## Phase 14 -- Cleanup and documentation

### Goal

Only the new path is left, no devai application container has a GPU device, and the documentation describes the result.

### Detailed steps

1. **Router.** Nothing is left to remove: Phase 9 removed the podman launch layer (I10).
2. **Probers.** Remove the container driver (`container_run_detached` and the functions around it, `MUTEX_CONTAINERS`), and `scripts/ops/probe-ollama.sh`'s throwaway `devai-ollama` container (Phase 9).
3. **Bench.** Remove `clean_slate.py`, the deadline path and `--legacy` in `bench-sync.py`, and the `bench-harvest` target. Rewrite `test_bench_clean_slate.py` and `test_bench_deadline.py` for the job path or delete them.
4. **Images.** Remove `deploy/Dockerfile.ollama` and `deploy/Dockerfile.vllm` (merged into `Dockerfile.engines`).
   - List the retired images with their sizes: the upstream ones and the old `devai-base-cpu`, `devai-lab-cpu`, `devai-base-gpu`, `devai-lab-gpu`, `devai-ollama`, `vllm-devai`.
   - Remove them from the local store only after the operator confirms: the store is shared with other projects on this host.
5. **Lab.** `make lab`, `make shell`, the lab actions and devai-agent stop attaching the GPU (D1). aiagent's `DEVAI_AIAGENT_GPU=share` goes, because its GPU work is trainer jobs; tell the aiagent session.
6. **Documentation.**
   - New: `docs/operator.md`, the operator's reference.
   - Updated: README, INSTALL.md, CLAUDE.md, docs/router.md, docs/backends.md, docs/bench-results.md ("Run protocol"), docs/laya-trainer.md ("The GPU hold"), docs/aiagent.md, docs/pipelock.md.
   - docs/slurm.md: its status becomes "built".

### Exit criteria

- The re-bench of Phase 10 has run, so nothing removed here is still needed for it.
- The upstream-images guard test passes with an empty transitional allowlist, and `make build` succeeds from a store that holds only the pinned Debian image.
- A test reading compose and the scripts finds a GPU device only on `devai-slurm`, `devai-engines`, `devai-laya-trainer`, `devai-operator` and bench containers (NVML reads only).
- A new user can go from the one-time host setup to a running stack through the browser, following the documentation alone.
- `make test-router`, `make test-python`, `make test-devai-tools`, `make test-operator` and `make test-slurm` pass.

---

## Phase 15 -- SGLang: keep or freeze

From sglang-backend-remediation, Phase 5. Its Phases 0-3 shipped; its Phase 4 re-probe is done by Phase 4 here, apart from the 7 GPTQ cells, which are removed instead (I17).

### Goal

Decide, on measured evidence, whether SGLang stays -- before Phase 4 spends its riskiest step compiling it -- and leave no stale SGLang claim behind either way. The picker stops offering an agent/backend pair known to fail.

It runs on today's images: upstream SGLang 0.5.16 is the version Phase 4 would compile (M5), so the measurement carries over. It must finish before Phase 4, step 4a. The SGLang plan ran this decision after its re-probe, and only if fewer than four SGLang models stayed servable; here it comes first, so a freeze saves Phase 4's SGLang build, and its documentation check runs again after Phase 4's re-probe (Phase 4's exit criteria).

### Deliverables

```
gpu-arbiter/main.go (sglangEntrypoint)   modify -- emit --enable-metrics (Phase 8 carries it into pkg/engineargs)
deploy/backend-flags.yaml                modify -- pin --enable-metrics and --disable-radix-cache
scripts/bench/bench_concurrency.py       modify -- run_multiturn gains `body_fields` (merged into the TOP level of every
                                                    request, warm-up included; never under an `extra_body` key, which
                                                    neither engine reads) and `on_turn(i)` (a callback before each
                                                    measured turn); with neither, it behaves exactly as today.
                                                    run_multiturn_sessions runs N sessions through it, each salted with a
                                                    per-run nonce, the engine and the session number, reads SGLang's
                                                    counters from --metrics-url in on_turn before turn 2 and again after
                                                    the last turn, and reports the per-session turn-2+ medians; CLI
                                                    --sessions, --salt-prefix, --metrics-url (I16)
scripts/bench/sglang_keep_check.py       new    -- the whole measurement (steps 2-3): launches each engine through the
                                                    probers' container driver, with their command args and recovery
                                                    flags, at one context and each probe cell's KV dtype (I16), after
                                                    checking that nothing holds the card (assert_no_active_backends);
                                                    steady TTFT and the sessions on each, SGLang also with
                                                    --disable-radix-cache; I16's statistics with scripts/stats/statlib.py;
                                                    a JSON report that M16 cites. Removed again after M16 (step 4)
Makefile                                 modify -- bench-concurrency passes SESSIONS, SALT_PREFIX, METRICS_URL;
                                                    sglang-keep-check (removed with the script)
deploy/operator/actions.yaml             modify -- once Phase 6 has landed, sglang-keep-check in `not_offered`: a one-shot
                                                    measurement for M16, run with the stack down
gpu-arbiter/testdata/argv-golden.txt     modify -- regenerated for --enable-metrics
scripts/probe-sglang-reasoning.py (sglang_command_args)
                                         modify -- --enable-metrics, as the router's argv; a --disable-radix-cache
                                                    switch for the radix-off run
scripts/model-picker.py                  modify -- `_resolve_agent` does not offer Codex on sglang (SGLang's /v1/responses
                                                    takes no function tools, so every Codex turn fails there)
CLAUDE.md, TODO.md, AGENTS.md, README.md, scripts/model-families.yaml (its SGLang status block),
docs/backends.md, docs/router.md (the tool-mode table), docs/bench-results.md (via make bench-report)
                                         modify -- every SGLang capability claim the caches contradict
docs/html/                               modify or remove -- 15 hand-maintained pages with no generator; router.html
                                                    still says "NVFP4 (broken)", bench-results.html "SGLang on hold"
docs/plans/platform-rebuild.md           modify -- the decision, its thresholds and measured inputs, recorded as M16
```

### Detailed steps

1. **The baseline (I15).** Probe Qwen3.5-9B-NVFP4 on vLLM 0.28 (`make probe-vllm-devai PROBE_REPO=...`); it already fits SGLang. If it does not fit vLLM 0.28, gpt-oss-20b takes its place after the same probe (neither has run on vLLM 0.28 yet): its weights are in both stores and it fits SGLang (131,072 tokens on 2026-10-05). If gpt-oss-20b does not load on vLLM 0.28 either, Gemma-4-26B-A4B-it-NVFP4 is tried the same way: its weights are in both stores, it fits SGLang (262,144 tokens, with its SGLang recovery flags), and its thinking is opt-in, so the reasoning-off body needs nothing more. If none of the three loads, Phase 15 measures nothing: SGLang is frozen by default, since I16 cannot be met without a baseline on the surviving vLLM (M4). M16 records the three failed probes, and the operator may overrule it. The keep thresholds and their measurement are I16, recorded before any run.
2. Emit `--enable-metrics` and pin it. The engines are launched outside the router, with the whole stack down (`make cache-down`; the check calls `assert_no_active_backends` before its first launch, as the probers do, because stopping only the router leaves Ollama's warm model or an engine on the card): `scripts/bench/sglang_keep_check.py` starts each through the probers' container driver -- SGLang with `sglang_command_args` (which emits `--enable-metrics` as the router does from this phase, and `--disable-radix-cache` for the radix-off run), vLLM 0.28 with the vllm-devai prober's arguments -- and runs the workload against it directly: the multi-turn sessions of `bench_concurrency.py`, with the cache-hit read-out. Before the measured run, an uncounted dry run checks the instrument: SGLang radix-on, 2 sessions, whose turns 2-5 must show a non-zero hit rate. Zero means the workload does not reuse prefixes: the scorer is fixed and the dry run repeated, before any measured run (I16).
3. Measure on one checkpoint served by both engines, exactly as I16 defines it (the same check script; MTP and reasoning off, no tools, one context): steady TTFT, turn-2+ TTFT over salted sessions, and SGLang's engine-reported cache hit rate.
4. Decide, and record it as M16:
   - **keep:** Phase 4 builds SGLang as planned;
   - **freeze:** Phase 4 skips steps 4a-4b and SGLang's stage in `Dockerfile.engines`; `sglang` leaves `_PICKER_BACKENDS` and the make targets; its entry leaves the router's backend table together with the then-unused `sglangURL`, so its router code stays but is unreachable (checked on a copy: the router builds and `go test` passes, 2026-10-06). Its prober (`scripts/probe-sglang-reasoning.py`) and its Python tests move to `attic/sglang/`, with a `RESTORE.md`, following the `attic/` precedent. Everything else that lists `sglang` as a backend drops it in the same change, so nothing points at a dead port:
     - `scripts/model-sync.py`: its `make probe-sglang` step;
     - `scripts/bench-sync.py`: `BACKENDS`;
     - `scripts/bench/bench_runner.py`: its backend choices, and the SGLang bench target;
     - `scripts/select-models.py`: the SGLang store-gap check;
     - `scripts/model-picker.py`'s `_BACKENDS`, and the `router-sglang` entries in config/codex/config.toml and config/opencode/opencode.json, so no agent keeps the provider.

     Check: `make model-sync DRY_RUN=1` and `make bench-plan` name no SGLang step or row. The fixes from sglang-backend-remediation's Phases 1-3 stay: most of them serve vLLM too. Every mention of SGLang in Phases 4-14, 16, 18 and 19 and in C1, C2, C5, C6 and C7 then lapses, and exit criteria that name SGLang read "if kept". The `RESTORE.md` carries the SGLang plan's open questions.
   - Either way, M16 records the unexplained Qwen3.5-9B HumanEval result (0.86 on SGLang against 0.68 on vLLM, same checkpoint).
   - Either way, once M16 is recorded, `scripts/bench/sglang_keep_check.py` and its target are removed. Git keeps them, and M16 cites the report and the commit. It is a one-time measurement, built on the vllm-devai prober (removed in 4e), the SGLang prober and the container driver (removed in Phase 14).
5. Correct every prose site named above, and regenerate or delete `docs/html/`: a published page with a wrong capability claim is worse than none.
6. Gate Codex off SGLang in the picker.

### Exit criteria

- The keep-or-freeze decision and its measured inputs are recorded here (M16), against the thresholds of I16.
- No file in the repository asserts an SGLang capability that the probe caches contradict.
- The picker never offers an agent/backend pair that is known to fail.

### Phase 15 risks

| Risk | Mitigation |
| ---- | ---------- |
| the multi-turn scorer measures something other than prefix reuse | step 2's uncounted dry run (a non-zero hit rate) before the measured run |
| no baseline loads on vLLM 0.28 | the three-candidate order in step 1; freeze by default, recorded in M16 |
| the measurement needs a GPU window | it is short (one checkpoint, two engines) and replaces part of Phase 4a's cost if the answer is freeze |

---

## Phase 16 -- Router short-circuit

From router-shortcircuit, Phases 1-4, and the leftovers of its Phase 5.

### Goal

Answer the throwaway requests Claude Code sends -- quota check, title generation, suggestion mode, command-prefix detection, file-path extraction -- in the router, without an engine. Today each can trigger a cold start whose answer is discarded as a UI label; after Phase 9 each would also cost an engine job.

The governing rule: a request may be short-circuited only when its correct answer does not depend on the conversation's content -- a constant, or a pure syntactic function of the request bytes. Anything needing judgment falls through to the model.

The feature is off by default, fail-open on every error, limited to `/v1/messages`, and calibrated on captured traffic before any pattern is trusted. Out of scope, as in the original plan: no tool execution, no URL fetching, no response cache.

It needs nothing. 16a should start early, so real traffic is captured. 16b-16d touch `makeRequestHandler`'s request path, which Phase 9 keeps; if they land before Phase 9, Phase 9 carries them.

### Deliverables

```
gpu-arbiter/fingerprint.go          new    -- 16a: fingerprintRequest(path, body): path, model, max_tokens, n_messages,
                                              has_system, system_sha256_12, has_tools, stream, body_bytes
gpu-arbiter/fingerprint_test.go     new    -- table-driven, including "no message content in the output"
gpu-arbiter/shortcircuit.go         new    -- 16b-16c: registry load, detector, Anthropic JSON and SSE synthesis,
                                              the command_prefix and filepaths responders
gpu-arbiter/shortcircuit_test.go    new    -- golden responses (JSON and the exact SSE byte sequence), no-match tables
gpu-arbiter/main.go                 modify -- fingerprint line when DEVAI_REQUEST_FINGERPRINT=on (and a content-revealing
                                              DEVAI_REQUEST_FINGERPRINT_RAW=on); load the registry at boot; call
                                              tryShortCircuit after the malformed-JSON guard, before the model allowlist;
                                              16e: /api/blobs/ added to ollamaMutationPaths
deploy/shortcircuit-probes.json     new    -- the pattern registry, "enabled": false by default
deploy/docker-compose.yaml          modify -- the registry mounted read-only; DEVAI_SHORTCIRCUIT, SHORTCIRCUIT_REGISTRY
tests/fixtures/claude-code-probes/  new    -- captured request bodies, one per probe
docs/router.md                      modify -- the fingerprint runbook; short-circuit as step 0 of the rewrite chain;
                                              the model-store mutation guard, which docs/router.md does not document yet
```

### Detailed steps

1. **16a -- Fingerprints, observe only.** One content-free `fingerprint:` line per POST when `DEVAI_REQUEST_FINGERPRINT=on`; the system text appears only as the first 12 hex digits of its SHA-256. The runbook: set it, run a real Claude Code session, bucket the lines by `(path, has_system, system_sha256_12, max_tokens, has_tools)`, and capture one body per frequent, deterministic bucket as a fixture.
2. **16b -- Constant answers.** Registry entries `quota_check`, `title_generation`, `suggestion_mode`, matched only when ALL their declared markers co-occur (high precision, biased to false negatives). The answer is synthesised as an Anthropic message object, or as the full SSE sequence (`message_start` ... `message_stop`) when `stream` is true. Both `DEVAI_SHORTCIRCUIT` and the registry's `"enabled"` must be on. Every hit logs a `shortcircuit:` line naming the probe. Placed before the model allowlist on purpose: a probe carrying Claude Code's hard-coded Haiku id is answered, not refused. The registry (read at boot from `SHORTCIRCUIT_REGISTRY`, default `/etc/devai/shortcircuit-probes.json`; a missing file means off):

   ```json
   {
     "enabled": false,
     "surface": "/v1/messages",
     "probes": [
       { "name": "quota_check",
         "match": { "max_tokens_eq": 1, "single_user_message": true, "content_contains_any": ["quota"] },
         "respond": { "kind": "constant", "text": "Quota check passed." } },
       { "name": "title_generation",
         "match": { "system_contains_all": ["title"], "system_contains_any": ["sentence-case title", "this session"] },
         "respond": { "kind": "constant", "text": "Conversation" } },
       { "name": "suggestion_mode",
         "match": { "user_content_contains_any": ["[SUGGESTION MODE:"] },
         "respond": { "kind": "constant", "text": "" } }
     ]
   }
   ```

   Matcher keys: `max_tokens_eq`, `single_user_message`, `content_contains_any`, `user_content_contains_any`, `system_contains_all`, `system_contains_any`, `body_contains_all`; every declared key must hold. The patterns mirror free-claude-code's and are calibrated against the 16a fixtures before the registry is enabled.
3. **16c -- Local-parse answers.** `prefix_detection` (markers `<policy_spec>` and `Command:`) and `filepath_extraction` (`Command:` and `Output:` plus a file-path marker), answered by pure functions (`respond.kind` `command_prefix` and `filepaths`) ported from free-claude-code's `extract_command_prefix` / `extract_filepaths_from_command` and pinned by the fixtures. An empty extraction falls through.
4. **16d -- count_tokens, optional.** Only if 16a shows Claude Code calling `/v1/messages/count_tokens` against the router; otherwise record it as shelved, with the reason.
   - **Forward it to the engine.** vLLM 0.28 and SGLang 0.5.16 serve that route themselves and count the fully rendered chat template, so the count is exact (read in both sources, 2026-10-06).
     - The router forwards it only when the request would be proxied without any launch. After the suffixes are peeled, the loaded launch (model, context, speculative spec) must be exactly this one, and serving: Ensure case 1, or before Phase 9 `bs.running` with the same `currentModel`, `currentContext` and spec.
     - It then calls the proxy directly, never `ensureBackendRunning` or Ensure, so a count can neither start nor switch anything.
     - First it runs the `/v1/messages` rewrite chain unchanged -- the system-message fold and `applyHFAnthropicMessagesPolicy` -- behind a path check that accepts both paths.
       - Under DISABLE that sends vLLM `chat_template_kwargs.enable_thinking=false` and SGLang `thinking: {type: disabled}`.
       - Under ENABLE or noop it drops `thinking` on SGLang; without the drop, a row with no reasoning parser answers 400.
       - vLLM's count ignores `output_config`, so on vLLM the router also passes the policy's effort as `chat_template_kwargs.reasoning_effort`.
       - SGLang's count request has no `chat_template_kwargs`, so an explicit effort cannot be carried there. The count then uses the template's default effort, which is what `/v1/messages` gets under `auto`.
   - **Not serving, or another model or context loaded:** the router answers 503 `count_tokens_unavailable` itself and does not forward. Forwarding would start or switch the engine, since any request that names a model goes through `ensureBackendRunning` (after Phase 9, Ensure).
   - **Ollama** v0.34.2 has no such route: on its port the router answers 404 `count_tokens_unsupported`.
   - Never an approximation: a wrong count drives wrong compaction.
5. **16e -- Guard leftovers.** Block `/api/blobs/` like the other store-changing Ollama paths (the original Phase 5 planned it; the shipped guard does not), and document the guard in docs/router.md.

### Exit criteria

- With `DEVAI_REQUEST_FINGERPRINT=on` every POST emits exactly one `fingerprint:` line, and none when it is unset; a unit test shows the raw prompt text never appears in it.
- Replayed fixtures return the correct canned answers, byte-validated against the golden JSON and SSE, and a normal chat, a summarisation-style request and a tool-call request all fall through.
- With the feature off (the default), the request path is byte-identical to today.
- Removing a required marker from a fixture flips it to fall-through.
- count_tokens returns the engine's own count on the fixtures (vLLM and SGLang), equal to the prompt tokens the engine reports for the same `/v1/messages` body under `auto` and under `::nothink`. It answers 503 `count_tokens_unavailable`, with the loaded model untouched, when another model or context is loaded or nothing is, and 404 `count_tokens_unsupported` on the Ollama port. Or it is recorded as shelved.
- `/api/blobs/` answers 403 on every listener; `make test-router` passes.

### Phase 16 risks

| Risk | Mitigation |
| ---- | ---------- |
| a real request matches a probe and gets a fabricated answer | all markers must co-occur; off by default; every hit logged; calibrated on captured traffic |
| Claude Code changes a probe's wording | patterns live in a config file; a miss falls through to the model; fixtures pin the patterns |
| a malformed synthesised answer makes Claude Code error | golden JSON and SSE tests |
| logging request shapes leaks content | hashes only; the raw mode is a separate, documented switch |

---

## Phase 17 -- Catalog and fit tooling

From odysseus-borrowed-ideas, CB-3 and the rest of CB-1. CB-1's odysseus source was read first-hand (`services/hwfit/fit.py:184-235,592-598`, `services/hwfit/models.py:26-62,192-207`); CB-3's only by agents (`routes/cookbook_routes.py:3823-3997`, `static/js/cookbook.js:342-491`) -- re-open those files before porting.

### Goal

- **17a, vendor launch recipes (CB-3):** a model starts probing from vLLM's published recipe -- parsers, engine flags, environment -- instead of from a bare command, and a new model gets a labelled parser guess before its first probe. Best done before Phase 4's re-probe, which then benefits.
- **17b, the estimator (the rest of CB-1):** rank every catalog row before probing, and fill the picker's empty speed and quality cells with a clearly labelled estimate. The KV-at-context term already shipped in `scripts/catalog-discover.py`.

It needs nothing. 17b's calibration uses the re-probed caches (Phase 4) and the re-bench (Phase 10).

### Deliverables

```
scripts/fetch-vllm-recipes.py     new    -- 17a: fetch vllm-project/recipes YAML per catalog repo, normalise base args and
                                            env, tool-calling and reasoning args, per-precision VRAM minimum and flags,
                                            min_vllm_version; cache 6-12 h; offline-first (a cached copy, or nothing)
deploy/recipe-seeds.yaml          new    -- 17a's output, one entry per catalog row with a recipe: parsers, engine flags
                                            and env, min_vllm_version, the recipe's commit; the probers read it as a
                                            parser seed (order below). scripts/model-families.yaml keeps its single
                                            writer, catalog-discover-add
scripts/recipe-apply.py           new    -- `make recipe-apply NAME=<row>`: after review, writes a row's recipe flags into
                                            deploy/recovery-flags.generated.json (I25) as one group with
                                            `"source": "recipe"`, only where min_vllm_version admits vLLM 0.28.0
                                            (D8), scoped to the backend that runs 0.28: `vllm-devai` before Phase
                                            4e, `vllm` after (backends match by engine, so a ["vllm"] group before
                                            4e would also reach stock 0.22.1; 4e rewrites the groups written
                                            before it). It writes only flags the argument builders never emit and
                                            reports the rest, since the last flag wins and such a flag would pin
                                            every launch (I18). The excluded set: every flag in
                                            gpu-arbiter/testdata/argv-golden.txt except what a case injects as
                                            RecoveryFlags (today only --enforce-eager, which stays writable: I18's
                                            own example), plus the speculative flags (vLLM --speculative-config;
                                            SGLang --speculative-algorithm, -num-steps, -num-draft-tokens,
                                            -eagle-topk, -draft-model-path) and the sleep flags
                                            (--enable-sleep-mode, --enable-memory-saver). Tested: --enforce-eager
                                            written, --max-model-len reported
deploy/recovery-flags.generated.json
                                  new    -- `{"version": 1, "models": {}}` (I25)
gpu-arbiter/recovery_flags.go, scripts/_probe_hf_common.py
                                  modify -- read deploy/recovery-flags.generated.json after the curated file, and
                                            re-read both at each launch when either changed (the router reads its
                                            registry once, at boot, today) (I25);
                                            tests: a group applies only to its backends, and removing it reverts
                                            exactly its flags. _probe_hf_common.py also gets the parser order
                                            below: recipe-seeds.yaml as a seed, `recipe` as a `tool_parser_source`
                                            value, the name heuristic, and I24's switch skipping both
deploy/docker-compose.yaml        modify -- the router mounts ${DEVAI_REPO}/deploy read-only at /etc/devai/deploy and
                                            reads both recovery files there, so it sees a rewrite (until Phase 9 it
                                            still builds the engines' argv); DEVAI_REPO as I25 says
scripts/_fit_estimate.py          new    -- 17b: estimate_memory_gb, with the KV-at-context term moved here from
                                            scripts/catalog-discover.py; a bandwidth-bound decode estimate keyed by
                                            the GPU's name (on the host `nvidia-smi --query-gpu=name`, which
                                            DEVAI_GPU_TYPE overrides; in the picker, the bench cache's); quant
                                            quality and speed tables. It imports only the stdlib and yaml
scripts/catalog-discover.py       modify -- imports its KV term (estimate_kv_gb) from _fit_estimate
scripts/select-models.py          modify -- `make model-rank` (tiers perfect / good / marginal / too_tight; VRAM= override)
Makefile                          modify -- fetch-vllm-recipes, recipe-apply, model-rank; export DEVAI_GPU_TYPE
.env.example                      modify -- DEVAI_GPU_TYPE
deploy/operator/actions.yaml      modify -- once Phase 7 has landed: fetch-vllm-recipes, recipe-apply and model-rank,
                                            each a `job` action with a scripts/ops/ script (Phase 7's coverage test
                                            needs all three)
scripts/model-picker.py           modify -- a recipe badge next to TOOLS; a labelled `est` value only in empty cells;
                                            the GPU's name from the bench cache (_meta.host_env_history, mounted
                                            already; the lab has no GPU after Phase 14), and no estimate when that
                                            names no GPU or one the table lacks (DEVAI_GPU_TYPE is for the host
                                            tools only: it reaches neither the lab nor devai-agent); _fit_estimate
                                            imported optionally, so a newer picker still starts in an older image
deploy/Dockerfile.lab, bin/devai-agent, Makefile (install, uninstall), tests/python/test_picker_container_imports.py
                                  modify -- _fit_estimate.py co-located with the picker at /usr/local/bin and listed
                                            in CO_LOCATED, with a test that imports it beside the picker and the
                                            CO_LOCATED files alone; deploy/recipe-seeds.yaml at /etc/devai, baked by
                                            Dockerfile.lab like models.yaml and overmounted by devai-agent (staged
                                            by make install)
docs/llm-tokens-and-speed.md      modify -- cross-link the estimator to the derivation
tests/python/test_fit_estimate.py, test_vllm_recipes.py
                                  new    -- tables, tiers, offline behaviour, the version gate
```

### Detailed steps

1. **17a:** a parser comes from, in order: the curated `parsers:` block; the card-derived value (`scripts/_card_hints.py`, which refuses rather than guesses); the recipe seed; a name heuristic, labelled "guessed, not probe-verified"; none. The probe is the final authority, and the probe cache records where a value came from (`recipe` joins `tool_parser_source`'s values). A recipe never injects a flag the pinned vLLM rejects (`min_vllm_version`). Phase 4's card-path check ignores the curated block, the recipe seed and the name heuristic (I24).
2. **17b:** probe and bench caches stay authoritative. An estimate appears only where no measurement exists, always labelled `est`. An unknown GPU type gives no estimate, never a wrong one. The odysseus idea of feeding predicted speed into cluster routing is dropped with cluster mode.
3. **Calibration (17b), after Phases 4 and 10:** compare the predicted decode speed of every benched model with its measured one, and record the error band.

### Exit criteria

- For a model with an official recipe, the parsers are seeded before any probe runs; after `make recipe-apply NAME=<row>`, its first probe launches with the recipe's flags that lie outside C2, and the rest are reported.
- The picker shows a recipe badge; a probe-verified value overrides a recipe-seeded one; a model without a recipe falls back to the labelled heuristic, then to today's behaviour.
- `make model-rank` prints a tier for every catalog row without starting a container, and `VRAM=80` re-ranks the catalog as on an 80 GB card.
- The picker shows `est` only where no bench value exists.
- The calibration's error band is recorded in docs/llm-tokens-and-speed.md.

### Phase 17 risks

| Risk | Mitigation |
| ---- | ---------- |
| a recipe lags or leads the pinned vLLM | the `min_vllm_version` gate; probes remain the authority |
| an estimate is mistaken for a measurement | the unmissable `est` label; measured values always win |
| the recipe repository is unreachable | offline-first: cached copy or no seed, never a failure |

---

## Phase 18 -- Launch-failure diagnosis and one-shot auto-fix

From odysseus-borrowed-ideas, CB-2. Its odysseus regex table was read by an agent, not by hand: re-verify it before porting.

### Goal

A failed model launch comes back with a diagnosis and a concrete fix, and a probe that fails on a recognised signature retries once with that fix applied and recorded. Built on the new launch path, so it waits for Phases 8 and 10: today's hook points (`detectLaunchFailure`, the probers' container code) are removed by Phases 9 and 14. It writes its fixes into the generated recovery file, which Phase 17a's readers and `devai-control` (since Phase 8) re-read at every launch (I25).

### Deliverables

```
devai-tools/internal/diagnose/    new    -- the error-signature -> fix table (KV or CUDA OOM, context too large, wrong
                                            or missing tool-call parser, trust-remote-code, lm_head quantisation,
                                            port in use, missing dependency, gated repository), each fix marked
                                            `applicable` (a flag safe to append) or `report` (I18); table-driven tests
devai-tools/internal/control/     modify -- a load_failed error carries `diagnosis {message, suggestions:[{kind, op,
                                            flag, value}]}`, an additive field on C1's error object: `kind` is
                                            `applicable` or `report` (I18), `op` is `append`, `replace` or
                                            `remove`, and an applicable suggestion is always an `append`
gpu-arbiter/slurm_launch.go       modify -- the 502 for a failed load carries the diagnosis, and X-DevAI-Warning names it
scripts/_probe_hf_common.py       modify -- the bus driver retries a load once with the first suggestion of `kind` `applicable` (I18): it adds
                                            a group to that model's list in deploy/recovery-flags.generated.json
                                            (I25) -- `backends` = the failing backend, `"source": "auto-fix"`, the
                                            diagnosis as `reason` -- logged and attributed; `report` suggestions go
                                            into the cell's evidence
tests/python/test_probe_autofix.py
                                  new    -- the retry without a GPU: the prober's bus driver against a stub
                                            devai-control that answers load_failed with a synthetic CUDA-graph OOM
                                            log tail (tests/fixtures/vllm/cuda-graph-oom.synthetic.log: no real case
                                            exists), then serving once the auto-fix group is in the generated file
```

### Detailed steps

1. Port the signature table from odysseus's `routes/cookbook_helpers.py:1208-1381` after re-reading it, and pin it with failure logs: `tests/fixtures/sglang/argparse-rejection.log`; new `tests/fixtures/vllm/` files cut from the persisted engine logs (`/var/cache/devai/logs/devai-vllm*.log`) and the strings `tests/python/test_probe_classify.py` already holds; and, for a signature with no captured case (the CUDA-graph OOM), a synthetic excerpt whose first line says so. Add the memory signatures Phase 8 recognises (vLLM's `less than desired GPU memory utilization` and `larger than the available KV cache memory`, SGLang's `Loaded weights leave no GPU memory for the KV cache`): today's router list has only `No available memory for the cache blocks`, so the router reports such failures as the generic `Engine core initialization failed`.
2. Mark each fix `applicable` or `report` (I18).
3. Wire the diagnosis into `devai-control`'s `load_failed`, the router's 502 and the prober's evidence.
4. Add the one-shot retry to the probers' bus driver (`devai-control` re-reads the recovery files at every load since Phase 8).

### Exit criteria

- A probe that fails on a CUDA-graph OOM is diagnosed, gets `--enforce-eager` in a new group of the generated recovery file (`backends` = its backend, `source` = `auto-fix`), and passes on the second attempt.
  - No model on disk is known to fail this way: every curated `--enforce-eager` entry belongs to a model whose weights are not on disk, and I17 downloads nothing.
  - So `test_probe_autofix.py` shows the retry, and Phase 8's exit criteria show `devai-control`'s re-read. A live run waits for a real case.
- A probe launched with a deliberately too-large context fails, is diagnosed as "context too large / KV OOM", and its evidence carries the suggestion; nothing is applied.
- A failure that matches no signature is reported verbatim: no silent wrong fix.
- The retry is capped at one attempt per probe.

### Phase 18 risks

| Risk | Mitigation |
| ---- | ---------- |
| a wrong auto-fix makes a probe "pass" | one retry only; every applied flag logged and attributed in the generated recovery file; unmatched failures reported verbatim |
| an over-broad signature misdiagnoses | the table is re-verified before porting and pinned by fixtures |

---

## Phase 19 -- Lanes on one port

From router-fanout, Phase 1 and the single-host items of its Phase 4. Its cluster demux and broadcast were dropped with cluster mode.

### Goal

An optional extra router port on which one client reaches several models: a request's `model` field (an agent's main and background slots, mapped through aliases) or an inline `@@<lane>` directive at the start of the last user message picks a lane, and the router serves that lane's model. Routing is explicit only; the router never interprets message content.

It waits for Phase 9: the lanes dispatch through the Slurm launcher. On one GPU every lane change is an engine switch, so the cost is a cold start whenever lanes alternate; the documentation says so plainly.

### Deliverables

```
gpu-arbiter/fanout.go             new    -- LaneConfig load; pure ResolveTargets(path, body): model-field and alias match,
                                            @@<lane>, @@<model>[@ctx][::reasoning][::mtp], the default lane; the
                                            directive stripped from the forwarded body
gpu-arbiter/fanout_test.go        new    -- every resolution branch, over OpenAI and Anthropic bodies
gpu-arbiter/main.go               modify -- with DEVAI_FANOUT_PORT set, a lanes listener that resolves, rewrites `model`
                                            and hands the request to the normal handler (the Slurm launcher's Ensure)
deploy/fanout-lanes.yaml          new    -- prefix, default_lane, lanes, aliases; the feature is inert without the file
deploy/docker-compose.yaml        modify -- DEVAI_FANOUT_PORT (11439 suggested; unset = off), FANOUT_LANES; the registry
                                            mounted read-only
docs/router.md                    modify -- the lanes surface, and that it parses explicit directives only
docs/fanout.md                    new    -- lanes, directives, agent recipes (Claude Code's ANTHROPIC_MODEL and
                                            ANTHROPIC_DEFAULT_HAIKU_MODEL on two lanes), the cold-start cost
```

### Detailed steps

1. The registry, `deploy/fanout-lanes.yaml` (example values):

   ```yaml
   prefix: "@@"            # the directive marker (configurable)
   default_lane: chat
   lanes:                  # each lane: backend, model, optional ctx, reasoning, mtp
     chat:   { backend: ollama, model: "qwen3.5:35b-a3b-ud-q3_k_xl" }
     code:   { backend: vllm,   model: "gpt-oss-20b", ctx: 65536 }
     reason: { backend: vllm,   model: "Qwen3.5-9B-NVFP4", reasoning: high }
   aliases:                # model-field value -> lane (an agent's slots)
     "claude-haiku-4-5-20251001": chat
   ```

   `@@<model>` names a model directly: when several backends serve that name, the first in the order vllm, sglang, ollama serves it, and `@@<backend>/<model>` picks one explicitly.
2. Directives are matched only at the first non-whitespace position of the last user message. `@@@@` is a literal `@@`. An unknown `@@token` answers 400 listing the valid lanes, and the directive never reaches a model.
3. `@@all` (broadcast) is refused with 400: on one GPU a broadcast is back-to-back cold starts.
4. Each lane's model is checked against the probe caches at boot; a lane without a fitting cell is skipped and logged.
5. Polish, each item separately shippable: stickiness (the latest `@@<lane>` in the resent history applies when the current message has none); Ollama-native `/api/chat` and text parts of multimodal content. A heuristic auto-lane stays parked: routing stays explicit.

### Exit criteria

- `ResolveTargets` returns the right target for a model-field match, an alias, each directive form and the default, and 400 for an unknown token -- over OpenAI and Anthropic bodies.
- The directive is stripped from the forwarded body.
- With `DEVAI_FANOUT_PORT` unset, the router is byte-identical to before.
- A real two-slot agent configuration sends its foreground and background requests to two different lanes.

### Phase 19 risks

| Risk | Mitigation |
| ---- | ---------- |
| alternating lanes on one GPU thrash cold starts | documented; the default lane and aliases make the common case one model |
| a legitimate message starting with `@@` is read as a directive | first-position match only; unknown token is a 400, not a silent route; `@@@@` escape |
| a lanes port departs from devai's one-port-per-backend rule | opt-in and off by default; the existing ports are untouched |

---

## Phase 20 -- Library service

From library-service. aiagent decides what to search, what to fetch and what to keep; the service makes every external request, politely, stores what aiagent accepts as plain files, and keeps search indexes that can be rebuilt from those files.

The field-level API contract is aiagent's `docs/design/library-service-needs.md` (devitops-com/aiagent), adopted with three explicit adjustments (see API below). The owner's decisions are LS1-LS30 above. The design was reviewed adversarially before the owner saw it (48 confirmed findings, all addressed). Nothing is implemented.

It needs Phase 2 (the pinned Debian image, pipelock and apt-cache built here), Phase 3 (`devai-base` on Debian) and Phase 6 (`devai-operator`: its `LIBRARY_DIR` mount, `make operator-up`, the stack page). Phase 12's `lab-start.sh` gets the `/library` mount once both have landed. It needs no GPU and no Slurm: none of its containers is a router backend. Its Makefile targets join the operator's registry (C8), with `scripts/ops/` scripts, once Phase 7 has landed; before that they go into C8's `pending` list. Sub-phases 20.1-20.5 run in order; 20.3 needs aiagent's side.

### Needs

- **aiagent's requirements document** (`docs/design/library-service-needs.md` in devitops-com/aiagent), the field-level contract, and aiagent's client implementation.
- **Owner actions, each blocking the phase named:**
  - the dedicated contact address (LS21) on the company's own domain: blocks 20.1's live exit (no external request without it), and every key application below;
  - an OpenAlex API key: optional, raises the daily budget from $0.10 to $1 (20.2);
  - a GitHub fine-grained token (20.4); a Hugging Face token (20.4); a Semantic Scholar key, requested only when 20.5 starts because unused keys are pruned after about 60 days; an OpenReview service account (20.5);
  - contacting arXiv before the library passes about 1,000 full-text downloads ("Politeness engine").
- **pipelock** (`docs/pipelock.md`): the service's outbound traffic goes through it for DLP.
- **The lab launchers** (`bin/devai-agent`, the Makefile), which gain an optional read-only `/library` mount.

Open for the owner: none (LS24-LS28 answered the original plan's questions). 20.1 still checks the arXiv PDF mirror's terms (LS27) and confirms the language detector's licence.

### Verified source facts (2026-09-25/26)

Researched from the providers' own documentation, terms and robots.txt files, then re-checked claim by claim by a second reader: 169 claims, of which 9 were refuted and corrected, 15 more confirmed only with corrections, and 1 left unverifiable. Several limits are undocumented or change often, so every number below is configuration, re-checked before the phase that uses it ships, and the service reads rate-limit headers at runtime rather than trusting these numbers.

| Source | Access | Limits and terms the service applies |
| --- | --- | --- |
| arXiv legacy APIs (`export.arxiv.org/api/query`, `oaipmh.arxiv.org/oai`) | open | Terms: one request every 3 s on one connection, for all legacy APIs together and across all of the operator's machines. GET only (POST draws 429s sooner); cache each query 24 h; at most 2000 results per call. 429 "Rate exceeded" happens even at compliant rates (server capacity); 503 means excessive use. Show "Thank you to arXiv for use of its open access interoperability." |
| arXiv files (`arxiv.org/pdf`, `/html`, `/abs`) | open | robots.txt: `Crawl-delay: 15`; `/pdf`, `/html`, `/abs` allowed; `/e-print`, `/src` disallowed. A descriptive User-Agent with contact. arXiv treats continued requests after a 403 as an attack. Contact arXiv before about 1,000 full-text downloads. Terms allow storing for personal or research use, not serving to others. |
| arXiv HTML | open | LaTeXML output, backfilled far back. Checked 2026-09-26: on `2609.27602` (numerical analysis) every formula carries its TeX (`annotation encoding="application/x-tex"`, 78 of 78), equations carry their printed numbers ("(2.1)") and LaTeXML ids (`S2.E1`), sections and paragraphs carry ids (`S1`, `S1.p1`); the authors' `\label` keys are not in the page. Even `hep-th/9901001` (1999) has HTML ("Generated by LaTeXML oxide 0.7.6"). What arXiv answers for a paper with no HTML was not observed. |
| arXiv metadata and licences | open | The API's Atom feed gives the latest version id and the dates of v1 and of the latest version only. The full version list, the withdrawal marker (best effort) and the licence come from OAI-PMH `arXivRaw`. Licences are per version and chosen by the submitter (CC BY, CC BY-SA, CC BY-NC-SA, CC BY-NC-ND, CC0, arXiv's non-exclusive licence); older records carry legacy URIs, and papers before April 2007 may have none. |
| OpenAlex | keyless or free key | Since 2026-02-13 metered by a daily budget: keyless $0.10/day, a free key $1/day. Singletons are free; structural filter lists (ids, `cites:`, dates, `is_oa`) cost $0.0001 per call; text matching costs $0.001 whether sent as `search=` or as a `*.search` filter; content downloads cost $0.01. Semantic search: 1 request/s, 50 results. The polite-pool `mailto` is ignored. Author metrics come from clustered profiles: advisory only. Data CC0. Its Terms of Service could not be read (403 to automated fetches): unverified. |
| Unpaywall | email parameter | DOI lookup only (`/v2/{doi}`); `/v2/search` returns 410 since 2026-09-18. Best open-access location, version and licence. |
| Semantic Scholar | key required in practice | The keyless pool is shared by all anonymous users and may be throttled at any time. A key allows 1 request/s across all endpoints. From its 2024 release notes (since discontinued): free-email domains are refused, approval took about a month, keys idle about 60 days are pruned. The licence grants use "to access and display" S2 data, with per-field licences such as CC BY-NC or ODC-BY, and requires attribution. |
| OpenReview | account required | API v2; anonymous calls get 403. Reviews and comments are CC BY 4.0. Rate limit undocumented. |
| Hugging Face papers API | open, token advised | `GET /api/papers/{arxiv_id}` gives `githubRepo`, which is partial and may be set by users or automatically (`githubRepoAddedBy`). Anonymous quota 500 requests per 5-minute window per IP, "subject to change"; RateLimit headers. |
| Papers with Code | frozen | Frozen since 2025-07-28 (archive, CC BY-SA). arXiv's own code widget (CatalyzeX) sits behind a Cloudflare challenge: not used. |
| GitHub | token | REST with a fine-grained token: 5,000/h, code search 10/min; a documented retry ladder. Pin to a commit sha; permanent links `blob/<sha>/<path>#Lx-Ly`. Licence detection reads the LICENSE file only; "none" or "other" is not permission. |
| Wikipedia | open, User-Agent with contact | At most 200 requests/min per client (target about 2/s), one in-flight Action API request. API paths allowed under LS5. |
| SSRN | link only (LS8) | Terms forbid automated queries "of any sort"; Cloudflare challenges every PDF; TDM rights reserved. The same work is looked up by DOI (prefix 10.2139) at other open locations. |
| Substack, Stack Exchange, Reddit, Drive-hosted Colab | link only (LS8) | Substack's terms forbid crawling and storing any significant portion (a Substack publication may sit on a custom domain: its feed's generator, hosting headers and CNAME give it away); Stack Exchange's policy bans gathering for AI tools; Reddit disallows everything in robots.txt and needs approval and local deletion sync; Drive-hosted Colab stays link only until the Drive API with the user's consent is researched. GitHub-hosted Colab notebooks go through the GitHub adapter. |
| Kaggle | official API only (LS8) | Its terms forbid scraping and storing significant portions. Only through the official API, only notebooks the user names, and stored only under a stated open licence (as LS10 requires for code); otherwise a link. |
| Medium | not decided | Not on the owner's restricted list. Its official API is unsupported (archived 2023) and never offered post content; public feeds are allowed by robots.txt; its terms could not be read (Cloudflare). Before 20.5: a manual read of the terms, then feed-only, on demand, for owner-approved authors. |

### Design

#### Components

- **`devai-library`**: the service. Python 3.14 (`devai-base`'s, Phase 3), a stdlib HTTP server and worker threads in the laya trainer's style, hash-locked dependencies with permissive licences only: `protego` (BSD), `lxml` (BSD), `onnxruntime` (MIT), `tokenizers` (Apache), SQLite (public domain), and a language detector whose model ships inside the package (for example `lingua-language-detector`, Apache-2.0; the choice and licence are confirmed in 20.1), since the download rule forbids fetching models at run time. No AGPL or GPL code: no PyMuPDF, poppler or pandoc. It never parses a fetched PDF or HTML page itself.
- **`devai-library-extract`**: the extractor. It turns PDF and HTML bytes into units: `pypdfium2` (Apache/BSD) and `pypdf` (BSD) for PDFs, `lxml` for HTML. It has no secrets, no mount of the store and no network except an internal one; bytes come in over HTTP and JSON goes out, and the service validates the JSON before writing anything.
- **`devai-grobid`**: GROBID (Apache-2.0), built here (LS29), for PDF structure (sections, paragraphs, formulas, references with DOI and arXiv ids, page coordinates). About 4 GB of RAM. Called with `consolidateHeader=0` and `consolidateCitations=0`, so it never contacts CrossRef or anything else.
- **The embedding model**: `bge-small-en-v1.5` (MIT, 384 dimensions, 512-token input), run with ONNX Runtime on the CPU inside the service. Downloaded only through `scripts/select-models.py`, from a pinned catalog row, into its own store (LS24).

None of them is a router backend: none needs the GPU, so the router's GPU exclusivity does not apply.

#### Network and egress

- **Networks.** `devai-library` sits on `devai-net` and `devai-lab-egress`, as the router does, with no host port; the lab reaches it as `http://devai-library:8080`, and the name joins the lab's `NO_PROXY` lists (Makefile `PIPELOCK_NO_PROXY`, `bin/devai-agent` `NO_PROXY_HOSTS`). A third network, `devai-library-internal`, created with `--internal` by `make cache-up` and declared external in compose like the others, is the only network of `devai-library-extract` and `devai-grobid`; only `devai-library` joins it too.
- **No URL ever comes from aiagent** (agreed with aiagent, its L11). Requests name a vetted source and a query, an identifier the service knows how to resolve (`arxiv:`, `doi:`), or an id the service issued (item, version, link). A link found in fetched content or in third-party metadata comes back as a link record, and the service alone decides whether it is fetchable.
- **Every hop is checked.** Automatic redirects are off; each redirect is checked like a first request (host list, robots.txt, the host's limits). IP-literal URLs, internal service names, and loopback, private, link-local and unique-local addresses are refused. A host that is not in the host list is not fetched: admission is default-deny.
- **Egress through pipelock**, so its DLP scans what leaves, including query text that aiagent wrote. The service trusts pipelock's CA. pipelock's header DLP will match real credentials (GitHub tokens, API keys): each is exempted with a per-pattern `exempt_domains` limited to its issuer's API host, in the phase that first sends it; keys travel in headers, never in URLs.
- **pipelock's own answers are not upstream answers.** A pipelock block (DLP) comes back as a 403, and its connect failures as 5xx, on what looks like the upstream connection. The service recognises pipelock's responses (its block body or marker, confirmed in 20.1) and never counts them as the source's answer, so a DLP hit on one query cannot open a source's breaker: a DLP block is `egress_blocked`, final for that request and never resent unchanged; a pipelock connect failure or 5xx is `egress_unavailable`, a wait retried within the item's retry budget.
- **pipelock does not restrict destinations.** Verified 2026-09-26 from a lab container: `example.com`, `arxiv.org` and `api.openalex.org` all answer 200 through pipelock, although only dev-tool domains are in its `api_allowlist`. So the service enforces its own host list; and other lab tools can still reach arXiv directly, spending the budget arXiv counts per operator, which the owner accepts (LS25).

#### Politeness engine

Scheduling is keyed on **rate groups**, defined in a versioned file, `deploy/library-sources.yaml`. A group spans the hosts that share one budget: `arxiv-legacy-api` (`export.arxiv.org` and `oaipmh.arxiv.org`: one connection, at least 3 s apart), `arxiv-files` (`arxiv.org`: robots' 15-second crawl delay, one connection), one group per other host. All jobs share a group's queue, first come first served in v1. Each group has:

- kind: `api` (governed by the provider's API terms) or `pages` (governed by robots.txt), LS5;
- minimum interval, maximum connections, daily budget (OpenAlex's in dollars, each call priced from its parameters and reconciled against the response's cost and budget headers);
- authentication: none, or a named key file (LS30);
- two kinds of breaker, keyed on the provider. A **denial** (an origin 403 from any host) opens a breaker that stays open until an operator clears it; for arXiv it stops all arXiv traffic and raises an alert, since arXiv treats continued requests after a 403 as an attack. **Overload** (sustained origin 429s or 5xx despite waiting, a configured count within a window, for example 5 within 10 minutes) pauses the provider for at least 15 minutes and then clears by itself. A single 429 or 503 is never a breaker event: arXiv sends 429 even at compliant rates.

Waiting on a single 429 or 503: `Retry-After` when given (wait reason `retry_after`), otherwise exponential backoff with jitter (`backoff`), both within the item's retry budget (5 attempts or 30 minutes, aiagent's section 3.1). Only failed attempts count against the budget, and its 30 minutes exclude time spent waiting for a rate slot, an overload pause or a `quota` reset, so a wait never fails an item. Rate-limit headers (OpenAlex's `X-RateLimit-*`, GitHub's, Hugging Face's) update the budgets at runtime. Every request carries `User-Agent: devai-library/<version> (+mailto:<contact>)`.

robots.txt is parsed with Protego and fetched by the service's own wrapper, following RFC 9309 with a stricter 4xx rule: an explicit User-Agent, a timeout, at most 5 redirects (each checked as above), a 500 KiB cap, a 24-hour cache. Outcomes and their reason codes: the rules forbid the path (`robots_disallowed`); robots.txt itself answers 401, 403 or a challenge (for example `cf-mitigated: challenge`), treated as disallow (`robots_denied`); robots.txt answers 429, 5xx or fails on the network, treated as disallow for now, with the cached copy kept and a retry later (`robots_unreachable`); 404, 410 and other 4xx mean "allow".

**Stops, and how they show.** Three conditions stop a source until someone acts: an open denial breaker (`breaker_open`), the missing contact address (`contact_not_configured`, all sources), and the arXiv full-text cap (`arxiv_fulltext_cap`: a configured count below 1,000 papers, each paper counted once whatever its formats; it stops arXiv PDF and HTML downloads only, not search or OAI-PMH metadata, until the owner records that arXiv was contacted). An overload pause and an exhausted daily budget are not stops: they end by themselves, so jobs wait for them (`backoff`, or `quota` until the budget's reset time).

- `/health` and `/v1/info` show a source `down` with its cause when all of its traffic is stopped (a missing contact, an open breaker), and `limited` with its cause when part of it is (the arXiv cap stops downloads, not search); the top-level status is `degraded` whenever any stop holds.
- A job-creating request whose every named source is stopped answers 503 `unavailable` with `error.details.cause`.
- A request naming a stopped source and a working one is accepted (202): the stopped source reports a per-source `error` in the search results, and fetch items that need a stopped source end promptly as `failed` with `upstream_unavailable`, still fetchable. The cause is always in `error.details.cause`, on 503 answers, per-source errors and per-item failures alike.
- A running job whose source stops meanwhile gives `eta: {basis: unknown, reason: source_down}`, and its remaining items for that source end the same way.
- Clearing a denial breaker, or recording the arXiv contact, is an explicit operator command, recorded under `state/`: `devai-library-admin breaker-clear <provider>` and `devai-library-admin arxiv-contact-recorded`, run inside `devai-library` and offered as C8 actions of kind `job` in the `library` group (job_kind action; each runs `podman exec devai-library devai-library-admin ...`). Alerts -- an opened denial breaker, the first 429 from a host -- are appended to `state/alerts.jsonl` and listed under `alerts` in `/health`; the operator's stack page shows them. Counters that must survive a restart (daily budgets, the arXiv total, breaker state) live under `state/`.

#### Configuration

- **`LIBRARY_DIR` in `.env`**, set once (LS15). Compose and the Makefile read it from `.env`. `bin/devai-agent` does not read `.env` by design, so `make install` stages it as `~/.devai/library`, a symlink to the directory, exactly as it stages the probe caches; the launcher mounts it and prints a warning naming `make install` when it is missing. A test checks that compose, the Makefile and the launcher resolve the same path.
- **Optional everywhere except in the service.** The lab mounts `/library` (read-only) only when the library is configured, as it mounts `/laya` only when that exists. `devai-library`, `devai-library-extract` and `devai-grobid` live in `deploy/compose.library.yaml`, which `make cache-up` and the stack-up action add only when `LIBRARY_DIR` is set, as `deploy/compose.gpu.yaml` is added only on a host with a GPU (Phase 11). devai-operator's own `LIBRARY_DIR` mount lives in `deploy/compose.operator-library.yaml`, which only `make operator-up` adds: the stack-up action runs inside devai-operator and must never recreate it (Phase 6). That file also sets `DEVAI_OPERATOR_LIBRARY_DIR` in devai-operator; after setting or changing `LIBRARY_DIR`, run `make operator-up` once. A profile with an inert default would not do: devai-operator runs all the time and is in no profile, and podman refuses a bind mount whose source does not exist. Without these files, compose never sees the variable. A host that starts the stack without `.env` (the systemd unit, where installed) simply runs without the library.
- **Three checks, no more (LS15).** `scripts/ops/stack-up.sh` (behind both `make cache-up` and the stack-up action) and the service check that `LIBRARY_DIR` is an absolute path, that the directory exists, and that it is readable and writable; a failure refuses the library's start and names the requirement that failed. Inside devai-operator, stack-up.sh first compares `.env`'s `LIBRARY_DIR` with `DEVAI_OPERATOR_LIBRARY_DIR`. When they differ, it refuses the library with "LIBRARY_DIR is not mounted in devai-operator: run make operator-up", and starts the rest of the stack. Nothing creates the directory, and nothing checks where it lives, whether it is its own volume, its filesystem or its free space. The service's containers run with the owner's host uid, so the owner's read and write permission is all they need. On its first start in a directory the service writes `LAYOUT.json` (the layout version) and its own top-level directories beside whatever is already there, and it never touches a file it did not write.
- **`LIBRARY_CONTACT`** in `.env`: the contact address (LS21). Unset, the service answers but makes no external request.

#### The store (`LIBRARY_DIR`)

The service owns the layout; aiagent uses only paths the API returns (its principle 8).

```
<LIBRARY_DIR>/
  LAYOUT.json                    layout version, written by the service on its first start
  objects/aa/bb/<sha256>         artifact bytes (PDF, HTML), content-addressed, write-once, 0444
  items/aa/bb/itm-<24 hex>/      one directory per item; aa/bb = the first 4 hex of the id
    key.json                     canonical key and first-seen time, write-once
    records/<ns>-<type>-<id>.json  append-only: metadata snapshots, fetch results, verdicts, topic
                                 assessments, aliases, redirects, tombstones, index status
    texts/tx-<24 hex>/           an accepted item's text versions: text.json, units.jsonl, SHA256SUMS
  staging/aa/bb/itm-.../         the same shape, before a verdict; purged on reject or after 30 days
  jobs/lbjob-<24 hex>/           job.json, events.jsonl, results.jsonl
  profiles/<name>/<version>.json research profile versions, opaque
  state/                         politeness counters, breakers, the arXiv contact record
  trash/<date>/                  artifacts of revoked items, kept 30 days (LS28)
  index/                         derived only: alias index, keyword index, embeddings, topic listings
```

- **The plain files are the only source of truth.** An item's current state is the fold of its `records/`; everything under `index/` is derived and can be deleted and rebuilt (aiagent's acceptance check 6).
- **Fan-out.** Two levels of two hex digits, as in git's object store but one level deeper: 65,536 leaf directories, so ten million items or objects leave about 150 entries per directory. The fan-out keeps every directory small whatever the filesystem, so `ls`, `find` and shell globs stay fast.
- **Writes are atomic** (write to a temporary name, then rename); files are 0444 and directories 0555 once final, as on the laya store. Standard tools work directly: `find items -name key.json`, `jq` over `records/`, `grep` over `units.jsonl`.
- **English only (LS16) in what stays.** Permanent records hold English only: for a non-English item, aiagent submits English metadata (`metadata_en`) with its verdict, and English renderings of evidence quotes (`quote_en`); the service stores those and keeps the original only as a link with its sha256. Staging, which is temporary, holds the original-language text until the verdict or the TTL.
- **Rejects and purges.** A reject deletes the item's staged artifacts and texts and writes tombstone records; unit addresses of a purged text answer 410 `text_purged` with the tombstone. A purge after the staging TTL is done by a maintenance job, which emits `item.purged` events like any other job.
- **Revoked items** (an accept turned into a reject) move their artifacts to `trash/<date>/`, where a maintenance job deletes them after 30 days (LS28); their record says where they are until then.

#### Text extraction

- **Versions are fixed first.** The fetch job resolves the latest version `vN` once, then fetches `/pdf/<id>vN` and `/html/<id>vN`, and the `arXivRaw` record (full version list, withdrawal marker, licence) as one paced request on the legacy-API group. Accept only copies the licence from that stored record; it makes no external request.
- **Format checks before hashing.** A PDF must start with `%PDF-`. A valid HTML page is 200, `text/html`, carries the LaTeXML generator marker, and names version `vN`. A definite "no HTML" is a 404 or 410, or a 200 without the LaTeXML marker: PDF-only staging at once, not a failure, with `quality.warnings` saying why. A 429, 5xx, network failure, or a 200 with the marker but the wrong content type or version is an error, retried within the item's retry budget like the PDF, and PDF-only staging with a warning if the budget runs out. HTML for a PDF-only item is fetched later only on request. Nothing that fails a check is stored or becomes a `sha256:` alias.
- **arXiv HTML is the primary text when it exists** (`origin: html`, `role: primary`): sections and paragraphs from LaTeXML's ids (`anchor.html_id` carries the unit's own id, for example `S3.SS2.p2`; `anchor.paragraph` comes from it; `anchor.block` is null), display equations as TeX with their printed numbers (`equation_number_method: html`; `equation_labels` empty, since arXiv's HTML drops the authors' labels; `refs.labels` point to LaTeXML ids), inline math as `$TeX$` with `math_spans` (offsets including the `$` delimiters), theorem-like environments (`anchor.env`: `{name, label, number}`), captions, and bibliography entries with parsed identifiers (`refs.cites` entries: `{html_id, printed, ids: {doi, arxiv, url}}`); a `heading` unit carries its section's `html_id`. `html_id`, `env`, `refs.cites` and `refs.labels` are v1 for HTML texts and null for PDF texts; section and paragraph anchors come from GROBID for PDF texts. The PDF page of each unit comes from aligning its text with the PDF's text layer (`page_method: text_align`), and the PDF text becomes a `fallback` text (`fallback_text_id`).
- **Licences of non-arXiv works** are recorded at fetch time from the open-access location the copy came from (OpenAlex's `locations[].license`, Unpaywall's `best_oa_location.license`), with `stated_by` naming that source (`stated_by` is one of `arxiv`, `openalex`, `unpaywall`); when none is given, `license.status` is `none_stated`. Accept never makes an external request.
- **Otherwise the PDF through GROBID** (`origin: pdf`): sections, paragraphs, formulas, references and page coordinates (`page_method: pdf_native`), with pypdfium2 as the fallback. A PDF without a text layer fails the fetch with `no_text_layer` (A-O7: OCR later).
- **Units** follow aiagent's rules (its section 3): structural kinds, at most 1,300 characters, split only when oversized, at sentence boundaries outside math and citations, no overlap, `prev`/`next` links, link records for every link found, and `lang` per unit from the language detector. The service counts characters only.

#### Index

- **Keyword:** SQLite FTS5 with the porter/unicode61 tokenizer, a derived file under `index/`, updated incrementally.
- **Vector:** embeddings stored as derived files keyed by (unit text sha256, model sha256), so a rebuild does not re-embed. Search starts as exact search over memory-mapped matrices (float32, later int8); faiss-cpu (MIT) is the next step when measurements call for it. Units longer than the model's 512 tokens are chunked internally; a hit always reports the unit.
- **Hybrid search** fuses the two lists mechanically (reciprocal-rank fusion) and returns the fused score (v1), later the component scores too.
- **Rebuild times** are measured, not promised, and reported by `/v1/info`.

#### Jobs and progress

As aiagent's section 8 specifies, in the laya trainer's style: every job-starting request returns 202 at once; `jobs/<id>/job.json` is rewritten atomically at every event, at most once per second, and `events.jsonl` is append-only. Waits carry their reason (`rate_limit`, `crawl_delay`, `retry_after`, `backoff`, `quota`, `concurrency`) and an `until` time. The first ETA comes within 2 s, computed from the schedule (requests left per rate group times the group's interval, plus measured transfer and extraction times), and a heartbeat at least every 5 s. `eta_inputs` and `/v1/info` report per rate group (`per_group`), since one source can span groups with different intervals (arXiv: 3 s and 15 s). After a restart a job resumes: every item step is idempotent, and unfinished items are queued again. For scale: 25 arXiv papers as PDF and HTML are about 50 files at arXiv's 15-second crawl delay, so about 12-13 minutes of waiting (the PDF mirror, if LS27's check allows it, would halve it).

#### API

aiagent's section 9.6 lists the endpoints and marks what v1 needs; this plan adopts it with three adjustments, which aiagent's document reflects:

1. arXiv artifacts are `pdf` and `html`, never `eprint` (LS4).
2. Sources restricted by LS8 come back as candidates or link records with a not-fetchable reason (`terms_disallow_automation`) and their link.
3. **HTML-primary text in v1**, as described under "Text extraction", instead of PDF-only text: `origin: html`, `page_method: text_align`, paragraph anchors from LaTeXML, `math_spans`, printed equation numbers. This serves the reason for LS4 (exact formulas for PDE and numerical analysis work).

Also agreed with aiagent: fetch keys `arxiv:`, `doi:` and `ssrn:` (an `ssrn:` key is accepted and ends `not_fetchable` with `terms_disallow_automation`, its open-access copy being fetched by `link_id`); the id namespace `openalex:W<id>` for works that have neither an arXiv id nor a DOI, minted after `ssrn` (`arxiv > doi > ssrn > openalex > gh > wiki > url`), and an OpenAlex id is always registered as an alias; a missing contact address shows in `/health` (`status: degraded`, `contact.configured: false`) and in `/v1/info` (each source `down`, cause `contact_not_configured`), and a job-creating request for an external source answers 503 `unavailable` with that cause in `error.details.cause`, so aiagent's acceptance step 0 fails fast.

| v1 | Later |
| --- | --- |
| `GET /health`, `GET /v1/info` | `POST /v1/items/{id}/translations` |
| profiles: `PUT`/`GET /v1/profiles/{name}/versions/...` | `POST /v1/items/match` |
| `POST /v1/searches`, `POST /v1/fetches` | `GET /v1/verdicts?basis=...&profile_version_lt=...` |
| `POST /v1/enrichments` (author metrics and citations from OpenAlex), `GET /v1/candidates/{id}` | `GET /v1/jobs/{id}/events/stream` (SSE) |
| jobs: `GET /v1/jobs[/{id}]`, `.../events`, `.../results`, `POST .../cancel` | repository fetches and excerpts |
| `GET /v1/items/{id}` (with tombstones), `GET /v1/items/resolve?key=` | |
| `GET /v1/items/{id}/texts/{text_id}/units` (410 with the tombstone for a purged text) | |
| `POST /v1/verdicts`, `POST /v1/verdicts/batch`, `GET /v1/verdicts/{id}` | |
| `POST /v1/items/{id}/topics` | |
| `GET /v1/library/topics[/{topic_id}/items]`, `POST /v1/library/search` | |

#### devai's answers to aiagent's questions (O-D1 to O-D13)

| # | Answer |
| --- | --- |
| O-D1 | A plain compose container, not a router backend: `http://devai-library:8080` on devai-net and devai-lab-egress. |
| O-D2 | The library root is mounted read-only at `/library` in the lab when configured (later `inbox/` read-write), from `LIBRARY_DIR` (staged for `bin/devai-agent` by `make install`). Secrets never live in the library directory. `state/` and `index/` live under the root; aiagent must not depend on their files. |
| O-D3, O-D11 | Replaced by arXiv HTML (LS4): paragraph and equation anchors and printed numbers come from LaTeXML's output; the authors' `\label` keys are not available. PDFs without HTML go through GROBID. |
| O-D4 | Dropped: the service counts no tokens. |
| O-D5 | `bge-small-en-v1.5`; embeddings cached by (unit text sha256, model sha256); exact search first; rebuild times measured. |
| O-D6 | SQLite FTS5, porter/unicode61, derived. |
| O-D7 | The arXiv legacy APIs (search and OAI-PMH) share one queue: one request per 3 s on one connection. arXiv files: robots' 15-second crawl delay, one queue. No e-print (LS4). |
| O-D8 | OpenAlex (discovery, open-access locations, citations, advisory author metrics), Unpaywall by DOI; SSRN link only (LS8); Semantic Scholar later, once a key is granted. |
| O-D9 | Jobs resume after a restart; per-group queues are first come, first served across jobs in v1. |
| O-D10 | Yes: revoked items' artifacts stay in `trash/` for 30 days (LS28, an owner-approved exception to LS12). |
| O-D12 | `devai-library/<version> (+mailto:<contact>)`; robots.txt as described above; candidates report `access.robots_allowed` with the reason. |
| O-D13 | Later: author pages from OpenAlex open-access locations, references parsed by GROBID and LaTeXML, and ORCID; blogs from an owner-approved list (LS22). No web-search API. |

#### GROBID, built here (LS29)

`deploy/Dockerfile.grobid` has two stages on the pinned trixie-slim. The build stage installs Debian's `openjdk-21-jdk-headless`, unpacks GROBID's source at a pinned release (sha256-pinned tarball) and builds the service distribution with GROBID's Gradle wrapper, whose `distributionSha256Sum` is pinned; every network step is tried 3 times, as in `scripts/build-vllm.sh`. The runtime stage holds Debian's `openjdk-21-jre-headless` and the distribution with its CRF models (they ship in the source's `grobid-home`), no DeLFT deep-learning models. The upstream-images guard (Phase 2) sees `localhost/devai-grobid` and needs no exception. Untested: whether the pinned release builds with JDK 21 (checked first in 20.1), and Gradle's dependency downloads from Maven Central, which the wrapper pin does not cover; the first build records them with their checksums (Gradle dependency verification).

#### Keys (LS30)

Each key the library uses is a plain file the owner writes once into `~/.config/devai/library/` (mode 0700, created by `make library-init`): `openalex.key`, `github.token`, `hf.token`, `s2.key`, `openreview.credentials`, each 0600. Each is mounted read-only into `devai-library` only. The service refuses a key file that group or others can read, never logs a key, and sends keys in headers, never in URLs. `devai-backup` includes the directory.

### 20.1 -- store, service core, politeness, arXiv

**Goal:** a running service that searches arXiv and fetches PDFs and HTML politely, stages English text units, and records verdicts, with jobs, progress and ETA.

**Deliverables:**
- Configuration: `LIBRARY_DIR` and `LIBRARY_CONTACT` in `.env.example`; `deploy/compose.library.yaml`, added by `make cache-up` and `scripts/ops/stack-up.sh`, and `deploy/compose.operator-library.yaml`, added by `make operator-up`, each only when `LIBRARY_DIR` is set; `make install` staging `~/.devai/library`; the optional lab mount in the Makefile, `bin/devai-agent` and Phase 12's `scripts/ops/lab-start.sh`, with `devai-library` in both NO_PROXY lists; `LIBRARY_DIR` among devai-operator's path-identity mounts (C6), so the stack-up action's checks and the alerts listing work from there; `~/.config/devai/library/` in devai-backup's manifest; LS15's three checks in `make cache-up` and the service.
- Images and services: `deploy/Dockerfile.library` and `deploy/Dockerfile.library-extract` (hash-locked, on `devai-base`), `deploy/Dockerfile.grobid` (LS29), `devai-library`, `devai-library-extract`, `devai-grobid`, the `devai-library-internal` network; `make build-library`, `make build-grobid`, `make test-library`, `make library-init` (LS30); each a C8 action once Phase 7 has landed, `library-init` of kind `host`.
- `deploy/library-sources.yaml` with the arXiv rate groups; the politeness engine, the RFC 9309 wrapper with the stricter 4xx rule, the redirect and address checks, the breakers, the stops, recognition of pipelock's responses.
- The store: layout, append-only records and their fold, alias resolution with a single writer, tombstones, the maintenance job (staging TTL, and the 30-day trash of revoked items, LS28).
- The arXiv adapter: search (GET, cached 24 h), version resolution, PDF, HTML, `arXivRaw` in the fetch job; format checks. The check of the anonymous PDF mirror's terms and robots.txt (LS27); the mirror is wired in only if both allow it.
- Extraction: HTML-primary units with text-aligned pages, GROBID for PDFs without HTML, the language detector.
- The v1 endpoints except enrichments and library search: verdicts with evidence checks, the one-current-verdict rule with A-O1's scoped exception, topic assessments, tombstones and 410.
- Jobs: events, waits, ETA per rate group, heartbeat, cancel, resume.
- The operator: the library-admin actions (`breaker-clear`, `arxiv-contact-recorded`) in the `library` group, and the stack page listing `<LIBRARY_DIR>/state/alerts.jsonl` when `LIBRARY_DIR` is set.
- Docs: `docs/library.md` (reference), CLAUDE.md, `docs/pipelock.md` (the finding that it does not restrict destinations).

**Exit criteria:**
- Unit tests with a fake upstream: intervals and crawl delays per rate group (search and OAI-PMH never overlap); 429 and 503 with and without `Retry-After` (a wait, never a breaker); sustained 429s (the self-clearing overload pause) and an origin 403 (the denial breaker, cleared only by the operator); a fake pipelock block that must not open arXiv's breaker; robots outcomes (`robots_disallowed`, `robots_denied`, `robots_unreachable`, 404 as allow) and the LS5 exemption; the stops' 503 and per-source error shapes; a cross-host redirect and a redirect to an internal name, both refused; every per-item outcome in aiagent's section 3.1; HTML present, absent and broken; verdict rules and the fold; the stops as `/health` and ETA show them; `LIBRARY_DIR` unset, relative, missing, and not writable (each refused, naming the requirement), and valid on an empty and on a non-empty directory (the layout written on first start, other files left alone).
- A layout test at scale: synthetic items and objects in the hundreds of thousands, timing `find` and a full fold.
- `LIBRARY_DIR` set in `.env` while devai-operator has no library mount: the stack-up action refuses the library naming `make operator-up`, not a missing directory.
- A live smoke run, once the contact address exists: one small arXiv query, a fetch of two papers (one with HTML, one without), units with anchors and pages, a reject that purges and answers 410, an accept that keeps, all within arXiv's limits.

### 20.2 -- OpenAlex, open-access locations, index

**Goal:** sources beyond arXiv (LS23), and search over the library.

**Entry:** a manual read of OpenAlex's Terms of Service in an ordinary browser.

**Deliverables:**
- The OpenAlex adapter: discovery (structural filters where they fit, text matching priced as search), open-access locations, `cited_by_count` and citing works, advisory author metrics with provenance (source, as-of date, match method), the dollar budget; `POST /v1/enrichments` and `GET /v1/candidates/{id}`.
- Unpaywall by DOI. Fetchable link records for open-access PDFs on allowed hosts (each host added to `library-sources.yaml` with its own robots and limits); "known, not held" link records otherwise, SSRN included (LS8).
- The key, if an OpenAlex key exists: `~/.config/devai/library/openalex.key` (LS30), mounted into `devai-library` only, the per-pattern DLP exemption for OpenAlex's host. Without a key, 20.2-20.3 run keyless at $0.10/day, and the acceptance test is sized to that.
- The keyword and vector indexes, library search; the embedding model's store (LS24: `/var/cache/devai/embed` on its own LV, set up with `deploy/setup-logs-volume.sh`), its catalog row in `deploy/embed-models.yaml` and its pull through `scripts/select-models.py`.
- The rebuild command: delete `index/` and rebuild it from the plain files.

**Exit criteria:** candidates from OpenAlex merge with arXiv records by identifier; a non-arXiv open-access PDF is fetched from an allowed host; an SSRN work comes back as a link with its reason; after a rebuild from the plain files the topic listing is byte-identical and every alias resolves as before (aiagent's check 6).

### 20.3 -- the acceptance test, with aiagent

**Goal:** the owner's test (LS23) end to end, as aiagent's section 10 walks it.

**Exit criteria:** aiagent's checks 1-7 pass on the AAD topic: check 2 as revised (a PDF, plus the arXiv HTML when the fetch staged it, otherwise a `quality.warnings` reason, and a license field, which may be `none_stated` for old arXiv papers), and check 7 (the run searches OpenAlex as well as arXiv, and its results hold at least one non-arXiv work, either an open-access copy fetched from an allowed host or a "known, not held" link with its reason). Progress never goes silent for more than 5 s. The measured times are recorded here.

### 20.4 -- code (verification level (a))

**Goal:** GitHub repositories that papers link to (LS6, LS10).

**Deliverables:** the GitHub adapter (fine-grained token, the documented retry ladder, commit pinning, licence detection, the tree with dataset-like files refused, fetches of chosen files only); code links from Hugging Face's papers API (token, `githubRepoAddedBy` recorded with each link, so user-set links are not treated as official) and from the papers themselves; excerpts stored as aiagent's section 3.6 specifies (at most 80 lines each, 10 per repository, licence required, A-O3), refreshed per LS14; the level (a) checks as data; per-pattern DLP exemptions for GitHub's and Hugging Face's API hosts; their tokens as key files (LS30).

**Exit criteria:** unit tests with a fake GitHub and Hugging Face API: the retry ladder, rate-limit headers, commit pinning, licence detection ("none" and "other" are not permission), dataset-like files refused, excerpts within A-O3's limits (at most 80 lines each, 10 per repository). A live run on one AAD paper's linked repository once the tokens exist: its excerpts pinned to a commit, with permanent links.

### 20.5 -- selected community sources

**Goal:** the remaining sources, within their terms (LS8, LS9, LS22).

**Deliverables:** the Wikipedia API (revision-pinned, cited by `oldid`); owner-approved blogs through their feeds, each blog's hosting platform classified first, so a Substack-hosted blog on its own domain inherits Substack's LS8 status; Semantic Scholar citation contexts (key requested at the start of this phase; storage of each field only under its licence); OpenReview reviews (a service account); Kaggle's API for notebooks the user names (stored only under a stated open licence); GitHub-hosted Colab notebooks; Medium after the terms are read. Link records for everything else.

**Exit criteria:** each adapter tested against a fake upstream (limits, licences, stops); a Wikipedia citation carries its `oldid`; a Substack-hosted blog on its own domain is classified as Substack and stays link-only; Semantic Scholar fields are stored only under their licences; a live smoke run per source once its key, account or terms reading exists.

### Later -- verification level (b), a separate plan

Sandboxed execution of chosen code, compared against the paper's reported results, as a Slurm job with the engine license, like the trainer jobs (Phases 8-9). It reuses the job model.

### Phase 20 risks

| Risk | Phase | Mitigation |
| --- | --- | --- |
| A ban from arXiv (403) after a policy mistake. | 20.1 | Rate groups, the crawl delay, the arXiv breaker, persisted counters, the full-text cap until arXiv is contacted. |
| A pipelock DLP block mistaken for a source's denial. | 20.1 | pipelock's responses recognised and recorded as `egress_blocked`; a fake-block test. |
| Other lab paths spend arXiv's per-operator budget. | 20.1 | Accepted by the owner (LS25); documented in `docs/library.md`; the breakers stop the library quickly if arXiv pushes back. |
| Limits change without notice (OpenAlex changed its model in 2026; arXiv's docs disagree with each other). | all | Every limit is configuration; runtime headers win; the first 429 from any host is logged and alerted. |
| Parser exploits in PDFs or HTML. | 20.1 | Parsing only in `devai-library-extract` and GROBID: no secrets, no store, internal network only; the service validates their JSON. |
| The service used to reach internal hosts (SSRF). | 20.1 | No URLs from aiagent; every redirect hop checked; internal names and private addresses refused; default-deny host admission. |
| arXiv HTML missing or broken for a paper. | 20.1 | Format checks; GROBID over the PDF; `quality.warnings` in `text.json`. |
| The library is written somewhere unintended (an unmounted mount point, a mistyped but existing path). | 20.1 | Accepted by the owner (LS15): only the three requirements are checked. The service logs the resolved path and its layout version at every start. |
| Real credentials blocked by pipelock's DLP. | 20.2, 20.4, 20.5 | Per-pattern exemptions limited to each issuer's API host, added in the phase that sends the credential. |
| Author metrics wrong (split or merged profiles). | 20.2 | Advisory only, with source, as-of date and match method (A-O5). |
| Millions of files slow the tools. | 20.1 | Two-level fan-out, measured in the layout test; indexes derived. |
| Licence contamination of the service. | 20.1 | Permissive dependencies only. |
| Credentials leaking into the lab. | 20.2, 20.4, 20.5 | Secrets only in `devai-library`; never in the library directory. |

---

## Phase 21 -- Loose ends

Small items left by finished plans. (The laya trainer's Phase 4, once listed here, was closed by the operator: I23.) Each is independent of the rest and can be done at any time.

| Item | From | What is left |
|---|---|---|
| pin pi | pi-coding-agent | `make fetch-cli` downloads pi from `releases/latest`; pin a version, as `dsh` is. (This row used to say every other download was pinned. It is not: 12 of the 13 `fetch-cli` downloads take the latest release. Pinning the others is not planned here.) |
| pi's two notes | pi-coding-agent | document the cold start a `/model` switch can cause (the GPU is exclusive), and how Pi picks up AGENTS.md |
| parser provenance | card-derived-hints-and-bench-sync | document the fields the probers stamp into the probe caches (`tool_parser_source`, `reasoning_parser_source`, `parser_disagreement`, `card_sampling`) in docs/backends.md. The card branch's first real run goes with Phase 4's re-probe |

Each row is its own small change; none depends on another phase.

### Exit criteria

- `make fetch-cli` pins pi; the two pi notes and the provenance fields are documented.

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| every upstream bump becomes a rebuild here | all | accepted; the same as Ollama and vLLM today |
| security fixes in Debian arrive only when the digest is bumped | all | a deliberate periodic bump and rebuild, documented next to the pin |
| renaming the lab images (`-gpu` suffix dropped) breaks scripts and habits that name them | 3 | every reference in the tree changes in the same phase; the old names leave the store only in Phase 14 |
| per-engine staleness code touches probers, bench and router at once | 4 | one change with tests; the Ollama-only rebuild exit criterion |
| a job vanishes without unloading (its script killed, `devai-slurm` restarted) | 5 | the epilog's `unload`; the 60 s lease; the guard's idle check |
| the bus is down longer than a lease, and engines unload under running jobs | 5 | supervisord restarts nats-server at once; the jobs end, and the router submits new ones |
| GPU accounting by sampling is coarser than Slurm's own | 5 | one GPU job at a time, so the whole card is the job's; 5 s samples plus NVML's energy counter |
| Slurm reports no GPU energy on NVIDIA | 5 | devai reads NVML's energy counter at the start and end of each job (I3) |
| the web login controls the host user, on the LAN | 6 | TLS, one htpasswd account, CSRF, no shell; login attempts logged |
| a sleeping model's VRAM leaves too little for the next load | 8 | for a vLLM load, `devai-control` lowers the utilization by the sleeper's `used_mib`, unless recovery flags set it (SGLang sizes from free memory; a pool below its context counts as a failure); otherwise it unloads the sleeper and loads again |
| a job signals or reads what it should not | 5 | jobs run as a subordinate uid: no key but their bus password, no signal to the host user's processes; no devai service shares that uid (I26) |
| an action outlives its job (its script killed, `devai-slurm` restarted) | 5, 10 | the action's lease: `devai-control` stops its process group 60 s after the last renewal |
| a recovery file rewritten by rename stays invisible | 8, 17, 18 | `${DEVAI_REPO}/deploy` mounted as a directory, and both files re-read at each launch (I25) |
| a job held after a prolog failure starts only after Slurm's requeue delay | 9 | the router cancels it; the next request submits a new job, never releasing a held one |
| a keep-warm engine job blocks a queued trainer or probe job forever | 9 | the router yields to a pending trainer or probe job |
| no fallback inside the router after the cut-over | 9 | Phase 8's live checks first; rollback by reverting Phase 9 (I10) |
| a sleeper nobody needs keeps host RAM and VRAM while the router is down | 9 | the router unloads it when it is back (I9) |
| an interrupt with another model costs a cold start (about 95 s for Qwen3.8-27B) | 10 | the sleep fast path for vLLM and SGLang; the same-model path needs nothing |
| submitting a job can run any registered action, destructive ones included | 5-10 | the JWT key only in `devai-slurm`, the router and `devai-operator`; nothing published; bus users limited to their subjects |
| SGLang is kept or frozen on a measurement of the wrong thing | 15 | the instrument's dry run before the measured run; thresholds fixed before the runs, and a failed condition is a freeze, not a re-run (I16) |
| a recipe or a guessed parser misleads a probe | 17 | curated and card-derived values win; the probe is the authority; recipe flags reach the generated recovery file only after review, and never a C2 setting |
| alternating lanes thrash cold starts | 19 | opt-in; documented |
| a short-circuit answers a real request with a canned string | 16 | all markers must co-occur; off by default; every hit logged |
| an auto-fix makes a probe pass for the wrong reason | 18 | one retry; every applied flag recorded and attributed |
| a ban from arXiv (403) after a policy mistake | 20 | rate groups, the crawl delay, the arXiv breaker, persisted counters, the full-text cap until arXiv is contacted |
| GROBID does not build from source with Debian's JDK | 20 | checked first in 20.1; the PDF text then comes from pypdfium2 alone until it builds |

## Migration / rollback story

- **Phase 1:** MCP and Open WebUI come back by reverting the commit; the AMD overlay and sops/age by their `attic/` RESTORE.md.
- **Phases 2-3:** the compose and Makefile image variables can point back at the upstream images and the old lab images, which stay in the local store until Phase 14.
- **Phase 4:** `make backup-create` runs before the re-probe. Rollback is restoring the probe caches and pointing the image variable back. Removing `vllm-devai` changes agent provider names (`router-vllm-devai`); the picker rewrites those sets at every launch.
- **Phases 5 and 8:** the new containers run in the `slurm` compose profile, and the stack is otherwise unchanged. Rollback is not starting the profile.
- **Phases 6-7, 12:** the host shell and the Makefile keep working throughout, calling the same scripts. Rollback is not starting `devai-operator` (from Phase 6 on, the log follower runs there too).
- **Phase 9:** there is no switch (I10). Rollback is reverting Phase 9's commits, which brings back the router's podman launch path and the per-engine containers. The permanent `engines` container and the old per-engine containers never run together.
- **Phases 10-13:** each can be reverted on its own. The bench and the probers keep their old paths (`--legacy`, the container driver; for Ollama, Phase 9's throwaway container) until Phase 14.
- **Phase 11:** CPU mode is entered only when the host check finds no GPU (or `DEVAI_GPU_DEVICE=` forces it), so a GPU host behaves as before.
- **Phase 14:** runs only after the re-bench on the new path; rollback is a revert.
- **Phase 15:** a freeze is undone through its `RESTORE.md`; the documentation corrections stand either way.
- **Phases 16 and 19:** off by default (`DEVAI_SHORTCIRCUIT`, `DEVAI_FANOUT_PORT`); turned off, the router behaves byte for byte as before.
- **Phases 17 and 18:** recipe flags and automatic fixes live only in `deploy/recovery-flags.generated.json` (I25), each group marked `recipe` or `auto-fix`; removing a group, or the file, reverts them, and the curated file is never touched.
- **Phase 20:** `deploy/compose.library.yaml` and `deploy/compose.operator-library.yaml` are added only with `LIBRARY_DIR` set; unset, the library is off. The library directory itself is the operator's and is never deleted.
- **Phase 21:** reverts change by change.
- **Existing hosts:**
  - delete the registry mirror stanza from `~/.config/containers/registries.conf` (Phase 2);
  - run `make setup-jobs` once, as root (Phase 5);
  - run `make slurm-init` (Phase 5) and `make operator-init` (Phase 6);
  - rebuild.

## Estimated effort

| Phase | Wall-clock |
| ----- | ---------- |
| 1 | 1 day |
| 2 | 1-2 days |
| 3 | 1-2 days, plus a GPU window |
| 4 | 4-7 days, plus a full re-probe |
| 5 | 4-6 days |
| 6 | 3-4 days |
| 7 | 3-5 days (124 targets) |
| 8 | 4-6 days, plus a GPU window for step 1 |
| 9 | 4-6 days |
| 10 | 5-7 days, plus the re-bench |
| 11 | 2-3 days |
| 12 | 1-2 days |
| 13 | 1-2 days |
| 14 | 2 days |
| 15 | 2-3 days, plus a GPU window |
| 16 | 3-4 days |
| 17 | 3-5 days |
| 18 | 2-3 days |
| 19 | 2-3 days |
| 20 | 13-18 days (20.1: 5-6, 20.2: 2-3, 20.3: 1 plus aiagent's side, 20.4: 2-3, 20.5: 3-5) |
| 21 | 1-2 days |

Estimates, not measurements. Phases 5-7, 15-17, 20 and 21 run beside others, so the wall-clock total is shorter than the sum.

## References

- [docs/slurm.md](../slurm.md) -- the architecture this plan builds.
- Slurm documentation (slurm.schedmd.com):
  - slurm.conf: `Licenses`, `PrologEpilogTimeout`, `BatchStartTimeout`, `KillWait`, `PrologFlags`, `HealthCheckProgram`, `SlurmdParameters`;
  - cgroup.conf (`IgnoreSystemd`), for the fallback;
  - accounting;
  - slurmrestd, and rest_api (JWT).
- NATS server authorization: users, publish and subscribe permissions, `allow_responses` (docs.nats.io); the Go client, github.com/nats-io/nats.go.
- supervisord's XML-RPC API (supervisord.org/api.html).
- vLLM sleep mode (`--enable-sleep-mode`, `VLLM_SERVER_DEV_MODE`); SGLang `--enable-memory-saver`.
- Apache mod_wsgi (Debian `libapache2-mod-wsgi-py3`); Flask (Debian `python3-flask`).
- pipelock: github.com/luckyPipewrench/pipelock (Go, Apache-2.0).
- SGLang 0.5.16 package metadata on PyPI (pinned dependencies); NVIDIA's CUDA repository for Debian 13 (the host's CUDA 13.1 packages).
- Router short-circuit (Phase 16): free-claude-code, github.com/Alishahryar1/free-claude-code (`api/detection.py`, `api/optimization_handlers.py`).
- Catalog tooling and diagnosis (Phases 17-18): odysseus, github.com/pewdiepie-archdaemon/odysseus (`services/hwfit/*`, `routes/cookbook_*`); vllm-project/recipes.
- Library service (Phase 20):
  - aiagent: `docs/design/library-service-needs.md` (devitops-com/aiagent), the field-level contract.
  - arXiv: API terms https://info.arxiv.org/help/api/tou.html; user manual https://info.arxiv.org/help/api/user-manual.html; https://arxiv.org/robots.txt and https://export.arxiv.org/robots.txt; robots guidance https://info.arxiv.org/help/robots.html; per-article downloads https://info.arxiv.org/help/ir.html; bulk data https://info.arxiv.org/help/bulk_data.html; OAI-PMH https://info.arxiv.org/help/oa/index.html; identifiers https://info.arxiv.org/help/arxiv_identifier_for_services.html; media types https://info.arxiv.org/help/mimetypes.html; the 2025-11 API replacement https://groups.google.com/a/arxiv.org/g/api/c/-WpHNxbaxU0.
  - OpenAlex: authentication https://help.openalex.org/api/authentication/; pricing https://help.openalex.org/access/pricing/; searching https://help.openalex.org/api/searching/; semantic search https://help.openalex.org/api/semantic-search/; deprecations https://help.openalex.org/guides/deprecations.
  - Unpaywall: https://unpaywall.org/products/api and https://unpaywall.org/legal/terms-of-service.
  - Semantic Scholar: https://www.semanticscholar.org/product/api, its licence https://www.semanticscholar.org/product/api/license, release notes https://github.com/allenai/s2-folks/blob/main/API_RELEASE_NOTES.md. OpenReview: https://docs.openreview.net/getting-started/using-the-api and https://openreview.net/legal/terms.
  - SSRN: https://www.ssrn.com/index.cfm/en/terms-of-use/ and https://papers.ssrn.com/robots.txt.
  - Wikimedia: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy, https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits, https://wikitech.wikimedia.org/wiki/Robot_policy.
  - Tooling: RFC 9309 https://www.rfc-editor.org/rfc/rfc9309.txt; Protego https://github.com/scrapy/protego; pypdfium2 https://github.com/pypdfium2-team/pypdfium2; GROBID https://grobid.readthedocs.io/en/latest/Grobid-docker/ and https://github.com/kermitt2/grobid/blob/master/doc/Coordinates-in-PDF.md; LaTeXML https://github.com/brucemiller/LaTeXML.
- In this repository:
  - the host-build pattern: scripts/build-vllm.sh, scripts/build-ollama.sh, deploy/Dockerfile.vllm, deploy/Dockerfile.ollama;
  - attic/README.md and attic/cluster-mode/RESTORE.md: the pattern for moving work to the attic;
  - gpu-arbiter/main.go (`ensureBackendRunning`, `stopOtherBackends`, `containerRecreate`, `computeLaunchConfig`);
  - scripts/_probe_hf_common.py, scripts/bench-sync.py, scripts/bench/clean_slate.py, laya-trainer/laya_trainer/controller.py;
  - docs/router.md, docs/bench-results.md "Run protocol", docs/laya-trainer.md.

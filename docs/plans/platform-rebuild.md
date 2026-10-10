# Platform rebuild

_Pull exactly one image from upstream; compile every engine here into one container that runs all the time; let Slurm, in its own container, decide what is on the GPU and record every job; connect the containers with a NATS bus; make the router a Slurm client; and run everything the Makefile does from a web service, as jobs. Alongside: decide SGLang's future, answer Claude Code's throwaway requests in the router, rank models before probing, diagnose failed launches, offer lanes on one port, and build the library service aiagent needs._

## Status

Draft (2026-10-05; engines and System One added 2026-10-06). Built since, on branch feat/platform-rebuild: Phases 1, 5 and 21, steps 4v, 16a, 16e and 17a (2026-10-07). Phase 5's live checks (`make test-slurm`) passed on the card: 75 of 76 on 2026-10-09, and the unkillable-holder check, which needs sudo, alone on 2026-10-10 (`make test-slurm ONLY=holders`, 10 of 10).

Plan-level review ends here (operator, 2026-10-06). After four readiness reviews, the last fix round was checked finding by finding and left the small items listed under "Left for implementation" (below Open questions). They are settled in the phase named, and each phase's tests and exit criteria are the check, not a further review of this plan.

- **devai's only plan.** On 2026-10-05 it replaced every earlier plan; all were deleted, and git history keeps them. `docs/plans/README.md` lists each and what happened to it.
  - Its core: `slurm-gatekeeper`, `minimal-external-images` and `devai-operator`. Their decisions are kept below under their original IDs: M1-M15 (images), D1-D16 (Slurm), O1-O9 (operator), quoted as recorded. In them, "the Slurm plan", "the image-reduction plan" and "the devai-operator plan" now mean the matching phases of this plan.
  - The other plans: their parts that conflict with the rebuild or that it makes moot were dropped; the parts it absorbs are listed under "Absorbed from earlier plans"; their compatible or independent parts became Phases 15-21. The library-service plan's decisions are kept as LS1-LS28 (its D1-D28); LS29-LS30 were added on 2026-10-05, when the plans were folded in.
- **The architecture is [docs/slurm.md](../slurm.md).** This plan follows it section by section (see Approach).
  - Where the code needs more detail than the architecture gives, the detail is fixed under Contracts (C1-C8).
  - Where that detail adds to or differs from slurm.md, it is recorded as an implementation decision (I1-I33), taken by the operator on 2026-10-05 and 2026-10-06 (I27-I32 with E1-E6, I33 after the fourth readiness review); those that shape the architecture are folded into slurm.md.
- **Engines and System One (2026-10-06).** The operator's E1-E6: vLLM moves to 0.30.0 (E1); one version per backend gets recorded exceptions (E2), the first bringing DiffusionGemma's System One back on stock vLLM 0.31.0, taking turns on the GPU (E3); devai serves laya students on the CPU (E4); System One reads on the teacher are measured before anything is planned for them (E5); laya moves to 0.3.28 (E6). On 2026-10-07 the operator dropped the 0.28.0 arm of step 4v: 0.30.0 is accepted on its own context and load probes (E7); and the runtime sets take no CUDA compiler from PyPI (E8). On 2026-10-08: plain directories under `/var/cache/devai`, no logical-volume steps (E9).
- **23 phases.** Phases 1-14 build the new architecture; Phases 15-21 are the independent work folded in from earlier plans; Phases 22-23 are the laya sync and System One (E3-E6). Each has its files, steps, tests and exit criteria; the order table under Approach says what can run beside what.

## Dependencies

None inside devai. Outside it:

- **aiagent** (devitops-com/aiagent): the library service's field-level contract is aiagent's `docs/design/library-service-needs.md` (Phase 20). The laya trainer's API toward aiagent does not change (D14; reference: docs/laya-trainer.md).
  - Phase 22 waits for aiagent to accept laya 0.3.28: its `LAYA_COMPAT` (`src/aiagent/system1/artifacts.py`) widened to 0.3.20 and 0.3.28, released, and that release in the lab image (E6).
  - Phase 23's served System One is usable by aiagent only with a remote client on its side: answer shapes adapted per backend (choice criteria as a dict; no `answer_confidence` from DiffusionGemma), 503 + `Retry-After` handled per call, and System One reads batched before escalations to the teacher. aiagent decides whether and how its gate uses them (LS1; E3, E4).
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
- **System One** (Phases 22-23): laya students served on the CPU beside any LLM; DiffusionGemma's structured reads on an engine of their own, taking turns on the GPU; a measured answer on reads from the teacher itself; laya 0.3.28 with its training fix.
- **Later:** several local GPUs, one engine per GPU, through Slurm's GPU allocation; sandboxed execution of a paper's code (the library's verification level (b)). Not in scope here.

## Out of scope

- **GPU and cluster.**
  - Multi-host clustering. Slurm could do it later; devai does not use it.
  - Several GPUs on one host.
  - Two models on the GPU at once. DiffusionGemma takes turns with every other model (E3); no VRAM split by license, shards, MPS or MIG.
  - GPUs other than this host's: the engines are compiled for compute capability 12.0 (sm120) only, so a host with another NVIDIA GPU needs a rebuild for its architecture.
  - Shielding the GPU from the host user or from software outside devai. Application containers stop using the GPU by convention (D1), not because of a security boundary. A production setup would add one (driver device-file mode plus a service account); this plan does not.
- **Engines and builds.**
  - Building PyTorch, FlashInfer, Triton or other generic dependencies from source (M3).
  - Upgrading any engine version, except vLLM's one move to 0.30.0 (E1) and the recorded exceptions of E2. Ollama and SGLang 0.5.16 keep their versions (M5); only how they are built changes.
  - CPU builds of vLLM or SGLang. Without a GPU, only Ollama and the laya trainer run (M15).
  - Developer tooling run ad hoc from images (gitleaks, hadolint, golangci-lint, actionlint in docs/security-ci.md). These mirror CI and are not part of the stack.
  - Bootstrapping Debian itself from the archive (mmdebstrap) to reach zero pulled images. The trust root would be the same Debian archive keys; it can be added later if an air-gapped host needs it.
- **Behaviour that stays as it is.**
  - In Phases 1-14, what the router does to requests (rewrites, reasoning policy, tool stripping, SSE keepalive): only its launch layer changes. Phases 16, 19 and 23 add a short-circuit, a lanes port and the System One port, all off by default.
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

Quoted as recorded in the three plans this one replaces, except E1-E6, which the operator took for this plan on 2026-10-06.

### Images (2026-09-30)

- **M1 -- Only Debian's official image comes from upstream,** pinned by digest. Everything else is built here from Debian packages, from hash-locked PyPI packages, or from source at a pinned tag -- the pattern `devai-ollama` and `vllm-devai` already follow.
- **M2 -- CUDA 13.1 only.** `cuda-nvcc-13-1` is the only nvcc the host's apt sources offer (`apt-cache policy`, 2026-09-30: no candidate for 12.9 or 13.0). A backend that cannot be built and run with it is dropped or upgraded. No stock builds of anything.
- **M3 -- What is compiled here, and where the rest comes from.** Compiled here: the engines and their own CUDA kernel packages -- Ollama, vLLM, and SGLang with its kernel packages. Every other dependency comes from the Debian package when its version satisfies the engine's pin, otherwise from the pinned, hash-locked PyPI wheel (as `scripts/build-vllm.sh` does for `torch==2.13.0` today). In practice nearly all of the ML stack comes from PyPI: Debian main carries torch 2.6.0 CPU-only (SGLang pins `torch==2.11.0` with CUDA) and does not package transformers; FlashInfer and Triton are not in the host's apt sources.
- **M4 -- One vLLM backend:** vLLM 0.28.0 + HyperQwen (today's `vllm-devai`), named `vllm`, on port 11435. Stock vLLM 0.22.1 (CUDA 12.9) is dropped; the `vllm-devai` backend name and port 11437 are retired. *(Version superseded by E1 on 2026-10-06: vLLM 0.30.0 with HyperQwen's current series.)*
- **M5 -- SGLang stays at 0.5.16 and is compiled here.** Only the build method changes, so a regression points at our build, not at a new release. `deploy/backend-flags.yaml` and the SGLang recovery flags stay valid. A version bump is a separate step.
- **M6 -- All backends in one image, `devai-engines`:** Ollama, vLLM, SGLang. The router and the laya trainer keep their own images.
- **M7 -- All Slurm components in one image, `devai-slurm`, run as one container** (slurmctld, slurmdbd, slurmrestd, slurmd, MariaDB), built by the Slurm plan. **No Slurm in the backends image:** jobs start engines inside `devai-engines` (the operator's answer to the Slurm session, 2026-09-30); the details are that plan's. *(Refined by D15: engines run all the time, and jobs only load and unload models.)*
- **M8 -- No per-model engine images.** The `image` field in `deploy/recovery-flags.json` (vLLM 0.25.1 for NVIDIA-Nemotron-Nano-9B-v2-NVFP4, the vLLM "gemma" build for diffusiongemma-26B-A4B-it-NVFP4) has no place in one image. A model that does not run on its backend's single version is dropped. *(E2-E3, 2026-10-06: diffusiongemma comes back on a backend of its own, `vllm-s1`, as a recorded exception; per-model images stay out.)*
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
- **D3 -- Job results live in `/var/cache/devai/jobs`** (a plain directory the operator creates, E9; it was to be its own volume).
- **D4 -- Use a published image if one exists.** *(Superseded by D7 on 2026-09-30.)* Docker Hub has none that is maintained (2026-09-29: the `slurm/` namespace holds two repositories last updated in 2017 and 2018; the rest are personal images). SchedMD, Slurm's maintainer, publishes official images on GHCR (`ghcr.io/slinkyproject/{slurmctld,slurmd,slurmdbd,slurmrestd}:26.05-ubuntu26.04`, Slurm 26.05.4, recipe github.com/SlinkyProject/containers). Its `slurmd` was built without the `gpu_nvml` plugin.
- **D5 -- Not a separate project.** One consumer, contracts that change together with the router and the workloads, no product without devai. A second repository would make every change a cross-repo release (the aiagent experience). It stays extractable: own directory, own tests, config under `deploy/slurm/`, a written job/result format.
- **D6 -- No containers inside containers.** Ruled out after spike round 1, which ran engines as podman containers nested in the slurmd container. (It also meant, until D10, that an engine ran as the job's own process. D10 replaced that with `podman exec` into a sibling container, which is still not nesting. Since D15 no job starts a process in another container at all.)

### Slurm (2026-09-30, second round)

The first architecture (one Slurm node per backend image) was judged far too complicated; a second one (Slurm and every backend in one image) mixed Slurm with the backends.

- **D7 -- Fewer moving parts, above all fewer images.** Follow the image-reduction plan: `debian:trixie` is the only upstream image, everything else is built here. For this plan: Slurm's daemons come from devai's own Slurm build and MariaDB from Debian, which **supersedes D4**. *(Its Slurm-build part is superseded by D12: Slurm comes from Debian's packages.)*
- **D8 -- One version per backend.** One vLLM (0.28.0 + HyperQwen), one SGLang, one Ollama, one laya trainer; a model that does not run on its backend's version is dropped. One vLLM backend, `vllm` on 11435; `vllm-devai` and 11437 are retired. *(Version superseded by E1; exceptions to one version per backend are recorded under E2, the first being E3.)*
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

### Engines and System One (2026-10-06)

The operator's answers after two research rounds on 2026-10-06: vLLM after 0.28, "System One" models beside an LLM, and the laya sync. The research itself is not kept in the tree; each fact the plan relies on is cited where it is used.

- **E1 -- vLLM moves to 0.30.0 with HyperQwen's current series.**
  - 0.30.0 (2026-09-22) is the version HyperQwen's main branch pins. 0.31.0 (2026-10-05) waits for HyperQwen's port: a text-only apply of its series to the 0.31.0 tree failed 12 of 46 patches (research of 2026-10-06, on a partial tree; some may only cascade, and nothing was built).
  - It replaces 0.28.0 in M4, in D8 and in the out-of-scope list. devai's own MTP patch (`deploy/vllm-patches/`) is still needed: upstream 0.31.0 still allocates the drafter's vocabulary before sharing it.
  - Phase 4 builds it first (step 4v), on today's `vllm-devai` backend, so that Phase 15 and Phase 8 step 1 measure the vLLM that will serve.
- **E2 -- One version per backend, with recorded exceptions.** D8's rule stands; the operator expects exceptions to it ("we shall probably need exceptions from the single vllm version rule anyway").
  - Each exception is recorded here, with the models it serves, its reason, and what ends it.
  - It is a backend of its own: its own name, port, probe cache, and environment inside `devai-engines`. So M6 and D9 stand: one image, one container, one supervisord.
  - It holds the GPU only under `engine:1`, like every engine. M8 stands: there are no per-model images.
  - Which exceptions are admitted, and how each is built, is set by I32.
- **E3 -- The first exception: `vllm-s1`, stock vLLM 0.31.0, for DiffusionGemma's System One.**
  - vLLM's System One is DiffusionGemma's structured read mode (vllm#57250, vllm#58216). It ships first in 0.31.0. Its `/v1/systemone` comes from an example server that devai vendors at the 0.31.0 tag.
  - It serves one model, diffusiongemma-26B-A4B-it (NVFP4, about 17.5 GiB of weights). That cannot share the 24 GB card with the 27B teacher or with any LLM devai has probed, so it takes turns with every other GPU model under `engine:1`. aiagent reads its System One answers in a batch before it escalates to the teacher (aiagent's side).
  - It is measured first (Phase 23, step 1). If it does not load, serve and stay up on this card, it is deferred, and the deferral is recorded here.
  - It ends when the `vllm` backend reaches a version with the structured read mode (a HyperQwen port to 0.31 or later), or when an in-engine `/v1/systemone` (vllm#59299, not merged) covers what aiagent uses.
  - It reverses M8's drop of diffusiongemma, for this backend only.
- **E4 -- devai serves laya students on the CPU, behind `/v1/systemone`.**
  - The trainer's format-1 artifacts run on onnxruntime's CPU provider: the same graph and the same provider as the golden check, so their calibration and aiagent's tau stay valid. No GPU, no license; pinned to the E-cores where the CPU has them (the question the operator answered named both; how, I28).
  - Third-party small System One models (OpenDecider-nano, Verdict, jevos) come later, each only after its licence and provenance are checked. TypeSafe's customer agreement (section 2.3(b)) forbids training on Jev's output, and several published models say they were.
  - Whether aiagent's gate uses a served student, and under which tau, is aiagent's decision (LS1). aiagent's own D2 and D3 keep its in-process path.
- **E5 -- System One reads on the resident teacher are measured first, then decided.** Reading next-token log-probabilities over the answer labels needs no extra VRAM. Two open vLLM bugs concern log-probabilities under speculative decoding, which MTP is: vllm#42592 (a crash) and vllm#44727 (wrong values). Phase 23 measures it on the `vllm` backend with MTP on and off. Whether devai plans a route for it is decided on that report.
- **E6 -- laya 0.3.28.**
  - The package moves from 0.3.20 to 0.3.28. The checkpoints stay at commit 55cf4c4: the only newer commit adds a root `config.json`, and a new revision would make every stored dataset fail its base check.
  - Dataset schema 1 and artifact format 1 stay.
  - aiagent goes first: it accepts 0.3.28 beside 0.3.20 and releases. Then devai bumps.
  - devai's training takes upstream's fix for the partial gradient-accumulation window now (laya gh#941). Soft cross-entropy and option shuffling come only if a held-out evaluation supports them (Phase 22, I31).

### vLLM 0.30.0 acceptance (2026-10-07)

- **E7 -- 0.30.0 is tested on its own; 0.28.0 is no longer measured.** The operator, 2026-10-07: "I am not really interested in 0.28 any more. just test 0.30."
  - 0.30.0 replaces 0.28.0 when both prepared Qwen3.8 rows keep their contexts with MTP on 0.30.0 -- Qwen3.8-27B-MTP-devai-NVFP4 at 118,784 and Qwen3.8-27B-W4A16-devai-AutoRound at 131,072 (probed at its 0.93 and, only on a miss, once at 0.96) -- and both pass the serving-time load probe at that context. A miss or a load-probe failure stops the move, and the report goes back to the operator, as under E1.
  - Dropped from step 4v: every 0.28.0 measurement -- the 0.28.0 load-probe baseline, the quality non-inferiority comparison (the paired gsm8k and humaneval runs with their 3- and 5-point margins), and the paired speed ratios and MTP acceptance lengths. `scripts/stats/vllm_upgrade_accept.py` keeps only its `retention` subcommand.
  - Kept, recorded without a threshold: gsm8k (1,319 items) and humaneval (164) on 0.30.0 with MTP, as the picker serves the teacher; the latency prompts on both rows; the FlashInfer check; the engine-log facts. Kept as before: the prefix-cache retention measurement (it compares two 0.30.0 settings) and the re-probes for Phase 15's baseline.
  - 0.28.0's wheelhouse and image tag stay until 4g passes, as the rollback for a failed move.
- **E8 -- No CUDA compiler from PyPI.** The operator, 2026-10-07: "we don't need to copy them verbatim from hyperqwen, if we have better solution."
  - The runtime sets leave out pip's CUDA compiler packages (`nvidia-cuda-nvcc`, `nvidia-cuda-crt`, `nvidia-cuda-cccl`, `nvidia-nvvm`), from HyperQwen's `docker/requirements.txt` and from vLLM's own `humming-kernels[cu13]` extra alike (the package stays, without the extra). HyperQwen pins pip's 13.0 set because its image has no toolkit; devai's carries the host's CUDA 13.1 (M2).
  - Checked in the 0.30.0 image on 2026-10-07: FlashInfer JIT-compiled its kernels with `/usr/local/cuda` (13.1), and humming-kernels, the only other package that compiles with nvcc at run time, picks the system toolkit over pip's when both are present. Neither Qwen3.8 row uses humming (NVFP4 on FlashInfer CUTLASS, int8 layers on Marlin).
  - It amends step 4v's runtime-set rule ("without its `vllm`, `flashinfer-cubin` and extra-index lines"): `scripts/build-vllm.sh` drops these four packages and those extras when it writes `<version>.in`.
  - Done 2026-10-07: both locks regenerated (each lost exactly the four packages); the 0.30.0 image rebuilt without them (`5ab31d2b6ebc`) and re-probed with the FlashInfer cache emptied -- both prepared rows keep their contexts with MTP and pass the load probe, Qwen3.5-9B-NVFP4 keeps 262,144, every value equal to the first build's (docs/backends.md "vLLM 0.30.0"); `:latest` moved to it.

### Storage (2026-10-08)

- **E9 -- Plain directories under `/var/cache/devai`; no logical-volume steps.** The operator, 2026-10-08: "remove this logical volume creation step. regarding thin-pool: this is planned, but is part of another project. for now, we just use plain directories."
  - `/var/cache/devai` is one volume (`vgais-cache`), and the folders under it are plain directories on it. A new one is a plain directory the operator creates. Per-folder thin volumes belong to that other project, not to this plan.
  - `make setup-jobs` is removed. The operator created `/var/cache/devai/jobs` on 2026-10-08. `make slurm-up`, the `devai-slurm` entrypoint and `tests/slurm/test-slurm.sh` check that it exists, not that it is a mount point. Amends D3, C4 and Phase 5 step 8.
  - The same holds for LS24's embedding store: `/var/cache/devai/embed` is a plain directory, and `make setup-embed` is not written (Phase 20.2).
  - No step of this plan checks for a mount point under `/var/cache/devai`.

## Implementation decisions (2026-10-05)

The plan proposed each of these as the smallest choice that lets the code be written, and the operator decided them on 2026-10-05. I1-I14 were decided first, three of them differently from the proposal (I4, I9, I10); I15-I24 the same day, after a readiness review; I25 and I26 after a second review, which also changed I24's model and defined I16's measurement. After a third review, on 2026-10-06, the operator revisited I26 and completed I16's measurement. Each is folded into slurm.md where it shapes the architecture, and each names those sections and the phase that needs it. I27-I32 were proposed on 2026-10-06 with E1-E6, and the operator accepted them the same day as proposed, together with Phase 4 step 4v's acceptance rule and "chat wins" over a loading `vllm-s1` (Phase 23).

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
  - The host calls it through a small wrapper, `bin/devai-jobs`, which runs `podman exec devai-slurm devai-jobs "$@"`. From Phase 10 on, it runs `submit`, `suspend`, `interrupt` and `service` in devai-operator instead (`podman exec devai-operator devai-jobs "$@"`), because only devai-operator mounts the C8 registry that a submission reads for an action's kind, license and time limit, and `suspend` and `interrupt` submit the cancelled bench task again; every other command stays in devai-slurm.
  - So neither slurmrestd nor NATS is published to the host, and the JWT key is never read on the host.
  - The operator web app calls the same binary, so there is one REST client (C8).
- **I14 -- Running without a GPU comes after the router moves to Slurm** (Phase 11).
  - The CPU mode of the router is written once, in the Slurm launcher, and never in the podman path that Phase 9 removes.
  - The earlier phases keep only the cheap no-GPU checks of the pieces they build (devai-slurm, devai-control).
- **I15 -- Phase 15's baseline is Qwen3.5-9B-NVFP4 on vLLM 0.28** (Phase 15; operator, 2026-10-05). No model fits both SGLang and the vLLM that survives (M4): the two models that fit vLLM 0.28 do not fit SGLang. Qwen3.5-9B-NVFP4 fits SGLang and its weights are on disk, so Phase 15 first probes it on vLLM 0.28 and then measures both engines on it. If it does not load on vLLM 0.28, gpt-oss-20b and then Gemma-4-26B-A4B-it-NVFP4 are tried; if none loads, SGLang is frozen by default (Phase 15, step 1). *(Since E1 the vLLM side is 0.30.0, built in Phase 4 step 4v; the three-candidate order stands.)*
- **I16 -- The SGLang keep thresholds** (Phase 15; operator, 2026-10-05). SGLang is kept only if both hold:
  - (i) its steady TTFT is within 3x of vLLM's on the same checkpoint;
  - (ii) its turn-2+ TTFT on an identical multi-turn workload is at least 25% better than vLLM's, with a non-zero engine-reported cache hit rate and a measurable gain over `--disable-radix-cache` (conditions (b)-(d) below).

  How it is measured (operator, 2026-10-05 and 2026-10-06, after the second and third readiness reviews; docs/statistics-primer.md conventions):
  - Both engines are launched by the check itself, outside the router (Phase 15), at one context (the smaller of the two probed maxima), with MTP off (the checkpoint declares `mtp:`, and the picker would launch it with `::mtp`).
  - Each engine runs with the KV-cache dtype its probe cell is stamped with, as the router would serve it: vLLM at fp8 (0.30.0 since E1; step 1's probe); SGLang without `--kv-cache-dtype`, since the checkpoint has no KV quantization (its default, bf16). The check passes the dtype explicitly, never a prober's default; the context is the smaller of the two maxima probed under those dtypes; the report records both. So the ratios compare each engine as served.
  - Every request carries the same body:
    - reasoning off, as `chat_template_kwargs: {"enable_thinking": false}` at the top level, which both engines read (for the gpt-oss-20b fallback, whose template ignores it, `reasoning_effort: "low"`);
    - the same sampling;
    - no tools (the two engines use different Qwen3.5 tool parsers).
  - Steady TTFT: the bench's latency prompts (`scripts/bench/data/latency_prompts.jsonl`), prompt 0 as the warm-up, then the median over prompts 1-30. They are distinct and short (about 4-55 tokens, sharing at most 13 characters), so this measures scheduling, streaming and a short prefill, not long-prompt prefill.
  - Turn-2+ TTFT: 20 sessions per engine and configuration (vLLM, SGLang radix-on, SGLang radix-off), each of five turns with run_multiturn's sizes: a 4,096-token system prefix, 4,096 tokens added per turn, 64 tokens out (its defaults, fixed here before the runs).
    - Each session has its own salt, made of a per-run nonce (bench_concurrency's `--salt`, Phase 4 step 4v), the engine and the session number. So every session starts on a new prefix: sessions are independent, and no cache is flushed between them.
    - A session's value is the median TTFT of its turns 2-5; the engine's is the median of its 20 session values.
  - Ratios are SGLang over vLLM, and radix-on over radix-off for the gain. Their 95% interval is a percentile bootstrap over session values (`boot_ci_two_sample` in scripts/stats/statlib.py, 10,000 resamples): sessions, not turns, because the turns of one session are not independent.
  - SGLang is kept only when all four hold; any other outcome is a freeze:
    - (a) the steady ratio is at most 3.0 (point estimate);
    - (b) the turn-2+ ratio, SGLang over vLLM, is at most 0.75 (point estimate), and its interval lies wholly below 1.0;
    - (c) the turn-2+ ratio of SGLang radix-on over radix-off has an interval lying wholly below 1.0 (radix-on faster): the "measurable gain over `--disable-radix-cache`" of (ii);
    - (d) the pooled hit rate of the radix-on sessions is above zero. It is the sum over sessions of the change in `sglang:cached_tokens_total` (summed over `cache_source`) over the sum of the change in `sglang:prompt_tokens_total`, each read before turn 2 and after the session's last turn (`--enable-metrics`); the per-session rates are reported too.
  - The measured run is not repeated: a failed condition is a freeze, not a reason to change the scorer. The scorer is checked before it, in an uncounted dry run (Phase 15, step 2).
  - Every measured request (steady prompts 1-30 and every session turn 1-5) must return 200 and stream at least one token; no session is dropped and no turn is skipped. A failed request, an engine exit or a launch failure in an SGLang configuration (radix-on or radix-off) ends the measured run, and the outcome is a freeze, recorded in M16 with the failing request. A failure in the vLLM arm voids the run: M16 records it, and the operator decides (no automatic re-run). *(Added after the fourth readiness review, 2026-10-06.)*
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
  - gpt-oss-20b has not run on devai's vLLM build yet (0.30.0 since E1). If it does not load there in 4g, the check runs on SGLang only (if kept) and records why.
  - The first choice, Qwen3.5-9B-NVFP4, rested on a wrong premise (that every vLLM/SGLang family had a curated block; 15 SGLang catalog rows have none). `_card_hints.py` refuses that checkpoint by design -- ambiguous Qwen3 XML tool markup, a bare `<think>` -- so it could only have exercised the refusal branch. The operator changed the model after the second readiness review.
- **I25 -- Machine-made recovery flags live in their own file** (C2, C6; Phases 8, 17 and 18; operator, 2026-10-05, after the second readiness review).
  - `deploy/recovery-flags.generated.json` holds every flag a program added -- recipe flags (Phase 17's `recipe-apply`) and automatic fixes (Phase 18) -- and nothing else. `deploy/recovery-flags.json` stays hand-curated, and its contract does not change.
  - Each model has a list of groups, `{backends, engine_flags, engine_env, source, reason, added}`. `backends` is required (there is no "every backend"); `source` is `recipe` or `auto-fix`; there is no `image` (M8). A missing file reads as empty.
  - Both readers -- the router's recovery registry (`pkg/engineargs` from Phase 8, used by the router and `devai-control`) and `scripts/_probe_hf_common.py` -- append, for a launch on backend B, the curated entry's flags and environment when it applies to B, then every generated group whose `backends` names B or B's engine (the curated file's rule: `appliesTo`, `_entry_applies_to_backend`), in file order. The last flag still wins. Phase 17a adds the file and both readers; Phase 8 needs 17a and carries the Go reader into `pkg/engineargs`.
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

**Decided on 2026-10-06, with E1-E6** (proposed by the plan, accepted by the operator as proposed).

- **I27 -- One System One port on the router** (Phase 23).
  - The router serves `POST /v1/systemone` and `GET /v1/models` on 11440 (`DEVAI_SYSTEMONE_PORT`; unset means off, and the router is then byte-identical to before).
  - It dispatches by `model`. DiffusionGemma's catalog name goes to `vllm-s1` through an engine job, as every GPU launch does: engine license and guard. While `vllm-s1` is not serving with that launch, the router starts the switch and answers 503 + `Retry-After` at once, holding no request open. No request waits on that switch, so the router watches the engine job itself: it keeps the pending job while requests for it keep arriving and cancels it once none has come for `HEALTH_TIMEOUT_SECONDS`, by a timer of its own rather than Phase 9's last-waiter rule; a failed launch charges the breaker as Phase 9 case 5 says and is reported once, to the next request (Phase 23). On a host without a GPU, DiffusionGemma's name answers the router's no-GPU error before anything is submitted, as every vLLM request does there (Phase 11). It is launched at one fixed context, `DEVAI_SYSTEMONE_CTX` (default 8192, confirmed by Phase 23 step 1). Any other name goes to the student service, which answers 404 for a name it does not serve; no job, no license.
  - The reply names the model that answered. Each backend's answer fields pass through unchanged: the router never maps one backend's `confidence` onto another field, and `answer_confidence` comes only from students. Error statuses are kept as they are, with no rewrite to 502.
- **I28 -- The student service runs in `devai-laya-trainer`** (Phase 23).
  - It is a supervisord program, `systemone`, managed by that container's `devai-control` (kind `program`), on port 11443 on `devai-net`.
  - CPU only: `CUDA_VISIBLE_DEVICES` empty and onnxruntime's CPU provider. It is pinned to the E-cores the kernel lists in `/sys/devices/cpu_atom/cpus`; where none are listed, it is not pinned.
  - It serves each succeeded job's artifact from `/laya/runs`, by its `fine_tuned_model`. An artifact is loaded on first use, after its file hashes check out and its golden rows are reproduced within 1e-3 (aiagent's own check). At most `SYSTEMONE_MAX_RESIDENT` (default 2) are loaded at once, the least recently used going first. It never writes into a run directory.
  - It accepts manifests whose laya version is 0.3.20 or 0.3.28, as aiagent does after E6.
- **I29 -- `vllm-s1` is a fourth environment in `devai-engines`** (Phases 4 and 23).
  - `scripts/build-vllm.sh` builds it from the record `deploy/vllm-builds/0.31.0.sh` (4v's `VLLM_VERSION`): its own sdist sha256 and kernel pins, no patch series, and its runtime set `deploy/vllm-builds/0.31.0.in`, hash-locked as `0.31.0.lock` by 4v's `make vllm-lock VLLM_VERSION=0.31.0`, from PyPI only (M3). `flashinfer-cubin`, published only on flashinfer.ai's index, is not installed, as 0.28.0's build does not install it: FlashInfer JIT-compiles its kernels with the image's nvcc (with run-time cubin downloads disabled, `FLASHINFER_CUBINS_REPOSITORY=file:///nonexistent/`; Phase 23 step 1). The result is its own stage in `Dockerfile.engines`.
  - It is its own service in the engines' `control.yaml`: the engine on 11441 and the vendored System One server on 11442, both started by one `load` and stopped by one `unload`.
  - It has its own probe cache, `deploy/.vllm-s1-reasoning-cache.json`, and `sleep_mode` false until Phase 23 has measured sleep on it.
  - It loads only on an empty card: its engine job's prolog unloads any sleeper first, as a probe job's does, unless the sleeper is this job's own launch on `vllm-s1` (the job's `load` then wakes it), and the router ends the job it replaces with a cancel, never USR1.
  - The server, `examples/features/structured_diffusion/structured_server.py` at the v0.31.0 tag, is vendored unmodified under `third_party/vllm-systemone/`, with a sha256 manifest, as HyperQwen's scripts are.
  - It is not offered by the picker: DiffusionGemma is reached only through `/v1/systemone` (I27).
  - Its dependencies follow M3: PyPI only, hash-locked (`deploy/vllm-builds/0.31.0.lock`). `flashinfer-cubin`, published only on flashinfer.ai's index, is left out, as in every vLLM build here (Phase 4 step 4v). It would come in only if Phase 23's gate shows it is needed and the operator records that under E3.
- **I30 -- The DiffusionGemma gate** (Phase 23, step 1). It is measured with the card to itself (before Phase 10 with the stack down; from Phase 10 on as a GPU action, C8), at 8192 context, at most 4 sequences, a served canvas of 64 and the vision tower off. DiffusionGemma goes on only if all of these hold:
  - it loads, and its peak VRAM (0.1 s sampler) stays at least 1 GiB below the card, under requests that fill the context, as Phase 23 step 1 builds them;
  - a soak of 2,000 System One reads mixed with 200 chat requests, at 1 and at 4 clients, ends with the engine still up, and 10 further separate launches (start, `/health`, 50 reads, 5 chat requests, stop) all serve to the end. 10 of 10 bounds the per-launch failure rate below 26% (one-sided 95% Clopper-Pearson); the June record is 23 of 34 launches lost;
  - every read is valid: each question answered, its probabilities summing to 1 within 1e-3.

  Its startup time and per-read latency (p50, p95) are recorded without a threshold. Flags are tuned only in uncounted shakedown launches. The counted runs (both soaks and the 10 launches) run once, on the final flag set. A failure in a counted run is a gate failure, not a reason to run again with the same flags. A counted run that the GPU action's time limit ends (from Phase 10 on) is neither a pass nor a failure: the report records it as "limit reached", with what was measured, and it goes to the operator; it is not run again automatically. The operator may run it again on the same flags by naming that attempt (Phase 23 step 1, `--after-limit`). A changed flag set restarts every counted run, and the report lists every counted attempt. If the gate fails, E3's deferral is recorded and Phase 23 goes on with the student service alone.
- **I31 -- When soft cross-entropy or option shuffling is adopted** (Phase 22).
  - Three arms on one stored campaign dataset, each with the gradient-accumulation fix: today's recipe (RLCD, options in their order), soft cross-entropy, and option shuffling. Three seeds per arm.
  - Each run is scored on the dataset's held-out split through its exported ONNX graph, against the teacher's gold labels.
  - Per item, take the share of an arm's seeds that answer correctly. For each variant, take the mean per-item difference from today's recipe, with a 97.5% percentile bootstrap interval over items (two variants, Bonferroni).
  - A variant is adopted only if that interval's lower bound is above -0.01, its point estimate is at least 0, and its expected calibration error (15 equal-width bins of the calibrated top probability, laya's and aiagent's bin edges, `statlib.ece_15bin`; averaged over the arm's seeds) is not worse than today's by more than 0.02: the upper bound of the 97.5% item-bootstrap interval of the difference is at most +0.02. Both margins are fixed before any run. Otherwise today's recipe stays. If both variants pass, the report shows both and the operator chooses; their combination is not measured.
  - Option shuffling also reports how often the student answers the first listed option, against how often the teacher does (laya gh#131).
- **I32 -- What an exception may be** (E2).
  - An exception is admitted for a capability that another version has and that the backend's own version cannot get yet.
  - It is compiled here from that version's source (M1-M3). Whether the main backend's patch series, or devai's own patches (`deploy/vllm-patches/`), apply is recorded with each exception. For `vllm-s1` neither applies: it is stock (E3).
- **I33 -- Where `devai-control` runs** (D16, C1; operator, 2026-10-06, after the fourth readiness review). It runs in `devai-engines`, `devai-laya-trainer` and `devai-operator` only: the containers whose services Slurm jobs or the operator drive (C1 lists them). `devai-router`, `devai-slurm` and the supporting containers (apt-cache, pipelock, the library's) have none; their processes are not bus services, no engines-page or `devai-jobs service` verb reaches them, and they are restarted as containers. D16's text stays as recorded; this is its reading.

## Open questions

1. Does SGLang 0.5.16 build and serve with nvcc 13.1 on sm120? -- Phase 4 step 4a answers it. The rule for a no is already decided: drop or upgrade (M2).
2. Does slurmd run in an unprivileged container with `proctrack/linuxproc`, now that job scripts only send bus commands? -- Phase 5 step 1 answers it; the measured privileged setup is the fallback. *(Partly answered, 2026-10-07: all of devai-slurm runs unprivileged in podman's default user namespace, with `CgroupPlugin=disabled` in `cgroup.conf` -- without it slurmd stops at start, failing to set a systemd scope. Still open: the `--pid=host` part, which `tests/slurm/test-slurm.sh` checks.)*
3. The sleep fast path: how long sleep and wake take, how much VRAM a sleeping vLLM or SGLang process keeps, and whether sleep support costs speed or memory while serving. -- Phase 8 step 1 measures it; `sleep_mode=false` by default is the fallback.

4. ~~What are the SGLang keep thresholds (Phase 15)?~~ Decided: I16.
5. ~~Short-circuit design (Phase 16)?~~ Decided: I21.
6. ~~Lanes design (Phase 19)?~~ Decided: I22.

None of them blocks coding: questions 1-3 are the first steps of their phases, each with a fallback, and the rest are decided.

### Left for implementation (2026-10-06)

The fourth readiness review's fixes were checked one by one: 70 of 103 were complete. These are the rest, mostly a second place that still says the old thing. Each is settled when its phase is coded; the review notes are not in the tree. A regression check of that fix round was stopped by the operator before it finished.

| Phase | Item |
|---|---|
| 7 | `scripts/ops/operator-up.sh` must keep handing devai-operator the lab's host context that Phase 6 passes through Make (`DEVAI_HOST_IP` and the rest). |
| 8 | **Phase 8 step 1 has no decision rule for `sleep_mode`'s default** (Open question 3): fix one before measuring; the fallback (sleep only for a launch a hold or a pending job names) stays. |
| 8 | A test that an unstamped SGLang cell (empty `kv_cache_dtype`) passes C2's check. |
| 8 | The controller's `POST /devai/v1/jobs/{id}/fail` writes job.json and belongs in the single-writer rule. |
| 8, slurm.md | slurm.md Sec. 4 still says every model process starts with sleep support (Phase 8: per launch, `sleep_mode`), and its sleeper retry leaves out "stop the new process first". |
| 8, slurm.md | slurm.md Sec. 3 and CLAUDE.md still say `devai-control` runs in each container (I33). |
| 9 | A test that the launch-time `sleep_mode` fallback is built. |
| 9, slurm.md | slurm.md Sec. 7's vLLM route list is narrower than C5's. |
| C1, 10 | C1's `lost` should cover a run devai-control could not adopt; Phase 10's exit criterion for `make operator-up` during a bench task. |
| C8 | The GPU-actions bullet omits Yielding's held-job exclusion; the bench-submission sentence's registry clause belongs to the action case. |
| 10 | bench.sh has no source for `HF_TOKEN_FILE`'s host path when run from the page. |
| 10 | A late-granted hold's `truncated` should be marked apart in the cache row (`late_hold`). |
| 10 | The conversion row lacks quant-local's script (4h). |
| 12, slurm.md | slurm.md Sec. 12 still names one fixed lab port pair; each lab has its own. |
| 13 | `devai-jobs prune` (C4 retention) is missing from Phase 13's devai-jobs deliverable. |
| 14 | docs/laya-trainer.md should name `runs/<job>/run.log`. |
| 15 | The freeze list misses the router's `probeCachePathByBackend` SGLang entry and the `bench-sglang`, `test-sglang` and `test-probe-sglang` targets. |
| 15 | I16's "any other outcome is a freeze" must exempt the new void outcome (a vLLM-side failure). |
| 20 | The `index/` "derived only" lines contradict the candidate cache kept there; the library can reach the router on devai-lab-egress, which one line denies. |
| 23 | The compose row mounts the `vllm-s1` probe cache as a single file; Phase 9 mounts `deploy/` as a directory. |
| 23 | Step 1's launch list has no KV-cache dtype. |
| 23 | The `vllm-s1` load waits a fixed 600 s instead of the load's `timeout_s`. |
| 23, I27, slurm.md | Two places say every not-serving answer starts a switch with `Retry-After`; with the breaker open it does neither. |
| slurm.md | Sec. 8 still says a re-run "needs the model", not "names the launch". |
| 23 | Kept by decision, not open: a chat request cancels a loading `vllm-s1` job ("chat wins", operator, 2026-10-06). |

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
| Sec. 3 Containers and images | `devai-slurm`, `devai-engines`, `devai-laya-trainer`, `devai-operator`, `devai-router` | 1-4 (images), 5 (`devai-slurm`), 6 (`devai-operator`), 8 (supervisord and `devai-control` in engines and trainer), 23 (`vllm-s1` in `devai-engines`, the student service in `devai-laya-trainer`) |
| Sec. 3.1 `devai-slurm` | image, privileges, processes, Slurm and NATS configuration | 5 |
| Sec. 4 How a job loads a model | bus verbs, what load / unload / sleep mean per engine, the engine job script | 5 (bus, `devai-control` core), 8 (engines, trainer) |
| Sec. 5 Jobs | kinds, licenses, time limits, comments | 5 (submission, scripts), 9 (engine, trainer jobs), 10 (probe, bench, action jobs) |
| Sec. 6 GPU guard | prolog, epilog, sampler, energy | 5 |
| Sec. 7 Router | Slurm client, blocked control routes, holds | 9, 23 (System One on :11440, I27) |
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
| 1 | lab agents -> router | unchanged; :11440 in 23 (I27) |
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
| no messaging between containers | NATS in `devai-slurm`; `devai-control` in devai-engines, devai-laya-trainer and devai-operator (I33) | 5, 8, 10 |
| no check that the card is free | the prolog guard | 5 |
| the router holds the podman socket and recreates containers | only `devai-operator` holds the socket; the router is a Slurm client and follows engine state on the bus | 9 |
| laya busy hold in the router | trainer jobs | 8, 9 |
| bench clean slate and deadline in `bench-sync.py` | bench runs as jobs, with a time limit and a hold | 10, 14 |
| probers start engine containers | probe jobs that load through the bus | 10, 14 |
| results scattered | slurmdbd + `jobs/results/` + the bus log + `devai-jobs` | 5, 13 |
| operations are Makefile targets in a host shell | `devai-operator`, a web service running devai's scripts as jobs; the Makefile stays for development | 6, 7, 10, 12 |
| the stack assumes an NVIDIA GPU | the same images with or without one | 11 |
| vLLM 0.28.0 + HyperQwen | vLLM 0.30.0 + HyperQwen's current series (E1) | 4 (step 4v) |
| SGLang's future undecided, with stale claims in the docs | a measured keep-or-freeze decision; the docs corrected | 15 |
| Claude Code's throwaway requests load models | answered in the router, off by default | 16 |
| an unprobed model has no estimate and no parser | vendor recipes and a labelled estimator | 17 |
| a failed launch needs a manual post-mortem | a diagnosis, and one recorded auto-fix retry in probes | 18 |
| one port per backend | an optional lanes port | 19 |
| no library service | the library service for aiagent | 20 |
| laya 0.3.20 | laya 0.3.28, with upstream's gradient-accumulation fix (E6) | 22 |
| DiffusionGemma dropped; students only inside aiagent's process | one System One port: students on the CPU, DiffusionGemma on `vllm-s1` in turns; reads on the teacher measured (E3-E5) | 23 |
| lab containers have the GPU | no GPU device in the lab | 14 |

### End state

| Image | Built from | Containers |
| ----- | ---------- | ---------- |
| `debian:trixie-slim` | pulled, digest-pinned | -- |
| `devai-engines` | trixie-slim + host builds | `devai-engines`: one permanent container whose supervisord keeps the engines running; from Phase 23 also `vllm-s1`, stock vLLM 0.31.0 for DiffusionGemma's System One, in its own environment (E3, I29) |
| `devai-slurm` | trixie-slim + Debian's Slurm 24.11.5, MariaDB, NATS | `devai-slurm` |
| `devai-operator` | trixie-slim + Debian's Apache, mod_wsgi, Flask; Docker's compose plugin; CUDA 13.1 compiler; Go, Rust, uv | `devai-operator`, which also runs the log follower (I20) |
| `devai-router` | empty image + host-built static binary | `devai-router` |
| `devai-laya-trainer` | `devai-base` | `devai-laya-trainer`, which from Phase 23 also serves the laya students on the CPU (I28) |
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
| 4 | `devai-engines` | 3 (its first step, 4v, needs nothing) | 5, 6, 7 |
| 5 | `devai-slurm` and the bus | -- (it uses Phase 2's Debian pin once that exists) | 1-4, 6, 7 |
| 6 | `devai-operator`: image, container, direct actions | 2, 3 (`scripts/host-gpu.sh`), and Phase 5's jobs directory (`/var/cache/devai/jobs`, created by the operator, E9: its runs write under it) | 4, 5 |
| 7 | Recipes into scripts | 6 | 4, 5, 8 |
| 8 | Engines and trainer under `devai-control` | 4, 5, 17a (the generated recovery file and its readers, I25); its step 1 needs only 4v, and runs before 4e retires `vllm-devai` | 7 |
| 9 | Router as Slurm client | 6 (`scripts/ops/probe-ollama.sh`; the log follower in devai-operator, I20), 8 | 7 |
| 10 | Workloads and operator actions as jobs; the re-bench | 7, 9 | -- |
| 11 | Running without a GPU | 10 | 12, 13 |
| 12 | Lab and host setup in the web | 6 | 11, 13 |
| 13 | History | 10 | 11, 12 |
| 14 | Cleanup and documentation | 10, 11, 12, 13 | -- |
| 15 | SGLang: keep or freeze | 4v (must finish before 4a) | 1-3, 5-7 |
| 16 | Router short-circuit | -- (16a early) | any |
| 17 | Catalog and fit tooling | -- (17a before 8, I25, and best before 4e and 4g; 17b's calibration after 4 and 10) | 17a beside 1-3, 5-7; 17b any time |
| 18 | Launch-failure diagnosis | 8, 10 | 11-13, 19 |
| 19 | Lanes on one port | 9 | 10-13, 18 |
| 20 | Library service | 2, 3, 6, 7 (20.1's `scripts/ops/operator-up.sh`, the rewritten systemd unit and its registry entries; 12 for the lab action's mount) | 4, 5, 8-19 |
| 21 | Loose ends | -- | any |
| 22 | laya 0.3.28 | -- (aiagent's release first, E6) | any |
| 23 | System One | 4, 8, 9 (step 1 needs only the `vllm-s1` build, with 4c's `Dockerfile.engines` and 4d's launcher written for it if 4a-4b are not done yet, and Phase 5's jobs directory (`/var/cache/devai/jobs`, created by the operator, E9: step 1 writes `attempts.jsonl` under it); step 2 only the trainer image; step 5 only 4v; its registry entries Phase 7, its jobs, GPU action and holds Phase 10) | 10-22 |

The critical path is 1 -> 2 -> 3 -> 4 -> 8 -> 9 -> 10 -> 13 -> 14. Phase 5 runs beside it and must be done before 8, Phase 6 before 9, and Phase 7 before 10. Phase 4's first step, 4v (vLLM 0.30.0, E1), needs nothing and comes before Phase 15, which runs beside 1-3 and must be done before 4a; 17a must be done before 8. Phases 15-22 are outside slurm.md's scope; Phase 23 adds a service to two containers and a router port, which slurm.md records.

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

**Commands** are request/reply on `devai.<container>.<service>.<verb>`. **Events** are published on `devai.events.<container>.<service>` by `devai-control`, on every state change and every 30 s as a heartbeat. On `actions` and `trainer`, the heartbeat also carries every run the service still keeps (until 10 minutes after its end). A refused request changes no state and publishes no event; its reply reaches only the sender. The guard publishes `devai.events.slurm.guard`. A subscriber treats a state older than 90 s as unknown.

| Container | Services | `devai-control` kinds |
|---|---|---|
| `engines` | `ollama`, `vllm`, `sglang`, `vllm-s1` (Phase 23: the engine and its System One server, one service, I29); `fake` in Phase 5's engines test container | engine; `fake` |
| `laya-trainer` | `trainer`; `controller`; `systemone` (Phase 23, I28) | trainer; program; program |
| `operator` | `actions`; `apache` | actions; program |

**Users and permissions.** Each user's password is a 0600 file (C7). Every client sets its own inbox prefix (`nats.CustomInboxPrefix`), so one user cannot read another's replies.

| User | Used by | May publish | May subscribe |
|---|---|---|---|
| `router` | the router | `devai.*.*.state`, `devai.engines.*.unload` (I9) | `devai.events.>`, `_INBOX.router.>` |
| `jobs` | job scripts, prolog, epilog, health check | `devai.engines.*.load`, `devai.engines.*.unload`, `devai.engines.*.sleep`, `devai.engines.*.lease`, `devai.engines.*.state`, `devai.laya-trainer.trainer.train`, `devai.laya-trainer.trainer.cancel`, `devai.laya-trainer.trainer.lease`, `devai.laya-trainer.trainer.state`, `devai.operator.actions.run`, `devai.operator.actions.stop`, `devai.operator.actions.lease`, `devai.operator.actions.state`, `devai.events.slurm.guard` | `devai.events.>`, `_INBOX.jobs.>` |
| `operator` | devai-operator's app, its actions (probers), `devai-jobs` | `devai.>` | `devai.events.>`, `_INBOX.operator.>` |
| `control-engines`, `control-laya`, `control-operator` | `devai-control` in that container | `devai.events.<container>.>` | `devai.<container>.>`, with `allow_responses: {max: 1, expires: "15m"}` (the plain `true` form allows a reply for 2 minutes only, nats-server's `DEFAULT_ALLOW_RESPONSE_EXPIRATION`, shorter than a cold `load`; Phase 5 checks that default in the 2.10.27 source) |
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
| `state` | `job` (for `actions` and `trainer`) | the service's state; for `actions` and `trainer`, that job's run |
| `start`, `stop`, `restart` | -- | the same on the supervisord program; then its state. On an engine or the trainer, see "Supervisor verbs" below |
| `reload` | -- | re-read the program's configuration; restart it if it changed. On an engine or the trainer, see below |
| `load` | `launch` (C2), `timeout_s` (default 600, at most 840; a larger value is refused with `bad_request`, so the reply always falls inside the 15-minute response permission above) | replies once serving, or `load_failed` with the log tail. The same launch already serving for the same job is a no-op; the same launch sleeping is woken |
| `unload` | `target` (`active`, the default, or `sleeper`), `force` (default false), `timeout_s` (default 60) | unloads the loaded model, or the sleeping one; replies once the program has exited and NVML's used memory is back to its level before that load (+256 MiB); with `target: sleeper`, once the program has exited and used memory has fallen, from the reading taken when the unload began, by at least the sleeper's `used_mib` minus 256 MiB, since another model may hold the card. A `loading` or `waking` model is unloaded the same way: for vLLM and SGLang the program is stopped; for Ollama the warm request is cancelled, then `keep_alive: 0` is sent and `devai-control` waits until `/api/ps` is empty. The status goes `unloading`, then `idle`, never `failed`, so neither `devai-bus hold` nor the router's breaker reads it as an engine failure; the waiting `load` is answered `load_failed` with message `unloaded`. An `unload` that arrives while the job's model is already `unloading` waits for that same end and gets the same reply. An `unload` from a job whose model is already gone (`idle`, or the service names no job) replies ok with the state: a no-op, like `cancel` once a run has ended |
| `sleep` | -- | vLLM `/sleep?level=1`, SGLang `/release_memory_occupation`; `unsupported` on Ollama and for any model whose launch has `sleep_mode` false (C2). Refused with `bad_request` unless this job's model is `serving`: the job then ends, and its epilog unloads the model, since the state is not `sleeping`. It replies once the engine has slept (vLLM `/is_sleeping` true, SGLang's call returned) and `used_mib` has been read; until then the status stays `serving`, and it becomes `sleeping` only with that reply. An `unload` from the owner (or with `force`) that arrives during a sleep waits for the sleep to end, then unloads the model as the sleeper (as with `target: sleeper`), and the `sleep` is answered `bad_request` with message `unloaded`. A sleep that fails leaves the model `serving`, and its reply carries the error. The sleeper stays until a load wakes or replaces it, or until an `unload` with `target: sleeper` (I9) |
| `lease` | -- (published, no reply) | renews the job's lease on its model, its trainer run or its action run; after 60 s without one, the model is unloaded, the trainer run cancelled (reason `lease`), or the action's process group stopped as `stop` with `TERM` and a 30 s grace would. The 60 s are counted from the later of the job's last `lease` and the moment its `load` (a wake included), `train` or `run` request was answered. No lease is needed while one of these is in progress: the job scripts start their hold only after the reply (C3), and a cold vLLM load takes minutes. A load whose job stops leasing still runs to its end, and the model is then unloaded 60 s after it reached `serving` unless a lease arrives. A sleeper has no lease (`sleep` row) |
| `train`, `cancel` | `ft_job`, `reason`; `cancel` also `force` and `timeout_s` (default 60) | start or stop one laya run. `train` starts the run's supervisord program and replies once it has started (`running`); `busy` while another run goes. `cancel` without `ft_job` stops the run of the sending job, or with `force` whatever run is going. It stops the program (TERM to its process group, SIGKILL after 30 s) and replies only once the program has exited, its record is final (Phase 8: `jobs.py`'s `finalize`, from the exit code and the cancel's `reason`) and NVML's used memory is back to its level before the run (+256 MiB). `reason` is optional; when given it is one short word (`^[a-z_]{1,32}$`), kept in the record's events; `time_limit` makes the record `failed` with `time_limit`, any other reason, or none, `cancelled`. A `cancel` while the run is `stopping` waits for that same end; one with no run going replies ok with the state, a no-op |
| `run`, `stop` | `run`: `action`, `params`, `kind` (the kind of the job sending it); `stop`: `signal` (`INT` or `TERM`), `grace_s`, and `force` with `run_job` (the job whose run to stop) | `run` starts a registered action (C8) for the job and replies once it has started (`running`); it is refused (`bad_request`) when the action's `job_kind` is not `kind`, or when a `stop` for that job came first. The run's end is an event: `exited` with `exit_code`, or `lost`. `stop` signals the run's process group and replies at once; SIGKILL follows after the grace |

A loaded model belongs to the job that loaded it, and a trainer or action run to the job that started it. `unload`, `sleep`, `stop`, `cancel` and `lease` from another job are refused with `not_owner`. The exception is `unload`, `stop` and `cancel` with `force`, which the guard and the operator send, and the router for a sleeper's `unload`.

**Supervisor verbs.** The ownership rule's `stop` is the `actions` verb. `start`, `stop`, `restart` and `reload` come only from user `operator` (the engines page, `devai-jobs service`; the permissions above keep `jobs` and `router` from sending them) and are not checked for ownership. On `actions`, `stop` is only the actions verb, and the other three do not apply. On a `program` service they act on its supervisord program. On an engine service (`ollama`, `vllm`, `sglang`, `vllm-s1` from Phase 23, and `fake` as `vllm`) they mean:

- `stop`: a forced `unload` of the loaded model, also while it is loading or waking, and of the sleeper. On `ollama` it also stops `ollama serve`, and the status becomes `stopped`; on the others it becomes `idle`. On `vllm-s1` the forced `unload` stops both programs, the System One server and then the engine, as its `unload` does (I29). `job` becomes null. The owning job ends: its hold sees the service no longer serving it and exits 0, or, while the model was still loading, its `load` is answered `load_failed` (`unloaded`) and it exits 1. Its epilog's `unload` is then a no-op. A `load` on a `stopped` Ollama starts `ollama serve` again, as a load with other server settings restarts it (I8): to keep jobs off the card, the operator holds them in Slurm.
- `start`: on `ollama`, starts `ollama serve` with no model, giving status `idle`. On the others it replies with the state: only a `load` starts a model process.
- `restart` and `reload`: refused with `busy` while a model is loaded, loading, waking or sleeping. Otherwise on `ollama` they restart `ollama serve`, and on the others they reply with the state.
- On the trainer, `stop` is a `cancel` with `force` and reason `operator`; `start`, `restart` and `reload` are as on `vllm`, with "a run going" in place of "a model loaded".

**State.** `{"container", "service", "status", "job", "launch", "since", "build", "sleeper", "error"}`:

- `status` is one of `idle`, `loading`, `serving`, `sleeping`, `waking`, `unloading`, `failed`, `stopped`. An `actions` service keeps one state per job instead -- `running`, `exited` with `exit_code`, or `lost` (a restarted `devai-control` adopted the run but cannot collect its exit status) -- which `state` with that `job` returns; an ended run's state is kept 10 minutes.
- The trainer's `status` is `idle`, `running` or `stopping` (a `cancel` is under way), with `job` and `ft_job` naming its run (null when idle); `launch` is null. A run's end is an event, `ended`, with `job`, `ft_job`, `outcome` (the record's final status: `succeeded`, `failed` or `cancelled`) and the record's `error`. Like `actions`, it keeps one state per job, which `state` with that `job` returns: `running`, `stopping`, or `ended` with `outcome`, kept 10 minutes. A run it no longer knows -- `devai-laya-trainer` restarted, and `devai-control` failed the record with `trainer_stopped` at its start (Phase 8) -- is gone from `state`; the hold reads that as lost (C3).
- `sleeper` is `{"job", "launch", "since", "used_mib"}` or null. `used_mib` is the card's used memory that `devai-control` read right after the model went to sleep, minus its reading before that model's load. The guard tolerates that much as the sleeper's (Phase 8).
- `error` is `{"code", "message", "log_tail"}` or null; from Phase 18 a failed load's error also carries `diagnosis` (the same object as on the `load_failed` reply), since the router learns of a failed load only from this state, never from the reply.
- `build` is the engine build id, read from `/etc/devai/engine-builds.json`, which `Dockerfile.engines` writes with the same content hashes as its `devai.engine.<backend>` labels (Phase 4, step 4c). The router compares it with the probe cache's stamp, without podman.

**Adoption after a restart.** `devai-control` records in its state file every engine program it starts, with its job, launch, start time and status (`loading`, `waking`, `serving`, `sleeping`). After a restart it adopts each program supervisord still runs. A program recorded `loading` or `waking` is watched to its end as the original load would have been -- `/health`, a fail-fast exit, and the rest of `timeout_s` counted from the recorded start -- and `devai-control` then publishes `serving` or `failed` for the same job. A program supervisord no longer runs is recorded `failed`. The reply to the original `load` is lost with the old process; the job's `devai-bus request` takes the event instead (C3).

**Memory readings.** `devai-control` reads the card with `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`; CDI puts `nvidia-smi` into every container that gets the GPU. Without a GPU it skips every memory check.

### C2 -- Launch settings (the `launch` object of a `load`)

| Field | Engines | Type and check | Today's source |
|---|---|---|---|
| `backend` | all | `ollama`, `vllm`, `vllm-s1` or `sglang` (`fake` for Phase 5's fake engine); must match the service. `vllm-s1`'s engine is vLLM, so it takes the `vllm` fields (I29) | -- |
| `model` | all | a catalog name passing the router's `isSafeModelName`; for vLLM and SGLang, `<store>/<model>` must exist | the request |
| `ctx` | all | int, 1024 to `MAX_CONTEXT_LEN` | `launchConfig.MaxContext` |
| `kv_cache_dtype` | all | the raw probe-cell stamp; empty is allowed and means what it means today. vLLM: empty (= `fp8`, the legacy hardcode), `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`. SGLang: empty (= no `--kv-cache-dtype` flag, the engine's default), `auto`, `fp8_e4m3`, `fp8_e5m2` (no bare `fp8`, which SGLang refuses). Ollama: empty or `f16` (= no `OLLAMA_KV_CACHE_TYPE`), `q8_0`, `q4_0`. The builders keep today's decoding of empty (`vllmEntrypoint`, the SGLang builder, `ollamaDynamicEnv`) | `KVCacheType` |
| `flash_attention` | ollama | bool or null | `FlashAttention` |
| `num_gpu` | ollama | 999 (GPU) or 0 (a host without a GPU) | `warmLoadOllama` |
| `mem_fraction` | vllm, sglang | 0.10 to 0.98 | `MemFraction` |
| `max_num_seqs` | vllm, sglang | int >= 0; 0 omits the flag | `MaxNumSeqs` |
| `tool_parser`, `reasoning_parser` | vllm, sglang | `^[a-z0-9_]{1,64}$` or empty | `ToolParser`, `ReasoningParser` |
| `mtp` | vllm, sglang | null, or the catalog's `mtp:` block (`configSpeculative`) | `Speculative` |
| `sleep_mode` | vllm, sglang | bool; default true (D15); false for `vllm-s1` until Phase 23 step 3 has measured sleep on it (I29) | -- |

Some settings are not in the request: `devai-control` resolves them from files mounted read-only (I1):

- recovery flags and environment, by backend and model: first `deploy/recovery-flags.json` (hand-curated; that file's own contract, also named C2), then `deploy/recovery-flags.generated.json` (machine-made, I25); both re-read at every load (I18, I25; `devai-control` from Phase 8);
- parser plugin paths, from `deploy/vllm-plugins.json`.

All three come from the `deploy/` directory, mounted read-only (C6). A recovery flag that sets a launch setting overrides the C2 field, since the last flag wins: 9 of the 14 curated entries set `--gpu-memory-utilization` or `--mem-fraction-static`, the Qwen3.8 rows among them. Machine-made flags never set one (I18; Phase 17).

The router and the probers compute the fields as they do today (`computeLaunchConfig`; `host_scaled_fraction` for a probe band). `gpu-arbiter -print-load <backend> <name>[::mtp][@ctx]` prints the body of the `load` request the router would send, `{"launch":{...}, "timeout_s":...}`, without `v` and `job` (which `devai-bus request` adds, C3); `timeout_s` is the router's `HEALTH_TIMEOUT_SECONDS` (default 600), and `-print-load` refuses a value above 840 (C1); an engine job carries it as `DEVAI_LOAD`, and tests use it.

Two launches are the same when backend, model, ctx, `kv_cache_dtype` and `mtp` are equal -- today's `currentModel` / `currentContext` / `currentSpec` comparison. For Ollama, a bare `<name>` matches whatever context is loaded; only `<name>@<ctx>` asks for a context (today's `ctxPinned` rule).

### C3 -- Jobs

**Submission.** `POST /slurm/v0.0.42/job/submit`, with the headers `X-SLURM-USER-NAME: devai` and `X-SLURM-USER-TOKEN: <JWT>`. The JWT is HS256, `{"sun":"devai", "iat":<now>, "exp":<now+300>}`, signed with `jwt_hs256.key`. The key could sign a token for any user, root included, so slurmctld accepts a job only from the Slurm user `devai`: `deploy/slurm/job_submit.lua` (`JobSubmitPlugins=lua`, Phase 5) refuses every other submitter, and root has no association. The field names below are checked against 24.11's own spec in Phase 5, step 2. *(Checked 2026-10-07, against the spec dumped from this slurmrestd, `deploy/slurm/openapi-v0.0.42.json`, which `pkg/slurmrest` follows: the body wraps the job, `{"job": {...}}`, with `job.script`; `time_limit` and `kill_warning_delay` are objects `{set, infinite, number}`; `kill_warning_signal` is a signal name (`INT`, `USR2`) and `kill_warning_flags` is `["BATCH_JOB"]`; the list paths end in a slash (`/slurm/v0.0.42/jobs/`, `/slurmdb/v0.0.42/jobs/`); `job_state` is a list; slurmdbd returns `comment` as `{administrator, job, system}`; a hold that devai sends over REST reads `JobHeldAdmin`. The names below stay as written.)*

Every job sets:

- `script`: two lines, `#!/bin/bash` and `exec /usr/local/lib/devai/jobs/<kind>.sh`. The job scripts ship in the `devai-slurm` image, so they change with it, not with the router.
- `name`: `engine-<backend>`, `trainer`, `probe-<backend>`, `bench-<task>` or `action-<id>`.
- `licenses`, `time_limit` and the warning signal, per kind (table below).
- `requeue`: false. A job whose node loses it (a `devai-slurm` restart, a host reboot) ends NODE_FAIL and never runs again (`JobRequeue=0`, Phase 5). The router submits again on demand; actions and bench tasks are re-run only by their submitter.
- `current_working_directory`: `/var/cache/devai/jobs/results`.
- `standard_output` and `standard_error`: `/var/cache/devai/jobs/results/%j/job.log`.
- `environment`: `PATH=/usr/local/bin:/usr/bin:/bin`, `DEVAI_BUS_URL=nats://127.0.0.1:4222`, `DEVAI_BUS_USER=jobs`, `DEVAI_BUS_PASSWORD_FILE=/run/devai/bus/jobs.pass` (the copy devai can read, I26), plus the kind's own (`DEVAI_BACKEND`, `DEVAI_LOAD`, `DEVAI_FT_JOB`, `DEVAI_ACTION`, `DEVAI_PARAMS`). A job gets only the environment its submitter passes (measured). `DEVAI_LOAD` is the body of the `load` request, as `gpu-arbiter -print-load` prints it: `{"launch":{...}, "timeout_s":...}`, with `launch` as in C2 and without `v` and `job`, which `devai-bus request` adds.
- `comment`: JSON, under 1 KiB: `{"v":1, "kind", "backend", "model", "ctx", "mtp", "requester", "git", ...}`. Per kind:
  - trainer: `ft_job`;
  - probe: `action`;
  - bench: `run`, `task`, `attempt`, `rerun_of`, and always `backend`, `model`, `ctx`, `mtp` (from the bench plan's target, C8), so its C5 hold is accepted and a held re-run names its launch (Phase 9 case 4);
  - bench, probe and action: `hold_key`, 32 random hex characters that `devai-jobs` generates at submit; C5's hold routes require it as `X-DevAI-Hold-Key`;
  - action: `action`, `gpu` (true for a GPU action, C8; the router reads the license from the job itself, Phase 9), and `backend`, `model`, `ctx`, `mtp` when its entry sets `comment` (C8).

| Kind | Licenses | Time limit and warning signal | Its script |
|---|---|---|---|
| engine | `engine` | none | `load` `$DEVAI_LOAD`; hold. TERM -> `unload`; USR1 -> `sleep` |
| trainer | `engine` | `LAYA_MAX_HOLD_S`, rounded up to whole minutes, Slurm's unit (0 = unlimited); `B:USR2@30` | `train`; hold. TERM -> `cancel`; USR2 -> `cancel` with reason `time_limit` |
| probe | `engine` | 60 min | `probe.sh`, one model (one model and band for Ollama): lease the model the prober loads with this job id (`devai-bus hold devai.engines.$DEVAI_BACKEND --job $J --lease-only`); `run` the per-model probe action `probe-<backend>-model` (C8) in `devai-operator`, and hold the run. TERM -> `stop` the action and wait for it, then `unload` the model (the job owns it) |
| bench | `bench` (I5) | 32 min; `B:INT@120` | `bench.sh`: `run` the bench-task action and hold the run. INT -> `stop {signal: INT, grace_s: 110}` and wait for the run to end; TERM -> `stop {signal: TERM}` and wait |
| action | none; `engine` for a GPU action (`gpu: true` in its registry entry, C8) | per action, the entry's `time_limit_min` (C8); none when the entry sets none, which a GPU action may not | `action.sh`: `run` and hold the run. TERM -> `stop` and wait |

These time limits are defaults; a C8 registry entry's `time_limit_min`, when set, wins for the jobs its action submits. A GPU action must set one: the partition has no maximum, and a GPU action without a limit could hold `engine:1` indefinitely.

`devai-bus request <subject> <json> --job N [--timeout D]` sends one command:
- It sends `<json>` with `v: 1` and `job: N` added, and prints the reply on stdout. `--job` is required; the operator's tools, which act for no job, pass `--job 0`.
- A body that already carries `job` or `v` is refused (exit 1), so the two never disagree.
- `--timeout` is the client's wait for the reply, separate from a verb's `timeout_s`; a request that times out exits 1.
- For `load` (a wake included), it subscribes to `devai.events.<container>.<service>` before it sends. A `serving` event for its job counts as an ok reply, and a `failed` event for its job as `load_failed` (exit 1), once no reply has followed it within 2 s (`ReplyGrace`). So a reply lost with a restarted `devai-control` (C1, "Adoption after a restart") or with a bus restart does not leave the job waiting out its `--timeout`, while a reply that arrives is what answers: `devai-control` publishes the event as the load ends, and without the grace the event answered every load of the first live run (2026-10-09), so the 15-minute response permission was never used.

`devai-bus hold <subject> --job $J --parent $$` holds a model, a trainer run or an action run:
- It subscribes to the service's events first and then asks for its state, so a run that ended before it started is not missed.
  - On an engine service it exits 0 once the service no longer serves this job (unloaded, gone to sleep, or another job's model), and 3 when it is `failed` for this job.
  - On `devai.operator.actions` it exits 0 when the action exits 0, 1 when it exits non-zero, and 3 when the run is `lost`, or when the service no longer knows a run the hold had seen (a `state` reply, or a heartbeat younger than 90 s, that does not list it: `devai-operator` was recreated). Only when the job has had no run since the hold started does it exit 0 at once.
  - On `devai.laya-trainer.trainer` it exits 0 when the run ends `succeeded` or `cancelled`, 1 when it ends `failed`, and 3 when the trainer no longer knows a run it had seen (lost, C1). When the job has no run, it exits 0 at once. A `cancel` with reason `time_limit` ends the run `failed`, but trainer.sh's USR2 trap sent that `cancel` and is still waiting for its reply, so the job ends with the trap's 0, not the hold's 1.
- It renews the lease every 10 s.
- It survives a bus outage. Its client reconnects without limit (nats.go `MaxReconnects(-1)` with `ReconnectWait` 1 s; nats.go's documented options, unverified here, Phase 5 tests them). The subscription is restored on reconnect, and the hold sends a lease at once. After every reconnect it asks for its service's state again, as at start (`state`, with `job` on `actions` and `trainer`), because core NATS delivers at most once and drops events published while it was away, an action's `exited` among them. The state it gets back decides as at start. A state older than 90 s (C1) is unknown, not an exit reason: the hold keeps holding. Exit 2 is only for a bus it cannot reach before its first `state` reply (it tries for 30 s). If the outage outlasts the 60 s lease, `devai-control` has already unloaded the model or stopped the run, and the hold learns this from that re-query (slurm.md Sec. 11).
- It ignores SIGTERM and SIGINT. On `scancel` and on the time limit, Slurm signals every process of the batch step, not only the script (24.11.5: slurmd's `_kill_all_active_steps` with SIGTERM; `proctrack/linuxproc` signals the whole process tree). The hold must outlive that signal to see its run end. Only `--batch` / `B:` signals reach the script alone. Slurm's SIGKILL after `KillWait` still ends it.
- It dies with its script: it sets `PR_SET_PDEATHSIG(SIGKILL)`, and exits at once if its parent is no longer the `--parent` pid. A hold whose script was SIGKILLed is re-parented to `tini -s` (C6) and so leaves the step's process tree; it would otherwise renew its lease for ever.
- With `--lease-only` it only renews a lease, and ends on SIGHUP from its script.
- `devai-bus request` keeps the default handling of SIGTERM, which Slurm sends to every process of the step, so a TERM trap is never held up behind a long `load`. A USR1 sent with `--batch` reaches bash alone, and bash runs a trap only after the foreground command returns (bash 5.2.37, checked 2026-10-06): its trap would wait until the `load` returns. That is why the router sends USR1 only to a job whose engine is `serving` (Phase 9, case 4), and why `sleep` on a model that is not serving is refused (C1).

If the script dies without its trap (SIGKILL, `devai-slurm` restarting), `devai-control` stops the run 60 s after the last renewal (C1). So no action outlives its job, and the prober of a dead probe job cannot go on loading models outside the engine license.

The action job script, `deploy/slurm/jobs/action.sh`:

```bash
#!/bin/bash
umask 0002
A=devai.operator.actions J=$SLURM_JOB_ID H=
stop() {   # stop replies at once (C1); the job ends only once the run has ended
  devai-bus request $A.stop "{\"signal\":\"$1\",\"grace_s\":$2}" --job "$J" --timeout 5s
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
- `probe.sh` sends kind `probe`. Before `run` it starts `devai-bus hold devai.engines.$DEVAI_BACKEND --job "$J" --lease-only --parent $$ &` for the model the prober loads with this job id. Once the run's hold has returned, on either path (its `stop()` and its normal end), it sends that model's `unload`, `devai-bus request devai.engines.$DEVAI_BACKEND.unload '{}' --job "$J" --timeout 25s` (the job owns it: no `force`; a no-op when the prober already unloaded it, C1), and ends the lease with `kill -HUP`. The epilog sends the `unload` again.

Exit codes, the same for `devai-bus` and for every job script, which passes them through:

- 0: done; a job ended normally, including after a cancel or a stop its own script sent;
- 1: refused or failed (an error reply such as `load_failed` or `busy`, an action that exited non-zero on its own, or a trainer run that ended `failed` on its own);
- 2: the bus could not be reached (for `hold`: only before its first `state` reply);
- 3: the service left its active state on its own (an engine `failed`, or an action run or a trainer run `lost`).

The engine job script, `deploy/slurm/jobs/engine.sh` (slurm.md Sec. 4 shows the same script):

```bash
#!/bin/bash
umask 0002
S=devai.engines.$DEVAI_BACKEND J=$SLURM_JOB_ID
trap 'devai-bus request $S.unload "{}" --job "$J" --timeout 25s; exit 0' TERM INT
trap 'devai-bus request $S.sleep "{}" --job "$J" --timeout 120s; exit 0' USR1   # refused unless serving (C1); the epilog then unloads
devai-bus request $S.load "$DEVAI_LOAD" --job "$J" --timeout 900s || exit $?   # timeout_s is at most 840 (C1, C2)
devai-bus hold $S --job "$J" --parent $$ &   # a lease every 10 s; ignores TERM and INT; exits 0 when the service stops serving this job, 3 on failed
wait $!
```

The trainer job script, `deploy/slurm/jobs/trainer.sh`:

```bash
#!/bin/bash
umask 0002
T=devai.laya-trainer.trainer J=$SLURM_JOB_ID
trap 'devai-bus request $T.cancel "{\"reason\":\"scancel\"}" --job "$J" --timeout 25s; exit 0' TERM INT
trap 'devai-bus request $T.cancel "{\"reason\":\"time_limit\"}" --job "$J" --timeout 25s; exit 0' USR2   # B:USR2@30
devai-bus request $T.train "{\"ft_job\":\"$DEVAI_FT_JOB\"}" --job "$J" --timeout 30s || exit $?
devai-bus hold $T --job "$J" --parent $$ &   # a lease every 10 s; ignores TERM and INT; exits 0, 1 or 3 with the run
wait $!
```

- `cancel` replies only once the run's program has exited, its record is final and the card is back at its level before the run (C1). So the job ends only once the GPU is free, and the epilog's idle check passes after a `scancel` or the time limit.
- The time limit reaches the record through `reason: time_limit`: `devai-control` finalizes the record after the program has exited, with that reason (Phase 8). `run_job` itself only receives SIGTERM.
- If the trap's `cancel` does not return within 25 s, the epilog repeats it (reason `epilog`) and waits up to 60 s.

How the router ends an engine job:

- To unload it, the router cancels it: Slurm sends SIGTERM, waits `KillWait` (30 s), then sends SIGKILL.
- To make it sleep, the router signals the batch script alone, `DELETE /slurm/v0.0.42/job/<id>?signal=SIGUSR1&flags=BATCH_JOB` (`scancel --signal=USR1 --batch`). It does so only for a job whose engine is `serving` and whose launch has `sleep_mode` true (C2): the script is then in `wait`, where a trap runs at once, and the process can sleep. A job whose engine is pending, loading or waking, or whose launch has `sleep_mode` false, is cancelled instead (Phase 9, case 4). The rule is the launch's field, never the backend's name.
- The trap's `unload` waits at most 25 s. The epilog sends `unload` again and waits up to 60 s, so a slow shutdown past `KillWait` is still cleaned up. The epilog skips that `unload` when the service reports `sleeping` for this job. For a trainer job the epilog sends `cancel` instead (`{"reason":"epilog"}` with `--job` = that job and no `ft_job`, so the run of that job; it waits up to 60 s; a no-op once the run has ended, C1).

### C4 -- Files under `/var/cache/devai/jobs`

This is a plain directory the operator creates (D3, E9). A job's directory is shared through a setgid group (I26). The prolog creates `results/<jobid>/` and its `artifacts/` owned by devai, group 0 (the host user's group), mode 2775. `job.log` belongs to devai's subordinate uid and everything else to the host user; every file carries the host user's group.

```
results/<jobid>/job.log          the job script's output (Slurm)
results/<jobid>/result.json      bench, probe and action jobs; its writer per kind is below
results/<jobid>/action.log       an action's output (devai-control in devai-operator)
results/<jobid>/gpu.json         written by the epilog of engine, trainer, probe and GPU-action jobs (C8)
results/<jobid>/guard.json       the prolog's verdict: idle, unloaded, killed or busy, with the holders
results/<jobid>/artifacts/       anything else the job keeps
engines/<backend>/<jobid>.log    a vLLM or SGLang model process's output (devai-control)
engines/ollama/serve.log         ollama serve's output, rotated by supervisord (100 MB x 5)
bus/<YYYY-MM-DD>.jsonl           every bus event, one per line, with its receive time (the bus log)
operator/<run>/                  direct actions: run.json and log (devai-operator)
mariadb/                         MariaDB's data directory
slurmctld/                       slurmctld's StateSaveLocation
```

- Retention: `devai-jobs prune --older-than <days>` (default 90) deletes `results/<jobid>/`, `engines/<backend>/<jobid>.log`, `operator/<run>/` and `bus/<date>.jsonl` older than that. It never deletes `mariadb/`, `slurmctld/`, or anything that belongs to a job still in the queue. It runs daily inside `devai-slurm` (a supervisord program that sleeps a day between runs), and Phase 13's tests pin it.
- `gpu.json`: `{"v":1, "job", "device", "interval_s":5, "started_at", "ended_at", "samples":[[t, used_mib, util_pct], ...], "peak_mib", "mean_mib", "mean_util_pct", "energy_j"}`. The epilog also puts `gpu peak=<MiB> mean=<MiB> util=<pct> energy=<J>` into the job's `AdminComment` (Phase 5 step 6 checks that 24.11 accepts this from an epilog). *(It does not: "Job has already finished", 2026-10-07. `gpu.json` is the record, and Phase 13 reads it there.)*
- `result.json`: `{"v":1, "job", "kind", "status", "started_at", "ended_at", "summary", "outputs"}`.
  - `status` is one of `ok`, `failed`, `truncated`, `cancelled`.
  - `outputs` depends on the kind. Bench: `{model, backend, ctx, task, n_planned, n_scored, score, cache_key}`. Probe: `{model, backend, cells_written}`. Action: `{exit_code}`.
  - Its writer, per kind:
    - action: `devai-control` in `devai-operator` writes it at the run's end, with status `ok`, `failed`, or `cancelled` after a `stop`, and outputs `{exit_code}`;
    - bench: `bench_task.py` writes it (Phase 10);
    - probe: the per-model prober action (`probe-<backend>-model`) writes it to `$DEVAI_RESULTS_DIR` (Phase 10);
    - fallback: for a bench or probe run, `devai-control` writes it only when the run ended without leaving one, with status `failed`, or `cancelled` after a `stop`, and outputs `{exit_code}`. It never overwrites an existing file.
  - Engine and trainer jobs have none: the bus log and, for the trainer, the controller's record keep their outcome.

### C5 -- The router's own API

The router serves `/devai/v1/` on every backend port itself and never proxies that path.

| Method and path | Behaviour |
|---|---|
| `POST /devai/v1/holds` `{job_id, model}` | `model` is written as a client sends it (`<name>[::mtp][@ctx]`). The hold is checked against Slurm: `job_id` must be RUNNING, of kind bench, probe or action, and its comment must name this backend and model (the comment's `model` is compared with the hold's name after the `::mtp` and `@ctx` suffixes are removed; the comment's `ctx` and `mtp`, when present, must match the hold's, and a comment that leaves them out covers every launch of that model); otherwise 403. The request must also carry `X-DevAI-Hold-Key` equal to the `hold_key` in the job's comment (32 hex characters, generated by `devai-jobs` at submit and handed to the job's run as `DEVAI_HOLD_KEY`; C3, C8); otherwise 403. A POST for a job that already has a hold is refused 409 unless it names the same launch; a repeated POST for the same job and launch renews the hold (below) and answers at once while the engine serves. Starts or keeps the engine and answers 200 with the GPU record once it serves (long-poll up to `HEALTH_TIMEOUT_SECONDS`). 409 when another active hold needs a different launch; 503 + `Retry-After` when a trainer, probe or GPU-action job holds the GPU or is PENDING and not held (any job other than an engine job that holds or requests the `engine` license, Phase 9 case 3). A new hold is checked by Ensure like any request, so it never starts or keeps an engine while such a job waits; a hold already active is kept until its job ends |
| `DELETE /devai/v1/holds/{job_id}` | 204 when `X-DevAI-Hold-Key` matches the job's `hold_key`; otherwise 403 and the hold stays. A hold is also dropped when its job leaves RUNNING, checked on every decision and every 30 s. Holds live only in the router's memory: a client keeps its hold by POSTing it again every 30 s, and the router drops a hold not renewed for 90 s, so after a router restart each hold returns with its client's next renewal |
| `GET /devai/v1/gpu` | the GPU record: the holder's job, kind, backend, launch and state; the holds; the sleeper; the pending jobs and the launches they name. Job comments and hold keys are never shown |
| `GET /devai/v1/metrics` | on the vLLM and SGLang ports, the serving engine's own `/metrics` for that backend, proxied read-only (GET only, the body passed through unchanged); 503 `engine_not_serving` when that backend's engine is not serving. It never starts, keeps or switches an engine. On every other port, 404 (Ollama has no `/metrics`). The bench reads engine metrics here (Phase 10), because bench-task containers are not on `devai-net` |
| `GET /devai/v1/ready` | 200 only when slurmrestd and the bus answer, the probe caches are loaded and the model stores are mounted; 503 naming what is missing. `/health` stays a liveness answer (Phase 9; from odysseus RT-6) |

The router also blocks the engines' control routes on every listener, before the proxy, as `makeMutationGuard` blocks Ollama's store-changing paths today. It answers 403 with a body naming `devai-control`.

- vLLM: every route that `VLLM_SERVER_DEV_MODE` turns on (`/sleep`, `/wake_up`, `/is_sleeping`, `/collective_rpc`, weight updates and the rest), plus the state-changing routes registered without it (`/scale_elastic_ep`, which 0.31.0 registers for every generating model) and those other settings turn on (the LoRA load and unload routes, `/start_profile`, `/stop_profile`, `/fault_tolerance/apply`).
- SGLang: every route its source marks `@auth_level(AuthLevel.ADMIN_OPTIONAL)` or `ADMIN_FORCE` (39 in the source read on 2026-10-06; step 1 reads 0.5.16's). That covers the memory and weight routes (`/release_memory_occupation`, `/resume_memory_occupation`, `/update_weights_from_disk`, ...) and also `/configure_logging` (it writes request dumps to a path the client names, as root), `/abort_request`, `/pause_generation`, `/continue_generation`, `/start_profile`, `/stop_profile`, `/flush_cache`, `/set_internal_state`, `/dumper/*`, `/hicache/*`, `/load_lora_adapter`, `/unload_lora_adapter` and `/slow_down`.

The exact lists are read in Phase 9 step 1 from the vLLM 0.30.0 (E1) and SGLang 0.5.16 sources, and pinned by a test. `vllm-s1` (I29) needs no list: no router listener proxies to its engine on 11441, and the System One port forwards only `POST /v1/systemone` to 11442, builds `GET /v1/models` itself, and answers 404 to any other path outside `/devai/v1/` (I27).

### C6 -- Ports and mounts of the permanent containers

| Container | Ports | Mounts |
|---|---|---|
| `devai-slurm` | 6820 (slurmrestd) and 4222 (NATS) on `devai-net`; nothing published | podman's default user namespace, so its root is the host user, with its default capabilities (I26); `tini -s` as its entrypoint, since under `--pid=host` tini is not PID 1; a tmpfs at `/run/devai`; `/var/cache/devai/jobs` read-write; its C7 files; the bus passwords of every user, to render the server configuration. Inside, these appear as root's 0600 files; only the jobs password is copied for devai (I26) |
| `devai-engines` | Ollama 11434, vLLM 11435, SGLang 11436 on `devai-net` -- one port per engine, since they share a network namespace; from Phase 23 `vllm-s1` 11441 and its System One server 11442 (I29) | the Ollama store at `/root/.ollama` read-write; the vLLM and SGLang stores at their host paths, read-only (`vllm-s1` reads the vLLM store); the FlashInfer cache volumes, one per backend, `vllm-s1`'s included; `jobs/engines/` read-write; `${DEVAI_REPO}/deploy` at `/etc/devai/deploy` and the plugin directory, read-only -- directories, so a file replaced by rename is seen, by absolute path (I25); its bus password; its processes run as root (I26) |
| `devai-laya-trainer` | the controller on 11434 and, from Phase 23, the student service on 11443 (I28), on `devai-net` | `/var/cache/devai/laya` at `/laya` (the student service only reads `runs/`); its bus password |
| `devai-operator` | `${OPERATOR_PORT:-8443}` published on all interfaces (O6) | runs as `devai` with `--userns=keep-id` (I19); the host's podman socket read-write; the repository, `/var/cache/devai`, the lab's home volume (`HOME_VOLUME`) and the host user's home (`HOST_HOME_DIR`) at their host paths (path identity), with `HOME` set to `HOST_HOME_DIR` in compose, so `~/.devai`, `~/.config/devai` (pipelock's CA key, C7's directories), `~/.cache` and `HF_TOKEN_FILE` resolve to the host's files; `LIBRARY_DIR` the same way through `deploy/compose.operator-library.yaml`, which `make operator-up` adds only when it is set (Phase 20); TLS files, htpasswd and the HF token read-only (the token at `/etc/devai/operator/hf-token`, named by `HF_TOKEN_PATH`); from Phase 10, the JWT key at `/etc/devai/slurm/jwt_hs256.key` and the bus passwords at `/etc/devai/bus/operator.pass` and `/etc/devai/bus/control-operator.pass` (host-user 0600 files, which `devai` reads under keep-id, I19); the GPU when present |
| `devai-router` | 11434, 11435, 11436, 11438 (11437 retired in Phase 4); `DEVAI_FANOUT_PORT` when Phase 19's lanes are on; 11440 (`DEVAI_SYSTEMONE_PORT`) when Phase 23's System One port is on (I27) | the JWT key; its bus password; `${DEVAI_REPO}/deploy` at `/etc/devai/deploy`, read-only (from Phase 9, or Phase 17a if it lands first: the probe caches, models.yaml and the recovery files, read by absolute path so a file replaced by rename is seen, I25); no podman socket (from Phase 9) |

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
| `HF_TOKEN_FILE` (default `~/.cache/huggingface/token`) | the Hugging Face token model pulls use today | `devai-operator`, at `/etc/devai/operator/hf-token`; `HF_TOKEN_PATH` points the `hf` CLI there. A script that hands the token to a container it starts passes `HF_TOKEN_FILE`'s host path as the bind source (visible under `HOME`, C6), never `/etc/devai/operator/hf-token`, which podman would look for on the host |

The router's environment, set in compose:

- `SLURM_REST_URL=http://devai-slurm:6820`, `SLURM_JWT_KEY_FILE`, `SLURM_USER=devai`;
- `DEVAI_BUS_URL=nats://devai-slurm:4222`, `DEVAI_BUS_USER=router`, `DEVAI_BUS_PASSWORD_FILE`.

devai-operator's environment, set in compose from Phase 10 (C6 names the files):

- `SLURM_REST_URL=http://devai-slurm:6820`, `SLURM_JWT_KEY_FILE=/etc/devai/slurm/jwt_hs256.key`, `SLURM_USER=devai`;
- `DEVAI_BUS_URL=nats://devai-slurm:4222`, `DEVAI_BUS_USER=operator`, `DEVAI_BUS_PASSWORD_FILE=/etc/devai/bus/operator.pass`.

`devai-jobs` and `devai-bus` read these in the app, in its direct runs and in every action `devai-control` runs (C8). `devai-control` itself uses `control-operator`, named in its own file below.

devai-slurm's environment, set in compose (for `devai-jobs` under `podman exec`, I13):

- `SLURM_REST_URL=http://127.0.0.1:6820`, `SLURM_JWT_KEY_FILE=/etc/slurm/jwt_hs256.key`, `SLURM_USER=devai`.

slurmd passes none of the container's environment to the prolog, the epilog or the health check. Each of `prolog.sh`, `epilog.sh` and `healthcheck.sh` sets the bus settings itself before it calls `devai-bus`: `DEVAI_BUS_URL=nats://127.0.0.1:4222`, `DEVAI_BUS_USER=jobs`, `DEVAI_BUS_PASSWORD_FILE=/run/devai/bus/jobs.pass` (the values of C3's job environment).

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
  # Phase 23 (I29): stock vLLM 0.31.0 for DiffusionGemma on 11441, and the vendored System One server on 11442
  vllm-s1: {kind: vllm-s1, port: 11441, systemone_port: 11442, canvas: 64, store: /var/cache/devai/vllm, log_dir: /var/cache/devai/jobs/engines/vllm-s1}
```

The trainer's copy (`deploy/laya-trainer/control.yaml`, Phase 8) has `container: laya-trainer`, the bus line with user `control-laya` (C1) and its password file, the same `supervisor`, `state_file` and `lease_timeout_s`, and `services: {trainer: {kind: trainer, store: /laya, catalog: /etc/devai/laya-models.yaml, device: cuda}, controller: {kind: program, program: controller, port: 11434}}`; `LAYA_TRAINER_DEVICE`, when set in the container, overrides `device`, as it overrides the controller's `--device` today.

devai-operator's copy (`deploy/operator/control.yaml`, Phase 10) has `container: operator` and the bus line `bus: {url: "nats://devai-slurm:4222", user: control-operator, password_file: /etc/devai/bus/control-operator.pass}`.

### C8 -- The operator's action registry (`deploy/operator/actions.yaml`)

Both the web app and `devai-control` (kind `actions`) read it and check every parameter against it. Nothing runs through a shell:

- the argument list is `[script] + args + the mapped parameters`;
- the environment is a fixed base plus the parameters mapped to variables. The base:
  - `PATH`, `HOME` (the host user's home, `HOST_HOME_DIR`, path identity; C6), `DEVAI_REPO`;
  - the `NAME=value` lines that `make -s -C $DEVAI_REPO ops-env` prints, read once per run: the Makefile's view of `.env` and of its defaults, for the variables in its `OPS_ENV_VARS` (Phase 6), so a page run and the same make target see the same values; a non-zero exit of `ops-env` refuses the action;
  - copied, after those lines, from `devai-control`'s own environment (for a direct run, the app's) when set: `CONTAINER_HOST`, `DEVAI_GPU_DEVICE`, `HF_TOKEN_PATH`, `DEVAI_BUS_URL`, `DEVAI_BUS_USER`, `DEVAI_BUS_PASSWORD_FILE`, `SLURM_REST_URL`, `SLURM_JWT_KEY_FILE`, `SLURM_USER` (C7), and the lab's host context (`DEVAI_HOST_IP`, `DEVAI_HOME_VOLUME`, `DEVAI_HOST_WORK_DIR`, `LAB_PORT`, `DSH_PORT`, `DEVAI_HOST_GITCONFIG`, `DEVAI_HOST_SSH_DIR`, which `make operator-up` hands to devai-operator, Phase 6, and Phase 12's lab-start reads), so a prober loads through the bus and a submitter submits through `devai-jobs`; a value copied here wins over the same name from `ops-env`;
  - for a `job` action also `DEVAI_JOB_ID`, `DEVAI_RESULTS_DIR` (`/var/cache/devai/jobs/results/<jobid>`) and `DEVAI_HOLD_KEY`, the `hold_key` of the job's comment (C3), which `devai-control` reads through `devai-jobs show <jobid> --json` before it starts the run; the hold key goes nowhere else, and a script that starts a container passes it on only where the container takes a C5 hold (Phase 10's bench-task.sh).

  supervisord passes the container's environment to `devai-control` and to Apache (a program inherits supervisord's environment, with its `environment=` added on top: supervisord's documentation, "Subprocess Environment"). That mod_wsgi's daemon process keeps Apache's environment is assumed, not checked. A registry test (Phase 6's "argv and env building") pins this list;
- a `host` action is never run, only shown with its command.

```yaml
version: 1
actions:
  - id: model-pull                    # [a-z0-9-]+, unique
    group: models                     # stack, build, models, probe, bench, test, backup, logs, lab, host, library, maintenance
    title: Pull a catalog model
    kind: job                         # job (a Slurm job) | direct (stack lifecycle, O2) | host (shown only, O3)
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
    # comment: {backend: vllm, model: <param>, ctx: <param>, mtp: <param>}   # optional, job_kind action only: C3's keys for a C5 hold
    confirm: false                    # true for destructive actions: the page asks again
    covers: [model-pull]              # the Makefile targets this action replaces
not_offered:
  - target: shell
    reason: interactive; a JupyterLab terminal replaces it (O4)
pending: []                           # targets not yet moved to scripts; Phase 7 empties it
```

`job_kind` is required on every `kind: job` entry and has no default. Both readers, `registry.py` and `devai-control`, refuse to load a registry where a job entry lacks it or names a kind other than `action`, `probe` or `bench`. `time_limit_min`, `gpu` and `comment` are the only optional job fields; `time_limit_min` is required when `gpu: true` (C3). `comment` (allowed only with `job_kind: action`) fills C3's base keys for a run that takes a C5 hold: `{backend: <literal>, model: <param name>, ctx: <param name>, mtp: <param name>}`, `backend` and `model` required in it, `ctx` and `mtp` optional. `devai-jobs submit`, which the web app calls too (I13), copies the literal and the checked parameter values into the job's comment, so the run can take a C5 hold on that launch. The registry test (Phase 6) pins that a `comment` naming an undeclared parameter, or set on another job kind, is refused.

**GPU actions.** `gpu: true` (default false; allowed only with `job_kind: action`) marks an action that computes on the GPU itself, such as Phase 22's `laya-recipe-eval`.

- Its job is submitted with the `engine` license and a `time_limit_min`, and its comment carries `gpu: true` (C3). The prolog guard and the epilog treat it as they treat a probe job (slurm.md Sec. 6): an empty card first (the sleeper unloaded), and at the end the action stopped if it still runs, `gpu.json` written and the card checked idle.
- The router treats it as it treats a trainer or a probe job (Phase 9): while one is PENDING, the router ends its engine job so the license comes free; while one runs, engine requests get 503 `gpu_held_by_job`. The router recognises such a job by the `engine` license it requests in `GET /jobs`, on any job that is not an engine job, so no list of kinds has to follow new ones.
- Whatever holds the card for it runs with the job's `DEVAI_JOB_ID` in its environment, so a holder left behind by a dead job is found and stopped like a stale prober. Its script starts any container with the job's label (`devai.job=$DEVAI_JOB_ID`) and `DEVAI_JOB_ID`, as `scripts/ops/bench-task.sh` does.
- From Phase 10 on, every one-shot GPU-exclusive measurement is a GPU action: Phase 4's `quant-local` (4h), Phase 22's `laya-recipe-eval` and Phase 23's `systemone-gate`. Its script, started outside a job (no `DEVAI_JOB_ID`, as from its Makefile target), only submits the action through `devai-jobs submit action <id>` (on the host `bin/devai-jobs`, which runs `submit` in devai-operator, I13) and prints the job id, so `make` and the page take the same path.
- `devai-jobs submit action`, `submit probe` and a bench run's submissions take the job kind, the license (`-L engine`, and `gpu: true` in the comment, for a GPU action), `time_limit_min` and, for an action, the `comment` keys from the action's entry. A bench run's submission (`devai-jobs submit bench --run <id> --task <t> --backend <b> --model <name> --ctx <n> [--mtp]`, one per (model, task) of `bench-sync.py --submit`'s plan) writes C3's `backend`, `model`, `ctx` and `mtp` from those arguments, the same launch the task holds (C5), read from `$DEVAI_REPO/deploy/operator/actions.yaml`. They refuse when no registry is found, as in devai-slurm, which does not mount the repository (C6).
- Before Phase 10 such a measurement runs with the stack down (`make cache-down`, and from Phase 8 on the `slurm` profile stopped too), after a card check on the host: `nvidia-smi --query-compute-apps=pid --format=csv,noheader` prints nothing, and `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` reads at most 512 (MiB; the prolog's idle rule, slurm.md Sec. 6). Otherwise the target refuses. A list of container names, such as the one the probers' `assert_no_active_backends` iterates, cannot see devai-engines (Phase 8) or a leftover throwaway container. Phase 10 converts each such script and entry that landed before it (Phase 10 deliverables); from Phase 10 on no target runs the card check.
- The registry test (Phase 6) pins that `gpu` is refused on any other job kind, and that a `gpu: true` entry without `time_limit_min` is refused.

**Coverage of new targets.** Every documented Makefile target that a later phase adds follows Phase 7's rule. Once Phase 7 has landed, it comes with its `scripts/ops/` script and an entry here with `covers`, or with a `not_offered` entry and a reason. After Phase 6 and before Phase 7, it sits in `pending`. Before Phase 6 there is no registry, and Phase 7's inventory picks it up.

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
devai-tools/internal/backup    modify -- the age key removed everywhere, or it warns on every backup: the IsAgeKey
                                         entry and the excludeAgeKey parameter (manifest.go), the "sops-age" root
                                         (manifest.go ArchiveRoots), the missing-key warning (snapshot.go), the
                                         0600 re-assert (restore.go); manifest_test.go, roundtrip_test.go and
                                         malicious_test.go updated (the root-depth escape case moves to another
                                         multi-segment root, or is dropped if none remains); restoring an older
                                         archive that contains sops-age/ skips that member with a warning, pinned
                                         by a test
devai-tools/cmd/devai-backup   modify -- --exclude-age-key removed
tests/test-backup-restore.sh   modify -- the fake age key and its content and mode assertions removed
docs/backup-restore.md         modify -- the age-key rows and the recovery warning removed
tests/python/test_vllm_devai_backend.py
                               modify -- the Dockerfile.mcp-modelstatus and modelcache/probecache.go assertions
                                         removed from test_launcher_mounts_its_cache_and_status_server_bakes_it
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
bin/devai-agent                     modify -- registry-cache out of NO_PROXY_HOSTS
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
                                          names nvidia.com/gpu=all), nothing otherwise; a DEVAI_GPU_DEVICE set
                                          in its environment (`${DEVAI_GPU_DEVICE+x}`) overrides it, and an
                                          empty one forces CPU
Makefile                        modify -- devai-base / devai-lab only; build-base-cpu, build-cpu, lab-cpu,
                                          shell-cpu removed; base = the pinned trixie-slim; `DEVAI_GPU_DEVICE ?=
                                          nvidia.com/gpu=all` removed: DEVAI_GPU_DEVICE is set from `$(shell
                                          scripts/host-gpu.sh)` unless `$(origin DEVAI_GPU_DEVICE)` is
                                          `environment` or `command line` (then kept, empty included), and is
                                          still exported; GPU_FLAGS = `$(if $(DEVAI_GPU_DEVICE),--device
                                          $(DEVAI_GPU_DEVICE) --security-opt=label=disable,)`, the `$(or
                                          ...,nvidia.com/gpu=all)` fallback removed, so `lab`, `shell`, the
                                          probers and `cache-up` follow the host check. A `DEVAI_GPU_DEVICE` line
                                          in `.env` (the removed gpu-vendor target wrote one) no longer counts as
                                          explicit and is deleted from this host's `.env`
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
- With `nvidia-smi` hidden from PATH and DEVAI_GPU_DEVICE unset, `make -n lab` shows no `--device`; `make -n lab DEVAI_GPU_DEVICE=` shows no `--device` on this host.
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

Its vLLM is 0.30.0 with HyperQwen's current series (E1). That version is built first, on today's `vllm-devai` backend (step 4v), and accepted there by a rule fixed before the runs. If the rule fails or is inconclusive, 0.28.0 stays and the rest of the phase builds it instead.

### Deliverables

```
scripts/build-vllm.sh                modify -- 4v: VLLM_VERSION (default 0.30.0) selects a build record; one build root
                                               per version; the HyperQwen and devai checks only where the record has them;
                                               for a HyperQwen record it writes <version>.in from the release's
                                               requirements/cuda.txt + common.txt plus HyperQwen's docker/requirements.txt,
                                               without vllm, flashinfer-cubin and index lines;
                                               fetch_wheels downloads the record's lock with --require-hashes, and the dist
                                               carries the lock as requirements.txt
deploy/vllm-builds/0.28.0.sh, deploy/vllm-builds/0.30.0.sh
                                     new    -- 4v: one record per version, sourced by build-vllm.sh: sdist sha256, torch
                                               pin, HyperQwen commit and skipped patches (or none), devai's patch series
                                               (or none), the runtime requirement set, the EXTERNALS table
deploy/vllm-builds/0.28.0.in, 0.28.0.lock, 0.30.0.in, 0.30.0.lock
                                     new    -- 4v: each record's runtime set and its hash lock (`make vllm-lock`, M3)
deploy/vllm-patches/qwen3_5-mtp-share-vocab-at-init.patch
                                     modify -- 4v: applies at --fuzz 0 on 0.30.0 after HyperQwen's series; its comment
                                               names both drafter-sharing paths (Model Runner V1 and V2)
scripts/vllm_plugins/deepseek_string_tool_parser.py, scripts/vllm_plugins/nemotron_json_tool_parser.py
                                     modify -- 4v: protocol classes from vllm.entrypoints.generate.base.protocol, falling
                                               back to the module 0.29 removed (vllm#54492), so one file serves every vLLM
deploy/Dockerfile.vllm               modify -- 4v: the HyperQwen check only when BUILD_INFO names a HyperQwen commit; a
                                               runtime set that is a lock installed with --require-hashes, then the
                                               vLLM wheel built here with --no-deps, then pip check
Makefile                             modify -- 4v: VLLM_VERSION; VLLM_DIST and VLLM_DEVAI_IMAGE_TAG follow it;
                                               `export VLLM_DEVAI_IMAGE`, so a value in .env reaches the probers and
                                               compose (as VLLM_IMAGE's export does);
                                               bench-concurrency passes SALT, MIN_TOKENS and METRICS_URL;
                                               probe-load-vllm passes PROBE_MTP; probe-load-vllm-devai (new: PROBE_MTP,
                                               PROBE_CACHE, PROBE_LEDGER; PROBE_FORCE as probe-load-vllm), which 4e removes
                                               with probe-vllm-devai; probe-vllm-devai passes PROBE_LEDGER;
                                               BENCH_RUN_FLAGS gains BENCH_CACHE (--cache), BENCH_LOG_DIR (--log-dir),
                                               BENCH_MTP (--mtp) and BENCH_MAX_CONNECTIONS (--max-connections); scratch
                                               paths lie under $(CACHE_DIR)/bench, the bench container's only read-write
                                               mount besides deploy/; vllm-lock (VLLM_VERSION=<v>): deploy/vllm-builds/<v>.lock
                                               from <v>.in, for every record, PyPI only, hash-locked (M3)
scripts/verify-backend-flags.py, deploy/backend-flags.yaml
                                     modify -- 4v: also checks VLLM_DEVAI_IMAGE, with the flags of the recovery entries
                                               scoped to its backend
gpu-arbiter/main.go (vllmEntrypoint), scripts/probe-vllm-reasoning.py (vllm_command_args),
gpu-arbiter/testdata/argv-golden.txt
                                     modify -- 4v: --prefix-cache-retention-interval None on the 0.30.0 backend, only if
                                               the retention measurement adopts it
scripts/bench/bench_concurrency.py   modify -- 4v, MODE=multiturn: --salt (default a random per-run nonce, printed in
                                               the report), so repeated sessions share no prefix; --min-tokens
                                               (`min_tokens` at the top level of every turn); --metrics-url: the
                                               engine's prefix-cache counters read before and after each turn, and
                                               each turn's completion tokens, in the per-turn records. Phase 15 builds
                                               its sessions on these, with no second salt or counter reader
scripts/bench/bench_runner.py        modify -- 4v: --mtp appends `::mtp` to the alias of a row whose cell records
                                               `mtp_fits: true`, and refuses any other row; --max-connections N
                                               overrides max_connections_for(backend) (default unchanged), and the run
                                               records the value it used, because a sample queued inside the engine is
                                               charged to its working_limit
scripts/bench/bench_latency_leak.py  modify -- 4v: `run` also returns per-prompt records (id, ttft_ms, effective_tokens,
                                               token_source, stream_seconds, finish_reason); bench_runner writes them to
                                               <log-dir>/latency-<alias>.jsonl, not into the cache row
scripts/stats/vllm_upgrade_accept.py new    -- 4v: the acceptance rule's statistics and verdicts, read-only, with
                                               statlib.py: per-item pairs from each build's inspect logs (time-outs
                                               per item from the samples' limit field; for the bound a time-out is
                                               wrong on 0.30.0 and right on 0.28.0, on the build where it occurred
                                               only; for McNemar a time-out is wrong, as the harness scores it),
                                               newcombe_paired bounds,
                                               mcnemar_exact with Holm, pass/fail/inconclusive; the speed ratios over
                                               latency-<alias>.jsonl (prompts 1-39, paired by id, boot_ci); the
                                               retention rule (a)-(b) over the 8 per-session bench_concurrency
                                               reports of each arm; a JSON report that docs/backends.md cites. It
                                               takes both arms' recorded image ids and refuses a verdict (exit
                                               non-zero, no Pass) when they are equal, or when an arm's recorded id
                                               differs from its tag's id
tests/python/test_stats_vllm_upgrade_accept.py
                                     new    -- 4v: the verdicts on fixed fake logs and reports (pass, fail,
                                               inconclusive, a task meeting both Pass's bound and Fail's test; a
                                               time-out on 0.30.0 only, on 0.28.0 only, and on both (the other
                                               build's observed outcome kept in the bound; time-outs wrong for
                                               McNemar); rule (a) failing in one session; equal image ids, and an
                                               id that differs from its tag's, both refused)
devai-tools/internal/backup/manifest.go, manifest_test.go, docs/backup-restore.md
                                     modify -- 4v: deploy/.vllm-devai-reasoning-cache.json joins the probe caches, so
                                               4v's `make backup-create` holds the cache its rollback restores
scripts/_probe_load.py, scripts/_probe_hf_common.py
                                     modify -- 4v: --mtp: the load probe relaunches a row whose cell records
                                               `mtp_fits: true` with the speculative config the fit probe builds, and
                                               stamps `serving_mtp: true` on the cell; without it, as today.
                                               --ledger PATH (Make PROBE_LEDGER): read and written in place of
                                               deploy/.model-status.json by the fit and load passes; I24 reuses it
tests/python/                        modify -- 4v: bench_concurrency's salt, min_tokens and counter read-out (stubbed
                                               /metrics), bench_runner's --mtp and --max-connections (the override and
                                               its default), the per-prompt records (stubbed stream), the load probe's
                                               --mtp; the Makefile's `export VLLM_DEVAI_IMAGE`; a load probe and an
                                               exact-context probe given --ledger leave deploy/.model-status.json
                                               byte-identical, and a load-probe OOM in one tag's scratch ledger does
                                               not stop the other tag's exact-context probe from probing the row
third_party/hyperqwen/README.md, tests/python/test_prepare_checkpoint.py
                                     modify -- 4v: the vendored scripts stay at c0c81bb, the commit the prepared
                                               checkpoints were made with; the build's commit is the record's
tests/python/test_build_vllm_script.py, test_vllm_image_build.py, test_vllm_devai_backend.py
                                     modify -- 4v: the records, the per-version build root, the conditional checks,
                                               every record's lock (it pins torch==<the record's torch pin> and
                                               flashinfer-python), a HyperQwen record's .in (the release's runtime set
                                               plus HyperQwen's file; no vllm, flashinfer-cubin or index line), a record
                                               without HyperQwen, `vllm-lock`'s command
CLAUDE.md, docs/backends.md          modify -- 4v: the vllm-devai backend's version; the record of 4v's runs
scripts/build-sglang.sh              new    -- host build of SGLang 0.5.16 and its kernel packages (M3, M5), like build-vllm.sh
deploy/Dockerfile.engines            new    -- trixie-slim; tini; Ollama, vLLM env, SGLang env, one CUDA 13.1 toolkit,
                                               the launcher, per-engine labels, /etc/devai/engine-builds.json;
                                               default command an idle loop (Phase 8 runs supervisord)
scripts/devai-engine                 new    -- the launcher: `devai-engine <backend> [<command> [args...]]` (4d)
gpu-arbiter/*.go                     modify -- entrypoints via the launcher; one engines image; vllm-devai removed;
                                               per-engine drift; engine cache volumes under the engine's HOME
scripts/_probe_core.py, _probe_hf_common.py, _probe_load.py, probe-check.py,
scripts/bench/bench_runner.py, _bench_core.py, scripts/bench-sync.py
                                     modify -- per-engine identity instead of image digest; cache volumes under HOME
scripts/probe-vllm-devai-reasoning.py remove -- merged into the vllm prober
scripts/prepare-checkpoint.py        modify -- its image is devai-engines, not docker.io/devai/vllm-devai; each step runs
                                               `--entrypoint devai-engine <image> vllm python3 <step>`
scripts/verify-backend-flags.py      modify -- 4d: runs each engine's `--help` through `devai-engine` against ENGINES_IMAGE
scripts/_probe_hf_common.py          modify -- PROBE_IGNORE_CURATED_PARSERS=1 with a scratch cache, and uses 4v's
                                               --ledger PATH so a scratch run leaves deploy/.model-status.json alone (I24)
scripts/_model_status.py             modify -- --exclude NAME::BACKEND --reason TEXT: a `manual` verdict, TEXT its detail, as --retire takes it (I17)
scripts/probe-cache-drop.py          new    -- removes named rows from a probe cache, atomically, as the probers save (I17)
scripts/quant/requirements.in, scripts/quant/requirements.lock, Makefile (quant-local, quant-cloud, QUANT_IMAGE)
                                     new / modify -- 4h: llm-compressor and datasets hash-locked into the image;
                                               quant-local through the launcher, a GPU action; quant-cloud not_offered
                                               or on its own public QUANT_CLOUD_IMAGE (operator)
tests/python/test_upstream_images.py modify -- the engine images, VLLM_DEVAI_IMAGE_TAG and QUANT_IMAGE leave the
                                               transitional allowlist
scripts/model-picker.py, scripts/probe-check.py
                                     modify -- a row whose weights are not in its store and whose cell carries
                                               another engine build is listed by probe-check as not re-probed,
                                               without failing, and hidden by the picker (I17)
scripts/model-picker.py, bin/devai-agent, config/*, scripts/generate-catalog.py, deploy/models.yaml,
scripts/laya-check.py, scripts/select-models.py, scripts/model-sync.py, scripts/bench-sync.py,
scripts/bench/bench_runner.py, scripts/bench/_bench_core.py, scripts/bench/clean_slate.py,
deploy/recovery-flags.json, deploy/recovery-flags.generated.json, deploy/docker-compose.yaml, Makefile, tests/
                                     modify -- vllm-devai -> vllm (M4); `image` field removed (M8)
deploy/.vllm-reasoning-cache.json, deploy/.vllm-devai-reasoning-cache.json,
devai-tools/internal/backup/manifest.go
                                     modify / remove -- 4e: the vllm-devai cache's rows merged into the vllm cache,
                                               then the file and its manifest entry removed
```

### Detailed steps

1. **4v -- vLLM 0.30.0 (E1).** It runs first: it needs nothing, and Phase 15 and Phase 8 step 1 measure on its build. `devai-engines` does not exist yet, so the vehicle is today's `make build-vllm` and the `vllm-devai` backend (port 11437). 4c installs the wheelhouse 4v accepts.
   - **The build record.** `scripts/build-vllm.sh` takes the version as a parameter, `VLLM_VERSION` (default 0.30.0), and sources `deploy/vllm-builds/<version>.sh`. The record holds what are constants at the top of the script today:
     - the sdist sha256 (0.30.0: `5f8f4e890c042ffa1c3e103f81c35e2d96f60a0a175ac43a4adbae7006bef62b`, from PyPI's JSON, read 2026-10-06; 0.28.0's there matches the script's);
     - the torch pin of the build venv (2.13.0 for 0.28.0 to 0.31.0, from each tag's requirements);
     - the HyperQwen commit and the patches it skips, or none (0.28.0: c0c81bb, skipping `dflash2-backport`);
     - devai's patch series directory, or none;
     - the runtime requirement set: `deploy/vllm-builds/<version>.lock` beside the record, hash-locked from `<version>.in`. For a record with HyperQwen, `<version>.in` is the release's own runtime set, written as for a record without HyperQwen (the sdist's `requirements/cuda.txt` with the `requirements/common.txt` it includes and the torch pin, without its extra-index and `flashinfer-cubin` lines), followed by HyperQwen's `docker/requirements.txt` at that commit without its `vllm`, `flashinfer-cubin` and extra-index lines *(and, since E8, without pip's CUDA compiler packages, from either file)*. Where both name a package, HyperQwen's pin must satisfy the release's specifier, or `vllm-lock` fails. HyperQwen's file alone holds no vLLM dependency: at c0c81bb it pins vllm, transformers, tokenizers, compressed-tensors, huggingface_hub, hf_transfer, ninja and pandas, and its `vllm` line is what pulled torch, Triton and FlashInfer in. `build-vllm.sh` writes the file, and the review of the lock's diff checks it. Today `fetch_wheels` downloads HyperQwen's file as it is, `==` pins without hashes and transitive dependencies resolved freely (`scripts/build-vllm.sh:147-150`), and the dist carries it verbatim (`:277`); from 4v it downloads the lock with `--require-hashes`, and the dist carries the lock as `requirements.txt`;
     - the EXTERNALS table: the kernel repositories, at the revisions that version's CMake files name.
   - **Every record** keeps its runtime set beside it: `<version>.in` (for a record without HyperQwen, written from that release's runtime requirements, `requirements/common.txt` and `cuda.txt`, with the torch pin) and `<version>.lock`, hash-locked.
     - `make vllm-lock VLLM_VERSION=<v>` (new in 4v) writes the lock, for every record, 0.28.0's included: `uv pip compile --generate-hashes` against PyPI alone, with no extra index (M3), for the interpreter `build-vllm.sh` builds with (`PYTHON`, Debian's python3), keeping every locked version the input still allows, as `make lab-lock` does. Its diff is reviewed.
     - The lock holds every runtime dependency and not vLLM itself, which is built here. The image installs the lock with `--require-hashes`, then the wheel with `--no-deps`, then runs `pip check`, for every record.
     - 0.28.0's kept wheelhouse and tag were built before 4v, from HyperQwen's file, and serve only the rollback. If the move stops, 0.28.0 is rebuilt from its record and lock before 4c installs it, so 4c's vLLM stage makes the same checks for either version; 4g re-probes that build as it would 0.30.0's.
     - `flashinfer-cubin` is not installed, by any record. Today's build does not install it either: 0.28.0's runtime set resolves `flashinfer-python` alone (the wheelhouse holds `flashinfer_python-0.6.16.post3` and no cubin wheel, checked 2026-10-06). Since 0.6.14 the package is published only on flashinfer.ai's index, which 0.30.0's and 0.31.0's `requirements/cuda.txt` add; a lock uses neither that index nor the package. `build-vllm.sh` drops a `flashinfer-cubin` line (HyperQwen's `docs/install.md` pins one) when it writes a HyperQwen record's `<version>.in`, and the record says so. FlashInfer JIT-compiles its kernels with the image's nvcc (measured 2026-09-21); the FlashInfer check below confirms that on 0.30.0, and Phase 23's gate on 0.31.0.
   - A record without HyperQwen, or without devai's series, skips that series, the wheel check that looks for it and `Dockerfile.vllm`'s HyperQwen check (`BUILD_INFO` gains a `devai_patches=` line). Phase 23 adds `deploy/vllm-builds/0.31.0.sh`, `0.31.0.in` and `0.31.0.lock` for `vllm-s1` this way, with no series at all (I29); `build-vllm.sh` needs no change for it.
   - `vllm-lock` and `probe-load-vllm-devai` (below) are documented Makefile targets. Once Phase 7 has landed, each has a `scripts/ops/` script and a C8 entry with `covers`, or a `not_offered` entry with a reason; after Phase 6 and before Phase 7 it sits in `pending`.
   - `check_pins` still refuses a pin the tree no longer names, and now also checks the repository URL: DeepGEMM moved to a vllm-project fork (0.30.0 release notes, vllm#56876).
     - 0.30.0 names cutlass v4.7.1 (`CMakeLists.txt:487`) and adds DeepSelect (`CMakeLists.txt:1575`), which CMake clones for every CUDA build, outside any retry. It gets an EXTERNALS row like the others, through `DEEPSELECT_SRC_DIR` (honoured in 0.31.0, `cmake/external_projects/deepselect.cmake:7-8`; 0.30.0's copy is read when the record is written).
     - The other revisions are read from 0.30.0's `cmake/external_projects/` when the record is written.
   - The build root becomes `~/.cache/devai/vllm-build/<version>/`. The Makefile's `VLLM_DIST` and `VLLM_DEVAI_IMAGE_TAG` follow `VLLM_VERSION` (`v<version>-cu131-debian13-hq.<7 hex>`, or `-stock` without HyperQwen).
     - Today's tree moves to `.../0.28.0/` before the first 0.30.0 build. The script deletes its own `dist/` on every build, and 0.28.0's wheelhouse is the rollback.
     - 4v builds with `make build-vllm VLLM_VERSION=0.30.0 VLLM_DEVAI_IMAGE=<the 0.30.0 tag>`, so `:latest` stays on 0.28.0 until the acceptance rule passes.
   - **HyperQwen.** The record pins the newest commit on HyperQwen's main branch that still pins vLLM 0.30.0 when 4v starts (`docker/requirements.txt`: `vllm==0.30.0`). Its 0.30 port, d88544b of 2026-09-28, is the first such commit.
     - `third_party/hyperqwen/` stays at c0c81bb. The prepared checkpoints were made with those scripts, and 4v does not prepare them again. The README says so, and `test_prepare_checkpoint.py` compares the vendored commit with the 0.28.0 record instead of the build's default.
   - **devai's MTP patch.** `qwen3_5-mtp-share-vocab-at-init.patch` is refreshed to apply at `--fuzz 0` after HyperQwen's series on 0.30.0. A dry run on 2026-10-06 applied it at offsets; nothing was built.
     - Model Runner V2 is the default since 0.29 (vllm#53183). It shares the drafter's vocabulary layers in `v1/worker/gpu/spec_decode/eagle/utils.py`, not in the proposer the patch's comment names.
     - In 0.31.0 that path shares whenever the drafter declares no copy of its own (`_should_share`, `:15-27`; `load_eagle_model`, `:86-118`), which the patch's placeholders satisfy. 4v re-reads it in 0.30.0, extends the patch if sharing there needs anything the placeholders lack, and names both paths in the patch's comment.
     - The patch stays: upstream still allocates both layers before sharing them (0.31.0 `qwen3_5_mtp.py:85-88`, `:246-250`; E1).
   - **The two tool-parser plugins.** `deepseek_string` and `nemotron_json` import `vllm.entrypoints.openai.engine.protocol`, which 0.29 removed (vllm#54492). With `--enable-auto-tool-choice`, the server then fails at startup, before the model loads (0.31.0 `launchers/launcher.py:233-239`).
     - The classes they import live in `vllm.entrypoints.generate.base.protocol` (the file is in the 0.29.0 and 0.30.0 trees; the classes were read in 0.31.0). Each plugin imports from there and falls back to the old module, so one file serves stock 0.22.1, 0.28.0 and 0.30.0.
     - `nano_v3` imports only modules that 0.30.0 still has (its file list).
     - Their rows: DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B (`deepseek_string`), whose weights are not on disk, so nothing is re-probed (I17); NVIDIA-Nemotron-Nano-9B-v2-NVFP4 (`nemotron_json`), whose weights are in both stores. Its vLLM launches use an image of their own today (vLLM 0.25.1, `deploy/recovery-flags.json`), which M8 removes, so 4g decides it on the single build.
     - Checked without a GPU: each plugin file is imported in the new image, and its parser name is then registered in `ToolParserManager`.
   - **Launch flags.** `make verify-backend-flags` reads only the stock image today. It now also checks `VLLM_DEVAI_IMAGE`, against `deploy/backend-flags.yaml` and the flags of every recovery entry scoped to that backend. It passes on the 0.30.0 image before any probe.
   - **FlashInfer** moves to 0.6.18.post1, HyperQwen's pin for 0.30.0 and vLLM 0.30.0's own, without `flashinfer-cubin` (above). That it still JIT-compiles at first use is checked, before the acceptance rule:
     - remove the volume `devai-engine-cache-vllm-devai-flashinfer`, launch with `FLASHINFER_CUBINS_REPOSITORY=file:///nonexistent/` (so no cubin can be fetched at run time, as 4d's launcher will set it), send the NVFP4 row's first request at fp8 KV, and list the volume;
     - compiled objects there mean it still compiles, and the host nvcc stays in `Dockerfile.vllm` and in 4c's image;
     - an empty volume means the kernels came from somewhere else (not a run-time cubin download, which the variable rules out). Where is found and recorded before the rule runs, and nvcc stays anyway: removing it is a change of its own, made only after an image without nvcc serves both prepared rows.
   - **Engine-log facts**, from the NVFP4 row's first launch: the NVFP4 GEMM path it selects (W4A4 is the default on SM120 since vllm#55170; this body declares 4-bit float activations, so it qualifies), and whether FlashInfer's GDN prefill (vllm#55715) is on. The wording of those log lines is read in the 0.30.0 source; it is not known here.
   - **Source re-reads in 0.30.0.** Only a partial 0.30.0 tree was read on 2026-10-06. So 4v re-reads, for the parts of this plan that cite 0.28:
     - the startup memory check (Phase 8, the sleeper's VRAM; unchanged in 0.31.0, `v1/worker/utils.py:539-558`);
     - one model per process (D15);
     - the dev-mode routes (C5, Phase 9 step 1);
     - the Anthropic count path and `build_chat_params` (Phase 16d).
   - **Run-time cubins off for every 4v launch.** Before the first probe, `FLASHINFER_CUBINS_REPOSITORY=file:///nonexistent/` is added to the engine environment of every `vllm-devai` launch: the router's `buildContainerSpec` and the probers' `container_run_detached`, for that backend only (a router test and a Python test pin it). Without it, vLLM GETs NVIDIA's artifactory at startup (0.28.0 `vllm/utils/flashinfer.py:367-414`) and, if it answers, may select the TRTLLM XQA decode kernels on SM12x. So every 4v measurement runs as 4d's launcher will serve: both tags' exact-context and load probes, the quality and speed runs, the retention arms and the re-probes, Phase 15's baseline included. Phase 8 moves the constant into `pkg/engineargs` and 4d's launcher. docs/backends.md records that the 2026-09-21 `vllm-devai` numbers were taken without it.
   - **Image selection reaches every consumer.** Today `VLLM_DEVAI_IMAGE` is set in the Makefile but not exported (`Makefile:70`), so a value in `.env` never reaches the probers or compose, which fall back to `:latest`, and both arms could silently run 0.28.0. 4v adds `export VLLM_DEVAI_IMAGE` next to `export SGLANG_IMAGE`, and every 4v target runs as `make <target> VLLM_DEVAI_IMAGE=<tag>`.
     - Before each arm, the record stores the tag's image id and manifest digest (`podman image inspect --format '{{.Id}} {{.Digest}}' <tag>`).
     - After each probe, the probed cache's `_meta.current_image_digest` must equal that digest (the probers stamp `.Digest`, `scripts/_probe_core.py` `image_digest_via_cli`).
     - After each bench arm, `podman inspect --format '{{.Image}}' devai-vllm-devai` must equal that id.
     - `scripts/stats/vllm_upgrade_accept.py` takes both ids and refuses a verdict when they are equal, or when an arm's recorded id differs from its tag's.
   - *(The acceptance rule below is superseded by E7, 2026-10-07: 0.30.0 is accepted on the Context condition alone, with the load probe run on 0.30.0 only; the 0.28.0 arm, the Quality condition and the paired speed comparison are dropped. The text stays as recorded.)*
   - **The acceptance rule**, fixed before the runs, and accepted by the operator on 2026-10-06 with its margins and the AutoRound row's utilisation limit of 0.96. 0.30.0 replaces 0.28.0 only if both conditions below pass. On a failure or an inconclusive result the move stops, 0.28.0 stays, and E1 goes back to the operator with the report. Quality and context come before speed. Every arm runs with MTP on, as the picker serves both rows (CLAUDE.md, "MTP is unconditional"), because 4v changes the drafter's path (devai's MTP patch, HyperQwen's port).
     - **Context.** Both prepared Qwen3.8 rows load with MTP at their current probed contexts: Qwen3.8-27B-MTP-devai-NVFP4 at 118,784 and Qwen3.8-27B-W4A16-devai-AutoRound at 131,072 (`deploy/.vllm-devai-reasoning-cache.json`, both cells `mtp_fits: true`). Their recovery flags stay as they are, except `--gpu-memory-utilization`, which may rise to at most 0.96 (the NVFP4 row's value today; the AutoRound row has 0.93). Each is confirmed with `make probe-vllm-devai PROBE_REPO=devai/<row> PROBE_CTX_EXACT=<ctx> PROBE_LEDGER=<scratch ledger>`, `VLLM_DEVAI_IMAGE` on the 0.30.0 tag, and its new cell must again record `mtp_fits: true`. The NVFP4 row is probed at its 0.96. The AutoRound row is probed first at 0.93; only if that misses is it probed once more at 0.96, the limit, and that probe decides. No other value is tried. A miss at the deciding value fails the rule. The record lists every probe with its utilisation, and the 0.30.0 load probe and every later 4v run use the deciding value. A row the exact-context probe skips, or for which it writes no cell, counts as a miss.
       - The fit probe is one launch and three short chats; it cannot see the serving-time transient that once OOM-killed an engine on its first long prompt after a clean load (CLAUDE.md, devai-vllm's persisted torch.compile cache, 2026-09-23). So each row also passes the serving-time load probe at that context, with MTP: `make probe-load-vllm-devai PROBE_REPO=devai/<row> PROBE_MTP=1 PROBE_CACHE=<scratch cache> PROBE_LEDGER=<scratch ledger>`, where each tag gets its own new copy of `deploy/.vllm-devai-reasoning-cache.json`, and its own new copy of `deploy/.model-status.json` as its ledger, used by that tag's exact-context and load probes, so no 4v verdict reaches the real ledger or survives a rollback (`_probe_load.py` saves the ledger on every pass, and an `oom` verdict there would make the other tag's exact-context probe skip the row). The 0.28.0 cache copy is taken before any 4v probe; the 0.30.0 copy is taken after that tag's exact-context probe, so it holds the new cell. Without a new copy, `_probe_load.py` skips a cell that already carries `serving_ok` (`cached`, `scripts/_probe_load.py:1026`).
       - It runs first with `VLLM_DEVAI_IMAGE` on the 0.28.0 tag and today's recovery flags, before any utilization change, because no current cell carries `serving_ok` and this gives the baseline, and then on the 0.30.0 tag, after the exact-context probe, with the flags that probe passed with (a raised utilization included). On 0.30.0 the cell must stay at the row's context with `serving_ok: true` and the needle recalled. A load-probe failure on 0.30.0 where 0.28.0 passed fails the rule. A failure on both is recorded; if 0.30.0 ran at a raised utilization, its failure fails the rule, and otherwise the rule then rests on the exact-context probe alone.
     - **Quality** (non-inferiority). gsm8k on its whole test split (1,319 items) and humaneval on all 164 items, on the teacher as the picker serves it, `Qwen3.8-27B-MTP-devai-NVFP4::mtp@118784`, one run per build, on the same day: `make bench-vllm-devai VLLM_DEVAI_IMAGE=<tag> BENCH_REPO='^devai/Qwen3\.8-27B-MTP-devai-NVFP4@' BENCH_TASKS=gsm8k,humaneval BENCH_N_GSM8K=1319 BENCH_N_HUMANEVAL=164 BENCH_NO_EARLY_DROP=1 BENCH_MTP=1 BENCH_MAX_CONNECTIONS=4 BENCH_CACHE=/var/cache/devai/bench/4v/<tag>/cache.json BENCH_LOG_DIR=/var/cache/devai/bench/4v/<tag>/logs` (the runner reaches devai-router only on devai-net, which the target's container joins), a scratch cache and log directory per build, so neither run writes `deploy/.bench-cache.json`. 4 is the row's `--max-num-seqs` (`deploy/recovery-flags.json`), so no sample waits inside the engine on its working-time clock (`bench_runner.py`, the comment above `BACKEND_MAX_CONNECTIONS`); both builds run with the same value. Without `--repo` the runner would bench every fitting row of the vllm-devai cache; without `--no-early-drop` a gsm8k score below 0.70 would skip humaneval and leave its comparison without data.
       - The margins, fixed now: 3 points for gsm8k and 5 points for humaneval. Per task, the paired difference 0.28.0 minus 0.30.0 is taken over the per-item pairs from the inspect logs (`scripts/stats/vllm_upgrade_accept.py`, with `scripts/stats/statlib.py`).
       - Its upper bound is the upper end of the two-sided 95% Newcombe interval (`newcombe_paired`), a one-sided 97.5% bound: Bonferroni over the two tasks. For this bound a time-out on 0.30.0 counts as wrong on 0.30.0 and a time-out on 0.28.0 counts as right on 0.28.0, the conservative assignment; the other build's observed outcome for that item stands (an item that timed out on both builds becomes right on 0.28.0, wrong on 0.30.0). Time-outs are assigned so because a score with time-outs mixes correctness and speed (docs/statistics-primer.md Sec. 7, informative truncation). Each build's time-outs are reported.
       - **Pass:** both upper bounds lie below their margins and no task meets Fail. **Fail:** an exact McNemar test (`mcnemar_exact`, Holm over the two tasks, time-outs scored as the harness scores them) shows a loss on 0.30.0 with an adjusted p below 0.05; Fail takes precedence, so a task that meets both (a small significant loss inside the margin) is a failure. **Anything else is inconclusive**, and the move stops as on a failure.
       - Fail's precedence follows the step's priority, quality and context before speed. Classic non-inferiority, under which such a loss passes and is reported, is the operator's to choose instead (E1); it is not chosen here.
       - Before the runs, the record states each task's expected half-width (`mde_paired_halfwidth` at 10% discordance): about 1.7 points for gsm8k and about 4.8 points for humaneval. So humaneval can be inconclusive even when nothing is lost.
     - **Speed, reported, not deciding.** On each of the two rows, one run per row, the bench's 40 latency prompts (the `leak` task) on each build, with MTP, as the picker serves them: `make bench-vllm-devai VLLM_DEVAI_IMAGE=<tag> BENCH_REPO='^devai/<row>@' BENCH_TASKS=leak BENCH_MTP=1 BENCH_MAX_CONNECTIONS=4` with the same scratch `BENCH_CACHE` and `BENCH_LOG_DIR` per build as the quality run.
       - Prompts are paired by id, per row, from the per-prompt records (`<log-dir>/latency-<alias>.jsonl`, new in 4v), prompts 1-39 only (prompt 0 stays excluded, as the cold start it was before the warm-up request below).
       - Two measures per prompt: the client's TTFT, and the decode rate, effective_tokens / stream_seconds, counted only where `token_source` is `usage` on both builds and at least 64 tokens were generated on each.
       - Each 0.30.0/0.28.0 ratio is the geometric mean of the per-prompt ratios, with a 95% bootstrap interval over prompts (`boot_ci`): the method of docs/llm-tokens-and-speed.md's paired build comparison, with client-side timing instead of the engine log. One run per build, so the intervals cover prompt-to-prompt variation only.
       - Each row's MTP mean acceptance length on each build (1 + accepted draft tokens / drafts) comes from the engine's speculative-decoding counters on `/metrics`; the counter names are read in the 0.30.0 source first. Counters are per engine process and the router relaunches the engine for each row, which is why the latency task runs once per row. Before each run, one request to `<row>::mtp@<ctx>` loads that row's engine. The counters (`podman run --rm --network devai-net <lab image> curl -s http://devai-vllm-devai:11434/metrics`, saved next to that build's logs) and the container's start time (`podman inspect -f '{{.State.StartedAt}}' devai-vllm-devai`) are then read before and after the run. If the start time changed, the engine was relaunched during the run, and that row's value is reported missing, not computed and not re-run.
       - An MTP-off latency run on each build is added only if the report needs to separate a build effect from a drafter effect.
     - The 0.28.0 arm runs on the kept tag, `docker.io/devai/vllm-devai:v0.28.0-cu131-debian13-hq.c0c81bb`. The router is started with `VLLM_DEVAI_IMAGE` set to each tag in turn (`make cache-up VLLM_DEVAI_IMAGE=<tag>`), and the image checks above confirm which build served.
     - `make backup-create` runs before the first probe. `:latest` moves to the 0.30.0 tag (`podman tag`) only after the rule passes. 0.28.0's wheelhouse and image tag are kept until 4g passes.
     - Rollback: restore `deploy/.vllm-devai-reasoning-cache.json` alone from the backup (`devai-backup`'s manifest gains this file in 4v; it lists only the ollama, vllm and sglang probe caches today). Not with `make backup-restore`, which has no path filter and would also revert the bench cache, the other probe caches, `.model-status.json`, `~/.devai` and the age key: `make backup-verify ARCHIVE=<archive>`, move the current file aside, then `tar -xzf <archive> deploy/.vllm-devai-reasoning-cache.json` from the repository root. Then revert 4v's `--gpu-memory-utilization` change in `deploy/recovery-flags.json` with git (the file is tracked, not in the backup), and, if the retention flag was adopted, remove its constant from `vllmEntrypoint`, `vllm_command_args`, `deploy/backend-flags.yaml` and the argv golden (0.28.0 has no `--prefix-cache-retention-interval` argument, vllm#52216 is in 0.29, so a 0.28.0 engine would fail at startup). Then tag 0.28.0 as `:latest` and set the script's default back to 0.28.0.
   - **Prefix-cache retention**, measured once the rule has passed.
     - 0.28 kept the prefix cache dense for hybrid models by default. 0.30.0 defaults `--prefix-cache-retention-interval` to 0 (vllm#52216, in 0.29; 0.29's automatic dense setting for MTP, vllm#55760, is not in 0.30.0). Under MTP, 0 reuses less of each previous turn: HyperQwen measured 864 more prefill tokens per turn, about 0.7-0.8 s on a 3090, and passes `None`, which is 0.29's dense behaviour (HyperQwen `docs/vllm-0.30.md`, "Prefix-cache retention"). The environment variable is gone in 0.30.0 (vllm#55353); only the flag counts.
     - The two settings differ only in how much of the previous turn's reply is kept: HyperQwen's formulas are `((P+R)//A)*A - A` cached tokens under dense retention against `(P//A)*A - A` at 0, P the previous turn's prompt, R its reply and A the attention block (432 under MTP on HyperQwen's 3090; `docs/vllm-0.30.md`), and its 864 tokens per turn came from replies of 1,000 tokens. With run_multiturn's "Reply with OK." turns, R is about 2 tokens and the two formulas almost never differ. So the replies are made long.
     - Measured on the teacher row with MTP, `Qwen3.8-27B-MTP-devai-NVFP4::nothink::mtp@118784`: `make bench-concurrency MODE=multiturn PORT=11437 TURNS=5 MAX_TOKENS=1024 MIN_TOKENS=1024 METRICS_URL=http://devai-vllm-devai:11434/metrics`, 8 sessions per setting, under the default and under `None`. Session i runs with `SALT=<run nonce>-<i>`, the same in both arms, so session i of each arm sees the same prompts. Between the arms the stack is restarted (`make cache-down`, then `make cache-up`), so the engine is relaunched with an empty cache; for the `None` arm the flag first goes into the row's `engine_flags`.
     - Per turn, the run records the engine's prefix-cache hits and queries (`--metrics-url`, read before and after the turn; the counter names are read in the 0.30.0 source first) and the reply's completion tokens.
     - Before the measured runs, one uncounted session under the default checks the instrument. Its salt is `<run nonce>-dry`, which no measured session uses. It passes if every reply has 1,024 completion tokens, and the hit counter moves on turns 2-5. Otherwise the workload is fixed (`min_tokens` refused or ignored under MTP, for example) and the dry run repeated.
     - The rule, fixed before the runs: `None` is adopted only if all three hold:
       - (a) in every session, its cached prompt tokens on turns 2-5 exceed the default's. This is a deterministic check (docs/statistics-primer.md Sec. 12); a difference of about two attention blocks per turn is expected (864 tokens in HyperQwen's run);
       - (b) the None/default ratio of the medians of the per-session turn-2+ TTFT medians has a 95% upper bound not above 1.0 (`boot_ci_two_sample` over sessions; with 8 sessions the bound is reported as approximate);
       - (c) both prepared rows still pass `PROBE_CTX_EXACT` at their contexts with the flag (Qwen3.8-27B-MTP-devai-NVFP4 at 118,784, Qwen3.8-27B-W4A16-devai-AutoRound at 131,072), each with its recovery flags; for this probe the flag goes into the AutoRound row's `engine_flags` too.

       Otherwise the default stays: the flag is removed from the teacher row's `engine_flags` in `deploy/recovery-flags.json` (and from the AutoRound row's, if (c) put it there), and each row whose cell (c) replaced is re-probed with `PROBE_CTX_EXACT` at its context without the flag. Nothing is emitted.
     - If it is adopted, the flag lives in the argument builders: the router's `vllmEntrypoint` and the vLLM prober's `vllm_command_args`, from one constant, for launches on the 0.30.0 backend only (`vllm-devai` until 4e, `vllm` after; stock 0.22.1 lacks the flag). It leaves both rows' `engine_flags`, is pinned in `deploy/backend-flags.yaml` and the argv golden, and moves into `pkg/engineargs` in Phase 8. So the probers, the router and `devai-control` launch identically (I1), and recipe-apply cannot write it (Phase 17).
   - **Re-probes now.** KV auto-sizing changed in 0.29 (vllm#53306) and YaRN now matches Transformers (vllm#56446, 0.30), so every cell is re-probed in 4g. 4v re-probes only what the next steps need, with `VLLM_DEVAI_IMAGE` on the 0.30.0 tag, after `make backup-create`:
     - the two prepared rows (the acceptance rule above); Phase 8 step 1 uses the AutoRound row;
     - Phase 15's baseline (I15), probed after the retention decision, so that its cell is measured with the flag if it was adopted: `make probe-vllm-devai PROBE_REPO=ykarout/Qwen3.5-9B-NVFP4`; then `PROBE_REPO=openai/gpt-oss-20b`, and then `PROBE_REPO=bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`, each only if the one before it does not load.
     - gpt-oss-20b is the only on-disk vLLM row with a YaRN block (factor 32 over 4,096, `max_position_embeddings` 131,072, from its `config.json`). So its probe, in 4v or in 4g, shows what vllm#56446 does to it.
   - **The record.** The HyperQwen commit, the contexts and the load probes on both tags, the margins and expected half-widths stated before the runs, the intervals, tests and time-outs, the speed ratios and MTP acceptance lengths, the FlashInfer check, the retention decision, the engine-log facts and the source re-reads go into docs/backends.md, section "vLLM 0.30.0".
2. **4a -- SGLang build spike** (Open question 1).
   - List SGLang 0.5.16's dependencies that contain compiled code and sort them by M3. Its pins include `sglang-kernel==0.4.5`, `sgl-deep-gemm==0.1.4.post1`, `humming-kernels[cu13]==0.1.10`, `flash-attn-4>=4.0.0b18`, `quack-kernels>=0.6.1` and `flashinfer_python[cu13]==0.6.14` (PyPI metadata). The ones that are SGLang's own kernel packages are compiled; the rest come from Debian if its version satisfies the pin, else from PyPI.
   - Build for sm120 with nvcc 13.1 against the pinned `torch==2.11.0`, install in a trixie-slim environment with the host toolkit, and serve one model.
   - The spike also settles how a Debian package satisfies a pin inside the engine's environment: Debian's python3 with system site-packages, or none used.
   - If SGLang cannot be built, and no upgrade builds either, it is dropped (M2) and the rest of this phase proceeds without it.
3. **4b -- `scripts/build-sglang.sh`**, on the pattern of `build-vllm.sh`:
   - sources sha256-pinned; external repositories fetched once, at the revisions the source names; every network step tried 3 times; sm120 only;
   - output: a wheelhouse plus a content hash, and a record of where each dependency came from (compiled, Debian, PyPI).
4. **4c -- `Dockerfile.engines`.**
   - One stage per engine, copied into the final stage least-changed first.
   - The vLLM stage installs the wheelhouse 4v accepted (0.30.0, or 0.28.0 if the move stopped), with the checks `Dockerfile.vllm` makes.
   - The CUDA 13.1 compiler pieces are copied once from the host toolkit, as `Dockerfile.vllm` does today (FlashInfer needs nvcc at run time).
   - Each engine gets an image label, `devai.engine.<backend>`, set to its build's content hash (the `DIST_ID` idea already used for vLLM). The same hashes are written to `/etc/devai/engine-builds.json` (C1 `build`), so code inside the container can read them without podman.
   - `tini` is PID 1 and the default command is an idle loop, so Phase 8 can run the image as one permanent container; the router's current launch overrides the command.
   - The stages are written so that one engine's environment builds and runs alone (`--target <backend>-run`, with the launcher); Phase 23 step 1 uses this for `vllm-s1` before the rest of the image is ready.
5. **4d -- The launcher.** `devai-engine <backend> [<command> [args...]]` sets `PATH` (that engine's environment first), `CUDA_HOME`, `LD_LIBRARY_PATH` and `HOME=/var/lib/devai-engines/<backend>` for that engine. It then `exec`s <command> with its args; with no command it `exec`s the engine's server. For Ollama the command is an `ollama` subcommand (`devai-engine ollama serve`, `devai-engine ollama pull <tag>`). 4c installs the ollama binary at `/usr/bin/ollama`, so `podman exec devai-ollama ollama ...` (`scripts/select-models.py`, the Makefile, `tests/test-model-matrix.sh`) keeps working until Phase 9 moves those commands to `devai-engine ollama`. The image sets no `PATH` into any one engine environment, so a caller that needs an environment's Python goes through the launcher: `scripts/prepare-checkpoint.py` runs each step as `--entrypoint devai-engine <image> vllm python3 <step>`, and `scripts/verify-backend-flags.py` runs `devai-engine vllm python3 -m vllm.entrypoints.openai.api_server --help` and `devai-engine sglang python3 -m sglang.launch_server --help` against `ENGINES_IMAGE`. A test pins these call sites and the launcher's no-command and with-command paths. For each vLLM environment (`vllm`, and `vllm-s1` from Phase 23) it also sets `FLASHINFER_CUBINS_REPOSITORY=file:///nonexistent/`, so no FlashInfer cubin is fetched at run time and every kernel comes from the image or its JIT compile (M3; vLLM 0.31.0 `vllm/utils/flashinfer.py:77-80, 564-614`). 4v measures with the same setting (step 4v, "Run-time cubins off for every 4v launch").
   - For Ollama it also sets `OLLAMA_MODELS=/root/.ollama/models`, so the store stays where it is mounted (C6) and where `scripts/select-models.py` expects it.
   - Each engine thus has its own `~/.cache`. The router's `engineCacheVolumes` and the probers' `engine_cache_volumes` mount the existing `devai-engine-cache-<backend>-*` volumes under that `HOME` (C6).
   - The router's `vllmEntrypoint` / `sglangEntrypoint` and Ollama's launch call the launcher instead of a bare `python3`; so do the probers.
   - One `ENGINES_IMAGE` replaces `OLLAMA_IMAGE`, `VLLM_IMAGE`, `VLLM_DEVAI_IMAGE` and `SGLANG_IMAGE`. It is exported in the Makefile (`export ENGINES_IMAGE`) and pinned by the same test as 4v's `export VLLM_DEVAI_IMAGE`.
6. **4e -- `vllm-devai` becomes `vllm`.** Run Phase 8 step 1 first: it needs `vllm-devai`, which this step retires. The rename covers the router's backend table, compose, the picker's backend lists and agent provider sets (`router-vllm-devai`), the probers, the derived catalog rows (`backend: [vllm]`), and the tests that pin the vllm-devai registration points. So do laya-check (`DEFAULT_TEACHER_PORT` 11435, and the Makefile help's `TEACHER_PORT=11435`), the bench harness (`bench_runner.py`'s caches, metrics URLs and `--backend` choices; `_bench_core.py`'s port map; `clean_slate.py`'s container list; `bench-sync.py`'s BACKENDS; the `bench-vllm-devai` target), and `select-models.py` and `model-sync.py`, which drop the vllm-devai probe cache and store entry. Port 11437 is retired. Every `vllm-devai` in a `backends` list becomes `vllm` in `deploy/recovery-flags.json`.
   - The rows of `deploy/.vllm-devai-reasoning-cache.json` move into `deploy/.vllm-reasoning-cache.json`, and where a key is in both, the vllm-devai row wins: it was measured on the build that serves. The move goes through the same atomic save the probers use, after `make backup-create`. Then the file is removed, with its compose bind mount, `bin/devai-agent`'s staging and mount, `make install`'s symlink and its entry in devai-backup's manifest. A test shows a merged cache keeping both prepared rows' cells.
   - Check: `grep -rnE 'vllm-devai|11437' scripts/ bin/ config/ gpu-arbiter/ deploy/*.yaml deploy/*.json Makefile` finds only comments recording history.
   - If Phase 17a has landed, the same is done in `deploy/recovery-flags.generated.json` (groups recipe-apply wrote before 4e; the file is rewritten whole and renamed, I25), and a test shows a pre-4e `["vllm-devai"]` group applying to a `vllm` launch.
   - If 17a has not landed, there is no generated file to rename and no such group to test; 17a then writes `vllm` from the start.
7. **4f -- Per-engine staleness.** Today the HF probe caches stamp `_meta.current_image_digest`, and three places compare it: `make probe-check`, the router's drift check (`probeCachePathByBackend`, `readProbedImageDigest`) and `make bench-plan` (`stale_image`).
   - With one image for every backend, rebuilding it for an Ollama change would mark every vLLM and SGLang cell and bench row stale.
   - So stamp and compare the engine's `devai.engine.<backend>` label instead. Ollama's cache gets the same stamp (it has none today).
8. **4g -- Re-probe everything** on the new image: `make probe`, `probe-vllm`, `probe-sglang`, `probe-load-vllm`, `probe-load-sglang`.
   - Every vLLM cell is re-probed on the version 4v left in service, including those 4v already probed: KV auto-sizing changed in 0.29 (vllm#53306), YaRN in 0.30 (vllm#56446), and the image is a new build.
   - Run `make backup-create` first.
   - Failures go to the exclusion ledger and the model is dropped from that backend (M8). Of the two models that had their own images, NVIDIA-Nemotron-Nano-9B-v2-NVFP4 is decided here on the single build. diffusiongemma-26B-A4B-it-NVFP4 has no weights on disk, so it is not pulled, not probed and gets no verdict (I17); it returns on `vllm-s1` in Phase 23 (E3).
   - `probe-load-vllm` runs with `PROBE_MTP=1`, so a row whose cell records `mtp_fits: true` is load-probed with its drafter, as the picker serves it (4v's `--mtp`).
   - The two prepared Qwen3.8 rows are not left to the 32K-grid search, which would bring Qwen3.8-27B-MTP-devai-NVFP4 back at 98,304: after `make probe-vllm` and before `make probe-load-vllm`, each is re-confirmed with `make probe-vllm PROBE_REPO=devai/<row> PROBE_CTX_EXACT=<ctx>` at the context 4v accepted (Qwen3.8-27B-MTP-devai-NVFP4 at 118,784, Qwen3.8-27B-W4A16-devai-AutoRound at 131,072), with the recovery flags 4v left in place. The load probe above then covers them at that context. A miss is recorded with the engine's estimate, and it goes back to the operator.
   - It also closes items absorbed from earlier plans. The SGLang cells without a launch fingerprint are re-probed with their load probe. The 7 SGLang cells that still carry a GPTQ "unsupported quantization" verdict (Gemma-4-31B-IT-NVFP4 among them) have no weights on disk and are not downloaded (I17): they are removed from the SGLang probe cache, and each model gets a `manual` ledger verdict saying why (`scripts/probe-cache-drop.py` and `scripts/_model_status.py --exclude`, after `make backup-create`).
   - Rows whose weights are not on disk are not re-probed and get no verdict (I17).
   - The card-derived parser path (`scripts/_card_hints.py`) gets its first real run (I24): gpt-oss-20b is re-probed on vLLM, and on SGLang if kept, with `PROBE_IGNORE_CURATED_PARSERS=1`, which makes the prober skip its curated `parsers:` block, its recipe seed and the name heuristic (Phase 17). The cells go to a scratch cache, with a scratch ledger (`--ledger`), and are compared with the curated cells: the same parsers, or a recorded disagreement. If gpt-oss-20b does not load on vLLM 0.30.0 (0.28.0 if 4v stopped the move), the check runs on SGLang only (if kept) and records why.
9. **4h -- Quantization.** `QUANT_IMAGE` points at `devai-engines`, and `quant-local` runs there through the launcher, `--entrypoint devai-engine $(QUANT_IMAGE) vllm python3 -u /quant/quant_smoke.py ...` (not `bash -lc`: a login shell resets `PATH`).
   - llm-compressor and datasets come from a hash lock, `scripts/quant/requirements.lock` (from `requirements.in`), installed at image build time with `--no-deps --require-hashes` into an overlay on the vLLM environment, then `pip check`; nothing is installed at run time (M1, M3), as `quant-local`'s `pip install` does today. A conflict with the vLLM lock fails the build; `quant-local` then moves to `not_offered` with that reason.
   - `quant-local` computes on the GPU, so it is a GPU action once the registry exists (C8: `job_kind: action`, `gpu: true`, `time_limit_min`; its container labelled `devai.job=$DEVAI_JOB_ID`). Until Phase 10 it runs with the stack down after C8's card check on the host; from Phase 10 on it is submitted with the `engine` license.
   - `quant-cloud` runs on HF Jobs, which can only pull a public image, so it cannot use `devai-engines`. It moves to `not_offered` with that reason and is removed with its `QUANT_IMAGE` use, unless the operator keeps it with its own pinned public image, `QUANT_CLOUD_IMAGE`, which the upstream-images guard then allowlists as cloud-only.

### Exit criteria

- 4v: the acceptance rule passed, and its record is in docs/backends.md; or the move stopped there (a failure or an inconclusive result), 0.28.0 stayed, and E1 went back to the operator with the report. *(Since E7: the rule is E7's -- both rows' contexts and load probes on 0.30.0.)* `devai-engines` serves the vLLM version 4v left in service.
- 4v: both tool-parser plugins import in the new vLLM environment, and `make verify-backend-flags` passes for every vLLM image in use.
- `make probe-check` is clean against `devai-engines` for every row whose weights are on disk, and lists the others as not re-probed (I17); the picker lists re-probed rows only.
- Both prepared Qwen3.8 rows keep the contexts 4v accepted (118,784 and 131,072) in `deploy/.vllm-reasoning-cache.json`, with `mtp_fits` and `serving_ok` true, or the miss went back to the operator; `deploy/.vllm-devai-reasoning-cache.json` no longer exists and nothing mounts or backs it up.
- `grep -rnE 'vllm-devai|11437'` over the paths 4e names finds only comments recording history, and `make laya-check` passes against the teacher on 11435.
- A picker-launched agent completes a turn on Ollama, vLLM and SGLang (if kept).
- `make test-router`, `make test-python`, `make test-vllm`, `make test-sglang` and `make test-e2e` pass.
- Rebuilding only the Ollama part leaves `make probe-check` clean for vLLM and SGLang.
- No file in the repository asserts an SGLang capability that the re-probed caches contradict (Phase 15's documentation check, run again on the new caches).
- Once 4g has passed, 0.28.0's wheelhouse (`~/.cache/devai/vllm-build/0.28.0/`) and image tag may leave the host; not before.
- Done before Phase 8 step 2 starts. Phase 8 step 1 is the exception: it runs on 4v's build before 4e.

### Phase 4 risks

| Risk | Mitigation |
| ---- | ---------- |
| vLLM 0.30.0 loses context or quality on the prepared Qwen3.8 rows | 4v's acceptance rule, fixed before the runs, with MTP on as served: the exact-context and load probes, and quality non-inferiority with stated margins; on a failure or an inconclusive result 0.28.0 stays and E1 goes back to the operator; 0.28.0's wheelhouse and tag are kept until 4g passes |
| HyperQwen's re-port is not mechanical: its own 0.29 -> 0.30 port shipped four defects despite a clean `--fuzz 0` apply (HyperQwen `docs/vllm-0.30.md`) | the newest commit that still pins 0.30.0, not the first; the rule judges the built image on real loads and bench items served with MTP, as the teacher is, not the patch apply |
| the `deepseek_string` and `nemotron_json` plugins stop the server at startup on 0.29 and later (vllm#54492) | ported with a fallback import, and imported in the new image before any probe; only one on-disk row uses either |
| 0.30.0's default prefix-cache retention (0) re-prefills part of every agent turn under MTP | measured in 4v on the teacher with 1,024-token replies, where the two settings can differ; `None` adopted only by a rule fixed before the runs, and then emitted by the argument builders, so probes and serving match (I1) |
| without `flashinfer-cubin`, a FlashInfer kernel comes from somewhere other than the image's nvcc | 4v's empty-cache first-request check, before the acceptance rule; nvcc stays in the image |
| the 0.30.0 build deletes 0.28.0's wheelhouse, the rollback | one build root per version; today's tree moved to `.../0.28.0/` before the first 0.30.0 build |
| a rollback to 0.28.0 keeps what 4v set for 0.30.0: an adopted retention flag 0.28.0 has no argument for, a raised utilization | 4v's rollback reverts both, and restores the vllm-devai probe cache from a backup whose manifest now holds it |
| a rollback after 4d points an image variable back, and launches exec a `devai-engine` the old images lack | the rollback reverts 4a-4h and restores the caches and the ledger from the archive taken before 4e (Migration / rollback story); the old images stay until Phase 14 |
| 4e's rename leaves `vllm-devai` or 11437 in a caller (laya-check, the bench harness, the sync scripts), which then targets a dead port | 4e names them, and its grep finds only history; `make laya-check` is in the exit criteria |
| SGLang does not build or run with nvcc 13.1 on sm120 | 4a answers it first; drop or upgrade (M2) |
| the upstream images carried components the source builds lack (SGLang's recovery flags for Gemma-4 rely on `flashinfer_cutlass`) | the re-probe finds them; the model is dropped or the component is added to the build |
| models lost to the single vLLM version (Nemotron-Nano-9B-v2; diffusiongemma until Phase 23's `vllm-s1`, E3) | accepted (M8); the re-probe decides Nemotron-Nano-9B-v2, and `make probe-check` lists diffusiongemma as not re-probed (I17) |
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
                                            devai-bus, devai-jobs; the files below. The build fails unless
                                            job_submit_lua.so (slurm-wlm-basic-plugins 24.11.5) is installed
deploy/slurm/slurm.conf.in        new    -- ClusterName=devai; SlurmctldHost=localhost; NodeName=devai
                                            NodeAddr=127.0.0.1 CPUs=@CPUS@ RealMemory=@MEMORY_MB@ (I4); PartitionName=devai
                                            Default=YES MaxTime=INFINITE; SelectType=select/cons_tres,
                                            SelectTypeParameters=CR_CPU; Licenses=engine:1,bench:1 (I5);
                                            AuthType=auth/slurm, CredType=cred/slurm, AuthAltTypes=auth/jwt,
                                            AuthAltParameters=jwt_key=/etc/slurm/jwt_hs256.key;
                                            ProctrackType=proctrack/linuxproc, TaskPlugin=task/none,
                                            JobAcctGatherType=jobacct_gather/none;
                                            AccountingStorageType=accounting_storage/slurmdbd,
                                            AccountingStoreFlags=job_comment,job_script;
                                            PrologFlags=Alloc,ForceRequeueOnFail (a guard refusal still requeues
                                            and holds the job); JobRequeue=0 (a job lost with its node never runs
                                            again, C3); JobSubmitPlugins=lua (job_submit.lua, below);
                                            Prolog/Epilog; HealthCheckProgram, HealthCheckInterval=30 (I7);
                                            PrologEpilogTimeout=180; BatchStartTimeout=240; KillWait=30; SlurmdParameters=config_overrides;
                                            ReturnToService=2 (a node set DOWN for any reason, "Node unexpectedly
                                            rebooted" after a host reboot included, returns when slurmd registers;
                                            a DRAIN set by the guard or an epilog stays the health check's, I7);
                                            StateSaveLocation=/var/cache/devai/jobs/slurmctld
deploy/slurm/job_submit.lua       new    -- slurm_job_submit refuses (slurm.ERROR, "devai jobs run as devai only")
                                            any job whose submit_uid or job_desc.user_id is not 1000 (devai);
                                            slurm_job_modify passes every change (root's scontrol, the epilog's
                                            AdminComment; a job's user cannot be changed). Installed as
                                            /etc/slurm/job_submit.lua
deploy/slurm/slurmdbd.conf.in     new    -- MariaDB on localhost, the password from mariadb.pass, auth/jwt as alt type
deploy/slurm/nats-server.conf.in  new    -- C1 users and permissions (the control users' `allow_responses` with
                                            `expires: "15m"`); passwords filled in at start into /run/devai (0600)
deploy/slurm/supervisord.conf     new    -- mariadb, slurmdbd, slurmctld, slurmrestd (user slurmrest, -a rest_auth/jwt,
                                            :6820), slurmd (-N devai; autostart=false, node-recover starts it),
                                            nats-server, the bus log, node-recover (one-shot) and accounting-init
                                            (one-shot: autorestart=unexpected, startsecs=0, exitcodes=0)
deploy/slurm/node-recover.sh      new    -- on every container start, before slurmd (added 2026-10-09): waits for
                                            slurmctld; when it lists jobs on the node as running (they died with the
                                            previous container), sets the node DOWN, so Slurm ends them NODE_FAIL at
                                            once; then starts slurmd, whose registration returns the node
                                            (ReturnToService=2). Slurm alone ended them only at a registration more
                                            than BatchStartTimeout after they were last seen: tens of minutes, with
                                            the engine license held (measured 2026-10-09)
deploy/slurm/entrypoint.sh        new    -- render the configs (the node's CPUs and memory from the host, I4); copy the jobs bus
                                            password to /run/devai/bus/jobs.pass (devai, 0400; I26); run directories for slurm, slurmrest, mysql; initialise
                                            MariaDB on an empty data directory (--innodb-snapshot-isolation=OFF);
                                            exec supervisord (the sacctmgr steps run as the program accounting-init)
deploy/slurm/accounting-init.sh   new    -- on every container start: waits up to 120 s for `sacctmgr -n show
                                            cluster` to succeed (slurmdbd answering), then idempotently (-i,
                                            "already exists" ignored) adds cluster devai, account devai, user devai
                                            in account devai, and sets devai AdminLevel=Operator (I6); root gets no
                                            association. Exits non-zero on the timeout, so supervisord logs it
deploy/slurm/devai-gpu            new    -- stdlib Python, NVML via ctypes (I3): `processes` (pid, used MiB),
                                            `used`, `sample --interval 5 --out F --pidfile P` (one JSON line per
                                            sample, flushed per line; exits 0 on SIGTERM), `energy`;
                                            DEVAI_GPU_FAKE=<json file> for tests
deploy/slurm/prolog.sh, epilog.sh, healthcheck.sh
                                  new    -- slurm.md Sec. 6, I7: the guard, the sampler and the idle check for
                                            engine, trainer, probe and GPU-action jobs only (C8); for every job the prolog creates
                                            results/<jobid>/ and its artifacts/ as devai:0, mode 2775 (I26);
                                            each exports C7's bus settings for user jobs itself (slurmd passes
                                            them no environment); the prolog starts the sampler detached
                                            (slurm.md Sec. 6, step 5) and the epilog stops it by its pid file.
                                            The epilog test switch: `DEVAI_TEST_EPILOG=off` in
                                            /run/devai/test/epilog.conf, a test-only config that only
                                            tests/slurm/test-slurm.sh (and the Phase 10 check) writes, through
                                            `podman exec`, and removes again (an EXIT trap; the tmpfs also drops
                                            it on restart). With it on, the epilog only writes gpu.json and exits
                                            0: no `unload`, `cancel` or action `stop`
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
devai-tools/cmd/devai-bus/        new    -- request (C3: `<subject> <json> --job N [--timeout D]`, --job required;
                                            adds v and job, refuses a body carrying either), publish, hold (C3:
                                            ignores TERM and INT, dies with its --parent, rides out a bus outage and
                                            asks for the state again after it; --lease-only ends on HUP; on
                                            devai.operator.actions it exits with the action, on the trainer with its
                                            run), watch, log; the exit codes of C3
devai-tools/cmd/devai-control/    new    -- main
devai-tools/internal/control/     new    -- config, supervisord XML-RPC client (getProcessInfo, startProcess,
                                            stopProcess, signalProcess, reloadConfig, addProcessGroup,
                                            removeProcessGroup), the state machine, verbs, ownership, the
                                            supervisor verbs on an engine service (C1), the lease timer (counted
                                            from the later of the last lease and the answer to the job's load,
                                            train or run request, C1), the state
                                            file (with the programs still loading or waking, adopted after a
                                            restart, C1), events; kinds `program`, `fake` (an `unload` also stops a load
                                            in progress; a repeated `unload` joins the one under way) and `actions`
                                            (registry C8, run with setsid, reply on start and publish `exited`,
                                            signal the group, stop a run whose lease lapsed 60 s ago, on exit
                                            remove containers labelled devai.job=<id> when a podman CLI is
                                            configured; at its start, publish `lost` for every run in the state
                                            file it cannot adopt and, with a podman CLI, remove every container
                                            labelled devai.job=<id> whose job has no adopted run (the container
                                            was recreated, and its runs died with it); action.log for every run,
                                            result.json at the end of an
                                            `action`-kind run, and for a `bench` or `probe` run only when it left
                                            none, never overwriting one, C4)
devai-tools/cmd/devai-jobs/       new    -- list, show, queue, cancel, node, submit; every command with --json
                                            (basic; Phase 13 completes it)
bin/devai-jobs                    new    -- `podman exec devai-slurm devai-jobs "$@"` (I13; from Phase 10, `submit`, `suspend`,
                                            `interrupt` and `service` run in devai-operator); installed by `make install`
tests/slurm/Dockerfile.control-test
                                  new    -- trixie-slim + supervisor + devai-control. test-slurm.sh runs it as two
                                            containers, so jobs use the production subjects, users and namespaces:
                                            devai-control-test-engines (default user namespace, root, the GPU;
                                            container `engines`, service `fake`: scripts/vram-ballast.py as the
                                            engine, whose `load` holds N MiB and is "serving" once it prints
                                            ready, after a load delay set in that container's control.yaml (0 by
                                            default; the lease-clock and cancel-while-loading checks set one)) and devai-control-test-operator (keep-id, user devai, no GPU;
                                            container `operator`, service `actions`, with a test registry); each
                                            with its own control.yaml
tests/slurm/test-slurm.sh         new    -- the live checks in the exit criteria; refuses to run while a production
                                            devai-control runs (devai-engines or devai-operator), since both would
                                            answer the same subjects and match the guard's command-line lookup, and names
                                            the commands that stop them. It runs from the host with the stack down:
                                            `make cache-down` (before Phase 9 also `make slurm-down`), and once devai-operator exists also `podman stop
                                            devai-operator` (stack-down leaves it running, Phase 6). It starts both
                                            test containers and removes them afterwards, and an EXIT trap brings
                                            devai-slurm back as compose defines it (`make cache-up`, with
                                            compose.gpu.yaml from Phase 11), then runs `make operator-up` if it
                                            found devai-operator stopped; the unkillable-holder
                                            check starts the ballast through `sudo -n` as root (skipped, exit 77,
                                            without passwordless sudo)
tests/python/test_slurm_config.py new    -- static: the C1 table equals nats-server.conf.in, `expires: "15m"` included; the
                                            slurm.conf.in keys and their rendering above (JobRequeue=0,
                                            ReturnToService=2, JobSubmitPlugins=lua among them); the compose service
                                            has no podman socket and publishes nothing;
                                            the epilog test switch is absent from the production slurm.conf.in,
                                            the entrypoint, the Dockerfile and compose
tests/python/test_devai_gpu.py    new    -- devai-gpu and the guard's decisions with DEVAI_GPU_FAKE
deploy/docker-compose.yaml        modify -- service `slurm` (container devai-slurm, hostname devai-slurm, devai-net
                                            only, pid: host, the default user namespace (I26), tmpfs /run/devai,
                                            the GPU when present, C6/C7 mounts, the environment of C7:
                                            SLURM_REST_URL, SLURM_JWT_KEY_FILE, SLURM_USER); profile `slurm`
                                            until Phase 9
Makefile                          modify -- build-slurm, slurm-init, slurm-up, slurm-down, test-slurm
scripts/slurm-init.sh             new    -- C7 (the Slurm part)
devai-tools/internal/backup/manifest.go
                                  modify -- ~/.config/devai/slurm/
devai-tools/internal/backup/snapshot.go
                                  modify -- MkdirAll(destDir, 0o700); the archive created with os.OpenFile(path,
                                            O_WRONLY|O_CREATE|O_EXCL, 0o600), so the umask cannot widen it: it
                                            now holds C7's 0600 files
devai-tools/internal/backup/roundtrip_test.go
                                  modify -- the snapshot archive's mode is 0600
docs/backup-restore.md            modify -- archives are written 0600, because they contain C7's secrets
```

### Detailed steps

1. **Open question 2 first.** Build the image and start it unprivileged (`--pid=host`, the GPU, podman's default user namespace: I26), with slurmd on `proctrack/linuxproc`.
   - Submit `sleep 600` with `-L engine` and cancel it; check that its process tree is gone.
   - If slurmd refuses to run this way, fall back to the measured setup: `--privileged --cgroupns=private`, the cgroup step in the entrypoint, `cgroup.conf` with `IgnoreSystemd=yes`, `proctrack/cgroup`.
   - Record which setup runs.
2. **The REST spec.** Dump it from this slurmrestd (`slurmrestd -d v0.0.42 -s slurmctld,slurmdbd --generate-openapi-spec`) into `deploy/slurm/openapi-v0.0.42.json`.
   - Check the names C3 and C5 use against it: `job.script`, `comment`, `licenses`, `requeue`, `time_limit`, `environment`, `standard_output`, the kill-warning signal, delay and batch flag (`kill_warning_*`), `signal` with `flags=BATCH_JOB`, hold and release, node `RESUME`.
   - Where 24.11 differs, correct the contract. The spike measured v0.0.44 on 26.05.
3. **Accounting.** MariaDB with `--innodb-snapshot-isolation=OFF`, slurmdbd, the `sacctmgr` steps (`accounting-init.sh`); `job_submit.lua` refuses a job from any user but devai. A REST-submitted job's JSON comment must arrive intact in `sacct` (measured on 26.05; re-checked here on 24.11).
4. **`pkg/slurmrest`.** Test it against an `httptest` server that checks every request against the spec dump. Test the JWT live in `test-slurm.sh`.
5. **`pkg/bus`, `devai-bus`, `devai-control`.** Go tests with an embedded nats-server started from the rendered C1 permissions (not an open configuration), a fake supervisord XML-RPC server and a 2 s lease. They cover:
   - every verb, and `not_owner`;
   - lease expiry;
   - the lease clock (C1): a fake engine whose load takes three test leases reaches `serving` and is not unloaded, while its job then leases; with the job script SIGKILLed during that load (no epilog), the model reaches `serving` and is unloaded one test lease later;
   - adopting a running program from the state file after a `devai-control` restart, including one still loading: with `devai-control` killed during a fake-engine load, the restarted `devai-control` publishes `serving` for the same job, the job's `devai-bus request ... load` returns ok (exit 0) from that event, and the hold starts;
   - an action hold whose `devai-control` is replaced by a fresh one with an empty state file exits 3 within one heartbeat;
   - the permissions (a `jobs` client cannot publish `devai.engines.ollama.restart`);
   - an `actions` run refused when its parameters break the registry; a run whose process group is signalled and then killed after its grace; a run stopped once its lease lapses; `run` replying on start and `exited` arriving as an event; a `run` refused after a `stop` for its job; a forced `stop` with `run_job` stopping another job's run, and `not_owner` without `force`; a hold whose parent shell is SIGKILLed exiting within 1 s, and one started with a dead `--parent` exiting at once.
   - `devai-bus request`: refused (exit 1) without `--job`, and for a body that carries `job` or `v`; the request on the wire carries `v: 1` and the `--job` value.
   - the embedded nats-server shut down and restarted on the same port: (i) for 1 s while a fake-engine hold and an action hold run, both renew within 2 s of the restart and the model stays loaded; (ii) with the action exiting during the outage, its hold exits with the action's code after the reconnect; (iii) for longer than the 2 s test lease, the engine hold exits 0 after the reconnect, because the model was unloaded; (iv) a hold that cannot reach the bus before its first `state` reply exits 2.
   - `unload` of a fake engine that is still loading stops the load, answers the waiting `load` `load_failed` (`unloaded`) and leaves `idle`, never `failed`; a second `unload` during the first returns with it; one after it returns ok at once.
   - the supervisor verbs (C1): `stop` on a fake engine owned by a job, sent as job 0 without `force`, unloads it, and that job's hold exits 0; `restart` on a serving one is refused with `busy`; `start` on an idle one replies with the state.
   - result.json (C4): an `action`-kind run gets devai-control's; a `bench`-kind run that writes its own keeps it after `exited`; one that writes none gets the fallback.
6. **The guard.** `devai-gpu`, the prolog, the epilog and the health check.
   - Unit tests with `DEVAI_GPU_FAKE`: idle; a devai holder that unloads; a foreign holder that is killed; a foreign holder of another user (exit 1); the energy delta; a bench or action job, whose prolog and epilog leave the card alone even when it is busy; a holder in the operator container whose job is gone (a fake environ carrying `DEVAI_JOB_ID`): a forced `stop` with `run_job`, then a kill; one without `DEVAI_JOB_ID`, treated as foreign; two `devai-control` processes in one cgroup, a configuration error (exit 1); a trainer job, whose epilog sends `cancel`, not `unload`.
   - The sampler: the prolog returns within 15 s with the sampler still running (its pid alive after the prolog exits); the epilog stops it and `gpu.json` holds at least one sample; a killed sampler gives `gpu.json` with `sampler_lost: true`, and the epilog still exits 0 when the card is idle.
   - Live tests on the card, in `test-slurm.sh`.
   - Mapping an NVML PID to a devai container: compare `/proc/<pid>/cgroup` with that of the `devai-control` process whose command line names the container. Both are read through `--pid=host`. Each container holds exactly one `devai-control`. A holder in devai-operator is matched to its run by the `DEVAI_JOB_ID` in its `/proc/<pid>/environ` (C8 sets it for every `job` action, and children inherit it). devai-slurm's root and devai-operator's devai are the same host uid, so it is readable (untested so far).
   - Check whether 24.11 accepts `scontrol update JobId=<id> AdminComment=...` from the epilog of a completing job. If it does not, the summary stays in `gpu.json`, and Phase 13 reads it from there.
   - Check whether a node drained by a failed prolog keeps a reason that the health check can recognise; otherwise the health check also reads `guard.json`.
7. **The bus log.** `devai-bus log --dir /var/cache/devai/jobs/bus` writes one JSON line per event, with the receive time.
8. **Compose, Makefile, init, backup.** `slurm-init`; `slurm-up` refuses without `/var/cache/devai/jobs` (a plain directory the operator creates, E9; `setup-jobs` and its LV were removed); the backup entry, and archives written 0600 (C7's secrets are in them).

### Exit criteria

- After `make slurm-init build-slurm slurm-up` (with `/var/cache/devai/jobs` created), `sinfo` shows `devai` idle, `sacctmgr show cluster` lists `devai`, `sacctmgr -n show user devai format=adminlevel` prints `Operator` (also after `podman restart devai-slurm`), and `podman port devai-slurm` lists nothing.
- A job submitted through slurmrestd from another container on `devai-net`, with a JWT signed by `pkg/slurmrest`, runs, and its JSON comment arrives intact in `sacct`. A second `-L engine` job waits as `PENDING (Licenses)`.
- A job submitted through slurmrestd with a JWT signed for `sun: root`, or for any user other than devai, is refused at submission and never runs.
- `bin/devai-jobs list --json` succeeds against the live devai-slurm, and a prolog's `devai.events.slurm.guard` event lands in the bus log (C7: devai-slurm's environment; the guard sets its own bus settings).
- `podman restart devai-slurm` while a fake-engine job and an action job are RUNNING leaves both NODE_FAIL in `sacct` as soon as `devai-slurm` is back (`node-recover`; not "once `BatchStartTimeout` has passed": Slurm alone took tens of minutes, measured 2026-10-09), the node returns to service, and neither runs a second time.
- After `scontrol update NodeName=devai State=DOWN Reason=test` and a restart of the slurmd program in devai-slurm, `sinfo -h -o %T -n devai` prints `idle` within 60 s without a manual resume. A node drained by the guard stays drained until the health check resumes it.
- Against the test containers:
  - an engine job loads the fake engine (2 GiB held), and `sacct` or `gpu.json` shows the GPU summary;
  - `scancel` unloads it and the card is back at its baseline;
  - when the job script is killed with SIGKILL, the epilog unloads;
  - when the job script is killed and the epilog is switched off (the epilog test switch, `DEVAI_TEST_EPILOG=off`, deliverables above), the lease unloads within 70 s;
  - an engine job whose fake engine takes 90 s to print ready, longer than the 60 s lease, reaches `serving` and stays loaded (C1: no lease is needed during a load); one that takes 150 s, run through nats-server with the rendered C1 permissions, has its `load` answered `serving`, and `devai-bus request` exits 0 (the 15-minute response permission, C1); with its job script killed during that load and the epilog switched off, the model reaches `serving` and is unloaded within 70 s after that;
  - when an action job's script is killed with SIGKILL, the action's process group is stopped within 70 s;
  - `scancel` of an action job whose action needs 15 s to stop on TERM: the job stays RUNNING or COMPLETING until the action's `exited` event; the same for a `scancel` during the `run` request. (An action job holds no license, so no queued job can show the release here.)
  - `scancel` of a bench job (kind bench, `-L bench`, running a test bench action in the operator test container that needs 15 s to stop on TERM): a second `-L bench` job, submitted before the scancel, shows `PENDING (Licenses)` and starts only after the first job's run has published `exited` and its epilog has ended. The same for a `B:INT` warning: the second job starts only after the first job's prefix-scoring run has published `exited`.
- Ownership and isolation (I26):
  - In `results/<jobid>/`, the operator test container's devai-control, running keep-id as the host user, writes `action.log` and, for an `action`-kind run, `result.json`. Every file carries the host user's group, and the host user can read every file and delete the directory.
  - A job cannot read `slurm.key`, nor signal a process of the host user.
  - The guard kills a GPU holder that runs as a non-root user in another rootless container.
- The next engine job's prolog starts only after the previous engine job's epilog has ended. Slurm returns the license only after the epilog, and starts nothing on a node with a completing job (read in the 24.11.5 source; checked here).
- A foreign VRAM holder (the ballast in a plain container) is killed by the next GPU job's prolog. A foreign holder the guard cannot kill (the ballast run as another host uid: `sudo -n python3 scripts/vram-ballast.py 2048` on the host; root is outside devai-slurm's user namespace and shows as 65534) drains the node and the job is held; the health check resumes the node within 60 s of the holder exiting (test-slurm.sh ends the ballast with `sudo -n kill`). Without passwordless sudo, test-slurm.sh skips this check with exit 77; the operator runs it with `sudo -v` followed by `make test-slurm ONLY=holders` (that section alone, added 2026-10-10: a full run reaches it after sudo's 15-minute default), and the result is recorded in docs/slurm.md.
- While the node is drained, a license-free action job submitted then stays PENDING (reason shown by `devai-jobs queue`) and starts within 60 s of the health check resuming the node.
- `devai-bus request devai.engines.fake.stop '{}' --job 0`, sent as user operator while a fake-engine job serves, unloads the fake engine; the job ends 0, its epilog's `unload` is a no-op, and every event lands in `jobs/bus/<date>.jsonl`. `restart` on the serving fake engine is refused with `busy`.
- An engine job cancelled while the fake engine is still loading, before it prints ready: the trap's `unload` stops the load, the card is back at its baseline, the epilog passes without draining the node, and the job ends 0. An epilog `unload` sent while the trap's is still running returns when that one ends; one sent after it returns ok at once.
- The bus rides out a restart: `supervisorctl restart nats-server` while a fake-engine job serves and an action job runs leaves both RUNNING and the fake engine loaded. With nats-server stopped for 70 s instead, longer than the lease, the lease unloads the fake engine, and once the bus is back the engine job's hold sees it and the job ends 0.
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
                                            build-ollama.sh list, from NVIDIA's debian13 repository. Go at the exact
                                            version both go.mod files name (1.27.x today; O8's 1.26 is the floor,
                                            not the pin), with GOTOOLCHAIN=local, so a go.mod bump fails the build
                                            instead of downloading a toolchain; the rustup toolchain
                                            rust-toolchain.toml names, and uv, as pinned
                                            downloads checked against their published checksums (O8; uv follows
                                            the same rule)
deploy/operator/requirements-hf.lock
                                  new    -- huggingface_hub with hashes; `make operator-lock` regenerates it
deploy/operator/apache.conf       new    -- one TLS virtual host on 8443; AuthType Basic with AuthUserFile
                                            /etc/devai/operator/htpasswd on every path (O7); WSGIDaemonProcess
                                            devai-operator, user devai; HSTS; `Header always set
                                            Content-Security-Policy "default-src 'self'; script-src 'self';
                                            object-src 'none'; base-uri 'none'; frame-ancestors 'none'"` and
                                            `X-Content-Type-Options nosniff` (static/ holds every script; no
                                            inline script)
deploy/operator/supervisord.conf  new    -- apache2 in the foreground; the log follower (deploy/logging.sh, I20);
                                            Phase 10 adds devai-control
deploy/operator/actions.yaml      new    -- C8; this phase's entries: stack-up, stack-down (direct), model-pull,
                                            probe-ollama, probe-vllm, bench, backup-create; everything else in
                                            `pending` or `not_offered`
devai-operator/devai_operator/    new    -- the Flask app: app.py (routes), registry.py (load and validate C8, build
                                            argv and env), runner.py (direct runs: each starts as `setsid python3
                                            -m devai_operator.runwrap <run-dir> -- <argv>`, outside Apache's
                                            processes. The wrapper is the session leader: it records its pid and
                                            its start time from /proc in run.json, starts the action with
                                            DEVAI_RUN_ID=<run> in its environment, waits for it, and writes
                                            `exit_code` and `ended_at`, and `cancelled` after a stop. A run whose
                                            wrapper pid is gone, or no longer has the recorded start time, with no
                                            `ended_at`, is shown `lost`. Stop signals the wrapper's process group
                                            only after checking that start time, so a reused pid is never
                                            signalled, and after the grace removes the containers labelled
                                            devai.run=<run>. run.json and the log live under jobs/operator/<run>/),
                                            runwrap.py, csrf.py, startup.py (refuses to start without path
                                            identity, without `$HOME` equal to `HOST_HOME_DIR`, or without the
                                            podman socket), templates/, static/ (no external CDN)
devai-operator/wsgi.py            new
devai-operator/tests/             new    -- registry validation (a value outside its pattern never reaches argv),
                                            argv and env building (with a `.env` that sets GPU_MEMORY_GB=16, the
                                            environment built for the probe action carries GPU_MEMORY_GB=16, the
                                            value `make probe` exports), output encoding (a run log and an alert
                                            containing `<img src=x onerror=alert(1)>` come back escaped in the page
                                            and verbatim as text/plain from the raw-log route), CSRF, the start-up
                                            checks, and the coverage test:
                                            every documented Makefile target is in some action's `covers`, in
                                            `not_offered` with a reason, or in `pending`
scripts/ops/stack-up.sh, stack-down.sh, model-pull.sh, probe-ollama.sh, probe-vllm.sh, bench.sh, backup-create.sh
                                  new    -- the recipes of this phase's actions, moved out of the Makefile (O5).
                                            backup-create.sh passes `--home-dir "$HOME" --repo-root "$DEVAI_REPO"`
                                            and refuses when `$HOME/.config/devai` is missing; bench.sh mounts the
                                            HF token by `HF_TOKEN_FILE`'s host path (C7)
scripts/operator-init.sh          new    -- C7 (the operator part): htpasswd -B for the one account, prompting for
                                            the password; TLS from mkcert files if present, else self-signed
deploy/docker-compose.yaml        modify -- service `operator` (container devai-operator, devai-net, userns keep-id and
                                            user devai (I19), the C6 mounts except the JWT key and the bus passwords
                                            (Phase 10), HOME=HOST_HOME_DIR (C6), CONTAINER_HOST,
                                            DEVAI_REPO=$(CURDIR) for path identity, the
                                            host check's DEVAI_GPU_DEVICE, and the lab's host context as Make
                                            evaluates it on the host: DEVAI_HOST_IP=$(HOST_IP),
                                            DEVAI_HOME_VOLUME=$(HOME_VOLUME), DEVAI_HOST_WORK_DIR (HOST_WORK_DIR
                                            resolved with readlink -f), LAB_PORT, DSH_PORT, and
                                            DEVAI_HOST_GITCONFIG / DEVAI_HOST_SSH_DIR only when HOST_HOME_DIR is
                                            set and the file or directory exists on the host; a changed value
                                            takes effect only after `make operator-up`, as LIBRARY_DIR does); the `logger` service removed (I20)
deploy/logging.sh                 modify -- runs in devai-operator, against its podman socket (I20)
devai-tools/internal/backup/manifest.go
                                  modify -- ~/.config/devai/operator/
tests/python/test_upstream_images.py
                                  modify -- the logger's image leaves the transitional allowlist
Makefile                          modify -- build-operator, operator-init, operator-up, operator-lock, test-operator; the moved
                                            recipes call their scripts; `OPS_ENV_VARS`, the variables the Makefile
                                            passes to scripts/ops/ (at least GPU_MEMORY_GB, MAX_CONTEXT_LEN,
                                            VLLM_IMAGE, SGLANG_IMAGE, OLLAMA_IMAGE, DEVAI_GPU_DEVICE, the proxy
                                            variables, DEVAI_REASONING, LIBRARY_DIR, HOST_HOME_DIR), which every
                                            recipe passes explicitly, and `ops-env`, which prints them as
                                            `NAME=value` lines after `-include .env` and the `?=` defaults (C8);
                                            `export DEVAI_REPO := $(CURDIR)` unless Phase 17a
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
2. **The app.** Registry, runner, CSRF and the pages. Parameters are checked against the registry and passed as an argument list; nothing goes through a shell. Every value that comes from a log, run, job, bus event, router answer or library alert is shown as text: Jinja autoescape stays on for every template (no `|safe` on such values), and page scripts insert polled text with textContent or createTextNode, never innerHTML. A raw log is served as `text/plain; charset=utf-8`.
3. **The container.** Path identity: the repository, `/var/cache/devai`, the lab's home volume and the host user's home (`HOME` = `HOST_HOME_DIR`) are mounted at their host paths, because the scripts pass host paths to podman. The start-up check refuses to start otherwise.
4. **This phase's actions.** Their scripts under `scripts/ops/`, and the Makefile targets calling them. Each script that starts a container passes `--label devai.run=$DEVAI_RUN_ID` when DEVAI_RUN_ID is set, beside the devai.job label that Phase 10 adds.
5. **The coverage test,** with the `pending` list that Phase 7 empties.
6. **The probers' imports** are checked against the image's Python (stdlib and PyYAML): this phase already runs probes from it.
7. **`stack-down` leaves `devai-operator` running**, with its direct runs; it stops every other service. `stack-up` likewise names every service but `devai-operator`, so neither action ever recreates the container it runs in. Rebuilding or restarting `devai-operator` itself is a host command (`make operator-up`; its recipe moves to `scripts/ops/operator-up.sh` in Phase 7, and the systemd unit runs it), which also runs the host check (`scripts/host-gpu.sh`) and passes its result in as `DEVAI_GPU_DEVICE`.

### Exit criteria

- After `make build-operator operator-init operator-up`, `https://<host>:8443` asks for the login. A wrong password is refused and appears in Apache's log; the right one shows the stack page.
- From the browser:
  - the stack goes down and up;
  - a small catalog model is pulled;
  - `probe-ollama` runs for one model;
  - a bench of one task runs;
  - each run's log is visible while it runs and afterwards, and stop works.
- A parameter outside its pattern is refused, and no process starts. A POST without the CSRF token is refused.
- Restarting Apache (`supervisorctl restart apache2`) does not kill a running direct action. A direct action that ends while Apache is being restarted shows its real exit code and `ended_at` in run.json. A bench stopped from the page leaves no bench container (`podman ps --filter label=devai.run=<run>` is empty).
- The log follower writes a file for every devai container from inside `devai-operator`, and no container other than `devai-operator` and the router mounts the podman socket (the router keeps it until Phase 9).
- `make test-operator` passes, including the coverage test.

### Phase 6 risks

| Risk | Mitigation |
| ---- | ---------- |
| the web login controls the host user (podman socket), and it is on the LAN (O6) | TLS only; one strong htpasswd password (O7); CSRF tokens; parameters never passed through a shell; login attempts logged; log, job and alert text shown as text only, under a strict Content-Security-Policy (step 2) |
| the toolbox image is large (CUDA 13.1 dev packages, Go, Rust) | one image, shared layers |
| a path mounted at a different place breaks every podman call that passes it | path identity for the repository, `/var/cache/devai`, the lab's home volume and the host user's home (`HOME` = `HOST_HOME_DIR`); the start-up check |
| a direct action dies with an Apache restart | started with `setsid` through `runwrap`, outside Apache's processes; the wrapper records the end |
| Docker's repository key changes | the fingerprint check fails the build loudly; the key is updated deliberately |

---

## Phase 7 -- Recipes into scripts

### Goal

Every documented Makefile target is either an action in the registry, or listed as not offered with a reason. Every offered target's recipe is a one-line call to its script (O5). A target that a later phase adds follows the same rule from the day it lands (C8, "Coverage of new targets"), so the coverage tests keep passing.

### Deliverables

```
scripts/ops/*.sh                  new    -- one per operation: `set -euo pipefail`; no defaults of its own: the
                                            variables it reads come from the Makefile (`OPS_ENV_VARS`, passed
                                            explicitly) or, for a page run, from `make ops-env` (C8); parameters as
                                            the registry maps them
scripts/ops/operator-up.sh        new    -- `make operator-up`'s recipe (Phase 6): the host check
                                            (scripts/host-gpu.sh), then compose for devai-operator alone; the
                                            override files Phases 11 and 20 add to it (compose.gpu.yaml,
                                            compose.operator-library.yaml) when they apply. A `host` action:
                                            it recreates the container the page runs in
deploy/systemd/devai-infra.service
                                  modify -- runs `make -C <repo> operator-up cache-up` (step 1); no compose
                                            arguments of its own
Makefile                          modify -- recipes call scripts/ops/; variables passed explicitly, every one of them
                                            in `OPS_ENV_VARS` (Phase 6), so `ops-env` prints them all; install-systemd
                                            writes the unit with this repository's path and stages nothing
deploy/operator/actions.yaml      modify -- filled; `pending` empty
tests/python/test_makefile_ops.py new    -- every offered target's recipe is one call to scripts/ops/ (or a dependency
                                            list); the registry's `covers` and `not_offered` together equal the
                                            documented targets
```

### Detailed steps

1. Inventory the 124 documented targets by group.
   - Interactive targets (`lab`, `shell`) become job actions in the `lab` group (Phase 12; `job_kind: action`, no license, O2) or are not offered.
   - Host-root targets (`setup-logs`, the volume set-ups) become `host` actions (O3).
   - Targets Phase 1 removed are gone already. `secrets-tmpfs` in O3 went to the attic with sops/age (M12).
   - `install-systemd` is already broken: its unit runs compose from a copy in `~/.config/devai`, which stages none of six current mounts. The unit now runs the same scripts as the Makefile, so each override file a later phase adds reaches it without anyone returning to `install-systemd`:
     - `WorkingDirectory=<repo>` and `Environment=DEVAI_REPO=<repo>` (I25);
     - `ExecStart=/usr/bin/make -C <repo> operator-up cache-up`: `scripts/ops/operator-up.sh` starts devai-operator, then `scripts/ops/stack-up.sh` starts every other service, each with its own override files, its host check and its checks (LS15 from Phase 20);
     - `ExecStop=/usr/bin/make -C <repo> cache-down`, which, like the stack-down action, leaves devai-operator running (Phase 6, step 7).

     The unit holds no compose arguments of its own. It goes through `make` rather than calling the two scripts directly because `.env` is Make syntax (`.env.example` has `HOST_HOME_DIR=$(HOME)`), which systemd's `EnvironmentFile=` cannot read, and the Makefile passes its variables to the scripts explicitly (risks below). In this phase it adds no override file, because none exists yet (compose.gpu.yaml comes in Phase 11, the library's in Phase 20). Running compose in the repository also keeps compose's `./` paths away from the key directories under `~/.config/devai` (C7). `install-systemd` becomes a `host` action, since it writes the host user's systemd directory, and `operator-up` one too, since it recreates the container the page runs in.
2. Move the recipes group by group: stack and build first (the most used), then models, probe and bench, then tests, logs and maintenance.
3. For each moved target, run the target before and after and compare what it does: its output, and the files it writes.

### Exit criteria

- The coverage test passes with `pending` empty.
- `make test-python` and `make test-operator` pass.
- `make cache-up`, `make probe PROBE_MODELS=<one>` and `make bench-plan` behave as before.
- After `make install-systemd`, `systemctl --user restart devai-infra` brings up devai-operator and the stack as `make operator-up cache-up` does, and `systemctl --user cat devai-infra` shows no compose arguments.

### Phase 7 risks

| Risk | Mitigation |
| ---- | ---------- |
| a recipe's Make variables (defaults, `?=`, `$(CURDIR)`) behave differently in a script | the Makefile passes `OPS_ENV_VARS` explicitly, and C8's runs read the same list through `make ops-env`; scripts declare no defaults of their own; step 3 compares behaviour |
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
                                            registries, store, port) -> argv + env, which appends the engine's sleep
                                            support when the launch's sleep_mode is true (vLLM: --enable-sleep-mode
                                            and env VLLM_SERVER_DEV_MODE=1; SGLang: --enable-memory-saver), after
                                            the recovery flags are merged, so a recovery flag cannot drop them. The
                                            router's own launch path does not read sleep_mode before Phase 9, so
                                            argv-golden.txt is unchanged; new FromLoad cases pin both values of
                                            sleep_mode on each engine; the router passes 11434
                                            (unchanged), devai-control the port from control.yaml
gpu-arbiter/main.go, recovery_flags.go, vllm_plugins.go
                                  modify -- use pkg/engineargs; argv_golden_test.go and flags_allowlist_test.go move
                                            with the builders, their corpus unchanged; `-print-load` (C2)
devai-tools/internal/control/engine_ollama.go
                                  new    -- load: restart serve on changed settings (I8), /api/chat warm load with
                                            num_gpu and keep_alive -1, check /api/ps; unload: keep_alive 0, then wait
                                            for /api/ps empty and the NVML baseline; while `serving`, a liveness check
                                            on every 30 s heartbeat and on supervisord's PROCESS_STATE events:
                                            `ollama serve` not RUNNING, or `/api/ps` no longer listing the loaded
                                            model, sets status `failed` with `error` = {code `engine_died`, the log
                                            tail}
devai-tools/internal/control/engine_hf.go
                                  new    -- vllm and sglang: write the program file (autostart=false,
                                            autorestart=false, stopasgroup, killasgroup, stopwaitsecs=30, the log
                                            file of C4), delete vLLM's compile cache (C6), start, wait for /health
                                            (fail fast if the program exits; the load's `timeout_s`, C1); sleep and
                                            wake; unload of the loaded model or the sleeper (I9); at most one
                                            sleeper; the sleeper's VRAM (below); a vLLM or SGLang load on a host
                                            without a GPU is refused with no_gpu; while a model is `serving` or
                                            `sleeping`, a liveness check on every 30 s heartbeat and on
                                            supervisord's PROCESS_STATE events: the program not RUNNING, or a
                                            serving model's `/health` failing 3 times in a row (SGLang: also
                                            `/health_generate`, because its HTTP server outlives its scheduler),
                                            sets status `failed` with `error` = {code `engine_died`, the log tail},
                                            and the program is stopped; the next `load` or `unload` from any job
                                            starts from `failed` as from `idle`
devai-tools/internal/control/trainer.go
                                  new    -- train and cancel (C1): `train` takes the store lock (jobs.py), refuses
                                            (`bad_request`, nothing started) a record that is already terminal,
                                            and marks it `devai.run_by: control` before it starts the program
                                            `python -m laya_trainer.run_job --store <store> --catalog <catalog>
                                            --job <ft_job> --device <device>` (the trainer service's values in
                                            control.yaml, C7; the controller passes the same four today,
                                            controller.py) (autostart=false, autorestart=false, stopasgroup,
                                            killasgroup, stopwaitsecs=30, stdout_logfile=/laya/runs/<ft_job>/run.log,
                                            redirect_stderr=true, stdout_logfile_maxbytes=0, so a traceback is kept
                                            beside the record); `cancel` keeps its reason, stops the program and
                                            waits for its exit; after every exit, `python -m laya_trainer.jobs
                                            finalize --store <store> --job <ft_job> --exit-code <n> [--reason <r>]`,
                                            then the NVML baseline, then the `ended` event and the `cancel` reply; the
                                            states idle, running,
                                            stopping, and one per job kept 10 minutes (C1); at its start, a
                                            non-terminal record whose run had started (run_job wrote its phase) and
                                            whose program it cannot adopt (the container restarted) is finalized
                                            as failed `trainer_stopped`
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
                                            spawns nothing; a job is in progress (409 `job_in_progress`) while any
                                            record in the store is non-terminal, read from the store, not from
                                            memory; `cancel` takes the store lock (jobs.py) and finalizes the
                                            record `cancelled` only if it is non-terminal and not yet marked
                                            `devai.run_by: control`; a marked record is returned unchanged and
                                            ends through the trainer job's cancel (C1 `cancel`, `finalize`), which
                                            the router sends next (Phase 9); no start-up reconcile (devai-control fails a run
                                            it found gone, trainer.go above; the router fails a job with no trainer
                                            job, Phase 9);
                                            POST /devai/v1/jobs/{id}/fail (called by the router only)
laya-trainer/laya_trainer/jobs.py modify -- a store lock (one `flock` on `<store>/runs/.lock`) shared by the
                                            controller's create and cancel and by devai-control's `train` mark and
                                            `finalize`; a `__main__` argparse CLI, `finalize --store <dir> --job
                                            <id> --exit-code <n> [--reason <r>]`; the
                                            `finalize` entry point for devai-control (changes only a non-terminal record):
                                            controller.py's `_write_final` rules, moved here and shared by both. A run
                                            that completed (exit 0 and outcome.json's exit_code 0) is `succeeded`
                                            whatever arrived meanwhile; else after a `cancel`, failed `time_limit`
                                            for that reason and `cancelled` for any other; else failed with
                                            outcome.json's code, or with the exit code when there is none; failed
                                            `trainer_stopped` for a run whose program is gone with no exit code.
                                            run_job.py is unchanged: on SIGTERM it dies as today, and the record is
                                            written after its exit, so job.json keeps one writer at a time
                                            (jobs.py's own rule)
deploy/docker-compose.yaml        modify -- profile `slurm`: service `engines` (devai-engines, the C6 mounts, the GPU
                                            when present); laya-trainer runs supervisord; both run as root (I26)
tests/slurm/test-engines.sh       new    -- the live checks below; starts the trainer with LAYA_RUN_BY=control, so
                                            the hand-submitted trainer jobs exercise the new path before Phase 9
                                            flips the default
deploy/slurm/prolog.sh, epilog.sh, healthcheck.sh, devai-gpu
                                  modify -- tolerate a sleeper: devai-engines' processes count as the sleeper's while
                                            that container reports a sleeper and no loaded model, up to its `used_mib`
                                            + 256 MiB (C1). A probe or GPU-action job's prolog (a GPU action: a
                                            job whose comment carries `gpu: true`, C3, C8) unloads the sleeper
                                            instead (`unload`, `target: sleeper`, `force`): a probe measures an
                                            empty card, and its ballast is sized for the full one; a GPU action
                                            computes on the whole card (Phase 23 adds an engine job for `vllm-s1`
                                            to this rule). Unit tests for each, the GPU-action case with a fake
                                            job comment
devai-tools/internal/control/nvml.go
                                  new    -- memory readings through nvidia-smi (C1, "Memory readings")
devai-tools/internal/control/recovery.go
                                  new    -- the recovery files and the plugin registry re-read at every load (I18, I25)
```

**The sleeper's VRAM** (`engine_hf.go`, `engine_ollama.go`). This covers a load on another service while a model sleeps. The two engines size memory in opposite ways (read in the vLLM 0.28 and SGLang 0.5.16 sources, 2026-10-06). vLLM's startup check is unchanged in 0.31.0 (`v1/worker/utils.py:539-558`), and Phase 4 step 4v re-reads it in 0.30.0, the version in service (E1):

- **vLLM** takes `--gpu-memory-utilization` of the card's total memory, and refuses to start when less than that is free.
  - `devai-control` lowers it by the sleeper's `used_mib` over the card's total, and records the change in its reply.
  - Not for a model whose recovery flags set it (C2: 9 of the 14 curated entries do, the Qwen3.8 rows among them). For those only the retry below applies.
- **SGLang** takes `--mem-fraction-static` of the memory free when it starts, so it starts with a sleeper on the card, and nothing is lowered.
  - But its KV pool and runtime headroom shrink by about that fraction of the sleeper's `used_mib`.
  - A pool smaller than the context does not fail the load: SGLang caps the request length instead (`max_req_len = min(context_len - 1, pool - 1)`, 0.5.16 `managers/tp_worker.py`) and refuses longer prompts only when they arrive.
  - So after an SGLang load while a sleeper exists, `devai-control` reads `max_req_input_len` from the engine's `GET /server_info`. An unconstrained pool reports `ctx - 6`. Anything lower counts as a memory failure (`kv_pool_below_ctx`, both numbers in the reply).
- **If the load then fails on memory, or an SGLang pool comes up below its context**, `devai-control` stops the new model process if it still runs, waits for it to exit, unloads the sleeper and loads once more, and says so in its reply. Ollama, whose fit llama.cpp decides, gets only this retry.
  - A memory failure is recognised by these substrings of the engine's log:
    - vLLM: `less than desired GPU memory utilization`, `No available memory for the cache blocks`, `larger than the available KV cache memory`;
    - SGLang: `Loaded weights leave no GPU memory for the KV cache`;
    - any engine: `CUDA out of memory`, `torch.cuda.OutOfMemoryError`;
    - Ollama: `cudaMalloc failed: out of memory`.
- A load on the sleeper's own service never meets this: the new model process replaces the sleeper first.
- If SGLang is frozen (Phase 15), only a vLLM model can sleep and only an Ollama load can meet it, so the vLLM lowering is never reached; it is then built and unit-tested with fake engines only (step 3).

### Detailed steps

1. **Open question 3 first, on Phase 4 step 4v's build, before Phase 4's 4e retires `vllm-devai`.**
   - On `vllm-devai`, which then runs vLLM 0.30.0 (E1; 0.28.0 if 4v stopped the move), run Qwen3.8-27B-W4A16-devai-AutoRound at the context 4v probed for it, with `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`, and measure:
     - how long `/sleep?level=1` and `/wake_up` take (5 times each);
     - VRAM after sleep (NVML);
     - host RAM (RSS);
     - decode tok/s and TTFT with and without sleep mode, on paired prompts (docs/statistics-primer.md conventions).
   - Do the same for one SGLang model with `--enable-memory-saver`.
   - If sleep mode costs speed or memory, `sleep_mode` defaults to false, and the router sets it true only for a launch that, when the router builds it (before it submits that launch's engine job, so that job never counts), an active C5 hold (bench, probe or action) or another PENDING job names: the launches D1's interrupt suspends and D13's re-run needs again. An interactive chat launch then gets false, and an interrupt of it pays a cold start. The step 1 report states which rule applies.
2. **Move the builders.** A pure refactor: `make test-router` passes unchanged and the golden corpus stays byte-identical.
3. **`devai-control`'s engine and trainer kinds.** Go tests with fake engines (an `httptest` server answering `/health`, `/api/ps`, `/sleep`, `/wake_up`) and the fake supervisor. For the trainer: `cancel` replies only after the program has exited, `finalize` has run and the fake NVML reading is back; a `cancel` during `stopping` returns with the first; reason `time_limit` gives failed `time_limit`, and a `cancel` without `reason` gives `cancelled`; a run that exits on its own publishes `ended` with the record's status; `train` on a terminal record is refused and starts nothing. For the engines: a fake engine whose program exits, or whose `/health` stops answering, while `serving` turns the service `failed` within one heartbeat plus 3 failed reads; a `/sleep` delayed by the fake engine (a test delay) with an `unload` sent during it: the `unload` replies only after the sleep has finished and the model has been unloaded, the status goes `serving` -> `unloading` -> `idle` and never `sleeping`, and no sleeper is left.
4. **The images.** supervisord and the configurations.
5. **The laya split.** Its tests go into `make test-laya-trainer`, among them `finalize`'s rules (`jobs.py` above) and a terminal record left unchanged. In control mode: a second create while a record is pending gets 409, and a create after it is terminal gets 200; a cancel before `train` finalizes the record, and a later `train` is refused; a cancel after `train`'s mark leaves the record to `finalize`.
6. **The live checks.** Stop the router's backends (`make cache-down`) and start the `slurm` profile.

### Exit criteria

- For Ollama, vLLM and SGLang (if Phase 4 kept it), in `devai-engines`:
  - an engine job submitted with the request `gpu-arbiter -print-load` prints loads the model;
  - a chat completion sent straight to `devai-engines:<port>` answers;
  - `scancel` frees the card;
  - the model process's command line equals the router's golden argv for the same request, apart from the store path, the port and, when the launch's `sleep_mode` is true, the engine's sleep flag (`--enable-sleep-mode` or `--enable-memory-saver`); its environment carries `VLLM_SERVER_DEV_MODE=1` exactly when a vLLM launch's `sleep_mode` is true.
- vLLM sleep and wake:
  - an engine job ended with USR1 leaves the model `sleeping`;
  - the next job with the same launch wakes it;
  - a job with another launch on the same service unloads the sleeper first (each engine serves one model process on its port);
  - an `unload` with `target: sleeper` unloads it;
  - a `sleep` to a model loaded with `sleep_mode` false answers `unsupported`, and the epilog unloads it.
- Ollama:
  - a load with another context length restarts `ollama serve`, and `/api/ps` then shows that context;
  - a `/v1/chat/completions` request without `num_ctx` does not make Ollama load the model again (its log shows one load).
- A sleeping model does not make the next engine job's prolog unload it, and the epilog's idle check tolerates it; a probe job's prolog unloads it, and so does a GPU-action job's (a hand-submitted job with the `engine` license and `gpu: true` in its comment).
- The sleeper's VRAM:
  - with an SGLang model sleeping (if SGLang is kept), a vLLM load starts with its utilization lowered by the sleeper's `used_mib`;
  - with a vLLM model sleeping (if SGLang is kept), an SGLang load at its probed context reports a `max_req_input_len` of at least ctx - 6, after a retry if the pool came up short, and a prompt filling 99% of that context succeeds;
  - with an SGLang model sleeping (if SGLang is kept), a vLLM load forced to fail on memory (a `--gpu-memory-utilization` too large, set in the test) unloads the sleeper and succeeds on the second attempt;
  - if SGLang is frozen: with a vLLM model sleeping, an Ollama load that cannot fit beside the sleeper fails with `cudaMalloc failed: out of memory`, unloads the sleeper and succeeds on the retry. The test takes the Ollama row and context from the probe cache whose forced-full VRAM (`num_gpu` 999) comes closest to the empty card's free memory, and records the margin (that free memory minus that VRAM) against the sleeper's `used_mib`. If the margin is larger than `used_mib`, the load would fit beside the sleeper; the check is then covered by step 3's fake-engine test alone, and the report says so.
- A recovery file replaced by rename is used by the next load, without restarting `devai-engines`.
- When `devai-control` is killed while vLLM serves, supervisord restarts it and it adopts the model: `serving`, the same job, the same process. When it is killed while vLLM is loading, the job's `load` returns ok once the model serves (C1, "Adoption after a restart"; C3), the job stays RUNNING with its hold, and no second load starts.
- `kill -9` of the serving vLLM process (and, if SGLang is kept, of SGLang's scheduler process alone, leaving its HTTP server up) turns the service `failed` within 120 s; the engine job's hold exits 3, and the job ends and releases `engine`.
- Trainer jobs on the reference dataset `ds-a6c9c8248242`, submitted by hand after the controller created the job:
  - one runs to the end and finishes `succeeded`, with a record shaped as `make laya-check` expects; its hold exits 0, and `sacct` shows exit 0;
  - `scancel` during training ends a job `cancelled`; the trap's `cancel` returns only once the card is back at its level before the run; the epilog's idle check passes, `sinfo` shows the node idle (not drained), and `sacct` shows exit 0;
  - the time-limit signal ends a job `failed` with `time_limit`; the epilog's idle check passes, the node stays idle, and `sacct` shows exit 0;
  - a run that fails on its own (`run_job` made to exit non-zero in the test) ends `failed`, and the job exits 1;
  - with nats-server restarted during training, the job stays RUNNING and the run goes on.
- When a trainer job's script is killed with SIGKILL, the epilog's `cancel` ends the run `cancelled`, and the card is idle.
- A `sleep` sent for an engine job whose vLLM model is still loading is refused with `bad_request`, and the model goes on loading to `serving`.
- With `devai-engines` started without the GPU device (`test-engines.sh` starts it so; the compose switch comes in Phase 11), Ollama loads with `num_gpu: 0`, and a vLLM load is refused with `no_gpu`.
- `make test-router`, `make test-devai-tools`, `make test-laya-trainer` and `make test-slurm` pass.

### Phase 8 risks

| Risk | Mitigation |
| ---- | ---------- |
| sleep mode costs speed or memory while serving | measured in step 1; if so, it is set only for a launch that an active C5 hold or another pending job names when the router builds it (step 1) |
| a sleeping model leaves too little VRAM for the next load | for a vLLM load, `devai-control` lowers the utilization by the sleeper's `used_mib`, unless recovery flags set it (SGLang sizes from free memory; an SGLang pool below its context counts as a failure); if a load still fails, it unloads the sleeper and loads once more |
| the refactor changes an argument | the golden corpus, compared byte for byte |
| one engine process crashing takes the others down | each is its own supervisord program; only a container restart affects all of them, and the router then submits new jobs |
| a model process dies after it reached `serving` (an OOM kill, SGLang's scheduler behind a live HTTP server) | the liveness check on every heartbeat (`engine_hf.go`, `engine_ollama.go`) sets `failed`; the job's hold exits 3 and the job ends |

---

## Phase 9 -- Router as Slurm client

### Goal

The router starts and ends engines only through Slurm jobs, follows their state on the bus, and has no podman socket. Today's launch layer and the per-engine containers are gone in the same step (I10). The 2026-09-28 failure cannot be reproduced.

### Deliverables

```
gpu-arbiter/slurm_launch.go       new    -- the Slurm launcher (below); replaces ensureBackendRunning and
                                            ensureOllamaRunning
gpu-arbiter/gpu_record.go         new    -- the GPU record
gpu-arbiter/holds.go              new    -- C5 holds (the `hold_key` check, renewal, the 90 s expiry) and
                                            /devai/v1/gpu
gpu-arbiter/ready.go              new    -- C5 /devai/v1/ready (from odysseus RT-6)
gpu-arbiter/engine_metrics.go     new    -- C5 /devai/v1/metrics: the serving engine's /metrics, read-only; tests:
                                            proxied while the engine serves, 503 while it does not, no job submitted
gpu-arbiter/control_routes.go     new    -- C5 blocked engine routes, registered by newBackendMux on every listener
gpu-arbiter/probe_caches.go       new    -- reads the probe caches (ollama, vllm, sglang; vllm-s1's from Phase 23)
                                            and models.yaml from /etc/devai/deploy, and rebuilds the model table
                                            under arbiter.mu when any file's inode or mtime changes; checked on
                                            every refresh of the pending list (every 2 s) and before the allowlist
                                            answers unknown. A file that fails to parse keeps the previous table
                                            and logs the error. Tests: a rename-replaced cache adds a row and
                                            changes a row's context and KV dtype without a restart; a truncated
                                            file keeps the old table
gpu-arbiter/trainer_jobs.go       new    -- :11438 create and cancel -> trainer jobs; the reconcile with the
                                            controller (at boot and on every pending refresh)
gpu-arbiter/main.go               modify -- the podman launch layer removed (containerRecreate, createAndStart,
                                            buildContainerSpec, containerStop / Remove / State, stopOtherBackends,
                                            unloadOllama, warmLoadOllama, backendVanished, detectLaunchFailure and
                                            its log readers, reconcileBackendState, the podman client); backend URLs
                                            devai-engines:<port>; the breaker charged once per engine-job submit and
                                            repaid only by a served non-5xx response, refunded for a guard refusal,
                                            exit 2 or a cancel (Ensure cases 4-5); /health
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
                                            profile becomes the default stack; the router mounts ${DEVAI_REPO:?}/deploy
                                            read-only at /etc/devai/deploy (as Phase 17a does, if it landed first;
                                            I25), PROBE_CACHE, VLLM_PROBE_CACHE, SGLANG_PROBE_CACHE and CATALOG_FILE
                                            default to that directory, and the single-file cache and models.yaml
                                            mounts are removed (a single-file bind mount keeps the old inode, so a
                                            probe job's rename would stay invisible)
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
tests/python/test_engine_cache_volumes.py
                                  modify -- the Go-side case (engine_cache.go is removed) re-pinned against the
                                            compose mounts of Phase 8
```

### What the Slurm launcher does

**The record** (under `arbiter.mu`):

- `holder`: job id, kind, backend, launch, and status from the bus;
- `sleeper`;
- `holds`: job id -> backend, launch and the time of its last renewal (C5);
- `pending`: PENDING jobs and the launches their comments name, refreshed from `GET /jobs` at most every 2 s.

**Boot.** The router rebuilds the record from Slurm's RUNNING and PENDING jobs of devai's kinds and from `state` of every engine service. A serving engine whose job is RUNNING is adopted with its launch -- unlike today, the model is known, so the next request for it is proxied without a load. A loaded engine with no RUNNING job is left to its lease. Holds are not rebuilt: they come back with their clients' next renewal (C5), at most 30 s later. Boot cancels every PENDING engine job, since the requests that waited on it died with the old process; a renewed hold or the next request submits again.

**Ensure** replaces `ensureBackendRunning` and `ensureOllamaRunning`:

1. The holder is this launch and `serving`: proxy.
2. The holder is this launch and pending, loading or waking: release `a.mu` and wait for the bus event, Slurm's terminal state or `HEALTH_TIMEOUT_SECONDS`, with SSE keepalive as today. The recreate-coalescing fields (`recreating`, `pendingModel`) keep a second request from submitting a second job.
3. A job other than an engine job holds the `engine` license -- a trainer, a probe or a GPU action (C8), recognised by the licenses `GET /jobs` lists for it, not by its kind -- or such a job is PENDING and not held (Yielding, below): 503 `gpu_held_by_job`, `Retry-After: 30`. An engine bound to an active hold for another launch: 503 `gpu_held_by_hold`.
4. Otherwise, switch:
   - drain, as `drainBackend` does today;
   - end the holder's job -- `SIGUSR1` when the holder's bus status is `serving`, the holder's launch has `sleep_mode` true (C2; the field the router recorded at launch, never the backend's name), a pending job names its launch, and the new launch is on another backend (on the same backend the new model process replaces it anyway); a cancel otherwise. A holder that is pending, loading or waking is always cancelled: a loading or waking holder's script is still in the foreground `load`, and bash runs a trap only after that command returns (C3), so a USR1 would wait for the whole load. The cancel's TERM kills the `load` request, the TERM trap's `unload` stops the load (C1), and the epilog unloads again; a model that never finished loading has nothing to keep;
   - wait until Slurm reports the job ended (polling every 500 ms, at most 150 s: longer than the USR1 trap's 120 s `sleep` and `KillWait`'s 30 s). If it has not ended by then, cancel it (harmless after USR1) and submit anyway: the new job waits PENDING (Licenses), since Slurm returns `engine` only after the old job's epilog (Phase 5), and case 5's `gpu_queue` ends the request if the license never comes;
   - submit the new engine job, with `DEVAI_LOAD` = the `load` request body (`-print-load`'s output: `launch` = C2, built from `computeLaunchConfig` and the rest of today's resolution, and `timeout_s`; without `v` and `job`, C3);
   - charge the breaker for the launch (`noteLaunchAttempt` for (model, ctx), once per engine-job submit, as today); only a served non-5xx response repays it (`noteLaunchSucceeded`, the proxy's `ModifyResponse`), so an engine that reaches `serving` and then fails (its job exits 3) before answering keeps the unit spent;
   - continue as in case 2.
5. How a launch can end:
   - The guard refuses it (the `devai.events.slurm.guard` verdict for the job is `busy`, or the job is held): cancel the job and answer 503 `gpu_held_by_foreign_process` naming the holders. The unit charged at submit is refunded. The next request submits a new job; a held job is never released (Slurm delays a requeued job -- one the prolog refused -- by about 2 minutes; measured).
   - `load_failed`: 502 with the log tail. The unit charged at submit stays spent.
   - The job is still waiting on the license when the request gives up: 503 `gpu_queue`, `Retry-After: 30`. When the last request waiting on that pending engine job gives up, the router cancels the job.
   - The job reaches a terminal state, or `HEALTH_TIMEOUT_SECONDS` passes, with no `serving` or `failed` event for it. This covers a refused `load` (`busy`, `unknown_model`, `bad_request`: it changes no state and so publishes nothing, C1), exit 2 (the bus unreachable) and a cancel from elsewhere. The router asks the service for its `state` once. If that state names this job with `failed`, the ending is treated as `load_failed`. Otherwise:
     - it answers 503 `engine_job_ended` with `Retry-After: 30`, naming the job id, its Slurm state and its exit code (from `GET /jobs`);
     - on a timeout, it cancels the job;
     - for exit 1 the unit charged at submit stays spent, since a refusal that repeats (weights gone, a bad launch) must stop the way a crash loop does; for exit 2 or a cancel, its own included (a holder cancelled while loading in case 4 ends 0), it is refunded.

   - The holder was `serving` and turns `failed` (devai-control's liveness check, Phase 8): the hold exits 3 and the job ends. Requests already proxied get the upstream error. The next request submits a new job, and the unit charged at submit stays spent, as for `load_failed`.

     The next request submits a new job.

**Yielding.** A PENDING job other than an engine job that requests the `engine` license -- a trainer, a probe or a GPU action (C8), read from its licenses in `GET /jobs` -- makes the router end its engine job, unless that engine is bound to an active hold: drain, then cancel. A PENDING job counts here and in case 3 only while Slurm can start it. A held job is ignored: priority 0, with a state reason of `JobHeldUser`, `JobHeldAdmin` or the requeue hold after a prolog failure (the field names are checked against 24.11's spec in Phase 5, step 2). This covers an operator's `hold` (D1, C1) and a job the guard refused. Until the waiting job has run, new engine launches get case 3. Without this, a keep-warm engine job would hold the license forever, and the waiting job would never start. The `pending` refresh keeps each job's licenses and hold state for this.

**The sleeper (I9).** When no PENDING job, no active hold and no RUNNING engine job names the sleeper's launch any more, and no RUNNING bench, probe or action job naming it is within 60 s of its start without a hold, the router sends `unload` with `target: sleeper` and `force`. It checks this on every refresh of the pending list and at boot.

**Idle sweep.** With `IDLE_TIMEOUT` > 0, the idle sweep cancels the engine job.

### Trainer on :11438 (I11)

- **`POST /v1/fine_tuning/jobs`** is forwarded to the controller. On a 2xx answer, the router submits a trainer job carrying the returned `id`. It does not end the engine job itself: once the trainer job is PENDING, Yielding ends the engine job (drain, then cancel), unless an active hold binds it, in which case the trainer job waits until the hold's job ends. If the submit fails, it tells the controller (`POST /devai/v1/jobs/{id}/fail`) and answers 503.
- **`POST /v1/fine_tuning/jobs/{id}/cancel`** is forwarded to the controller, which finalizes a job that has not started, and the trainer job is cancelled. The answer is the record once it is terminal, waiting at most 15 s, as today.
- **Reads** are proxied to the controller, which now always runs. The "status reads never launch the trainer" rule disappears together with the launch.
- **Reconcile** (at boot and on every refresh of the `pending` list): a controller job that is not terminal, whose trainer job the router submitted (or, at boot, any non-terminal controller job), and whose trainer job is neither PENDING nor RUNNING in `GET /jobs` (cancelled while pending by `devai-jobs cancel`, `clear` or the web, or gone) is failed through `POST /devai/v1/jobs/{id}/fail` with reason `trainer_stopped`, naming the job's Slurm state. Such a job never started, so neither `devai-control` nor an epilog finalizes it. A job whose run had started is left to `devai-control`'s `finalize`. A controller job whose trainer job has not yet appeared in Slurm (the submit is still in flight) is not touched.
- **A refused trainer job:** when the guard refuses the router's own trainer job (its `devai.events.slurm.guard` verdict is `busy`, or the `pending` refresh sees it under the requeue hold after a prolog failure; an operator's `hold` is left alone), the router cancels it and fails the controller job (`POST /devai/v1/jobs/{id}/fail`, reason `gpu_held_by_foreign_process`, naming the holders from `guard.json`). As for an engine job, a held trainer job is never released; the client submits again.
- **The hold:** a RUNNING trainer job holds the GPU (Ensure case 3). `/health` on 11438 keeps its `job_runner` shape, with `hold_until` = the job's start + its time limit. `/health` on each engine port keeps `running`, `current_model`, `current_context` and `current_spec` (scripts/laya_check/drive_job.py reads them to restore the teacher).

### Detailed steps

1. Read the control routes from the sources in `devai-engines`: vLLM 0.30.0's (E1; 0.28.0's if Phase 4 step 4v stopped the move) and SGLang 0.5.16's (`vllm-s1` needs none, C5). Pin them (C5) with a test that fails when the installed source registers a route the list does not have: for SGLang, any endpoint carrying `ADMIN_OPTIONAL` or `ADMIN_FORCE`; for vLLM, any route registered under `VLLM_SERVER_DEV_MODE`, plus `/scale_elastic_ep` and the routes C5 names behind other settings.
2. Remove today's launch layer (Deliverables). Port the tests of the behaviour the Slurm launcher keeps before deleting the rest.
3. The record, Ensure, yielding, holds, the trainer (a trainer job held after a prolog failure is cancelled and its controller job failed with `gpu_held_by_foreign_process`; one under an operator hold is left alone) and boot. Unit tests run against an `httptest` slurmrestd (requests checked against the spec dump) and an embedded nats-server with fake `devai-control` responders. They cover:
   - every Ensure case;
   - a switch away from a `serving` holder whose launch a pending job names sends USR1; the same switch while the holder is `loading` or `waking` cancels it instead, and that holder's unit is refunded; a `serving` holder whose launch has `sleep_mode` false is cancelled, never sent USR1, on any backend;
   - an engine job that ends with exit 1 after an `unknown_model` refusal (503 `engine_job_ended`, the unit stays spent), one that ends with exit 2 (refunded), and one cancelled from elsewhere while loading (refunded);
   - an engine that reaches `serving` and then fails before any non-5xx response (job exit 3), three times for one (model, ctx): the breaker opens and the next request answers 503 `launch_breaker_open` without submitting a job; one non-5xx response in between resets it;
   - a guard refusal refunding its unit;
   - boot adoption;
   - a hold refusing a switch, and a hold dropped when its job ends or when it is not renewed for 90 s;
   - a hold POST or DELETE without the job's `X-DevAI-Hold-Key`, or a POST re-pointing another job's hold, answers 403 or 409 and leaves the hold in place;
   - a hold requested while a trainer, probe or GPU action is PENDING and not held answers 503 `gpu_held_by_job`; a trainer created while a hold binds the engine stays PENDING until the hold's job ends, then starts;
   - a trainer job cancelled while PENDING (as `devai-jobs clear` does): within one pending refresh its controller job is failed `trainer_stopped` and the next create answers 2xx, not 409;
   - boot cancelling the PENDING engine jobs it finds;
   - a sleeper chosen only when a pending job names the launch;
   - a sleeper unloaded once nothing names it, including at boot, but not while an active hold or a RUNNING engine job names it, nor while a just-released re-run is about to take its hold;
   - yielding to a pending probe, and to a pending GPU action (an action job requesting the `engine` license); a pending CPU action (no license) causes no yielding;
   - a held pending probe, trainer or GPU action (an operator hold, or the requeue hold after a prolog failure) causes no yielding and no 503 `gpu_held_by_job`; once it is released, it does;
   - `gpu_held_by_job` while a trainer, a probe or a GPU action runs.
4. Compose and the live checks.

### Exit criteria

- The router container has no podman socket, the per-engine containers are gone, and a picker-launched agent completes a turn on Ollama, vLLM and SGLang.
- The 2026-09-28 failure: a foreign process holds 4 GiB, and a request for a vLLM model arrives. Either the holder is killed and the model loads, or the client gets 503 naming the holder and the node drains. In both cases the model's breaker count does not change.
- Requests alternating between an Ollama model and a vLLM model give exactly one engine job per switch, and never two engine processes on the card at once (the samples in `gpu.json`; no overlapping engine jobs in `sacct`).
- After a router restart while vLLM serves, the next request for the same model is proxied without a load.
- After a router restart during a held bench, a request for another model gets 503 `gpu_held_by_hold` within 30 s of the bench's next renewal, the bench's engine job keeps running, and no PENDING engine job from before the restart remains.
- `make laya-check` passes through the router on Slurm: swap, 503 hold, job, parity, teacher restored.
- `/devai/v1/ready` reports not ready with slurmrestd or NATS stopped, while `/health` still answers.
- `make test-router`, `make test-e2e`, `make test-vllm` and `make test-sglang` pass.

### Phase 9 risks

| Risk | Mitigation |
| ---- | ---------- |
| a model relied on something only the podman path provides (environment, mounts) | Phase 8's argv and environment parity check; `make test-vllm` |
| waits under `a.mu` stall other backends | the release-and-wait pattern `waitForHealthy` already uses |
| a REST call on every request | the record is the cache; Slurm is asked only on a switch, a hold, and every 2 s for pending jobs |
| a lab client calls `/devai/v1/holds` | holds are checked against Slurm (C5), a lab client cannot create a job, and POST and DELETE require the job's `hold_key`, which only the job's run and its Slurm comment (read with the JWT) carry |
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
                                            podman CLI for the devai.job label cleanup; result.json per C4's
                                            writer rule) and `apache` (kind program); user control-operator
deploy/docker-compose.yaml        modify -- devai-operator: the control-operator and operator bus passwords, the JWT key
                                            (at the C6 paths); its environment as C7 lists it: SLURM_REST_URL,
                                            SLURM_JWT_KEY_FILE, SLURM_USER, DEVAI_BUS_URL, DEVAI_BUS_USER=operator,
                                            DEVAI_BUS_PASSWORD_FILE
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
scripts/ops/laya-recipe-eval.sh, scripts/ops/systemone-gate.sh, scripts/systemone/gate.py,
scripts/ops/systemone-teacher-reads.sh, scripts/ops/systemone-cpu-load.sh, deploy/operator/actions.yaml, Makefile
                                  modify -- only for what Phase 22 or Phase 23 landed before this phase; without
                                            them, nothing here. The from-Phase-10 forms those phases specify (C8):
                                            laya-recipe-eval and systemone-gate move from `not_offered` to GPU
                                            actions (`job_kind: action`, `gpu: true`, their `time_limit_min`), and
                                            outside a job their scripts and targets only submit the action through
                                            devai-jobs; C8's pre-Phase-10 card check and `cache-down` are dropped
                                            (Phase 14 step 2 relies on this). systemone-teacher-reads becomes
                                            one action job for the whole run, taking its three holds in turn,
                                            and systemone-cpu-load one action job for its one launch; both entries
                                            set C8's `comment` and take their holds (C5; Phase 23, Holds)
scripts/ops/bench-task.sh         new    -- the bench-task action: `podman run --label devai.job=$DEVAI_JOB_ID ...
                                            devai-lab /scripts/bench/bench_task.py ...` with the GPU device for NVML
                                            reads and CUDA_VISIBLE_DEVICES= (I12), and $DEVAI_RESULTS_DIR mounted at
                                            the same path (C8); forwards INT and TERM to it. The container sits on
                                            devai-lab-egress only, never on devai-net, with the lab's proxy
                                            environment as `make lab` sets it: it reaches the engines only through
                                            the router (devai-router:<port>, its metrics at C5's /devai/v1/metrics),
                                            and runs model-generated code (HumanEval). Its environment carries
                                            DEVAI_JOB_ID, DEVAI_RESULTS_DIR and DEVAI_HOLD_KEY (C8) and no bus or
                                            JWT file; the scripts and deploy/ are mounted read-only and only
                                            $DEVAI_RESULTS_DIR read-write (the inspect log, result.json, the cache
                                            row); the HF token file is mounted only for the gated GPQA task. After
                                            the container exits, bench-task.sh merges the task's row into
                                            deploy/.bench-cache.json (`_bench_core.update_row`, in devai-operator)
scripts/bench/bench_task.py       new    -- take the hold (C5, with `X-DevAI-Hold-Key: $DEVAI_HOLD_KEY`) and renew it
                                            every 30 s while the task runs; run one task with bench_runner; on SIGINT
                                            stop inspect and score the unbroken prefix (harvest_truncated's
                                            functions, imported); write result.json and the task's cache row with
                                            the job id to $DEVAI_RESULTS_DIR (bench-task.sh merges the row); release
                                            the hold. On SIGTERM: stop inspect, result.json `cancelled`, no cache
                                            row. A 503 or 409 on the hold: wait (`Retry-After`, else 30 s) and ask
                                            again until the job is stopped. A SIGINT or SIGTERM that arrives before
                                            the hold is granted writes result.json `cancelled` with summary
                                            `no_hold` and outputs n_scored 0, and no cache row, so bench-sync plans
                                            the task again; prefix scoring applies only after a grant. result.json
                                            records `hold_granted_at`, and a task whose grant came later than 5 min
                                            after the job's start is written `truncated` even if it finishes. A 403
                                            on the hold: result.json `failed` (reason `hold_forbidden`), no cache
                                            row, exit 1. A renewal that answers 409 or 503 (the hold was lost, as
                                            after a router restart, and another launch or job now has the card)
                                            ends the task as `cancelled`, with no cache row. Reads DEVAI_JOB_ID,
                                            DEVAI_RESULTS_DIR and DEVAI_HOLD_KEY (C8)
scripts/bench/tasks/humaneval.py  modify -- `_run_check_in_subprocess` runs the check with a minimal environment
                                            (`PATH`, `HOME=/tmp`, `_BLAS_ENV`), not `os.environ`, so the
                                            model-generated code sees no token, key or proxy setting
                                            (humaneval_plus.py imports it); pinned by a test that sets a sentinel
                                            variable and asserts the child does not see it
scripts/bench/bench_runner.py, scripts/bench/_bench_core.py
                                  modify -- engine metrics from the router's `GET /devai/v1/metrics` (C5) on the
                                            backend's port instead of the per-engine containers; the bench no
                                            longer joins devai-net
scripts/bench-sync.py             modify -- `--submit`: the same plan, submitted as one bench job per (model, task) in
                                            plan order (`devai-jobs submit bench --run <id> --task <t> --backend <b>
                                            --model <name> --ctx <n> [--mtp]`), with the run id and the target's
                                            launch in the comments; a `truncated` row (bench_task.py) is not
                                            counted as current, so the plan keeps it; the clean-slate and deadline
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
                                            model and band for Ollama) and one target per run (the -model actions);
                                            in a probe job (`DEVAI_RESULTS_DIR` set), write result.json there before
                                            exiting (C4: kind probe, status ok, failed or cancelled on SIGTERM,
                                            outputs {model, backend, cells_written})
scripts/ops/probe-*.sh, scripts/ops/bench.sh
                                  modify -- submit jobs; no cache-down needed
scripts/model-sync.py             modify -- no cache-down / cache-up bracket around the probes
devai-tools/cmd/devai-jobs/       modify -- submit (action, probe, bench run), hold, release, suspend, resume,
                                            interrupt, clear; `service <container> <service> start|stop|restart|reload`
                                            (D15's service control, over the bus as user operator; refused inside
                                            devai-slurm, naming devai-operator); submit reads the action's C8 entry
                                            for kind, license, gpu flag, time limit and `comment`, and refuses
                                            without a registry; a bench submission writes C3's backend, model, ctx
                                            and mtp from its arguments; every bench, probe and action submission
                                            writes a fresh `hold_key` into the comment (C5)
bin/devai-jobs                    modify -- `submit`, `suspend`, `interrupt` and `service` run `podman exec
                                            devai-operator devai-jobs "$@"`, every other command `podman exec
                                            devai-slurm devai-jobs "$@"` (I13; `service` too, since it sends
                                            supervisor verbs as bus user `operator`, whose environment, C7, only
                                            devai-operator has)
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
   - The hold is taken before the first sample, so inspect's clock never runs during a load, and renewed every 30 s (C5).
   - A task stopped before its hold is granted is not scored and leaves no row (`no_hold`); one granted more than 5 min late is `truncated` and planned again.
   - At 30 min Slurm sends SIGINT, 120 s before the limit. The task scores the unbroken prefix, as `harvest_truncated.py` does today.
4. **`devai-jobs suspend <run>`** (D13, I5) holds the run's pending tasks and cancels the running one; its `result.json` gets status `cancelled` and it is not scored. It then submits the cancelled task again, held, with `attempt` + 1 and `rerun_of`.
   - **`resume`** releases the re-run, waits until it is RUNNING, then releases the rest.
   - **`interrupt <run>`** suspends the run and prints the `resume` command; it does not prompt, since `bin/devai-jobs` runs without a terminal and the web calls it with `--json`. In between, the operator runs the quick job (an agent turn, another job), then resumes. If the quick job's model is on another backend, the router's Ensure finds the re-run pending and ends the bench's engine job with `sleep` (Phase 9, case 4). On the same backend the bench's model is unloaded instead: each engine serves one model process on its port.
   - **`clear`** cancels every PENDING job (D1); `hold`, `release` and `cancel` act on one job.
   - The web's buttons call the same commands.
5. **The re-bench.** With every exit criterion above met, run the stopped re-bench through `bench-sync.py --submit`, once, on the final images.

### Exit criteria

- The engines page and `devai-jobs service` restart an idle Ollama service, and `stop` a serving vLLM service: the owning engine job ends 0, its epilog's `unload` is a no-op, and the state events show both. A `restart` of the serving vLLM service is refused with `busy` (C1, "Supervisor verbs").
- An action started from the browser appears in `devai-jobs list` with its log. Cancelling it kills its process group and removes the containers it started. A stack-lifecycle action still runs direct while `devai-slurm` is down.
- A bench of one model and two tasks runs as two jobs. A task given `--time=3 --signal=B:INT@60` is cut off and scores its prefix, as the deadline path does today. Its `result.json` and cache row carry the job id.
- A bench job's comment in `sacct` carries the backend, model, ctx and mtp of its target, and its hold answers 200.
- A bench task submitted while a probe job holds the engine license for longer than its time limit ends `cancelled` (`no_hold`) and leaves no cache row.
- A bench-task container is on devai-lab-egress only (`podman inspect` lists no devai-net), its row carries the engine metrics read through the router's `/devai/v1/metrics`, and a HumanEval check run with a sentinel variable in the container's environment does not see it.
- A bench and an interactive agent session run side by side on the same model without failures.
- `devai-jobs interrupt`:
  - with the same model, nothing is loaded (the bus log shows no load);
  - with a model on another backend, the bench's model sleeps and wakes, and the cancelled attempt is in the history, not scored;
  - issued while the bench's model is still loading, with the quick job's model on another backend: the router cancels the bench's engine job instead of sending USR1 (Phase 9, case 4), the half-loaded model is unloaded, nothing is left `sleeping`, and the quick job is served within 150 s of its request plus its own load.
- Probing one vLLM model and one Ollama model through probe jobs gives the same cell as the container prober for the same (model, backend, band). These must be identical: the fit verdict (`fits` / `fully_on_gpu`), the context, `kv_cache_type`, `flash_attention` (Ollama) and the reasoning/tool parsers. The measured VRAM (`actual_vram_gb` / peak) must agree within max(0.25 GiB, 2 %) of the container prober's value. That margin is an engineering margin fixed before the run, not an estimate. Timestamps, evidence text and job ids are not compared. A cell that differs in any exact field, or in VRAM by more than the margin, fails the criterion, and the report shows both cells.
- The re-bench has run on the Slurm path. Its rows carry job ids, and `bench-report` renders them.
  - It covers every pickable row at the context the picker offers, SGLang's included if SGLang is kept.
  - No model trips the launch breaker in a loop while it runs (sglang-backend-remediation's 24-hour soak check).
- A live `make model-sync` run onboards one new catalog row: download, probe jobs, ledger verdict; the running router then serves the row (no restart), and a re-probe that changes a cell's context or KV dtype takes effect on the next load.
- A per-model probe action started by `devai-control` in a probe job loads and unloads its model through the bus as user `operator` (C8's base environment).
- A submitter action (`probe-vllm`) submits its probe jobs through `devai-jobs submit` from inside devai-operator.
- Run inside devai-slurm, `devai-jobs service` is refused, naming devai-operator; `bin/devai-jobs service` from the host runs it in devai-operator.
- `bin/devai-jobs submit action` with a `gpu: true` entry, run from the host, yields a job with `-L engine`, `gpu: true` in its comment and the entry's time limit; an entry with C8's `comment` yields a job whose comment carries its backend and the parameters' model, ctx and mtp. Run inside devai-slurm, `devai-jobs submit action`, `submit probe` and `suspend` are refused, naming the missing registry; `bin/devai-jobs interrupt <run>` from the host resubmits the cancelled task, held.
- When a probe job's script is killed with SIGKILL, its epilog stops the action (the run's `exited` is in the bus log before the job leaves COMPLETING), and the next GPU job's prolog finds an idle card (guard.json).
- The stale-prober path of the prolog, with Phase 5's epilog test switch on: start an Ollama probe for a band below the host's (`PROBE_VRAMS=16G`), so its ballast holds VRAM in devai-operator, with an engine job already queued behind it (`PENDING (Licenses)`). Once the ballast prints `ready`, kill the probe job's script with SIGKILL. The engine job starts at once, since the switched-off epilog returns the license without stopping anything. Its prolog, which runs well inside the run's 60 s lease (slurm.md Sec. 6: a 10 s idle wait, then step 2), finds the ballast through the `DEVAI_JOB_ID` in its environ and sends a forced `stop` with `run_job`. The run ends, `guard.json` and the `devai.events.slurm.guard` event name the stopped run, and the card is idle before the engine job's script starts. (Only the ballast is visible to this path: a prober loads through the bus and reads VRAM with nvidia-smi, so what it loaded is an engine holder.)
- `make test-python` (the bench, probe and model-sync tests updated), `make test-devai-tools` and `make test-operator` pass.

### Phase 10 risks

| Risk | Mitigation |
| ---- | ---------- |
| the probers need a Python package that `devai-operator` lacks | checked in Phase 6; the probers use the stdlib and PyYAML |
| a bench container with a GPU device computes on it | `CUDA_VISIBLE_DEVICES=` empty; the device is there for NVML reads only (I12) |
| a held re-run starts late after its release | untested; it is a fresh submission, not a requeue, which the spike saw start in about 1 s |
| a probe job's cell is invisible to a running router | the router re-reads the caches and catalog from the directory mount on change (Phase 9, probe_caches.go) |
| model-generated code in a bench-task container reaches an engine's control routes, a token or devai's files | the container is on devai-lab-egress only and reaches the engines through the router, which blocks those routes (C5); the HumanEval check runs with a minimal environment; the HF token is mounted only for GPQA; only `$DEVAI_RESULTS_DIR` is writable. The code still runs as the container's user beside the task's own files (accepted) |
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
scripts/ops/operator-up.sh, scripts/ops/stack-up.sh
                                  modify -- the host check runs on the host at `make operator-up` (inside
                                            devai-operator the CDI spec is not visible); its result reaches the
                                            stack-up action as DEVAI_GPU_DEVICE. The systemd unit runs `make
                                            operator-up cache-up` (Phase 7), so it adds compose.gpu.yaml under the
                                            same rule; install-systemd needs no change
deploy/docker-compose.yaml (router)
                                  modify -- `${DEVAI_GPU_DEVICE-nvidia.com/gpu=all}` without the colon, so an empty
                                            value is kept and means CPU
gpu-arbiter/slurm_launch.go       modify -- empty DEVAI_GPU_DEVICE = CPU mode: vLLM and SGLang requests (and, from
                                            Phase 23, DiffusionGemma reads on the System One port) refused
                                            before anything is submitted, with an error naming the reason; Ollama
                                            launches with num_gpu 0 (I14)
gpu-arbiter/main.go               modify -- CPU mode: synthesizeFromCache reads the `cpu` band, not the band keyed
                                            by GPU_MEMORY_GB, and takes a cpu cell's fit from its `fits` field (step
                                            3), not from `fully_on_gpu`; KV type, flash setting and probed max ctx
                                            come from the cpu cells. With no cpu cells, no Ollama model is
                                            registered, and the log names `make probe` (cpu band) as the fix
scripts/probe-ollama-reasoning.py, scripts/ops/probe-ollama.sh
                                  modify -- a `cpu` band: models loaded with no GPU layers, resident memory measured
                                            against the host's RAM, cells stamped with the band
scripts/model-picker.py           modify -- without a GPU: reads the `cpu` band, hides vLLM and SGLang rows
tests/                            modify -- CPU mode of the router (unit: a cache holding both a "24" and a `cpu`
                                            band; CPU mode registers rows, ctx and KV type from the `cpu` cells
                                            only; with no cpu cells it registers none), picker and probe band
```

### Detailed steps

1. **Devices.** Compose cannot make a device conditional, so the device list moves to `deploy/compose.gpu.yaml`, which `make cache-up` includes only when the host check prints a device.
2. **The router.** The device is exactly what compose hands over, and an empty one means CPU mode. (The old podman path fell back to `nvidia.com/gpu=all`; it is gone since Phase 9.)
3. **The CPU probe band.** Ollama on the CPU keeps weights and KV cache in RAM.
   - The band's budget is the host's RAM minus a named reserve for everything else on the host (a unified-memory host shares that RAM with any GPU it may have).
   - The probe decides each model's context as on the GPU; the compose default `OLLAMA_CONTEXT_LENGTH=262144` is not assumed to fit.
   - A cpu cell records `fits` (true when the model's resident memory, `actual_total_gb`, stayed within the band's budget) in place of `fully_on_gpu`, which a CPU load never sets. The router and the picker read `fits` for the `cpu` band and `fully_on_gpu` for the GPU bands.
4. **The picker.** Without a GPU it reads the `cpu` band. The bench stays GPU-only, so CPU rows show no bench columns.

### Exit criteria

- On this host, with the GPU forced off (`DEVAI_GPU_DEVICE=`):
  - `make cache-up` works;
  - `make probe` fills the `cpu` band;
  - the picker lists Ollama rows only;
  - an agent completes a turn on the CPU;
  - a request on port 11435 or 11436 returns the no-GPU error;
  - the router's startup log reports its Ollama rows from the `cpu` band, and the served context of the agent's model equals its `cpu` cell's, not its 24G cell's.
  - if Phase 23 has closed: a DiffusionGemma read on 11440 answers the no-GPU error with no job submitted (`devai-jobs list` shows none).
- With DEVAI_GPU_DEVICE unset and the host check returning nothing (nvidia-smi hidden), `make cache-up` renders no compose.gpu.yaml and the router logs CPU mode.
- `make test-laya-trainer` passes without a GPU (its tests already run on the CPU).
- After `make install-systemd`, a restart of the unit brings up the stack with the same overlays as `make cache-up` and `make operator-up`: `compose.gpu.yaml` for devai-slurm, devai-engines, devai-laya-trainer and devai-operator on this host.
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
                                  new    -- what `make lab` does, as two `job` actions (O2). lab-start: parameters
                                            name (^[a-z0-9][a-z0-9-]{0,30}$), lab_port and dsh_port (1024-65535;
                                            lab_port absent = LAB_PORT, dsh_port absent = no dsh). It runs
                                            `podman run -d` (no -it, no --rm) of devai-lab, named devai-lab-<name>,
                                            with the run flags `make lab` uses (user mapping, the egress network and
                                            proxy, the model and probe cache mounts), -p 0.0.0.0:<lab_port>:8888 and,
                                            with dsh_port, -p 0.0.0.0:<dsh_port>:3080 -e DSH_PORT=<dsh_port>, as
                                            `make lab` publishes them. It generates a token (openssl rand -hex 24),
                                            passes it as JUPYTER_TOKEN, and labels the container devai.lab=<name>,
                                            devai.lab.port, devai.lab.dsh_port and devai.lab.token -- never
                                            devai.job, so the job's end and its cleanup leave the lab running. It
                                            takes the host IP, the home volume, the work directory, LAB_PORT,
                                            DSH_PORT and the gitconfig and ssh bind sources only from its
                                            environment (DEVAI_HOST_IP, DEVAI_HOME_VOLUME, DEVAI_HOST_WORK_DIR,
                                            LAB_PORT, DSH_PORT, DEVAI_HOST_GITCONFIG, DEVAI_HOST_SSH_DIR, which
                                            `make operator-up` hands to devai-operator and C8 copies), never from
                                            `hostname -I`, `$HOME` or `.env`, which inside devai-operator name the
                                            container's own. It refuses to start, naming `make operator-up`, when
                                            DEVAI_HOST_IP or DEVAI_HOME_VOLUME is unset, and refuses a name or a
                                            port that a running devai.lab container already uses. Two labs share
                                            the one home volume, as two `make lab` runs would. lab-stop: parameter
                                            name; removes devai-lab-<name>. The read-only /library mount once Phase
                                            20.1 has landed
deploy/operator/actions.yaml      modify -- lab-start, lab-stop (kind job, job_kind action, group lab; no license, no
                                            gpu); the host items (kind host): setup-logs, the model-store
                                            volumes, install, and the jobs and embed directories (E9)
devai-operator/devai_operator/lab.py, host.py
                                  new    -- the lab page lists containers labelled devai.lab and shows
                                            http://<host>:<port>/lab?token=<devai.lab.token> and, when set, the dsh
                                            address http://<host>:<dsh_port>/ (dsh mints its own token per launch and
                                            prints it in the JupyterLab terminal that starts it; the page says so),
                                            with start and stop; the host-setup page (each item's state and command)
tests (make test-operator)        new    -- lab-start's argv and labels from the registry and a fake environment; the
                                            refusal of a used name or port, and of an unset DEVAI_HOST_IP
```

### Detailed steps

1. **The lab page** lists running `devai-lab` containers by their label and shows their links. Start and stop are `job` actions (`job_kind: action`, no license, O2), so they are in the job history: they hold no GPU, so they never wait for `engine:1`. If this phase lands before Phase 10, they run direct until Phase 10 makes them jobs, as every non-lifecycle action. The script starts the `devai-lab` container detached and returns. The container is labelled `devai.lab`, not with the job's label, so the end of the job and its cleanup leave it running.
2. **The host-setup page.** For each item it checks what the container can see. The mounts under `/var/cache/devai` are visible through the bind mount, so `setup-logs`, the jobs directory and the store volumes can be checked. The page then prints the command. `install` writes the host's `~/.local/bin`, which the container does not see, so it is shown with its command and "state not checked".

### Exit criteria

- From the browser, two labs with different names and ports start side by side (from Phase 10 on, each as a job in the history) and stay up after their jobs have ended. Each one's JupyterLab link opens with its token. A third lab reusing a name or a port is refused, and both labs stop.
- A lab started from the browser has the host's git identity (`git config user.email` inside it matches the host's), uses the same home volume as `make lab`, and its printed JupyterLab and dsh URLs carry the host's LAN IP; dsh accepts the LAN URL.
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

- `devai-jobs list --since yesterday` (on the host through `bin/devai-jobs`, and as the history page) shows every job with its kind, model, state, exit, times, GPU peak, mean and energy (engine, trainer, probe and GPU-action jobs; the others have no GPU record), and result status.
- `devai-jobs show <id>` adds the job's log tail, its `gpu.json` summary and its bus events.
- `make test-devai-tools` passes. There is no MCP tool (M10).

---

## Phase 14 -- Cleanup and documentation

### Goal

Only the new path is left, no devai application container has a GPU device, and the documentation describes the result.

### Detailed steps

1. **Router.** Nothing is left to remove: Phase 9 removed the podman launch layer (I10).
2. **Probers.** Remove the container driver (`container_run_detached` and the functions around it, `MUTEX_CONTAINERS`), and `scripts/ops/probe-ollama.sh`'s throwaway `devai-ollama` container (Phase 9).
   - `assert_no_active_backends` goes with `MUTEX_CONTAINERS`, which it iterates (`scripts/_probe_hf_common.py`). No target still calls it by then. The one-shot GPU-exclusive measurements -- Phase 22's `laya-recipe-eval` and Phase 23's `systemone-gate` -- never call it: before Phase 10 they use C8's card check, and from Phase 10 on they run as GPU actions, which hold the `engine` license instead (C8, Phase 10 deliverables). Phase 15's `sglang-keep-check` uses C8's card check too (it always runs before Phase 10, since Phase 15 finishes before 4a) and is removed when M16 is recorded.
   - A test pins that no script under `scripts/` calls it any more.
   - The tests that pin the removed driver are rewritten, or their cases deleted, in the same change: test_probe_classify.py, test_probe_load.py and test_probe_exact_ctx.py stop saving and stubbing `assert_no_active_backends`; test_laya_router_wiring.py's `ProberMutexTest` is deleted; and test_engine_cache_volumes.py's prober cases (`container_run_detached` with `backend=spec.name`) are replaced by a case pinning the `devai-engine-cache-<backend>-*` mounts on compose's `devai-engines` service.
3. **Bench.** Remove `clean_slate.py`, the deadline path and `--legacy` in `bench-sync.py`, and the `bench-harvest` target, with its entry in `deploy/operator/actions.yaml` (`covers` or `not_offered`) in the same change, or test_makefile_ops.py fails. Rewrite `test_bench_clean_slate.py` and `test_bench_deadline.py` for the job path or delete them.
4. **Images.** Remove `deploy/Dockerfile.ollama` and `deploy/Dockerfile.vllm` (merged into `Dockerfile.engines`).
   - List the retired images with their sizes: the upstream ones and the old `devai-base-cpu`, `devai-lab-cpu`, `devai-base-gpu`, `devai-lab-gpu`, `devai-ollama`, `vllm-devai`.
   - Remove them from the local store only after the operator confirms: the store is shared with other projects on this host.
5. **Lab.** `make lab`, `make shell`, `make test-agents`, `make test-bench-smoke`, the lab actions and devai-agent stop attaching the GPU (D1). aiagent's `DEVAI_AIAGENT_GPU=share` goes, because its GPU work is trainer jobs; tell the aiagent session.
6. **Documentation.**
   - New: `docs/operator.md`, the operator's reference.
   - Updated: README, INSTALL.md, CLAUDE.md, docs/router.md, docs/backends.md, docs/bench-results.md ("Run protocol"), docs/laya-trainer.md ("The GPU hold"), docs/aiagent.md, docs/pipelock.md.
   - docs/slurm.md: its status becomes "built".

### Exit criteria

- The re-bench of Phase 10 has run, so nothing removed here is still needed for it.
- The upstream-images guard test passes with an empty transitional allowlist, and `make build` succeeds from a store that holds only the pinned Debian image.
- A test reading compose and the scripts finds a GPU device only on `devai-slurm`, `devai-engines`, `devai-laya-trainer`, `devai-operator`, bench containers (NVML reads only), and containers that a GPU action's script starts (`gpu: true`, C8) with the job's `devai.job` label and `DEVAI_JOB_ID`. The test reads those scripts from the registry's `gpu: true` entries, together with the modules they run that start a container (`scripts/systemone/gate.py` for `systemone-gate`); each of them, when `DEVAI_JOB_ID` is set, starts its container with the label `devai.job=$DEVAI_JOB_ID` and `DEVAI_JOB_ID`.
- No script calls `assert_no_active_backends` (step 2).
- A new user can go from the one-time host setup to a running stack through the browser, following the documentation alone.
- `make test-router`, `make test-python`, `make test-devai-tools`, `make test-operator` and `make test-slurm` pass. `make test-slurm` runs from the host with the stack down: `make cache-down`, then `podman stop devai-operator` (its `devai-control` answers the subjects the test containers use, and stack-down leaves it running, Phase 6 step 7), and afterwards `make cache-up operator-up`.

---

## Phase 15 -- SGLang: keep or freeze

From sglang-backend-remediation, Phase 5. Its Phases 0-3 shipped; its Phase 4 re-probe is done by Phase 4 here, apart from the 7 GPTQ cells, which are removed instead (I17).

### Goal

Decide, on measured evidence, whether SGLang stays -- before Phase 4 spends its riskiest step compiling it -- and leave no stale SGLang claim behind either way. The picker stops offering an agent/backend pair known to fail.

It runs on today's SGLang image and on Phase 4 step 4v's vLLM build: upstream SGLang 0.5.16 is the version Phase 4 would compile (M5), and 4v's vLLM 0.30.0 is the version that will serve (E1), so the measurement carries over. It must finish before Phase 4, step 4a. The SGLang plan ran this decision after its re-probe, and only if fewer than four SGLang models stayed servable; here it comes first, so a freeze saves Phase 4's SGLang build, and its documentation check runs again after Phase 4's re-probe (Phase 4's exit criteria).

### Deliverables

```
gpu-arbiter/main.go (sglangEntrypoint)   modify -- emit --enable-metrics (Phase 8 carries it into pkg/engineargs)
deploy/backend-flags.yaml                modify -- pin --enable-metrics and --disable-radix-cache
scripts/bench/bench_concurrency.py       modify -- run_multiturn gains `body_fields` (merged into the TOP level of every
                                                    request, warm-up included; never under an `extra_body` key, which
                                                    neither engine reads); with it unset, it behaves exactly as after
                                                    Phase 4 step 4v. run_multiturn_sessions runs N sessions through it.
                                                    Session i's salt is `<salt>-<engine>-<i>`, where `<salt>` is the
                                                    per-run nonce of 4v's --salt (random when unset); a dry run's
                                                    sessions are salted `<salt>-dry-<engine>-<i>`, so none shares a
                                                    prefix with a measured session. 4v's per-turn
                                                    counter read-out (--metrics-url) also reads SGLang's counters; the
                                                    hit rate is taken over turns 2-5. It reports the per-session turn-2+
                                                    medians; CLI --sessions (I16). A session with any failed or
                                                    tokenless turn is reported as failed, never as a
                                                    shorter median. --salt and --metrics-url stay 4v's:
                                                    one salt and one counter reader for both measurements
scripts/bench/sglang_keep_check.py       new    -- the whole measurement (steps 2-3): launches each engine through the
                                                    probers' container driver, with their command args and recovery
                                                    flags, at one context and each probe cell's KV dtype (I16), after
                                                    C8's card check (no compute process, memory.used at most 512 MiB)
                                                    before each launch;
                                                    steady TTFT and the sessions on each, SGLang also with
                                                    --disable-radix-cache; I16's statistics with scripts/stats/statlib.py;
                                                    I16's failure rule (an SGLang failure = freeze, a vLLM
                                                    failure = void), pinned by a test with a fake engine that
                                                    fails one turn in each arm;
                                                    a JSON report that M16 cites. Removed again after M16 (step 4)
Makefile                                 modify -- bench-concurrency passes SESSIONS (SALT and METRICS_URL are 4v's);
                                                    sglang-keep-check (removed with the script)
deploy/operator/actions.yaml             modify -- once Phase 6 has landed, sglang-keep-check in `not_offered`: a one-shot
                                                    measurement for M16, run with the stack down; removed again with
                                                    the target (step 4)
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

1. **The baseline (I15).** Qwen3.5-9B-NVFP4 is probed on vLLM 0.30.0 in Phase 4 step 4v (`make probe-vllm-devai PROBE_REPO=ykarout/Qwen3.5-9B-NVFP4`, E1); it already fits SGLang. If 4v stopped the move, 0.28.0 stays in service and the same probes run on it. If Qwen3.5-9B-NVFP4 does not fit the vLLM in service, gpt-oss-20b takes its place after the same probe (neither has run on devai's vLLM build yet): its weights are in both stores and it fits SGLang (131,072 tokens on 2026-10-05). If gpt-oss-20b does not load there either, Gemma-4-26B-A4B-it-NVFP4 is tried the same way: its weights are in both stores, it fits SGLang (262,144 tokens, with its SGLang recovery flags), and its thinking is opt-in, so the reasoning-off body needs nothing more. If none of the three loads, Phase 15 measures nothing: SGLang is frozen by default, since I16 cannot be met without a baseline on the surviving vLLM (M4). M16 records the three failed probes, and the operator may overrule it. The keep thresholds and their measurement are I16, recorded before any run.
2. Emit `--enable-metrics` and pin it. The engines are launched outside the router, with the whole stack down (`make cache-down`; before each launch the check runs C8's card check -- `nvidia-smi --query-compute-apps=pid --format=csv,noheader` prints nothing and `memory.used` reads at most 512 MiB -- and refuses otherwise, because a container-name list such as `assert_no_active_backends`'s misses a GPU lab container and a leftover throwaway container, and stopping only the router leaves Ollama's warm model or an engine on the card): `scripts/bench/sglang_keep_check.py` starts each through the probers' container driver -- SGLang with `sglang_command_args` (which emits `--enable-metrics` as the router does from this phase, and `--disable-radix-cache` for the radix-off run), vLLM 0.30.0 (Phase 4 step 4v, E1; 0.28.0 if 4v stopped the move) with the vllm-devai prober's arguments -- and runs the workload against it directly: the multi-turn sessions of `bench_concurrency.py`, with the cache-hit read-out. Before the measured run, an uncounted dry run checks the instrument: SGLang radix-on, 2 sessions (salted `<salt>-dry-<engine>-<i>`, outside the measured set), whose turns 2-5 must show a non-zero hit rate. Zero means the workload does not reuse prefixes: the scorer is fixed and the dry run repeated, before any measured run (I16).
3. Measure on one checkpoint served by both engines, exactly as I16 defines it (the same check script; MTP and reasoning off, no tools, one context): steady TTFT, turn-2+ TTFT over salted sessions, and SGLang's engine-reported cache hit rate.
4. Decide, and record it as M16:
   - **keep:** Phase 4 builds SGLang as planned;
   - **freeze:** Phase 4 skips steps 4a-4b and SGLang's stage in `Dockerfile.engines`; `sglang` leaves `_PICKER_BACKENDS`, the make targets and, where Phase 6, 7 or 10 has added them, their `deploy/operator/actions.yaml` entries (probe-sglang, probe-load-sglang, probe-sglang-model, probe-load-sglang-model, with their scripts/ops/ scripts), because test_makefile_ops.py requires `covers` and `not_offered` to equal the documented targets; its entry leaves the router's backend table together with the then-unused `sglangURL`, so its router code stays but is unreachable (checked on a copy: the router builds and `go test` passes, 2026-10-06, with the prober still in place). Its prober (`scripts/probe-sglang-reasoning.py`) and its SGLang-only Python tests (tests/python/test_picker_sglang.py, and test_bench_sglang_gate.py with the SGLang bench target) move to `attic/sglang/`, with a `RESTORE.md`, following the `attic/` precedent. Everything else that lists `sglang` as a backend drops it in the same change, so nothing points at a dead port:
     - `scripts/model-sync.py`: its `make probe-sglang` step;
     - `scripts/bench-sync.py`: `BACKENDS`;
     - `scripts/bench/bench_runner.py`: its backend choices, and the SGLang bench target;
     - `scripts/select-models.py`: the SGLang store-gap check;
     - `scripts/probe-check.py`: its sglang row;
     - `scripts/model-picker.py`'s `_BACKENDS`, and the `router-sglang` entries in the three agent seeds, config/codex/config.toml (the provider table and the comment on line 6), config/opencode/opencode.json and config/pi/models.json, so no agent keeps the provider;
     - tests/python/test_picker_opencode.py and tests/python/test_picker_pi.py: their expected provider lists drop `router-sglang` (both assert that the picker writes one provider per `_BACKENDS` entry), or `make test-python` fails after the freeze;
     - the tests that read the SGLang prober: `gpu-arbiter/flags_allowlist_test.go` (the `.4f` check) reads only `probe-vllm-reasoning.py`; the SGLang cases of tests/python/test_probe_classify.py (`self.sglang` in `TestKvCacheDtypeThreading`) and tests/python/test_needle_warning.py (`self.s`, `test_sglang_prober_emits_max_running_requests`) move to `attic/sglang/`, and their setUp stops loading the SGLang prober, or `make test-router` and `make test-python` fail.

     Check: `make model-sync DRY_RUN=1` and `make bench-plan` name no SGLang step or row, and `grep -rnE 'router-sglang|:11436' config/` finds nothing, and `make test-router` and `make test-python` pass. The fixes from sglang-backend-remediation's Phases 1-3 stay: most of them serve vLLM too. Every mention of SGLang in Phases 4-14, 16, 18 and 19 and in C1, C2, C5, C6 and C7 then lapses, and exit criteria that name SGLang read "if kept". The `RESTORE.md` carries the SGLang plan's open questions.
   - Either way, M16 records the unexplained Qwen3.5-9B HumanEval result (0.86 on SGLang against 0.68 on vLLM, same checkpoint).
   - Either way, once M16 is recorded, `scripts/bench/sglang_keep_check.py`, its `sglang-keep-check` target and, if Phase 6 has added it, its `not_offered` entry in `deploy/operator/actions.yaml` are removed in the same change (Phase 7's test_makefile_ops.py requires `covers` and `not_offered` to equal the documented targets). Git keeps them, and M16 cites the report and the commit. It is a one-time measurement, built on the vllm-devai prober (removed in 4e), the SGLang prober and the container driver (removed in Phase 14).
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
| no baseline loads on the vLLM in service (0.30.0 from Phase 4 step 4v, E1) | the three-candidate order in step 1, probed in 4v; freeze by default, recorded in M16 |
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
   - **Forward it to the engine.** The vLLM in service (0.30.0, E1) and SGLang 0.5.16 serve that route themselves and count the fully rendered chat template, so the count is exact.
     - Read in vLLM 0.28 and SGLang 0.5.16 on 2026-10-06, and in vLLM 0.31.0 (`entrypoints/anthropic/api_router.py:89-119`, `entrypoints/anthropic/serving.py:1142-1171`).
     - Only a partial 0.30.0 tree was read here, without these files, so Phase 4 step 4v re-reads them in 0.30.0.
     - The router forwards it only when the request would be proxied without any launch. After the suffixes are peeled, the loaded launch (model, context, speculative spec) must be exactly this one, and serving: Ensure case 1, or before Phase 9 `bs.running` with the same `currentModel`, `currentContext` and spec.
     - It then calls the proxy directly, never `ensureBackendRunning` or Ensure, so a count can neither start nor switch anything.
     - First it runs the `/v1/messages` rewrite chain unchanged -- the system-message fold and `applyHFAnthropicMessagesPolicy` -- behind a path check that accepts both paths.
       - Under DISABLE that sends vLLM `chat_template_kwargs.enable_thinking=false` and SGLang `thinking: {type: disabled}`.
       - Under ENABLE or noop it drops `thinking` on SGLang; without the drop, a row with no reasoning parser answers 400.
       - vLLM's count reads neither `output_config` nor `thinking`: its request model has neither field, and the shim skips both for a count (0.31.0 `entrypoints/anthropic/protocol.py:321-344`, `entrypoints/anthropic/serving.py:598-599`, `:620-621`). Only `chat_template_kwargs` reaches the template. So on vLLM the router copies vLLM's own derivation, from the body the chain left:
         - let `e` be the `output_config.effort` in that body: under ENABLE the router's value, under noop the client's own, under DISABLE absent;
         - if `e` is set, the router puts `chat_template_kwargs.reasoning_effort = e`, and, unless `chat_template_kwargs.enable_thinking` is already present, `enable_thinking = (e != "none")`.
         - That is what `build_chat_params` does for `/v1/messages`: 0.31.0 `entrypoints/openai/chat_completion/protocol.py:568-588` (read here), and 0.28 `:575-588` (as cited by the 2026-10-06 review, not re-read here). The request's own unset values do not override the client's kwargs (0.31.0 `renderers/params.py:36-48`), so the count renders the same kwargs. Phase 4 step 4v re-checks this in 0.30.0's `build_chat_params`.
         - The router already sets `enable_thinking` under ENABLE and DISABLE (`gpu-arbiter/anthropic_effort.go:100`, `:109`), so the rule adds it only on the noop path.
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
- count_tokens returns the engine's own count on the fixtures (vLLM and SGLang), equal to the prompt tokens the engine reports for the same `/v1/messages` body under `auto` and under `::nothink`. On vLLM the same equality, under both, also holds on one row whose template's thinking is opt-in and whose probe cell says `capability: none`, with a captured Claude Code body that carries `output_config.effort`: Gemma-4-26B-A4B-it-NVFP4 if it loads on the vLLM in service (its `chat_template.jinja` treats an absent `enable_thinking` as false; it is `capability: none` on stock vLLM 0.22.1 today). It answers 503 `count_tokens_unavailable`, with the loaded model untouched, when another model or context is loaded or nothing is, and 404 `count_tokens_unsupported` on the Ollama port. Or it is recorded as shelved.
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
- **17b, the estimator (the rest of CB-1):** rank every catalog row before probing, and fill the picker's empty TPS cells with a clearly labelled bandwidth-bound estimate. Quality cells (CODE%, CODE+%, MMLU%, GPQA%, TOOLS) are never estimated: the quant quality table gives only a relative penalty, which orders `make model-rank`'s output within a tier and has no unquantized score to subtract from. The arch-based KV term already shipped in `scripts/select-models.py` (kv_layers, per-backend dtype); catalog-discover's weight-keyed term is the fallback for rows without `arch`.

It needs nothing, but 17a must land before Phase 8 (I25): Phase 8 moves 17a's reader of the generated recovery file into `pkg/engineargs`. 17b's calibration uses the re-probed caches (Phase 4) and the re-bench (Phase 10).

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
                                            `"source": "recipe"`, only where min_vllm_version admits the vLLM in
                                            service: 0.30.0 from Phase 4 step 4v (E1; 0.28.0 before it, or if 4v
                                            stopped the move), read from build-vllm.sh's default VLLM_VERSION, which
                                            tests already pin (D8). Scoped to the backend
                                            that runs that version: `vllm-devai` before Phase
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
scripts/_fit_estimate.py          new    -- 17b: estimate_memory_gb(row, ctx, kv_dtype). A row with `arch` is costed
                                            with select-models' kv_per_token_bytes, resolve_kv_dtype and KV_BYTES
                                            (kv_layers, per-backend dtype), which move here; select-models imports
                                            them back. Only a candidate without `arch` (catalog-discover) uses the
                                            weight-keyed estimate_kv_gb, which also moves here. A bandwidth-bound
                                            decode estimate keyed by the GPU's name (on the host `nvidia-smi
                                            --query-gpu=name`, which DEVAI_GPU_TYPE overrides; in the picker, the
                                            bench cache's); a quant speed table, and a quant quality-penalty table
                                            used only for model-rank ordering. It imports only the stdlib and yaml
scripts/catalog-discover.py       modify -- imports its KV term (estimate_kv_gb) from _fit_estimate
scripts/select-models.py          modify -- `make model-rank` (tiers perfect / good / marginal / too_tight; VRAM= override);
                                            ranks at CONTEXT (default: each row's largest standard tier within its
                                            position limit), and states each tier's bounds as fractions of VRAM in
                                            the tool's help text
Makefile                          modify -- fetch-vllm-recipes, recipe-apply, model-rank; export DEVAI_GPU_TYPE;
                                            `export DEVAI_REPO := $(CURDIR)` unless Phase 6 added it first (I25)
tests/test-router-idle.sh, tests/test-model-matrix.sh
                                  modify -- 17a: they call compose themselves, outside make, so each defaults
                                            DEVAI_REPO to the repository root before its first compose call:
                                            `: "${DEVAI_REPO:=$(cd "$(dirname "$0")/.." && pwd)}"; export DEVAI_REPO`
                                            (I25; tests/test-gpu-vendor.sh is in the attic since Phase 1, M11)
.env.example                      modify -- DEVAI_GPU_TYPE
deploy/operator/actions.yaml      modify -- once Phase 7 has landed: fetch-vllm-recipes, recipe-apply and model-rank,
                                            each a `job` action (job_kind action) with a scripts/ops/ script (Phase
                                            7's coverage test needs all three)
scripts/model-picker.py           modify -- a recipe badge next to TOOLS; a labelled `est` value only in empty TPS cells;
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
                                  new    -- tables, tiers, offline behaviour, the version gate; model-rank's fits /
                                            too_tight split agrees with model-fit's estimate_total_gb for every
                                            catalog row at the same VRAM and CONTEXT
```

### Detailed steps

1. **17a:** a parser comes from, in order: the curated `parsers:` block; the card-derived value (`scripts/_card_hints.py`, which refuses rather than guesses); the recipe seed; a name heuristic, labelled "guessed, not probe-verified"; none. The recipe seed and the name heuristic are guesses for a model nobody has curated: they fill only a row with no curated block for that engine, and a kind a curated block leaves out stays out unless the card derivation fills it (the operator, 2026-10-10, choosing this narrow rule over the literal order: Gemma-4's measured "no reasoning parser" stands although its vLLM recipes name `gemma4`). The probe is the final authority, and the probe cache records where a value came from (`recipe` joins `tool_parser_source`'s values). A recipe never injects a flag the pinned vLLM rejects (`min_vllm_version`). Phase 4's card-path check ignores the curated block, the recipe seed and the name heuristic (I24).
2. **17b:** probe and bench caches stay authoritative. An estimate appears only where no measurement exists, always labelled `est`. An unknown GPU type gives no estimate, never a wrong one. The odysseus idea of feeding predicted speed into cluster routing is dropped with cluster mode.
3. **Calibration (17b), after Phases 4 and 10:** compare the predicted decode speed of every benched model with its measured one, and record the error band.

### Exit criteria

- For a model with an official recipe, the parsers are seeded before any probe runs; after `make recipe-apply NAME=<row>`, its first probe launches with the recipe's flags that lie outside C2, and the rest are reported.
- The picker shows a recipe badge; a probe-verified value overrides a recipe-seeded one; a model without a recipe falls back to the labelled heuristic, then to today's behaviour.
- `make model-rank` prints a tier for every catalog row without starting a container, and `VRAM=80` re-ranks the catalog as on an 80 GB card.
- The picker shows `est` only in the TPS column and only where no bench value exists; no quality column ever shows `est`.
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
devai-tools/internal/control/     modify -- a load_failed error -- the reply's, and the `error` of the `failed` state
                                            that `devai-control` publishes as an event and returns to `state` --
                                            carries `diagnosis {message, suggestions:[{kind, op,
                                            flag, value}]}`, an additive field on C1's error object: `kind` is
                                            `applicable` or `report` (I18), `op` is `append`, `replace` or
                                            `remove`, and an applicable suggestion is always an `append`
gpu-arbiter/slurm_launch.go       modify -- the 502 for a failed load carries the diagnosis read from the `failed` state's
                                            `error` (event or `state` query, Phase 9 case 5), and X-DevAI-Warning
                                            names it; a test feeds a stub `failed` event carrying a diagnosis and
                                            finds it in the 502
scripts/_probe_hf_common.py       modify -- the bus driver retries a load once with the first suggestion of `kind` `applicable` (I18): it adds
                                            a group to that model's list in deploy/recovery-flags.generated.json
                                            (I25) -- `backends` = the failing backend, `"source": "auto-fix"`, the
                                            diagnosis as `reason` -- logged and attributed. If the retry also
                                            fails, it removes that group again (whole-file rewrite and rename,
                                            I25) and records both failures and the diagnosis in the cell's
                                            evidence. A suggestion whose flag is already in the launch's effective
                                            argument list is reported, not applied. `report` suggestions are
                                            printed with the diagnosis on the exact-miss and fits-nowhere paths,
                                            and the fits-nowhere path also stores them in the evidence of the cell
                                            it keeps (an exact miss still restores the previous cell untouched)
tests/python/test_probe_autofix.py
                                  new    -- the retry without a GPU: the prober's bus driver against a stub
                                            devai-control that answers load_failed with a synthetic CUDA-graph OOM
                                            log tail (tests/fixtures/vllm/cuda-graph-oom.synthetic.log: no real case
                                            exists), then serving once the auto-fix group is in the generated file;
                                            and a stub that answers load_failed twice, after which the generated
                                            file is byte-identical to how it started
```

### Detailed steps

1. Port the signature table from odysseus's `routes/cookbook_helpers.py:1208-1381` after re-reading it, and pin it with failure logs: `tests/fixtures/sglang/argparse-rejection.log`; new `tests/fixtures/vllm/` files cut from the persisted engine logs (`/var/cache/devai/logs/devai-vllm*.log`) and the strings `tests/python/test_probe_classify.py` already holds; and, for a signature with no captured case (the CUDA-graph OOM), a synthetic excerpt whose first line says so. Add the memory signatures Phase 8 recognises (vLLM's `less than desired GPU memory utilization` and `larger than the available KV cache memory`, SGLang's `Loaded weights leave no GPU memory for the KV cache`): today's router list has only `No available memory for the cache blocks`, so the router reports such failures as the generic `Engine core initialization failed`.
2. Mark each fix `applicable` or `report` (I18).
3. Wire the diagnosis into `devai-control`'s `load_failed` reply and its `failed` state, the router's 502 and the prober's evidence. The router learns of a failed load only from the `failed` state (event or `state` query), never from the reply, so the state must carry it.
4. Add the one-shot retry to the probers' bus driver (`devai-control` re-reads the recovery files at every load since Phase 8).

### Exit criteria

- A probe that fails on a CUDA-graph OOM is diagnosed, gets `--enforce-eager` in a new group of the generated recovery file (`backends` = its backend, `source` = `auto-fix`), and passes on the second attempt.
  - No model on disk is known to fail this way: every curated `--enforce-eager` entry belongs to a model whose weights are not on disk, and I17 downloads nothing.
  - So `test_probe_autofix.py` shows the retry, and Phase 8's exit criteria show `devai-control`'s re-read. A live run waits for a real case.
- A probe launched with a deliberately too-large context (`make probe-vllm PROBE_REPO=<row> PROBE_CTX_EXACT=<a value above the engine's estimate>`, a probe job from Phase 10 on) fails non-zero; its `[miss]` line in the job log names the diagnosis "context too large / KV OOM" and its `report` suggestion; the probe cache is byte-identical to before and deploy/recovery-flags.generated.json is unchanged (nothing is applied).
- A failure that matches no signature is reported verbatim: no silent wrong fix.
- The retry is capped at one attempt per model and backend per prober run, across all of that run's context-search launches. When the retry fails, the generated recovery file ends byte-identical to how it started (test_probe_autofix.py: the stub answers load_failed twice).

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
gpu-arbiter/main.go               modify -- with DEVAI_FANOUT_PORT set, a lanes listener with its own mux: the same guards
                                            as newBackendMux (ollamaMutationPaths -> makeMutationGuard, and C5's
                                            block of the engines' control routes) registered before anything
                                            else; lanes resolved on POST /v1/chat/completions, /v1/messages and
                                            /v1/messages/count_tokens, which rewrite `model` and hand the request
                                            to the normal handler (the Slurm launcher's Ensure); GET /v1/models
                                            lists the lanes and aliases; /health and /devai/v1/ready answer as on
                                            the other ports; every other path answers 404 until a polish item in
                                            step 5 adds it
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
4. Each lane's model is checked against the probe caches at boot; a lane without a fitting cell is skipped and logged. If the default lane is skipped (or `default_lane` names no lane), the lanes listener does not start, and the log says why; the other ports are unaffected.
5. Polish, each item separately shippable: stickiness (the latest `@@<lane>` in the resent history applies when the current message has none); Ollama-native `/api/chat` and text parts of multimodal content. A heuristic auto-lane stays parked: routing stays explicit.

### Exit criteria

- `ResolveTargets` returns the right target for a model-field match, an alias, each directive form and the default, and 400 for an unknown token -- over OpenAI and Anthropic bodies.
- The directive is stripped from the forwarded body.
- With `DEVAI_FANOUT_PORT` unset, the router is byte-identical to before.
- A real two-slot agent configuration sends its foreground and background requests to two different lanes.
- On the lanes port, POST /api/pull, /api/create and /api/delete answer 403, an unlisted path answers 404, and a registry whose default lane has no fitting cell leaves the port closed with a logged reason.

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

It needs Phase 2 (the pinned Debian image, pipelock and apt-cache built here), Phase 3 (`devai-base` on Debian), Phase 6 (`devai-operator`: its `LIBRARY_DIR` mount, the stack page) and Phase 7 (`scripts/ops/operator-up.sh`, which adds `compose.operator-library.yaml`, and the rewritten `devai-infra` unit that 20.1's systemd exit criterion restarts). Phase 12's `lab-start.sh` gets the `/library` mount once both have landed. It needs no GPU and no Slurm: none of its containers is a router backend. Its Makefile targets join the operator's registry (C8) with `scripts/ops/` scripts. Sub-phases 20.1-20.5 run in order; 20.3 needs aiagent's side.

### Needs

- **aiagent's requirements document** (`docs/design/library-service-needs.md` in devitops-com/aiagent), the field-level contract, and aiagent's client implementation.
- **Owner actions, each blocking the phase named:**
  - the dedicated contact address (LS21) on the company's own domain: blocks 20.1's live exit (no external request without it), and every key application below;
  - an OpenAlex API key: optional, raises the daily budget from $0.10 to $1 (20.2);
  - the embedding store's directory, `/var/cache/devai/embed` (a plain directory the operator creates, LS24, E9): blocks 20.2's index;
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
- **`devai-library-extract`**: the extractor, and the only caller of `devai-grobid`. It serves one endpoint on `devai-library-internal`, `POST /v1/extract` (body: the raw bytes; query: `kind=pdf|html`, `sha256`, `grobid=0|1`), and returns `{units, links, quality}` in the shapes of the text records under "Text extraction". It converts GROBID's TEI to units itself, so no XML or HTML from outside is ever parsed in `devai-library`. Libraries: `pypdfium2` (Apache/BSD) and `pypdf` (BSD) for PDFs, `lxml` for HTML and TEI. It has no secrets, no mount of the store and no network except `devai-library-internal`. The service validates every response against a JSON Schema shipped in its own package (`extract-response.schema.json`) before writing anything; an invalid response is `extraction_failed`.
- **Limits on fetching and extraction**, all configuration in `deploy/library-sources.yaml` and reported by `/v1/info`: a download larger than `max_fetch_bytes` (default 64 MiB; checked on `Content-Length` first, then counted while streaming) is abandoned and fails the item as `too_large`; one extraction call (GROBID included) gets `extract_timeout_s` (default 120 s), after which the service drops the connection and the item fails as `extraction_failed` (final, not retried, still fetchable); compose sets `mem_limit` on `devai-library-extract` (2 GiB) and `devai-grobid` (6 GiB), and an out-of-memory restart during a call is also `extraction_failed`.
- **`devai-grobid`**: GROBID (Apache-2.0), built here (LS29), for PDF structure (sections, paragraphs, formulas, references with DOI and arXiv ids, page coordinates). About 4 GB of RAM. Called with `consolidateHeader=0` and `consolidateCitations=0`, so it never contacts CrossRef or anything else.
- **The embedding model**: `bge-small-en-v1.5` (MIT, 384 dimensions, 512-token input), run with ONNX Runtime on the CPU inside the service. Downloaded only through `scripts/select-models.py`, from a pinned catalog row, into its own store (LS24).

None of them is a router backend: none needs the GPU, so the router's GPU exclusivity does not apply.

#### Network and egress

- **Networks.** `devai-library` sits on `devai-lab-egress` (internal, no route out) and `devai-library-internal` only, with no host port, so pipelock is its only path to the internet and it has no route to devai-net's services (the router, `devai-slurm`, the engines); the lab reaches it as `http://devai-library:8080`, and the name joins the lab's `NO_PROXY` lists (Makefile `PIPELOCK_NO_PROXY`, `bin/devai-agent` `NO_PROXY_HOSTS`). A third network, `devai-library-internal`, created with `--internal` by `make cache-up` and declared external in compose like the others, is the only network of `devai-library-extract` and `devai-grobid`; only `devai-library` joins it too.
- **No URL ever comes from aiagent** (agreed with aiagent, its L11). Requests name a vetted source and a query, an identifier the service knows how to resolve (`arxiv:`, `doi:`), or an id the service issued (item, version, link). A link found in fetched content or in third-party metadata comes back as a link record, and the service alone decides whether it is fetchable.
- **Every hop is checked.** Automatic redirects are off; each redirect is checked like a first request (host list, robots.txt, the host's limits). IP-literal URLs, internal service names, and loopback, private, link-local and unique-local addresses are refused. A host that is not in the host list is not fetched: admission is default-deny.
- **Egress through pipelock**, so its DLP scans what leaves, including query text that aiagent wrote. The service trusts pipelock's CA: `deploy/compose.library.yaml` sets `HTTPS_PROXY`/`HTTP_PROXY=http://pipelock:8888` and `NO_PROXY=devai-library-extract` on `devai-library`, and mounts `deploy/pipelock-ca.crt` read-only, named by `SSL_CERT_FILE`, which the service's HTTP client loads. pipelock's header DLP will match real credentials (GitHub tokens, API keys): each is exempted with a per-pattern `exempt_domains` limited to its issuer's API host, in the phase that first sends it; keys travel in headers, never in URLs, except OpenAlex's if 20.2 finds it accepts only the `api_key` query parameter.
- **pipelock's own answers are not upstream answers.** A pipelock block (DLP) comes back as a 403, and its connect failures as 5xx, on what looks like the upstream connection. The service recognises pipelock's responses (its block body or marker, confirmed in 20.1) and never counts them as the source's answer, so a DLP hit on one query cannot open a source's breaker: a DLP block is `egress_blocked`, final for that request and never resent unchanged; a pipelock connect failure or 5xx is `egress_unavailable`, a wait retried within the item's retry budget.
- **pipelock's response cap.** pipelock buffers each intercepted response up to `tls_interception.max_response_bytes` (5 MiB in deploy/pipelock.yaml), and many PDFs are larger. 20.1 first observes what pipelock returns above the cap (a block, a 5xx or a truncated body) and whether its block body carries a marker once `explain_blocks: true` is set. Phase 20 then raises `max_response_bytes` above the service's `max_fetch_bytes` ("Components" above; the setting is pipelock-wide, so the lab's traffic gets it too), and the service refuses a larger body itself as `too_large`. If pipelock's 403 still cannot be told apart from an origin 403, a 403 is treated as an origin denial (the breaker opens), never the other way round.
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
- Clearing a denial breaker, or recording the arXiv contact, is an explicit operator command: `devai-library-admin breaker-clear <provider>` and `devai-library-admin arxiv-contact-recorded`, run inside `devai-library` and offered as C8 actions of kind `job` in the `library` group (job_kind action; each runs `podman exec devai-library devai-library-admin ...`). The service is the only writer of `state/`. The admin CLI does not touch the files: it sends the command to an admin listener that the service binds to 127.0.0.1 inside its own container, which is reachable from `podman exec` and from no network. The service changes its in-memory state, persists it atomically under `state/`, appends the change to `state/alerts.jsonl`, and answers. If the service is not running, the command fails non-zero and changes nothing. Alerts -- an opened denial breaker, the first 429 from a host -- are appended to `state/alerts.jsonl` and listed under `alerts` in `/health`; the operator's stack page shows them. Counters that must survive a restart (daily budgets, the arXiv total, breaker state) live under `state/`.

#### Configuration

- **`LIBRARY_DIR` in `.env`**, set once (LS15). Compose and the Makefile read it from `.env`. `bin/devai-agent` does not read `.env` by design, so `make install` stages it as `~/.devai/library`, a symlink to the directory, exactly as it stages the probe caches; the launcher mounts it and prints a warning naming `make install` when it is missing. A test checks that compose, the Makefile and the launcher resolve the same path.
- **Optional everywhere except in the service.** The lab mounts `/library` (read-only) only when the library is configured, as it mounts `/laya` only when that exists. `devai-library`, `devai-library-extract` and `devai-grobid` live in `deploy/compose.library.yaml`, which `make cache-up` and the stack-up action add only when `LIBRARY_DIR` is set, as `deploy/compose.gpu.yaml` is added only on a host with a GPU (Phase 11). devai-operator's own `LIBRARY_DIR` mount lives in `deploy/compose.operator-library.yaml`, which only `make operator-up` adds: the stack-up action runs inside devai-operator and must never recreate it (Phase 6). That file also sets `DEVAI_OPERATOR_LIBRARY_DIR` in devai-operator; after setting or changing `LIBRARY_DIR`, run `make operator-up` once. A profile with an inert default would not do: devai-operator runs all the time and is in no profile, and podman refuses a bind mount whose source does not exist. Without these files, compose never sees the variable. The systemd unit (Phase 7) runs the same two scripts through `make operator-up cache-up`, so it starts the library and devai-operator's library mount under the same conditions as `make cache-up` and `make operator-up`.
- **Three checks, no more (LS15).** `scripts/ops/stack-up.sh` (behind both `make cache-up` and the stack-up action) and the service check that `LIBRARY_DIR` is an absolute path, that the directory exists, and that it is readable and writable; a failure refuses the library's start and names the requirement that failed. Inside devai-operator, stack-up.sh first compares `.env`'s `LIBRARY_DIR` with `DEVAI_OPERATOR_LIBRARY_DIR`. When they differ, it refuses the library with "LIBRARY_DIR is not mounted in devai-operator: run make operator-up", and starts the rest of the stack. Nothing creates the directory, and nothing checks where it lives, whether it is its own volume, its filesystem or its free space. The service's containers run with the owner's host uid, so the owner's read and write permission is all they need. On its first start in a directory the service writes `LAYOUT.json` (the layout version) and its own top-level directories beside whatever is already there, and it never touches a file it did not write.
- **`LIBRARY_CONTACT`** in `.env`: the contact address (LS21). `.env.example` ships it empty (`LIBRARY_CONTACT=`); LS21's placeholder is that empty value, never a made-up address. The service counts the contact as configured only when the value is one addr-spec (`local@domain`) with no whitespace or control characters and with a domain that is not `example.com`, `example.net` or `example.org` and not under `.invalid`, `.example`, `.test` or `.localhost`. Anything else (unset, empty, malformed) counts as not configured: the service answers but makes no external request (`contact_not_configured`, `contact.configured: false`). A value that is set but rejected is also logged once at start, with the reason.

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
  jobs/lbjob-<24 hex>/           job.json (with the request's Idempotency-Key and body hash, honoured 24 h), events.jsonl, results.jsonl
  profiles/<name>/<version>.json research profile versions, opaque
  state/                         politeness counters, breakers, the arXiv contact record
  trash/<date>/                  artifacts of revoked items, kept 30 days (LS28)
  index/                         derived only: alias index, keyword index, vector matrices, topic listings, and the candidate
                                 cache (candidate records kept 30 days, search results 24 h, enrichment results with as_of)
  cache/embeddings/              derived: unit embeddings keyed by (unit text sha256, model sha256); kept by a rebuild
```

- **The plain files are the only source of truth.** An item's current state is the fold of its `records/`; everything under `index/` and `cache/` is derived and can be deleted and rebuilt; a rebuild deletes only `index/` (aiagent's acceptance check 6).
- **Fan-out.** Two levels of two hex digits, as in git's object store but one level deeper: 65,536 leaf directories, so ten million items or objects leave about 150 entries per directory. The fan-out keeps every directory small whatever the filesystem, so `ls`, `find` and shell globs stay fast.
- **Writes are atomic** (write to a temporary name, then rename); files are 0444 and directories 0555 once final, as on the laya store. Standard tools work directly: `find items -name key.json`, `jq` over `records/`, `grep` over `units.jsonl`.
- **English only (LS16) in what stays.** Permanent records hold English only: for a non-English item, aiagent submits English metadata (`metadata_en`) with its verdict, and English renderings of evidence quotes (`quote_en`); the service stores those and keeps the original only as a link with its sha256. Staging, which is temporary, holds the original-language text until the verdict or the TTL.
- **The candidate cache and job keys.** The candidate cache under `index/` is disposable (aiagent's principle 9): a search refills it, and check 6 deletes it with the other indexes. Like staging it may hold original-language metadata, until it is pruned after 30 days. A candidate gets an `item_id` but no item directory: one is created only by a fetch (under `staging/`) or by a verdict or an assessment (under `items/`), and a metadata-only reject writes its `metadata_en` snapshot and verdict there, never the original-language text. An `Idempotency-Key` lives with its job in `job.json`, which is kept 30 days, longer than the 24 h the key is honoured.
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
- **Vector:** embeddings stored as derived files under `cache/embeddings/` keyed by (unit text sha256, model sha256), so a rebuild does not re-embed. Search starts as exact search over memory-mapped matrices (float32, later int8); faiss-cpu (MIT) is the next step when measurements call for it. Units longer than the model's 512 tokens are chunked internally; a hit always reports the unit.
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
| O-D1 | A plain compose container, not a router backend: `http://devai-library:8080` on devai-lab-egress (and devai-library-internal); no devai-net. |
| O-D2 | The library root is mounted read-only at `/library` in the lab when configured (later `inbox/` read-write), from `LIBRARY_DIR` (staged for `bin/devai-agent` by `make install`). Secrets never live in the library directory. `state/`, `index/` and `cache/` live under the root; aiagent must not depend on their files. |
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

Each key the library uses is a plain file the owner writes once into `~/.config/devai/library/` (mode 0700, created by `make library-init`): `openalex.key`, `github.token`, `hf.token`, `s2.key`, `openreview.credentials`, each 0600. Each is mounted read-only into `devai-library` only. The service refuses a key file that group or others can read, never logs a key, and sends keys in headers, never in URLs, except OpenAlex's if 20.2 finds it accepts only the `api_key` query parameter. `devai-backup` includes the directory.

### 20.1 -- store, service core, politeness, arXiv

**Goal:** a running service that searches arXiv and fetches PDFs and HTML politely, stages English text units, and records verdicts, with jobs, progress and ETA.

**Deliverables:**
- Configuration: `LIBRARY_DIR` and `LIBRARY_CONTACT` (empty) in `.env.example`; `deploy/compose.library.yaml`, added by `scripts/ops/stack-up.sh` (behind `make cache-up` and the stack-up action), and `deploy/compose.operator-library.yaml`, added by `scripts/ops/operator-up.sh` (behind `make operator-up`), each only when `LIBRARY_DIR` is set, which also brings both to the systemd unit (Phase 7) without a change to `install-systemd`; `make install` staging `~/.devai/library`; the optional lab mount in the Makefile, `bin/devai-agent` and Phase 12's `scripts/ops/lab-start.sh`, with `devai-library` in both NO_PROXY lists; `LIBRARY_DIR` among devai-operator's path-identity mounts (C6), so the stack-up action's checks and the alerts listing work from there; `~/.config/devai/library/` in devai-backup's manifest; LS15's three checks in `make cache-up` and the service; the proxy environment and the read-only pipelock CA mount on `devai-library`.
- Images and services: `deploy/Dockerfile.library` and `deploy/Dockerfile.library-extract` (hash-locked, on `devai-base`), `deploy/Dockerfile.grobid` (LS29), `devai-library`, `devai-library-extract`, `devai-grobid`, the `devai-library-internal` network; `make build-library`, `make build-grobid`, `make test-library`, `make library-init` (LS30); each a C8 action: `build-library`, `build-grobid` and `test-library` `job` actions (job_kind action), `library-init` of kind `host`.
- `deploy/library-sources.yaml` with the arXiv rate groups; the politeness engine, the RFC 9309 wrapper with the stricter 4xx rule, the redirect and address checks, the breakers, the stops, recognition of pipelock's responses, pipelock's response cap raised above `max_fetch_bytes`, `too_large` enforced by the service; the admin listener on 127.0.0.1.
- The store: layout, append-only records and their fold, alias resolution with a single writer, tombstones, the maintenance job (staging TTL, and the 30-day trash of revoked items, LS28).
- The arXiv adapter: search (GET, cached 24 h), version resolution, PDF, HTML, `arXivRaw` in the fetch job; format checks. The check of the anonymous PDF mirror's terms and robots.txt (LS27); the mirror is wired in only if both allow it.
- Extraction: HTML-primary units with text-aligned pages, GROBID for PDFs without HTML (called by the extractor only), the language detector; `POST /v1/extract`, `extract-response.schema.json`, `extract_timeout_s`, and the compose memory limits.
- The v1 endpoints except enrichments and library search: verdicts with evidence checks, the one-current-verdict rule with A-O1's scoped exception, topic assessments, tombstones and 410.
- Jobs: events, waits, ETA per rate group, heartbeat, cancel, resume.
- The operator: the library-admin actions (`breaker-clear`, `arxiv-contact-recorded`) in the `library` group, and the stack page listing `<LIBRARY_DIR>/state/alerts.jsonl` when `LIBRARY_DIR` is set.
- Docs: `docs/library.md` (reference), CLAUDE.md, `docs/pipelock.md` (the finding that it does not restrict destinations).

**Exit criteria:**
- Unit tests with a fake upstream: intervals and crawl delays per rate group (search and OAI-PMH never overlap); 429 and 503 with and without `Retry-After` (a wait, never a breaker); sustained 429s (the self-clearing overload pause) and an origin 403 (the denial breaker, cleared only by the operator); `breaker-clear arxiv` against a running service turns arXiv from `down` to `ok` in `/health` without a restart, and the cleared state survives a restart; `arxiv-contact-recorded` lifts `arxiv_fulltext_cap` the same way; both commands fail and change nothing when the service is down; a fake pipelock block that must not open arXiv's breaker; robots outcomes (`robots_disallowed`, `robots_denied`, `robots_unreachable`, 404 as allow) and the LS5 exemption; the stops' 503 and per-source error shapes; a cross-host redirect and a redirect to an internal name, both refused; every per-item outcome in aiagent's section 3.1; a body over `max_fetch_bytes` (`too_large`, with and without `Content-Length`); an extractor that hangs past `extract_timeout_s` and one that returns schema-invalid JSON (both `extraction_failed`); a source test that `devai-library`'s package never calls GROBID (only the extractor does); HTML present, absent and broken; verdict rules and the fold; the stops as `/health` and ETA show them; `LIBRARY_DIR` unset, relative, missing, and not writable (each refused, naming the requirement), and valid on an empty and on a non-empty directory (the layout written on first start, other files left alone); `LIBRARY_CONTACT` unset, empty, an `example.com` address, one with an embedded CR or LF, and a valid address (only the last counts as configured).
- A layout test at scale: synthetic items and objects in the hundreds of thousands, timing `find` and a full fold.
- `LIBRARY_DIR` set in `.env` while devai-operator has no library mount: the stack-up action refuses the library naming `make operator-up`, not a missing directory.
- After `make install-systemd`, a restart of the unit brings up the stack with the same overlays as `make cache-up` and `make operator-up`: with `LIBRARY_DIR` set, `compose.library.yaml` and `compose.operator-library.yaml`, after LS15's checks.
- From inside `devai-library`: a direct TCP connection to an internet address with no proxy fails; a fetch of an allowed host through pipelock succeeds over TLS; `devai-router` and the `devai-slurm` ports cannot be reached.
- A live smoke run, once the contact address exists: one small arXiv query, a fetch of two papers (one with HTML, one without), units with anchors and pages, one arXiv PDF over 5 MiB fetched through pipelock and stored byte-identical to a direct download, a reject that purges and answers 410, an accept that keeps, all within arXiv's limits.

### 20.2 -- OpenAlex, open-access locations, index

**Goal:** sources beyond arXiv (LS23), and search over the library.

**Entry:** a manual read of OpenAlex's Terms of Service in an ordinary browser.

**Deliverables:**
- The OpenAlex adapter: discovery (structural filters where they fit, text matching priced as search), open-access locations, `cited_by_count` and citing works, advisory author metrics with provenance (source, as-of date, match method), the dollar budget; `POST /v1/enrichments` and `GET /v1/candidates/{id}`.
- Unpaywall by DOI. Fetchable link records for open-access PDFs on allowed hosts (each host added to `library-sources.yaml` with its own robots and limits); "known, not held" link records otherwise, SSRN included (LS8).
- The key, if an OpenAlex key exists: `~/.config/devai/library/openalex.key` (LS30), mounted into `devai-library` only. First check whether `api.openalex.org` accepts the key in a request header (one keyed request; it must show the $1/day budget in the response's rate-limit or budget headers). If it does, send it in that header and add no DLP exemption: no pipelock pattern matches an OpenAlex key. If it does not, send it as the `api_key` query parameter that OpenAlex documents, to `api.openalex.org` only: add `api.openalex.org` to the `exempt_domains` of pipelock's `Credential in URL` pattern (deploy/pipelock.yaml), strip `api_key` from every URL the service logs or stores, and record the exception in `docs/library.md`. Without a key, 20.2-20.3 run keyless at $0.10/day, and the acceptance test is sized to that.
- The keyword and vector indexes, library search; the embedding model's store (LS24: `/var/cache/devai/embed`, a plain directory the operator creates (E9; LS24 had its own LV and `make setup-embed`), written out in `scripts/select-models.py`'s storage layout and pinned by `StorageLayoutTest`, mounted read-only into `devai-library` at `/embed`; the service refuses to start search when the pinned model file is absent there), its catalog row in `deploy/embed-models.yaml` and its pull through `scripts/select-models.py`.
- The rebuild command: delete `index/` and rebuild it from the plain files, reusing `cache/embeddings/` (a unit whose key is cached is not re-embedded); `--reembed` also deletes `cache/embeddings/`. Every listing is in a total order: the requested sort key, then `item_id` ascending. A cursor encodes the last row's (sort value, `item_id`), not an offset or a row id, so a page does not depend on insertion order.

**Exit criteria:** candidates from OpenAlex merge with arXiv records by identifier; a non-arXiv open-access PDF is fetched from an allowed host; an SSRN work comes back as a link with its reason; after a rebuild from the plain files the topic listing is byte-identical and every alias resolves as before (aiagent's check 6), tested on a fixture with tied relevance scores, whose items are written in a different order than they are read back, and compared across at least two pages, the second reached through `next_cursor`; a second rebuild with an unchanged library embeds zero units (counted in the rebuild's report); with an OpenAlex key, a keyed request reports the $1/day budget.

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
| Parser exploits in PDFs or HTML. | 20.1 | Parsing only in `devai-library-extract` and GROBID (TEI included; only the extractor calls GROBID): no secrets, no store, internal network only; the service validates their JSON against its schema; a size limit, a time limit per call and memory limits, so a crafted file fails one item instead of stalling it. |
| A PDF larger than pipelock's response cap blocked or truncated on the way in. | 20.1 | Behaviour above the cap observed first; the cap raised above `max_fetch_bytes`; `too_large` decided by the service; a PDF over 5 MiB in the live smoke run. |
| The service used to reach internal hosts (SSRF). | 20.1 | No URLs from aiagent; every redirect hop checked; internal names and private addresses refused; default-deny host admission; no devai-net, so no route to the router, `devai-slurm` or the engines, and pipelock its only way out. |
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

## Phase 22 -- laya 0.3.28

From the operator's E6 (2026-10-06) and the proposal I31. It replaces the trainer's one laya version, so it needs no exception under E2 (D8), and the fine-tuning API toward aiagent does not change (D14). Facts about laya 0.3.28 are cited from its sdist (checked against PyPI's sha256) or from github.com/NandhaKishorM/laya (`gh#N`). Nothing has run under 0.3.28 in devai's trainer image yet (torch 2.14.0, transformers 5.17.0, Python 3.14.7).

### Goal

The laya trainer runs laya 0.3.28 instead of 0.3.20, with upstream's fix for the partial gradient-accumulation window, and nothing that crosses between devai and aiagent changes.

- **The package moves; the checkpoints stay.**
  - laya goes from 0.3.20 (tag commit 23a1752, today's `LAYA_COMMIT`) to 0.3.28 (tag commit a4a8921afe).
  - The base checkpoints stay at convaiinnovations/laya@55cf4c4 (`deploy/laya-models.yaml`). The only newer commit, 7b928d8 (2026-10-03), adds a 160-byte root `config.json` and leaves all 10 pinned files unchanged. laya 0.3.28's own reviewed pin is the same commit (`laya/revisions.py:27-31` in the 0.3.28 sdist).
  - A new revision would make all four stored datasets (ds-4f0ddf475621, ds-57c9c09510f6, ds-a6c9c8248242, ds-e9443d334cb2) fail `check_base` with exit 4 (`laya-trainer/laya_trainer/dataset.py`): each manifest binds the base's name, revision and hashes. `make laya-check` would also need a new reference dataset under a new id.
- **Dataset schema 1 and artifact format 1 stay.**
  - Token sequences are unchanged: 200 of 200 cases identical with the real mmBERT tokenizer (research of 2026-10-06, outside the trainer image; the English ModernBERT tokenizer was not checked).
  - Checkpoint keys and shapes, the ONNX graph's 5 inputs and 2 outputs, and the answer keys are unchanged in the code. Only the manifest's `laya.version` and `laya.commit` change.
  - No student has to be retrained.
- **aiagent goes first.** aiagent refuses every manifest whose laya version is not in `LAYA_COMPAT` (`src/aiagent/system1/artifacts.py`; `{"0.3.20"}` at v0.8.1). It widens the set to both versions and releases; only then does devai bump. 0.3.20 must stay in the set: aiagent re-reads an installed student's manifest on every run, so dropping it would silently send the installed round-1 student back to the LLM.
- **Students of both versions are served.** Phase 23's student service accepts manifests whose laya version is 0.3.20 or 0.3.28 (I28), so artifacts trained before and after this phase are served alike.
- **New training options are measured, not taken.** Soft cross-entropy and option shuffling (both in 0.3.28's `laya.train`) are evaluated on a stored campaign dataset under I31's rule (step 5). Nothing from the evaluation is installed or served.

### Deliverables

```
laya-trainer/requirements.in      modify -- laya==0.3.28
laya-trainer/requirements.lock    modify -- `make laya-trainer-lock`. The recipe has no --upgrade and laya's core dependency
                                            floors did not change, so only laya's own hashes are expected to change; the
                                            diff is reviewed, and any other change is explained in the commit
laya-trainer/laya_trainer/contract.py
                                  modify -- LAYA_VERSION = "0.3.28", LAYA_COMMIT = "a4a8921" (the v0.3.28 tag's commit,
                                            a4a8921afe, in the 7-hex form contract.py records today)
laya-trainer/laya_trainer/tests/test_pipeline.py
                                  modify -- test_manifest_contract asserts {"version": "0.3.28", "commit": "a4a8921"}
laya-trainer/laya_trainer/tests/test_versions.py
                                  new    -- importlib.metadata.version("laya") == LAYA_VERSION. The manifest records the
                                            version twice (`laya.version` from contract.py, `trainer.versions.laya` from
                                            the installed package, package.py), and nothing ties the two together today
laya-trainer/laya_trainer/fixture.py
                                  modify -- build_checkpoint initialises the model itself (step 2)
laya-trainer/laya_trainer/train.py
                                  modify -- the accumulation window's divisor (step 3); `Recipe` (the loss, the shuffled
                                            question types), default today's recipe, passed only by recipe_eval.py
                                            (step 5); the docstring names laya 0.3.28 and gh#941
laya-trainer/laya_trainer/dataset.py
                                  modify -- train items also keep the internal question, the state's token ids and the
                                            option count, so a shuffled epoch can re-encode them (step 5); nothing else
laya-trainer/laya_trainer/tests/test_train.py
                                  new    -- the window sizes and the last window's gradient (step 3); the default Recipe
                                            trains as before; a shuffled item's target follows its options (step 5)
laya-trainer/laya_trainer/recipe_eval.py
                                  new    -- the evaluation's nine runs (step 5): validate, train, export, calibrate, check
                                            parity and score the held-out split through ONNX Runtime, per arm and seed;
                                            sets <eval-id> = <ds-id>-<UTC yyyymmddThhmmssZ> once at start and refuses
                                            an existing one; writes /laya/eval/<eval-id>/<arm>-<seed>/scores.jsonl and
                                            summary.json (with the run's exit_code and message, also for a failed
                                            run), then, in a `finally` block whether the run succeeded or not,
                                            deletes everything else the run wrote there: checkpoint/, tokenizer/,
                                            model.onnx and its external-data file model.onnx.data (1.2 GB in the
                                            stored campaign run ftjob-a40f596f637ba6c331490aea, beside a 1.3 GB
                                            checkpoint/; nine runs would leave about 11 GB on the root cache volume,
                                            where /var/cache/devai/laya lives). At its start it deletes those four
                                            from every directory under /laya/eval/ that has no summary.json (a run
                                            the action's time limit killed). Never writes under runs/
laya-trainer/laya_trainer/tests/test_recipe_eval.py
                                  new    -- one arm and seed end to end on the fixture, on the CPU; afterwards the run's
                                            directory holds exactly scores.jsonl and summary.json; nothing under runs/;
                                            the controller refuses a recipe in `hyperparameters` with 400 (D14); a
                                            forced parity failure leaves only summary.json, with its exit code, in
                                            the run's directory; an existing eval-id is refused
scripts/stats/statlib.py          modify -- ece_15bin(conf, correct, bins=15): the expected calibration error of laya's
                                            `ece_score` (`laya/common.py:763` in the 0.3.28 sdist), which aiagent's
                                            `expected_calibration_error` mirrors (`src/aiagent/distill/gates.py:151-167`),
                                            in stdlib Python: edges as `np.linspace(0, 1, bins + 1)` gives them (i/bins
                                            computed as i x (1/bins), the last edge exactly 1.0); the first bin closed
                                            at 0, the others (lo, hi]; each bin's |mean conf - mean correct| weighted
                                            by its share of rows; 0.0 for no rows, as aiagent (laya returns NaN).
                                            laya_stats.py has no such function: its `ece_15bin` only copies aiagent's
                                            reported value
tests/python/test_stats_statlib.py
                                  modify -- ece_15bin pinned against values laya's `ece_score` and aiagent's function
                                            return on a fixed input, written into the test (numpy is not imported),
                                            including rows at exactly 0, at 1 and on bin edges
scripts/stats/laya_recipe_eval.py new    -- I31's statistics and verdict over the nine score files, with statlib.py;
                                            stdlib only, read-only on its inputs; writes <eval-dir>/report.json and
                                            report.md
tests/python/test_stats_laya_recipe_eval.py
                                  new    -- synthetic score files with a known verdict: adopted; kept on the interval; kept
                                            on the point estimate; kept on the ECE interval's upper bound; both variants
                                            passing; the first-option table; a missing or failed score file gives
                                            `incomplete`, not a verdict
Makefile                          modify -- laya-recipe-eval DATASET=<ds-id> [SEEDS=42,43,44] (step 5). Before Phase 7 its
                                            recipe holds the steps (the GPU check, the runs in a throwaway trainer
                                            container, then the statistics); once Phase 7 has landed it is one call to
                                            scripts/ops/laya-recipe-eval.sh
scripts/ops/laya-recipe-eval.sh   new    -- once Phase 7 has landed: the target's recipe (O5). Inside a job
                                            (DEVAI_JOB_ID set, from Phase 10 on): the runs in a throwaway trainer
                                            container with the GPU device, the job's label (devai.job=$DEVAI_JOB_ID)
                                            and DEVAI_JOB_ID, then the statistics, as scripts/ops/bench-task.sh starts
                                            its container. Outside a job: before Phase 10, C8's card check (no
                                            compute process, memory.used at most 512 MiB), then the same; from
                                            Phase 10 on, it only submits the GPU action through `devai-jobs submit
                                            action laya-recipe-eval --dataset $DATASET [--seeds $SEEDS]` and prints
                                            the job id (C8); inside a job it passes --dataset and --seeds on to
                                            `python -m laya_trainer.recipe_eval`. Its exit trap deletes the model
                                            files of any run directory without summary.json
deploy/operator/actions.yaml      modify -- C8's coverage rule. After Phase 6 and before Phase 7: laya-recipe-eval in
                                            `pending`. From Phase 7 until Phase 10: in `not_offered` (GPU-exclusive,
                                            run with the stack down, step 5). From Phase 10 on: a GPU action
                                            (`kind: job`, `job_kind: action`, `gpu: true`, `time_limit_min: 60`, C8),
                                            script scripts/ops/laya-recipe-eval.sh, `params`: `dataset` (string,
                                            pattern '^ds-[0-9a-f]{12}$', required, to {flag: --dataset}) and `seeds`
                                            (string, pattern '^[0-9]{1,9}(,[0-9]{1,9}){2}$', optional, to {flag:
                                            --seeds}; absent = 42,43,44), `covers: [laya-recipe-eval]`, so
                                            its job holds the engine license (step 5). 60 minutes is about three times
                                            the derived 18-21 minutes (step 5, "Time"): an engineering margin, not an
                                            estimate
scripts/laya-check.py             modify -- REFERENCE's printed figures and date, and the comment naming the lab's aiagent
                                            version, from the first passing run under 0.3.28; the checked values (dataset,
                                            counts, 8172 trained tokens, 12 golden rows) stay. Its teacher port is
                                            Phase 4 step 4e's (DEFAULT_TEACHER_PORT 11437 before 4e, 11435 after,
                                            with the Makefile help's TEACHER_PORT); this phase does not change it
docs/laya-trainer.md              modify -- the re-measured numbers (step 4); the store's eval/ directory; the evaluation
                                            and its result
CLAUDE.md                         modify -- the trainer's test count; the laya store's eval/ directory
```

### Detailed steps

1. **aiagent's side, done in aiagent.** `LAYA_COMPAT` becomes `{"0.3.20", "0.3.28"}` in `src/aiagent/system1/artifacts.py`, and aiagent releases. devai checks that release before it changes anything, without a GPU and without touching a running container:
   - `make fetch-cli`, which takes aiagent's latest release (the cached installer is v0.8.1 today), then rebuild the lab image: `make build-gpu` today, `devai-lab` once Phase 3 has landed;
   - in a throwaway container of the rebuilt lab image, aiagent's own Python (the interpreter its launcher's shebang names, found as `scripts/laya-check.py` finds it) prints `sorted(aiagent.system1.artifacts.LAYA_COMPAT)`, and both versions are in it.

   aiagent may also port 0.3.28's stricter question checks and regenerate its own test fixture under 0.3.28. Neither blocks this phase.
2. **The bump.**
   - `laya==0.3.28` in `requirements.in`, then `make laya-trainer-lock`, and review the lock's diff.
   - `contract.py`, `test_pipeline.py` and the new version test, as listed.
   - **The fixture.** Since 0.3.21, `build_model(..., pretrained=False)` initialises no parameter: the encoder is built under `no_init_weights`, the head on the meta device and then materialised empty (gh#462; `laya/common.py:570-603,690-713` in the 0.3.28 sdist). `fixture.build_checkpoint` calls exactly that and saves the result, so under 0.3.28 it would save uninitialised memory. The mechanism is verified in the code (two same-seed builds matched in 0 of 36 head tensors in the research); that devai's tests then fail is inferred, not run.
   - The fixture therefore builds the model as aiagent's `tools/system1/make_fixture.py` does (`save_checkpoint`, lines 371-395 at v0.8.1): the encoder with its normal initialisation, `DecisionModel(encoder, head_layers, n_act)` without `no_init`, then every weight matrix drawn from N(0, 0.3) under the fixture's seed. aiagent chose 0.3 because at the default 0.02 every answer is uniform and the argmax flaky (`make_fixture.py:59`).
   - A test builds the fixture twice with one seed: `model.safetensors` is byte-identical and every tensor finite.
   - `train.load_model` needs no change: it loads every weight strictly right after building (`train.py`, `load_model`).
3. **The gradient-accumulation fix** (gh#941).
   - Today `train.train` adds `accum = ceil(batch_size / micro_batch_size)` micro-batches into one update, and divides every micro-batch's loss by `accum`. When an epoch's chunk count is not a multiple of `accum`, its last window has fewer micro-batches, and its update gets only that fraction of the gradient (before clipping).
   - With today's defaults (batch 32, micro-batch 8, so `accum` 4; derived from the code, one question per row): campaign round 0's 1,448 training items make 181 chunks, so each epoch's last update carried a quarter of the usual weight; the reference job's 50 items make 7 chunks, so three quarters. Round 2's 1,789 items make 224 chunks and no partial window.
   - Upstream divides by the window's actual size, `min(grad_accum, steps_per_epoch - window_start)` (`laya/train.py:656-658` in 0.3.28). devai does the same with its own names (`chunks`, `accum`). The learning-rate schedule (`total_updates`) does not change.
   - The unit test pins it: for 7 chunks at `accum` 4 the window sizes are 4, 4, 4, 4, 3, 3, 3; and on the fixture model, with a deterministic loss (soft cross-entropy alone, step 5) and a deterministic forward pass (every dropout probability in the fixture model set to 0: each nn.Dropout's p and the head attention's dropout attribute, because train() puts the model in train mode and DecisionModel defaults to dropout 0.1), clipping out of reach and the optimizer's step replaced by a recorder, the gradient of a final one-chunk window equals that chunk's own loss gradient (torch.allclose at float32 tolerance).
   - Every job after this trains slightly differently from before, by design.
4. **Verification.**
   - `make build-laya-trainer` (on `devai-base-gpu` today, on `devai-base` after Phase 3), then `make test-laya-trainer`: CPU, no network, as root and as uid 1000. Today 77 tests pass (docs/laya-trainer.md, 2026-09-24); this phase adds its own.
   - A whole CPU job on the real `laya-multilingual` base, through the controller in the image, as on 2026-09-24: import, validation, training, ONNX export, calibration, parity and golden rows. Its parity (maximum absolute probability difference, same top option n/n) and its golden rows are recorded.
     - This is where 0.3.28's head attention shows up. The head's self-attention is now `_DynamicMultiheadAttention` (gh#559; same parameters, `F.linear` plus SDPA; `laya/common.py:502-552,589-591`), which changes the exported graph's structure. Parity within 1e-3 under torch 2.14 has not been shown.
     - Above 1e-3 the job fails with exit 7, and the bump stops here.
   - `make laya-check` (GPU-exclusive; it evicts the teacher for about 3 minutes), on whichever launch path is live: the router's job runner before Phase 9, the trainer job after it; against the teacher on 11437 before Phase 4 step 4e and on 11435 after it (4e moves laya-check's `DEFAULT_TEACHER_PORT`). It checks the dataset, the counts 50/10/12/8, 8172 trained tokens, parity, and the lab aiagent's verify and golden steps; the last two pass only with step 1 done. 8172 is expected to hold, because it depends only on tokenisation and the epoch count; it has not been measured under 0.3.28.
   - Re-measured and dated in docs/laya-trainer.md:
     - "Functional checks": the rows for the unit tests, the CPU job on the real base, and the reference check;
     - "Deterministic numerical checks": new rows for the CPU job and the first `make laya-check` under 0.3.28 (the 0.3.20 rows stay as history);
     - "Reference check": the measured table (job request with the swap, phase times, peak VRAM, parity, teacher warm-up) and the first-run paragraph, with `scripts/laya-check.py`'s REFERENCE to match.
   - The campaign's "Measurements" and "Quality of the installed student" describe 0.3.20 runs. They stay, labelled so, until a new campaign replaces them. Training is not bit-reproducible against 0.3.20 (other random draws at initialisation, and step 3), so no older number is expected to repeat exactly.
5. **The evaluation of soft cross-entropy and option shuffling (I31).** The rule is I31's, fixed before any run; this step gives it a concrete form.
   - **The switch.** `train.Recipe(loss="rlcd" | "soft-ce", shuffle=())`; `Recipe()` is today's recipe.
     - Soft cross-entropy is the cross-entropy term that `rlcd_loss` already contains, alone (upstream's `soft_ce_loss`, `laya/train.py:472-475`).
     - Shuffling re-encodes each train item of a shuffled type every epoch with a random option order, through 0.3.28's `build_sequence(..., option_order=...)` (`laya/common.py:155-161`), and permutes its target with it (`encode_item` and `draw_option_order`, `laya/train.py:426-449`). Its generator is seeded from the run's seed.
     - Only `choice` questions are shuffled, as in upstream's own example (`laya/train.py:14`). A score answer is the expected level index, so score levels keep their order.
     - Calibration, held-out and golden items are always encoded in the listed order.
     - Only `recipe_eval.py` passes a `Recipe`. The controller, `run_job` and the fine-tuning API never do, and a `hyperparameters` key naming one is refused like any unknown key (400). The API toward aiagent does not change (D14).
   - **The dataset: campaign round 2's** (1,789 / 277 / 334 rows; its id is `dataset.id` in `runs/ftjob-a40f596f637ba6c331490aea/manifest.json`).
     - The three campaign datasets share the largest held-out split in docs/laya-trainer.md's table: the same 334 documents in every round, since the split is by document hash (the reference dataset has 12). So the held-out set is the same whichever is chosen, and round 2's has the most training rows.
     - It has one question per row: aiagent's `polarity` skill, a choice over four options (`src/aiagent/builtin_skills/polarity/skill.py`).
     - Its held-out labels are the teacher's (the top option of the mean of 3 samples), so "correct" means agreement with the teacher, not correctness (docs/laya-trainer.md, "What is estimated").
   - **The runs.** Three arms -- today's recipe (RLCD, options in their order), soft cross-entropy, and option shuffling -- each with step 3's fix, each at seeds 42, 43 and 44 (42 is the trainer's default; the same three seeds in every arm): nine runs, at the trainer's default hyperparameters (4 epochs, batch 32, lean mode).
     - Each run validates the dataset, trains, exports, fits its temperature on the calibration split through ONNX Runtime and checks parity, exactly as a job's phases do (`run_job.run`). It then writes one line per held-out item: the item, its calibrated probabilities, its top option and that option's position in the listed order, and the gold label and its position.
     - No golden rows and no manifest are written. Each run's directory is `/laya/eval/<eval-id>/<arm>-<seed>/`, where `<eval-id>` is `<ds-id>-<UTC yyyymmddThhmmssZ>`, set once by `recipe_eval.py` at start; an existing `<eval-id>` directory is refused. The directory is in the laya store but outside `runs/`, so neither the trainer's API nor the student service (I28) sees it and aiagent cannot install it. Whether the run succeeds or fails (non-zero exit, parity above 1e-3, divergence, an exception), everything in it but `scores.jsonl` and `summary.json` is deleted in a `finally` block: the checkpoint, the tokenizer, `model.onnx` and `model.onnx.data`. `summary.json` always records the run's `exit_code` and message. A run ended by SIGKILL (the action's time limit) cannot clean up: the next `recipe_eval.py` start, and the script's own exit trap, delete those four from every directory under `/laya/eval/` that has no `summary.json`.
   - **The statistics** (`scripts/stats/laya_recipe_eval.py`, with `scripts/stats/statlib.py` and docs/statistics-primer.md's conventions):
     - per item, each arm's share of seeds whose top option is the gold label (0, 1/3, 2/3 or 1);
     - for each variant, the mean per-item difference from today's recipe, with a 97.5% percentile bootstrap interval over items (`statlib.boot_ci` at level 0.975, 10,000 resamples, a fixed seed; primer Sec. 4.5). 97.5% because two variants are compared (Bonferroni). A dataset with several questions per row is resampled by row (primer Sec. 11, cluster bootstrap);
     - each arm's expected calibration error: 15 equal-width bins of the calibrated top probability (laya's `answer_confidence`, max(p)) against agreement with the gold label, `statlib.ece_15bin`: the first bin closed at 0 and the others (lo, hi], the edges of laya's `ece_score` (`laya/common.py:763`, 0.3.28), which aiagent's `expected_calibration_error` mirrors (`src/aiagent/distill/gates.py:151-167`). Each run's ECE is computed on its own, then averaged over the arm's three seeds;
     - for each variant, the ECE difference (variant minus today's, each averaged over seeds), with a 97.5% percentile bootstrap interval over items: each resample draws one set of items and recomputes every run's ECE on it, with the same `statlib.boot_ci` settings as the accuracy interval (by row for a dataset with several questions per row);
     - **the verdict:** a variant is adopted only if its accuracy interval's lower bound is above -0.01, its point estimate is at least 0, and the upper bound of its ECE-difference interval is at most +0.02. Both margins are fixed now, before any run. +0.02 is an engineering margin, not an estimate: it allows a variant that calibrates as well as today's to pass despite noise, and bounds how much worse a passing one can be. A point-estimate comparison ("not above today's") would reject an equally calibrated variant about half the time (derived: two noisy estimates of one value, not measured). Otherwise today's recipe stays. I31 does not say what happens if both variants pass: the report then shows both, and the operator chooses; their combination is not measured;
     - for option shuffling, also: per arm, the student's first-option rate as the mean over items of the share of seeds whose top option is the first listed one, with a 95% percentile bootstrap interval over items (`statlib.boot_ci`, same resampling as the accuracy interval; the three seeds answer the same items, so their answers are not pooled), against the gold label's first-option rate over the 334 items with a Clopper-Pearson interval (gh#131: laya-multilingual never picked the first listed score option, 0 of 290 in English).
     - What it can show (derived from a Wald approximation, not measured): with 334 items, if the arms disagree on 10-20% of them, the interval's half-width is about 0.039-0.055 (`statlib.mde_paired_halfwidth` at z = 2.24). A variant then clears the -0.01 bound only with a point estimate of about +0.03 (10% discordance) to +0.045 (20%) or more; a smaller gain keeps today's recipe, by design.
     - A verdict is given only when all nine runs have succeeded. Otherwise report.md says `incomplete` and lists each run's outcome, and today's recipe stays. No run is repeated on its own or automatically, since keeping a rerun is a choice made after looking (docs/statistics-primer.md). A new evaluation runs all nine under a new `<eval-id>`, and the record under E6 lists every attempt.
   - **Where it runs.** It is GPU-exclusive.
     - Before Phase 10: `make laya-recipe-eval DATASET=<ds-id>`, with the stack down (`make cache-down`). The target first checks that nothing holds the card, with C8's card check (no compute process, `memory.used` at most 512 MiB), then runs `python -m laya_trainer.recipe_eval` in a throwaway trainer container with the GPU device and the laya store, then the statistics on the host.
     - From Phase 10 on: a GPU action, `laya-recipe-eval` (`job_kind: action`, `gpu: true`, `time_limit_min: 60`, C8): its job holds the `engine` license, so the guard and the license keep every other model off the card, and the router ends a serving teacher's engine job for it while it waits (Phase 9, Yielding). Its script runs the same container with the job's label and `DEVAI_JOB_ID`, as `scripts/ops/bench-task.sh` runs the bench task. `make laya-recipe-eval` then only submits the action (C8); no `cache-down` and no card check. If this phase lands before Phase 10, Phase 10 converts the script, the target and the entry (Phase 10 deliverables).
   - **Time** (derived from single runs, not measured): a campaign job held the GPU 123-140 s (docs/laya-trainer.md, "Measurements"; training was 39-47 s of it). Nine runs therefore take about 18-21 minutes, plus the shuffled epochs' re-encoding (not measured), plus, before Phase 10, the teacher's cold start afterwards (122-135 s).
   - **The result** is recorded as a note under E6: the dataset id, the nine runs' parity, each variant's estimate and interval, each arm's ECE and each variant's ECE-difference interval, the first-option table, and the verdict. If a variant is adopted, `Recipe()`'s default changes in a separate commit, and docs/laya-trainer.md says from which job on. Artifact format 1 and aiagent are unaffected either way.

### Exit criteria

- aiagent's release in the rebuilt lab image accepts laya 0.3.20 and 0.3.28 (step 1), before the trainer image changes.
- The trainer image runs laya 0.3.28: the version test passes, and a new artifact's manifest records `{"version": "0.3.28", "commit": "a4a8921"}`.
- The lock's diff changes only laya, or each other change is explained in the commit.
- `make test-laya-trainer` passes on the CPU, as root and as uid 1000, with the fixture and accumulation tests.
- A CPU job on the real base succeeds with parity within 1e-3, and `make laya-check` passes, the lab aiagent's verify and golden steps included.
- docs/laya-trainer.md carries the re-measured numbers, dated, and labels the 0.3.20 ones as such.
- The evaluation has run on the nine runs, and its verdict -- adopted, not adopted, or incomplete -- is recorded under E6.
- The registry test pins that laya-recipe-eval's entry declares `dataset` (required, ds-id pattern), and a submission without it is refused.
- After an evaluation, each run's directory under `/laya/eval/<eval-id>/` holds only `scores.jsonl` and `summary.json`.
- `statlib.ece_15bin` reproduces laya's `ece_score` and aiagent's `expected_calibration_error` on the test's fixed input.
- If Phase 10 has landed: `laya-recipe-eval`, submitted while a teacher engine job serves, starts once the router has ended that job; while it runs, an engine request answers 503 `gpu_held_by_job`; its job carries the `engine` license, `gpu: true` in its comment and a 60-minute limit.
- The coverage tests pass with `laya-recipe-eval` in the registry, as C8's coverage rule places it for the phases that have landed.

### Phase 22 risks

| Risk | Mitigation |
| ---- | ---------- |
| devai moves before aiagent: the aiagent step of `make laya-check` and every campaign install refuse 0.3.28 artifacts | step 1 checks `LAYA_COMPAT` in the rebuilt lab image before anything changes on devai's side |
| the exported ONNX graph changes under torch 2.14 (the head's `_DynamicMultiheadAttention`), and parity within 1e-3 is not yet shown | the trainer refuses an artifact above 1e-3 (exit 7); step 4's CPU job shows it before any GPU window; reverting the bump restores 0.3.20 |
| laya's classifiers name Python 3.10-3.13 (`pyproject.toml:21-24` in the 0.3.28 sdist), and its CI tests those; on 3.14 it is untested upstream | the image's own tests and the CPU job run on 3.14.7 before any GPU run |
| devai calls laya's private helpers (`_load_tokenizer`, `_check_question`, `_to_internal`, `_fix_tokenizer_config`, `_verify_compatibility`); upstream shipped 8 releases in 9 days without a changelog | each sync reviews these five in the upstream diff. For 0.3.28: `_load_tokenizer`, `_fix_tokenizer_config` and `_verify_compatibility` are unchanged (AST comparison); `_check_question` is stricter; `_to_internal` now also passes a question's `option_order` through (sdist diff) |
| the stricter `_check_question` rejects rows 0.3.20 accepted (a blank qid or instructions, null or non-scalar labels, duplicate labels, null score levels, a bad `option_order`; `laya/agent.py:968-1108`): such a dataset fails with exit 3 | rows aiagent generates should not take these forms (inferred from its question builder); a hand-written dataset is fixed at its source, and the message names the row |
| training is not bit-reproducible against 0.3.20, so documented numbers drift | step 4 re-measures and dates them; the 0.3.20 numbers stay, labelled |
| the evaluation adopts a recipe on noise, or rejects one on too little data | I31's rule is fixed before the runs (three seeds per arm, Bonferroni, ECE not worse beyond a fixed +0.02 margin, by interval); step 5 states what 334 items can show |
| an evaluation run is taken for a student | its runs live under `eval/`, write no manifest and keep no model; nothing serves or installs from there |
| an evaluation's model files fill the root cache volume (`/var/cache/devai/laya` is not its own LV) | `recipe_eval.py` deletes everything but the two score files, `model.onnx.data` included; a test and an exit criterion check it |

---

## Phase 23 -- System One

Implements E2-E5 with I27-I30. New work, from the two research rounds of 2026-10-06; no earlier plan.

### Goal

- aiagent can ask devai for System One answers on one router port, `POST /v1/systemone` on 11440, instead of loading every model in its own process (I27):
  - laya students trained by devai answer on the CPU, beside any LLM, with no job and no license (E4, I28);
  - DiffusionGemma's structured reads answer on `vllm-s1`, stock vLLM 0.31.0, taking turns on the GPU under `engine:1` like every engine (E3, I29) -- only if it passes its gate on this card (I30).
- System One reads on the resident teacher are measured, with MTP on and off, so that the operator decides E5 on data.
- Nothing is installed in aiagent or switched on there. Whether and how its gate uses any of this is aiagent's decision (LS1, E4).

### Needs

- **Phase 4:** the engines image (`deploy/Dockerfile.engines`, the `devai-engine` launcher) and 4v's version parameter in `scripts/build-vllm.sh` (I29).
  - Step 1 needs only the `vllm-s1` build and Phase 5's jobs directory (created by the operator, E9), because it writes `/var/cache/devai/jobs/systemone-gate/attempts.jsonl`. The build: 4v's parameter and its `make vllm-lock`, the 0.31.0 record (`deploy/vllm-builds/0.31.0.sh`), the engines image's `vllm-s1` stage, which builds on its own (`--target vllm-s1-run`), and the `devai-engine` launcher (4d). If 4c and 4d have not been done, step 1 writes `Dockerfile.engines` and the launcher for `vllm-s1` alone, and 4c and 4d add the other engines later.
  - Step 5 needs only 4v: it reads from the teacher on the `vllm` backend (on `vllm-devai`, port 11437, until 4e).
- **Phase 8:** `devai-control`, for step 3 (`vllm-s1` as a service) and for the student service as a supervisord program of `devai-laya-trainer`. Step 2 needs only the trainer image.
- **Phase 9:** the router's Slurm launcher, for step 4: the System One port reaches DiffusionGemma through an engine job, and a switch into `vllm-s1` is a rule of that launcher.
- **Phase 7 and Phase 10:** the registry entries. Once Phase 7 has landed, each offered target's recipe is a `scripts/ops/` script with a C8 entry; between Phase 6 and Phase 7 the targets sit in `pending`. Phase 10 brings the probe and action jobs, GPU actions (C8 `gpu: true`) and holds for action jobs (C5). Before Phase 10, steps 1, 3 and 5 run from the operator's shell as Makefile targets: step 1 and step 3's probe with the stack down, step 5 with the stack otherwise idle. If this phase lands before Phase 10, Phase 10 converts what it left in that form: the GPU action, the holds, and the card check dropped (Phase 10 deliverables).
- **aiagent:** a remote client of its own (Dependencies). Nothing in this phase waits for it.

### Deliverables

```
gpu-arbiter/systemone.go              new    -- the System One listener (I27), on DEVAI_SYSTEMONE_PORT only (unset: no
                                                listener, and nothing else changes): POST /v1/systemone dispatched by
                                                `model`; GET /v1/models, the student service's list plus DiffusionGemma
                                                when it has a fitting vllm-s1 cell; C5's /devai/v1/ as on every
                                                listener; any other path 404. A DiffusionGemma read is forwarded to
                                                devai-engines:11442 under vllm-s1's backend state, as every port's proxy
                                                does: it counts in vllm-s1's in-flight requests, so a switch away drains
                                                it first; MAX_CONCURRENT_REQUESTS caps it (429); a non-5xx reply repays
                                                vllm-s1's breaker through the same ModifyResponse hook. In CPU mode
                                                (empty DEVAI_GPU_DEVICE, Phase 11, I14) a DiffusionGemma read answers the
                                                no-GPU error that 11435 returns, before anything is submitted, and
                                                GET /v1/models omits DiffusionGemma
gpu-arbiter/main.go, gpu_record.go, slurm_launch.go
                                      modify -- `vllm-s1` in the backend table only when DEVAI_SYSTEMONE_PORT is
                                                set (unset: the table, probeCachePathByBackend and every listener are
                                                as before, I27). Its entry has ListenPort 0, and runSingleHost's
                                                per-backend listener loop skips an entry with ListenPort 0 (new), so no
                                                proxy mux reaches 11441 (C5); only systemone.go's 11440 listener
                                                serves it, forwarding to devai-engines:11442. Its BackendURL,
                                                devai-engines:11441, is used only for the engine's /health and the
                                                backend state. Its own probe cache
                                                (VLLM_S1_PROBE_CACHE), one launch context (DEVAI_SYSTEMONE_CTX, default
                                                8192); engineOf maps it to vllm; probeCachePathByBackend lists it, so
                                                the drift check covers it; its launches carry `sleep_mode` false (C2)
                                                until step 3 sets it true, so Phase 9 case 4 never sends it USR1
                                                before then. A detached Ensure for 11440:
                                                case 4 (drain, end the holder's job, wait for it, submit) in a
                                                goroutine registered in the coalescing fields, the request answered
                                                503 systemone_loading before the drain; a per-job last-request timer in
                                                place of the last-waiter rule, case 5's outcome kept for the next
                                                request. A switch
                                                into vllm-s1 ends the holder's job with a cancel, never USR1 (its prolog
                                                unloads any sleeper but its own). A System One request never switches away from a
                                                holder that is pending, loading or waking for another launch
gpu-arbiter/pkg/engineargs/           modify -- FromLoad takes backend vllm-s1 with the vLLM builders; -print-load
                                                accepts it
gpu-arbiter/systemone_test.go         new    -- dispatch by name; 503 + Retry-After with no forward while vllm-s1 is
                                                pending, loading or held; a student request submits no job; answers
                                                and error statuses (404, 422, 502) byte-identical to the backend's;
                                                with the port unset, no vllm-s1 table entry, no extra listener, the
                                                drift-check cache map and the argv golden corpus unchanged; with it
                                                set, vllm-s1 has no listener of its own and only 11440 forwards to it
                                                (to 11442, never 11441); a System One request that starts a switch
                                                answered 503 systemone_loading before the drain completes, and a
                                                second one during the switch starting nothing; a chat request that
                                                arrives during the switch's drain or old-job wait served after it,
                                                with the vllm-s1 job it submitted cancelled (case 4);
                                                the pending job kept while requests keep arriving and cancelled after
                                                HEALTH_TIMEOUT_SECONDS without one; a load_failed answered once, 502
                                                with the log tail, on the next request, and an exit-1 ending as 503
                                                engine_job_ended, both charging the breaker; repeated load_failed opens
                                                the breaker, after which a read submits no job (503
                                                launch_breaker_open, with no Retry-After and no claim that another
                                                model or another ctx clears it); a guard refusal cancelled, reported as
                                                gpu_held_by_foreign_process and not charged; a loading teacher job not
                                                cancelled by a System One request (503 gpu_busy); a switch into vllm-s1
                                                cancels a serving teacher whose launch a pending job names; a read in
                                                flight delays a switch away until it ends; 429 beyond
                                                MAX_CONCURRENT_REQUESTS; a 200 from 11442 repays vllm-s1's breaker; in
                                                CPU mode a DiffusionGemma read gets the no-GPU error with no job
                                                submitted, and /v1/models does not list it; any path other than
                                                POST /v1/systemone, GET /v1/models and /devai/v1/ answers 404 without a
                                                forward (C5)
deploy/vllm-builds/0.31.0.sh          new    -- the 0.31.0 record in 4v's format, sourced by scripts/build-vllm.sh
                                                (VLLM_VERSION=0.31.0, I29): sdist sha256, torch 2.13.0, no HyperQwen, no
                                                devai series, the EXTERNALS pins the 0.31.0 tree names (DeepSelect
                                                new); build root and DIST_ID follow the version as 4v made them.
                                                build-vllm.sh itself needs no change
deploy/vllm-builds/0.31.0.in, deploy/vllm-builds/0.31.0.lock
                                      new    -- the runtime set of vLLM 0.31.0's requirements/cuda.txt without its
                                                flashinfer.ai index line and without flashinfer-cubin, hash-locked
                                                from PyPI only (M3) by 4v's `make vllm-lock VLLM_VERSION=0.31.0`;
                                                flashinfer-python 0.7.0.post1 from PyPI. FlashInfer JIT-compiles its
                                                kernels with the image's nvcc, as on 0.28.0, with run-time cubin
                                                downloads disabled (step 1 checks it)
deploy/Dockerfile.engines             modify -- a vllm-s1 stage: its own environment, installed offline from its
                                                wheelhouse; the vendored server, checked against its MANIFEST.sha256;
                                                label devai.engine.vllm-s1 and its line in /etc/devai/engine-builds.json;
                                                copied in before vLLM's stage (it changes less); `--target vllm-s1-run`
                                                gives that environment alone with the launcher, for step 1
scripts/devai-engine                  modify -- vllm-s1: its environment, HOME=/var/lib/devai-engines/vllm-s1 and
                                                FLASHINFER_CUBINS_REPOSITORY=file:///nonexistent/ (no cubin fetched at
                                                run time, so production runs as the gate measured, M3);
                                                `devai-engine vllm-s1 systemone <args>` runs the vendored server with
                                                that environment's Python, CUDA_VISIBLE_DEVICES= and HF_HUB_OFFLINE=1
third_party/vllm-systemone/           new    -- structured_server.py from vLLM v0.31.0, unmodified; LICENSE (vLLM's
                                                Apache-2.0); README (tag, sdist and file sha256, why it is here, how
                                                to update); MANIFEST.sha256
tests/python/test_vllm_systemone_vendor.py
                                      new    -- fails when a vendored file drifts from MANIFEST.sha256, as
                                                test_vendored_scripts_match_their_manifest in
                                                tests/python/test_prepare_checkpoint.py does for third_party/hyperqwen/
devai-tools/internal/control/engine_s1.go
                                      new    -- the vllm-s1 kind (below): one load starts the engine, then the server;
                                                serving only when both answer; one unload stops both; its own check of
                                                the engine; a load on a host without a GPU refused with `no_gpu`, as
                                                engine_hf.go refuses one; C1's supervisor verbs; tests with a fake
                                                engine and a fake server, among them `stop` from user operator on a
                                                serving vllm-s1 stopping both programs (the owning job's hold exits 0)
                                                and `restart` while serving refused with `busy`
deploy/engines/control.yaml           modify -- service vllm-s1 (C7)
deploy/slurm/prolog.sh                modify -- an engine job whose backend is vllm-s1 unloads the sleeper first
                                                (`unload`, `target: sleeper`, `force`), as a probe job's does (Phase 8,
                                                slurm.md Sec. 6), unless the sleeper is that job's own launch on
                                                vllm-s1 (its load wakes it): DiffusionGemma's gate sizes it for the
                                                whole card; a unit test for both cases
deploy/docker-compose.yaml            modify -- router: DEVAI_SYSTEMONE_PORT, DEVAI_SYSTEMONE_CTX, the vllm-s1 probe
                                                cache; engines: the devai-engine-cache-vllm-s1-flashinfer volume
scripts/probe-vllm-s1-reasoning.py    new    -- the vllm prober's spec under its own name, cache and image target; one
                                                cell, at exactly DEVAI_SYSTEMONE_CTX (--ctx-exact)
deploy/.vllm-s1-reasoning-cache.json  new    -- committed as {}: a missing bind source becomes a directory
scripts/_probe_hf_common.py, scripts/model-picker.py
                                      modify -- engine_of and _ENGINE_OF map vllm-s1 to vllm
scripts/select-models.py, scripts/model-sync.py, scripts/probe-check.py
                                      modify -- vllm-s1 as an HF backend on the vLLM store (HF_STORES, the backend
                                                lists, its cache), so `make model-pull NAME=<row>` takes a row that
                                                lists vllm-s1 only
scripts/model-families.yaml           modify -- diffusiongemma: `backends: [vllm-s1]`; its comment brought up to date
scripts/generate-catalog.py           modify -- a family's `backends:` replaces HF_BACKENDS for its rows; tests
deploy/models.yaml                    modify -- `make catalog-regen`: the diffusiongemma row lists vllm-s1 only
deploy/recovery-flags.json            modify -- diffusiongemma: `backends: ["vllm-s1"]` (its `image` field is gone
                                                since 4e, M8); flags re-tuned from step 1; its comments rewritten
deploy/.vllm-reasoning-cache.json     modify -- the 2026-07 diffusiongemma cell, measured on the "gemma" image,
                                                removed (scripts/probe-cache-drop.py, after make backup-create)
scripts/ops/build-vllm-s1.sh, probe-vllm-s1.sh, systemone-gate.sh, systemone-score.sh,
systemone-teacher-reads.sh, systemone-cpu-load.sh
                                      new    -- once Phase 7 has landed: the recipes of this phase's targets (O5);
                                                each Makefile target then calls its script. systemone-gate.sh is the
                                                GPU action's recipe from Phase 10 on; before Phase 7 the recipes stay
                                                in the Makefile
deploy/operator/actions.yaml          modify -- build-vllm-s1 (job_kind action, group build); probe-vllm-s1 (job_kind
                                                action: it only submits) and probe-vllm-s1-model (job_kind probe);
                                                systemone-score (job_kind action; students need no hold);
                                                systemone-teacher-reads and systemone-cpu-load (job_kind action; one
                                                job per run, holding its launches in turn, step 5 and Holds); each
                                                entry sets C8's `comment` so that its run can take each C5 hold:
                                                `backend: vllm` and `model` from the run's parameters for
                                                systemone-teacher-reads, whose three launches share the model, and
                                                `backend: vllm` with `model`, `ctx` and `mtp` from the run's
                                                parameters for systemone-cpu-load; systemone-gate in not_offered
                                                until Phase 10 (run with the stack down), then a GPU action
                                                (`job_kind: action`, `gpu: true`, `time_limit_min: 240`, an
                                                engineering margin, not measured; param `mode`, `shakedown` |
                                                `counted`, and the optional `after_limit` (step 1); C8),
                                                converted by Phase 10 if this phase lands first.
                                                Each offered entry has its scripts/ops/ script and `covers`; between
                                                Phase 6 and Phase 7 the targets sit in `pending`
scripts/bench-sync.py, scripts/bench/bench_runner.py, scripts/model-picker.py
                                      modify -- vllm-s1 skipped: it is not a chat backend (I29); tests pin that
                                                bench-plan and the picker name no vllm-s1 row
laya-trainer/laya_trainer/systemone.py
                                      new    -- the student service (below); stdlib http.server, like controller.py
laya-trainer/laya_trainer/tests/test_systemone.py
                                      new    -- CPU only: golden equality; lazy load and least-recently-used eviction;
                                                404 for an unknown name; nothing written into a run directory;
                                                manifests of laya 0.3.20 and 0.3.28 accepted, others refused; the same
                                                `input_ids` for a state sent as a dict and as its `serialize_state`
                                                string; the image's laya `serialize_state` gives the pinned fixture
                                                of tests/python/test_systemone_*.py (below), so a laya bump that
                                                changes it fails `make test-laya-trainer`
deploy/laya-trainer/supervisord.conf, deploy/laya-trainer/control.yaml
                                      modify -- program systemone; control.yaml services trainer, controller and
                                                systemone (kind program)
deploy/laya-trainer/systemone-run.sh  new    -- the program's command: pins it with `taskset -c` to the CPUs that
                                                /sys/devices/cpu_atom/cpus lists, unpinned when there are none;
                                                CUDA_VISIBLE_DEVICES=
scripts/systemone/gate.py             new    -- step 1, the DiffusionGemma gate (I30): its container, the shakedown
                                                and counted launches (`--mode shakedown|counted`), the report; inside
                                                a job (DEVAI_JOB_ID set, from Phase 10 on) its container carries
                                                devai.job=$DEVAI_JOB_ID and DEVAI_JOB_ID (C8), which Phase 14's
                                                GPU-device test reads here as well as in systemone-gate.sh; reads
                                                deploy/systemone-gate-flags.json and appends to
                                                /var/cache/devai/jobs/systemone-gate/attempts.jsonl (step 1); refuses
                                                to start when /var/cache/devai/jobs does not exist
                                                (E9); builds each launch's argv as production will (step 1)
deploy/systemone-gate-flags.json      new    -- the gate's flag set (step 1), seeded from the launch list; tests pin
                                                that gate.py refuses a file without `--kv-cache-dtype`, refuses
                                                `counted` on a sha256 with a pass or deferral line, refuses it after
                                                a "limit reached" line without `--after-limit`, and accepts it with
                                                `--after-limit` naming that line; and that it refuses to start when
                                                /var/cache/devai/jobs does not exist
scripts/systemone/score.py            new    -- one scorer for any /v1/systemone backend against a stored dataset's
                                                held-out gold labels, and `--golden` (steps 2 and 5)
scripts/systemone/teacher_reads.py    new    -- step 5 (E5): a /v1/systemone-shaped reader over the teacher's chat
                                                log-probabilities, and the MTP checks
scripts/systemone/cpu_load.py         new    -- step 2: the teacher's decode rate and TTFT with and without student load
tests/python/test_systemone_*.py      new    -- the four scripts against fake servers; their statistics against
                                                scripts/stats/statlib.py; score.py's state string for a non-ASCII
                                                held-out row equals a pinned fixture: `json.dumps(state,
                                                ensure_ascii=False)` for a dict or list, the string unchanged for a
                                                str (the body of laya 0.3.20's and 0.3.28's `serialize_state`,
                                                `laya/common.py`; tests/python is stdlib-only and has no laya, so
                                                laya_trainer/tests/test_systemone.py checks the same fixture
                                                against the image's laya);
                                                teacher_reads.py and cpu_load.py take and release their hold against a
                                                fake router, and wait on its 503 or 409; teacher_reads.py takes its
                                                three holds in order, one at a time, and refuses a report in which two
                                                launches share an engine job; gate.py's report for a counted run the
                                                time limit ends says "limit reached"
Makefile                              modify -- build-vllm-s1, probe-vllm-s1, systemone-gate, systemone-score,
                                                systemone-teacher-reads, systemone-cpu-load. systemone-score,
                                                systemone-teacher-reads and systemone-cpu-load run their script in a
                                                throwaway devai-lab container on $(DEVAI_NETWORK) with scripts/ mounted
                                                read-only and the laya store's runs/ and datasets/ read-only at /laya,
                                                as bench-concurrency does. Before Phase 10, systemone-gate
                                                (MODE=shakedown|counted, AFTER_LIMIT=) runs C8's card check and then
                                                gate.py on the host, as the probers run there
docs/router.md                        modify -- the System One port
docs/systemone.md                     new    -- listed now, written in step 4: the port, the dispatch, the answer shapes
                                                per backend, errors and Retry-After (no retry on
                                                launch_breaker_open), and what aiagent's client must
                                                adapt (choice criteria as a dict; the state as laya's
                                                `serialize_state(state)` string, which is `json.dumps(state,
                                                ensure_ascii=False)` for a dict, never its `text` field alone; no
                                                answer_confidence from DiffusionGemma; a 503 handled per call; reads
                                                batched before escalations)
```

### The vendored server, and what devai does about it

The server is `examples/features/structured_diffusion/structured_server.py` at vLLM's v0.31.0 tag (I29). It is not part of the vLLM wheel (`pyproject.toml` packages only `vllm*`), so devai vendors it unmodified, as it vendors HyperQwen's scripts. Its imports beyond the standard library are `transformers` and `pybase64`, both vLLM dependencies already (research round 2). In the 0.31.0 sdist (sha256 `dde46f1efbe846e5bd0c5c4ec6ec71842c42254c7ff3d9bebb2a57028715d380`, equal to PyPI's digest) the file is 1,544 lines, sha256 `9d085496ccaeec833ddf4ffe4b89212512121a6790f5844891b389cbeb02d36e`. Read there on 2026-10-06 (`ss` = that file):

| Its behaviour | Source | What devai does |
|---|---|---|
| a dict `state` is written with `json.dumps` and its default `ensure_ascii=True`, so Croatian, German or Polish text reaches the model as `\u` escapes | `ss:1157-1166` | clients send `state` as the string laya's `serialize_state(state)` gives, `json.dumps(state, ensure_ascii=False)` for a dict (docs/systemone.md), and devai's scorer does; the router does not rewrite the body (I27). The accuracy effect of escapes is unmeasured |
| the request's `model` is ignored; the reply carries the server's `--model` (default `dgemma`) | `ss:1350`, `:1500` | `devai-control` starts it with `--model <catalog name>`, so the reply names the model that answered (I27). `--model` is also the model in every upstream request (`ss:478`, `:504`, `:546`, `:601`), so it must equal the engine's `--served-model-name`, which `pkg/engineargs` sets to the catalog name; step 1's launch passes both |
| an upstream HTTP error becomes 502, any other failure 500, a connection error included | `ss:1304-1320` | the router forwards only to a serving engine and otherwise answers 503 + `Retry-After` itself (I27); statuses that do come back pass through unchanged |
| it waits up to 600 s on the upstream | `ss:435-450` | clients set their own, shorter timeout (docs/systemone.md); a dead engine shows as `failed` through `devai-control`'s own check |
| `GET /health` always answers ok | `ss:1216-1219` | never taken as evidence: `devai-control` checks the engine on its own port |
| Gemma constants are hard-coded (`VOCAB = 262144`, `TURN_CLOSE = 106`, the thought scaffold) | `ss:105-116` | only DiffusionGemma's name is dispatched to it |
| constrained reads are on unless `--no-constrained` is given | `ss:1509-1514`; its README:15-16 | left on. They also cut the unconstrained [rows, 262144] fp32 logits to at most 128 columns (research round 2, `diffusion_gemma.py` at v0.31.0) |
| it also serves `/v1/chat/completions` and `/v1/raw/chat/completions` | `ss:1263-1281` | not reachable through the router: 11440 forwards only `POST /v1/systemone` |
| the environment variable `API_KEY`, when set, guards the POST routes | `ss:100-102` | left unset: the server is on `devai-net` only |

### `vllm-s1` under devai-control (I29)

- **One `load`** with `backend: vllm-s1`:
  1. starts the engine program as `engine_hf.go` does for vLLM (the argv from `pkg/engineargs`, the recovery files, port 11441) and waits for its `/health` (fail fast if it exits; 600 s);
  2. then starts the server program: `devai-engine vllm-s1 systemone --upstream http://127.0.0.1:11441 --model <catalog name> --tokenizer <store>/<model> --canvas <canvas> --host 0.0.0.0 --port 11442`;
  3. then sends one read, a single `noul` question, and requires a 200 with a value between 0 and 1.
  - Only then is the service `serving`. A failure in any step stops both programs and replies `load_failed` with both log tails.
  - The canvas comes from `control.yaml` (`canvas: 64`). A launch whose `--diffusion-config` names another `canvas_length` is refused with `bad_request`: the server and the engine must agree.
  - **No sleeper beside it.** A `vllm-s1` engine job starts on an empty card: its prolog unloads any sleeper first, as a probe job's does (slurm.md Sec. 6), unless the sleeper is this job's own launch on `vllm-s1`, which its `load` wakes (possible only once step 3 has set `sleep_mode` true), since the gate sizes DiffusionGemma for the whole card, and the router ends the job it replaces with a cancel, never USR1 (below). So Phase 8's handling of a sleeper's VRAM never applies to a `vllm-s1` load.
  - **Without a GPU** a load is refused with `no_gpu`, as `engine_hf.go` refuses one. The router never submits one there (below); this covers a job submitted by hand.
- **One `unload`** stops the server, then the engine, and replies as C1 says (the program gone, NVML back to its level before the load).
- **Its own check.** On every heartbeat (30 s) `devai-control` reads the engine's `/health` on 11441 and checks that the server process is alive. Either one gone means `failed`, with the log tails. The server's own `/health` is never used.
- **Logs:** `jobs/engines/vllm-s1/<job>.log` for the engine and `<job>.systemone.log` for the server (C4's `engines/<backend>/`).
- **`sleep`** answers `unsupported` while `sleep_mode` is false (C1, I29). Step 3 measures sleep before that can change.
- **Supervisor verbs** are C1's for an engine other than Ollama: `stop` is a forced `unload` (the server, then the engine); `restart` and `reload` are refused with `busy` while the model is loaded or loading.
- **The guard.** The server runs with `CUDA_VISIBLE_DEVICES` empty and holds no GPU memory, so the guard sees one holder, the engine, in `devai-engines`: a devai engine service like the others (slurm.md Sec. 6).

**Engine-keyed code.** `vllm-s1`'s engine is vLLM: the router's `engineOf`, the probers' `engine_of` and the picker's `_ENGINE_OF` map it so, and request shapes, parser plugins and the memory heuristic follow vLLM's. The recovery files' `appliesTo` (router) and `_entry_applies_to_backend` (probers, `scripts/_probe_hf_common.py`) match a backend by name or by engine. So an entry scoped to `"vllm"` also applies to a `vllm-s1` launch. That is harmless: entries are per model, only the diffusiongemma row lists `vllm-s1`, and its curated entry names `["vllm-s1"]`. The reverse does not hold: an entry scoped to `"vllm-s1"` never applies to `vllm`.

### The student service (I28)

- **Process:** `python -m laya_trainer.systemone`, a supervisord program of `devai-laya-trainer` under its `devai-control` (kind `program`), on 11443 on `devai-net`. Standard library `http.server`, threaded, as `controller.py` is.
- **Routes:**
  - `POST /v1/systemone` `{model, state, questions}`. `model` is a run's `fine_tuned_model` (`ft:<base>:devai:<suffix>:<12 hex>`, docs/laya-trainer.md). Questions are laya's question objects; choice criteria may be a dict or a list, as laya accepts. Each answer is laya's answer without `action`: `type`, `choice` / `score` / `noul`, `probabilities`, `confidence`, `answer_confidence`. The reply carries `model` (the run's name) and `usage {input_tokens, output_tokens: 0}`.
  - `GET /v1/models`: the succeeded runs whose manifest is format 1 with laya 0.3.20 or 0.3.28.
  - `GET /health`.
- **Errors:** 404 for a name it does not serve; 422 for an invalid question, or for a state that does not fit the artifact's `max_len` (the dataset contract never truncates a state, so a 200 always means the whole state was read); 409 `artifact_rejected`, naming the failed check; 400 for a body that is not JSON.
- **Loading,** on first use of a name:
  - every file in `manifest.files` matches its sha256; `format_version` is 1; the manifest's laya version is 0.3.20 or 0.3.28;
  - every row of `golden.jsonl` is reproduced: `input_ids` exactly, each answer within 1e-3 (`golden.py`'s `TOLERANCE`, aiagent's own check);
  - a failed check answers 409 and is not tried again until the manifest file changes.
- **Inference.** The image's own laya encodes and decodes: `build_sequence` and the answer helpers (`answer_confidence`, `confidence_from_probs`), the code the golden rows came from. Those are unchanged from 0.3.20 to 0.3.28 (the laya research: 200 of 200 token sequences identical with the real tokenizer, the helpers identical as code). The artifact's applied temperatures come from its `rl_agent_config.json`. The graph runs on onnxruntime 1.30.0 (`laya-trainer/requirements.lock`) with `providers=["CPUExecutionProvider"]` passed explicitly.
  - laya's own `ONNXAgent` and laya-serve are not used: `ONNXAgent` prefers the CUDA provider when one is present, and laya-serve routes an unknown name to a stock checkpoint fetched from the Hub (laya research, `L28 onnx_agent.py:165-170`, `router.py:902-926`).
- **Residency:** at most `SYSTEMONE_MAX_RESIDENT` (default 2) artifacts are loaded; the least recently used goes first.
- **Read only:** it opens run files read-only and writes no cache; a test runs it over a read-only copy of a run and compares the tree's hashes before and after.
- **CPU only:** `CUDA_VISIBLE_DEVICES` empty and the CPU provider. `deploy/laya-trainer/systemone-run.sh` pins it with `taskset -c` to the CPUs that `/sys/devices/cpu_atom/cpus` lists: 8-23 on this host (read 2026-10-06; `cpu_core` lists 0-7). Where the file is absent or empty it is not pinned. `taskset` comes from Debian's util-linux; that the trainer image carries it is unchecked, and the image build fails without it.

### The router's System One port (I27)

- **Off unless `DEVAI_SYSTEMONE_PORT` is set** (11440 in compose). Unset, the router opens no listener and is byte-identical to before.
- **`POST /v1/systemone`**, dispatched by the body's `model`:
  - **DiffusionGemma's catalog name** -- a row that lists `vllm-s1` and has a fitting `vllm-s1` cell -- goes to `vllm-s1`, launched at `DEVAI_SYSTEMONE_CTX`:
    - serving with that launch: the request is forwarded to `devai-engines:11442`. Reads on 11440 are `vllm-s1`'s requests for the drain, the concurrency cap and the breaker credit, exactly as requests on 11435 are `vllm`'s;
    - otherwise the router starts the switch (Phase 9's Ensure: drain, end the holder's job, submit an engine job for `vllm-s1`) and answers at once 503 with the reason (`Retry-After: 30` on every reason except `launch_breaker_open`, which carries none) (`systemone_loading`, `gpu_held_by_job`, `gpu_held_by_hold`, `gpu_queue`, `gpu_busy`, `engine_job_ended`, `gpu_held_by_foreign_process` or `launch_breaker_open`). It holds no request open: the server would wait up to 600 s and turn a missing upstream into a 500;
    - the switch is detached: the handler sets the coalescing fields (`recreating`, `pendingModel` = the `vllm-s1` launch) under `a.mu`, starts a goroutine that runs case 4 up to its submit (drain, end the holder's job, wait for it to end, submit), and answers 503 `systemone_loading` before the drain begins; the goroutine clears the fields once the job is submitted and does not wait on the load. A System One request that finds these fields naming `vllm-s1` starts nothing and answers 503 `systemone_loading`. A request on another port waits for that goroutine as it waits for any switch, then finds the `vllm-s1` job pending or loading and cancels it by case 4 (below), so chat wins, at the cost of one job submitted and cancelled. Since no request waits on the load, Phase 9's last-waiter rule would cancel the job at once, so it does not apply here. Instead the record keeps, per pending or loading `vllm-s1` job, the time of the last System One request that named it, and cancels the job once none has come for `HEALTH_TIMEOUT_SECONDS`;
    - the router watches that job itself, from the bus and Slurm, and classifies its ending as Phase 9 case 5 does: the submit spent one breaker unit (case 4), which a 200 from 11442 repays; after `load_failed`, an exit-1 `engine_job_ended`, or a serving engine that fails before any 200, the unit stays spent; a guard refusal or a held job is cancelled at once and its unit refunded. The ending is kept, and the next DiffusionGemma request answers with it once (502 with the log tail; 503 `engine_job_ended`; 503 `gpu_held_by_foreign_process` naming the holders) before a new switch starts. While the breaker is open for that launch, a read starts nothing and answers 503 `launch_breaker_open` with no `Retry-After`. `vllm-s1` serves one model at one context, so neither of the breaker's own ways out (a served response, a launch under another key) can happen: it stays open until the router restarts. The refusal says so, names the failed launches' log tails and the remedy (restart `devai-router`, from the stack page or the host shell), and does not use `refusalMessage`'s "request a different model" or "smaller `@<ctx>`" text. docs/systemone.md tells clients not to retry on `launch_breaker_open`;
    - a System One request never switches away from a holder that is pending, loading or waking for another launch: it answers 503 `gpu_busy`, `Retry-After: 30`. Without this, a reader retrying every 30 s and a waiting chat client would cancel each other's loading jobs, and neither engine would ever serve. The rule is one-way: a request on any other port still goes through Phase 9 case 4 and cancels a pending, loading or waking `vllm-s1` job, so chat wins. DiffusionGemma serves only in a gap in other GPU traffic longer than its load plus the reader's retry interval (about 90-115 s + 30 s on the old image), and each System One read that switches away from a serving teacher costs the next chat request a teacher cold start. A serving holder is switched away from as Phase 9 says;
    - a switch into `vllm-s1` ends the holder's job with a cancel, never USR1: its prolog would unload the sleeper anyway (above).
  - **Without a GPU** (empty `DEVAI_GPU_DEVICE`, Phase 11): a DiffusionGemma read answers the router's no-GPU error at once, as every vLLM request does there, with no job and no 503. Student reads are unaffected.
  - **Any other name** goes to `devai-laya-trainer:11443`: no job, no license. A 404 for a name the student service does not serve passes through.
- **Unchanged:** answers, bodies and statuses pass through as the backend sent them. The router maps no backend's `confidence` onto another field and adds no `answer_confidence` (I27).
- **`GET /v1/models`:** the student service's list, plus DiffusionGemma when it has a fitting `vllm-s1` cell and the host has a GPU.
- **Otherwise a switch to or from `vllm-s1`** is an ordinary engine switch: guard refusals, holds, and yielding to any pending, unheld job other than an engine job that requests the `engine` license (a trainer, a probe or a GPU action, C8) work as Phase 9 says. A hold on 11440 names DiffusionGemma, and its job's comment names `vllm-s1` (C5).

### Detailed steps

**Holds.** From Phase 10 on, a measurement that reads from a launch the router serves holds that launch (C5), as `bench_task.py` does. `teacher_reads.py` (step 5) and `cpu_load.py` (step 2) take the hold on 11435 for `<name>::mtp@<ctx>` or `<name>@<ctx>` before their first request and release it at their end; a 503 or 409 on the hold means waiting (`Retry-After`, else 30 s) and asking again. Each sends `X-DevAI-Hold-Key: $DEVAI_HOLD_KEY` (C5, C8) and renews its hold every 30 s while it reads (C5), sharing bench_task.py's hold helper; a renewal that answers 409 or 503 ends the run as `cancelled`, and its report is not written. A hold names one launch, and one job takes its holds one at a time: `teacher_reads.py` runs step 5's three launches (MTP off, MTP on, MTP off) in one job of `systemone-teacher-reads`, in that order, releasing each hold before it takes the next. The three launches are on one backend, so each new hold makes the router cancel the previous launch's engine job and start a new one (Phase 9 case 4 sends USR1 only for a switch to another backend), and every launch is a new engine process; the report records each launch's engine job from the GPU record (C5), and two launches with the same job void the run. It writes one report covering all three launches. `cpu_load.py` likewise holds its one launch in one job of `systemone-cpu-load`. Each entry's `comment` (C8) names backend `vllm` and the teacher's model; `cpu_load.py`'s also names the launch's `ctx` and `mtp`, while `teacher_reads.py`'s leaves them out, so that it covers all three launches (C5). The student service needs no hold. Before Phase 10 these steps run with the stack otherwise idle, and their reports say so.

1. **The DiffusionGemma gate (I30)**, with the card to itself, before any router work.
   - **Build:** `make build-vllm-s1`. What 0.31.0 changes in the build, read in its sdist on 2026-10-06:
     - FlashInfer 0.7.0.post1 (`requirements/cuda.txt:17-18`). `flashinfer-python` is on PyPI and goes into `deploy/vllm-builds/0.31.0.lock` (4v's `make vllm-lock VLLM_VERSION=0.31.0`) like everything else. `flashinfer-cubin` is not on PyPI since 0.6.14 (the file names flashinfer.ai's index for it, `:14-16`). It is not installed (M3), as 0.28.0's build does not install it: that build's runtime set, resolved from HyperQwen's `vllm==0.28.0` line, holds `flashinfer-python` only. FlashInfer then JIT-compiles its kernels with the image's nvcc (measured on 0.28.0, 2026-09-21; with run-time cubin downloads disabled, see the Measure step), which the gate's first launch confirms on 0.31.0 (Measure, below);
     - kernel repositories that CMake clones during configure, without retries, at new pins: cutlass v4.7.1 (`CMakeLists.txt:487`) and a new repository, DeepSelect (`cmake/external_projects/deepselect.cmake:21-22`, commit `d96d33afe1fab0d6066da49cdc91e64c2bee65ea`), included for every CUDA build (`CMakeLists.txt:1559`); flash-attention, FlashMLA, MSA, tml-fa4, FlashKDA, qutlass, DeepGEMM (now a vllm-project fork with a `deep_jit` submodule, research round 1) and triton at the revisions their cmake files name. All are fetched first, three tries each, and handed over through their `*_SRC_DIR` variables (DeepSelect reads `DEEPSELECT_SRC_DIR`, `deepselect.cmake:7-8`), as `build-vllm.sh` does today;
     - unchanged: torch 2.13.0 and Rust 1.95 (`rust-toolchain.toml:2`).
   - **Model:** add `backends: [vllm-s1]` to the diffusiongemma family, run `make catalog-regen`, then `make model-pull NAME=diffusiongemma-26B-A4B-it-NVFP4` (the row in `deploy/models.yaml`; 17.53 GB into the vLLM store). The `select-models.py` deliverable lands first.
   - **Where it runs:**
     - before Phase 10: `make systemone-gate`, with the stack down. The target first checks that nothing holds the card, with C8's card check (no compute process, `memory.used` at most 512 MiB);
     - from Phase 10 on: the GPU action `systemone-gate` (`job_kind: action`, `gpu: true`, `time_limit_min: 240`, C8; an engineering margin, not measured: read latency on `vllm-s1` is not yet known, and the 10 counted launches alone take about 17 min of startup at the old image's 90-115 s each; the report records each job's wall time). Its parameter `mode` (`shakedown` | `counted`; before Phase 10 the target's `MODE=`, gate.py's `--mode`) selects the run. The flags come from one tracked file, `deploy/systemone-gate-flags.json` (the engine's arguments after `--model`, and its environment), which gate.py reads at start and copies, with its sha256, into the report. The file must name `--kv-cache-dtype` explicitly; gate.py refuses one that does not. gate.py builds each launch's argv as production will build it for backend `vllm-s1` at `DEVAI_SYSTEMONE_CTX`: the vllm-s1 prober's `vllm_command_args`, which mirrors the router's builder (I1; from Phase 8 the same argv as `gpu-arbiter -print-load` through `pkg/engineargs`), so `--trust-remote-code`, `--max-num-seqs` = `MAX_CONCURRENT_REQUESTS`, the computed utilisation and the family's curated parsers (`--reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4`), and then the file's flags and environment last, as the recovery entry will be appended (the last flag wins). The report records the full argv and environment of every launch. A shakedown job runs launches on that file and counts nothing; the operator edits and commits it between shakedowns. A counted job runs both soaks and the 10 launches on it. Every counted run appends one line (the flags' sha256, the flags, the job id or `host`, start, outcome: pass, deferral or limit reached) to `/var/cache/devai/jobs/systemone-gate/attempts.jsonl`; gate.py lists every line in its report and refuses `counted` when the current flags' sha256 already has a pass or deferral line. When the sha256's only lines are "limit reached", it refuses `counted` unless it is given `--after-limit <job id or host>` naming the latest such line (the target's `AFTER_LIMIT=`, the action's parameter `after_limit`); the new line records that id, and the report lists both attempts. gate.py refuses to start when `/var/cache/devai/jobs` does not exist (a plain directory on the cache volume, E9); a test pins the refusal. Its job holds the `engine` license, so the guard and the license keep every other model off the card, and its container carries the job's `devai.job` label and `DEVAI_JOB_ID`, as Phase 22's `laya-recipe-eval` does. No target runs the card check then, and Phase 10 converts the pre-Phase-10 form if this phase lands first;
     - either way, `gate.py` starts the container from `--target vllm-s1-run` under the fixed name `devai-systemone-gate`, on `devai-net`, with 11441 and 11442 published on 127.0.0.1 only, and removes it on every exit path. It reaches it at 127.0.0.1 from the host and by name from `devai-operator`.
   - **Launch**, with the engine and the server started as `devai-control` will start them:
     - `--served-model-name diffusiongemma-26B-A4B-it-NVFP4`, the name the server is given with `--model` (the table above);
     - the flags the research names: `--limit-mm-per-prompt '{"image":0,"video":0}'` (no vision tower, 1.07 GiB less), `--max-model-len 8192`, `--max-num-seqs` 4 or less, `--diffusion-config '{"canvas_length": 64}'`, `--max-logprobs 32`, `--enable-prefix-caching`; constrained reads (the server's default);
     - from the June and July runs on the old image (`deploy/recovery-flags.json`): `--quantization modelopt`, `--max-num-batched-tokens 512` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, as starting points;
     - a utilisation sized so that the KV pool holds 4 x 8192 tokens with room to spare, not the old 0.90, so that the VRAM rule tests the transients and not the budget. The research estimates the whole process at 17.9-20.7 GiB for 4 sequences (estimated, not measured);
     - if the FP4 MoE path fails on sm120, `VLLM_USE_FLASHINFER_MOE_FP4=0` is tried, as reported for an RTX 5090 (a comment on vllm#57250; not reproduced).
   - **Measure:**
     - FlashInfer without `flashinfer-cubin`: the first launch starts with an empty FlashInfer cache volume and with `FLASHINFER_CUBINS_REPOSITORY=file:///nonexistent/` (the launcher sets it, as in production), so no cubin can be fetched at run time. Without that, vLLM 0.31.0 checks NVIDIA's artifactory at startup and, when it is reachable, may use downloaded TRTLLM XQA decode kernels on SM12x (`vllm/utils/flashinfer.py:77-80, 564-614`; `setup.py:1342-1346`; read 2026-10-06), and devai-net reaches the internet. Its first read and its first chat request must answer 200, and the time to them is recorded. A kernel it cannot JIT-compile fails here: a deferral;
     - VRAM with the probes' 0.1 s sampler, from the container's start to the end of the soak;
     - startup time: container start to the engine's `/health`, and to the first valid read;
     - the soak, run twice, at 1 client and at 4 clients: 2,000 System One reads mixed with 200 chat requests (one after every 10 reads, straight to the engine's `/v1/chat/completions`, the bench's latency prompts, `max_tokens` 256). The reads are built from the reference dataset `ds-a6c9c8248242` (criteria as a dict, the state as laya's `serialize_state(state)` string, as docs/systemone.md says, `samples` 1). That dataset's states and the latency prompts are a few dozen tokens each, so one read in 10 and one chat in 10 are made long: the state (for a chat, the prompt) is padded with text from the load probe's corpus (`scripts/_probe_load.py`) up to 8192 tokens less the canvas (64) for a read or `max_tokens` for a chat, less the request's own tokens, and begins with a salt unique to the request so that prefix caching cannot serve it (as `bench_longctx.py` does). Each response's prompt token count is recorded (`usage.input_tokens` on a read, `usage.prompt_tokens` on a chat). The report gives the largest count served, how many requests were long, and the prefix-cache hit share from the engine's `/metrics` counters, read before and after each soak. A soak in which no request reached at least 7,000 prompt tokens is void, not a pass;
     - 10 separate launches on the same flags, each: container start, `/health`, 50 reads, 5 chat requests (one after every 10 reads), of which 5 reads and 1 chat are long, as in the soak, stop. Startup time is recorded for each;
     - per-read latency, p50 and p95 with their intervals (docs/statistics-primer.md).
   - **Pass rule:** I30's three conditions, made concrete:
     - peak `memory.used` at most 23443 MiB, 1 GiB below the card's 24467 MiB (`nvidia-smi --query-gpu=memory.total`);
     - after both soaks, the engine process has its first pid and its `/health` answers, and each of the 10 launches stays up with its first pid through its last request;
     - every read is a 200 in which each question is answered (no `ask_if` is used), choice and score probabilities sum to 1 within 1e-3, and each `noul` value lies between 0 and 1.
   - **Counting:** the flags in the launch list are tried in uncounted shakedown launches. The counted runs (both soaks and the 10 launches) run once, on the final flag set. A failed counted run is a deferral under the flags it ran with. A counted run that the job's time limit ends (TERM, then `stop`; from Phase 10 on) is neither a pass nor a deferral: `gate.py`'s report records it as "limit reached", with what was measured up to then, and it goes to the operator; it is not run again automatically. The operator may run it again on the same flags, with a raised `time_limit_min` if needed, by passing `--after-limit` (step 1, Where it runs). A new flag set (a new sha256 of `deploy/systemone-gate-flags.json`) restarts all counted runs, and `gate.py`'s report lists every counted attempt in `attempts.jsonl` with its flags.
   - **Result:** `scripts/systemone/gate.py` writes the report (JSON plus a summary). The operator records it under E3: a pass, with the flags that passed, or the deferral.
     - On a deferral: `vllm-s1`'s build record, stage, service, prober, recovery entry and catalog change are not merged; the family change is reverted. Its 2026-07 cells are removed from the vLLM and SGLang probe caches (`scripts/probe-cache-drop.py`, after `make backup-create`), and it gets a `manual` ledger verdict on `vllm` and on `sglang` (`scripts/_model_status.py --exclude diffusiongemma-26B-A4B-it-NVFP4::<backend>`, reason "E3 deferral: gate failed", naming the report), so no prober, `make model-sync` or the picker takes it up on those backends. Its weights stay in the vLLM store for a later gate, unless the operator removes them, which the report asks. The phase goes on with steps 2 and 5, and step 4 dispatches to the student service alone.
2. **The student service (I28),** on the CPU. It needs only the trainer image: before Phase 8 it runs in a throwaway container from that image, named `devai-systemone-check`, on `devai-net`, with the laya store read-only; the scripts reach it by name.
   - **Golden equality:** for every succeeded run in `/laya/runs`, the service reproduces `golden.jsonl` (`make systemone-score GOLDEN=1`: `score.py --golden` in a lab container on `devai-net`, reading `/laya/runs` read-only).
   - **Latency:** p50 and p95, with intervals, at 1 and 4 clients, for the student aiagent installed from the 2026-09-25 campaign (`ft:laya-multilingual:devai:polarity-classify:d0e411c9b85d`, round 1, run `ftjob-d0e411c9b85d1168ff7ea1fa`), over the 334 held-out rows of its dataset `ds-57c9c09510f6`. That `heldout.jsonl` is byte-identical (sha256 `d8b68d26...`) to `ds-e9443d334cb2`'s, which step 5 and docs/laya-trainer.md use (read 2026-10-06 in the store). The state is sent as `serialize_state(state)`, as in step 5. The first request's load time is reported apart.
   - **The teacher's cost** (`scripts/systemone/cpu_load.py`, holding the teacher's launch from Phase 10 on): the teacher serving as the picker launches it (MTP on), three blocks in this order: no load (A), load (L), no load (B). Each block sends the bench's latency prompts 1-30 once, in order, one teacher request at a time. During L only, 4 clients send student reads to the student service continuously (this step's 334 held-out rows, cycled), starting before the block's first teacher request and stopping after its last. Decode tok/s (the engine's own token count) and TTFT: each prompt's L value is paired with the mean of its A and B values. The report gives the ratio of medians (L over the A/B mean) with a paired percentile bootstrap over prompts (`statlib.boot_ci` on the per-prompt tuples), beside the same statistic for B over A, which gives the test-retest floor (primer Sec. 4.4). No threshold is fixed: the report goes to the operator, who decides whether E-core pinning suffices as is, or whether onnxruntime's intra-op threads are capped or the service restricted, and records that under E4.
3. **`vllm-s1` under devai-control** (after Phase 8; only if the gate passed).
   - The recovery entry scoped to `["vllm-s1"]`, with the flags that passed step 1.
   - The probe cell, with that entry in place: `make probe-vllm-s1 PROBE_REPO=nvidia/diffusiongemma-26B-A4B-it-NVFP4 PROBE_CTX_EXACT=8192 PROBE_KV_CACHE_TYPE=<the gate file's --kv-cache-dtype>` (a probe job from Phase 10 on). The cell is stamped with `devai.engine.vllm-s1`. `-print-load` needs this fitting cell, so it comes first.
   - An engine job submitted by hand, with the request that `gpu-arbiter -print-load vllm-s1 diffusiongemma-26B-A4B-it-NVFP4@8192` prints: both programs come up and the service is `serving`; `scancel` stops both and frees the card; the engine's command line equals the router's golden argv apart from the store path and the port, and equals the argv the gate's counted runs recorded apart from the store path and the port. Any other difference voids the gate: E3 is not recorded as a pass, and step 1 runs again on a flag file that reproduces the production argv.
   - **Sleep,** measured on `vllm-s1` as Phase 8 step 1 did for vLLM: with `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`, whether `/sleep?level=1` and `/wake_up` work (5 times each, with their times); `used_mib` after sleep; host RSS; the same read, same seed, before sleep and after wake; per-read latency with and without sleep mode, paired by read: the geometric mean of the per-read ratio (on over off), with a one-sided 95% upper bound (`statlib.boot_ci` at level 0.90, 10,000 resamples, a fixed seed; primer Sec. 5). The non-inferiority margin is 1.05, fixed before the run. The before-sleep and after-wake reads are a tolerance check (primer Sec. 12): over every item read, the largest absolute difference per option probability must be at most 1e-3, and the top option must be the same. `sleep_mode` stays false unless sleep and wake succeed all 5 times, the upper bound is at most 1.05, and the tolerance check passes. The report gives the observed ratio, its bound and the largest difference, and the operator decides.
   - **Host RAM** while DiffusionGemma sleeps and the teacher serves (`MemAvailable`, and swap in use), read only if sleep works on `vllm-s1`, and given to the operator with the sleep result. The reverse does not arise: the teacher never sleeps beside `vllm-s1`, since a switch into `vllm-s1` cancels its job and the prolog unloads any sleeper (step 4).
4. **The router's System One port** (after Phase 9), as above (I27): `gpu-arbiter/systemone.go`, the detached switch and the switch rules in `slurm_launch.go`, the prolog's sleeper rule for `vllm-s1`, their tests, compose, docs/router.md and docs/systemone.md. Then the live checks of the exit criteria.
5. **Teacher reads (E5),** on the `vllm` backend (needs only 4v), with MTP on and off. Before Phase 10 as `make systemone-teacher-reads` with the stack otherwise idle; afterwards as the `systemone-teacher-reads` action, one job for the whole run, holding its three launches in turn (above).
   - **Launches,** in this order: MTP off, MTP on, MTP off (`<name>@<ctx>`, `<name>::mtp@<ctx>`, `<name>@<ctx>`). MTP is a launch setting, so each is a new engine process, and the two MTP-off launches give a test-retest floor that includes a relaunch (primer Sec. 4.3, 4.4).
   - **How a read is made:** one chat request per question, with the state as `serialize_state(state)` and the options listed as letters; `max_tokens` 1, `logprobs: true`, `logprob_token_ids` the letters' token ids (each must be one token in the model's own tokenizer, or the script refuses), reasoning off (`chat_template_kwargs: {"enable_thinking": false}` at the top level, as in I16), temperature 0. An option's probability is the softmax over the returned log-probabilities of those ids. `teacher_reads.py` answers in the `/v1/systemone` shape, so `score.py` scores it like any backend.
   - **The field:** in the 0.31.0 sdist, `logprob_token_ids` is a field of the chat request (`vllm/entrypoints/openai/chat_completion/protocol.py:288`) and needs `logprobs: true` (`:791-794`). For 0.28.0 and 0.30.0 the research reports it present; the local trees of those tags are partial and do not contain the entrypoints, so that is not re-checked here. The step's first request checks it on the served build, through the router (unverified that the router passes the field through untouched; its rewrite chain re-encodes only what it changes).
   - **Measure:**
     - crashes (vllm#42592): 1,000 reads at 4 clients while two other clients stream long decodes, in the MTP-on launch and in the first MTP-off launch; HTTP 500s and engine failures counted; each 200 is also checked for the requested ids, as below;
     - wrong values (vllm#44727: under speculative decoding the engine returns top-N log-probabilities instead of those of the requested ids). In the 0.31.0 source, a step in which any request carries draft tokens goes through the rejection sampler, which gathers the top `max_num_logprobs` and never reads `logprob_token_ids`. This holds in both model runners: Model Runner V2, the default in 0.31.0 (`vllm/v1/worker/gpu/model_runner.py:1598-1610`; `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:177-218` calls `compute_topk_scores`, `vllm/v1/worker/gpu/sample/logprob.py:116-150`, without `logprob_token_ids_state`), and V1 (`vllm/v1/worker/gpu_model_runner.py`, `_sample`; `vllm/v1/sample/rejection_sampler.py`, `_get_logprobs_tensors`), where only the plain sampler reads the ids (`vllm/v1/sample/sampler.py:111-121`). Read 2026-10-06. So a `max_tokens` 1 read is exposed while another request decodes, and on an idle engine its MTP-on values should equal MTP-off's (derived from the source, not measured). In this step the teacher's launch log is checked for vLLM's "using the V1 model runner instead" warning, that runner's path is read again in the 0.30.0 source, and the report records which runner served. In each launch every held-out item is read twice: with the engine otherwise idle, and while two other clients stream long decodes, as in the crash run. For each MTP-on read: whether the returned ids are exactly the requested letter ids (a read missing one counts as a wrong value and is not renormalised), the largest absolute probability difference, and whether the top option agrees. MTP-on is compared with each MTP-off launch, per condition, beside the two MTP-off launches compared with each other; the report lists every other difference between the launches (image, flags, cache state);
     - prefix-cache hits per question: one client, the engine's `/metrics` hit and query counters read before and after each request (the metric names checked in the 0.30.0 source in this step; 0.30 changed the hybrid-model retention default, research round 1);
     - accuracy against the gold labels of `ds-e9443d334cb2`'s held-out split (byte-identical to `ds-57c9c09510f6`'s, step 2). These labels are the teacher's own (the top option of the mean of 3 samples at temperature 0.7, docs/laya-trainer.md), so this measures how well one read reproduces the teacher's sampled answer, not correctness.
   - The report goes to the operator, who records the decision under E5.

**Later, not steps here (E4):** third-party small System One models (OpenDecider-nano, Verdict, jevos), each only after its licence and provenance are checked; anything distilled from Jev is ruled out by TypeSafe's customer agreement (section 2.3(b)). Each would be one more name in the student service's dispatch or a service of its own, with its own record.

### Exit criteria

- **The gate:** the report from step 1 exists, and E3 records the pass or the deferral (read E3).
- **The student service:**
  - `make test-laya-trainer` passes with `test_systemone.py`;
  - `make systemone-score GOLDEN=1` (`score.py --golden` in a lab container on `devai-net`, against `devai-laya-trainer:11443`, reading `/laya/runs` read-only) reproduces every golden row of every succeeded run in the store;
  - the process's CPU affinity is the E-core list (`taskset -cp <pid>` prints 8-23 here), and it has no GPU context (`nvidia-smi` lists no process of that container);
  - step 2's latency and the teacher's cost are in its report, and the operator's decision on the teacher's cost is recorded under E4.
- **`vllm-s1`** (if the gate passed):
  - an engine job for it brings both programs up and the service reports `serving`; `scancel` leaves the card idle (the job's `gpu.json`);
  - with a `vllm` model sleeping, an engine job for `vllm-s1` serves after its prolog has unloaded the sleeper (`guard.json` names it), and DiffusionGemma's breaker count is unchanged;
  - if step 3 set `sleep_mode` true: with that launch sleeping on `vllm-s1`, an engine job for the same launch wakes it (the service goes `waking`, then `serving`), and its `guard.json` records no unload;
  - killing the engine process while the server runs turns the service `failed` within 30 s;
  - the probe cell at `DEVAI_SYSTEMONE_CTX` is in `deploy/.vllm-s1-reasoning-cache.json`, and `make probe-check` is clean for it;
  - `make test-python` passes with the vendor drift test, and `make bench-plan` and the picker name no `vllm-s1` row;
  - step 3's sleep result is with the operator.
- **The router:**
  - with `DEVAI_SYSTEMONE_PORT` unset, nothing listens on 11440 and `make test-router` passes with the golden corpus unchanged;
  - with it set, a student read answers with no Slurm job (`devai-jobs list` shows none), and its reply names the student;
  - a DiffusionGemma read while the teacher serves answers 503 with `Retry-After`; after the switch it answers 200 naming DiffusionGemma, and `sacct` shows exactly one new engine job, whose predecessor ended by cancel; a teacher request afterwards switches back;
  - an unknown name answers the student service's 404 unchanged;
  - on a host without a GPU, or with it forced off (`DEVAI_GPU_DEVICE=`), once Phase 11 has closed (if this phase closes first, Phase 11's exit criteria check it): a student read on 11440 answers 200, a DiffusionGemma read answers the router's no-GPU error at once with no Slurm job submitted (`devai-jobs list` shows none), and `GET /v1/models` on 11440 does not list it;
  - `make test-router` and `make test-devai-tools` pass.
- **Measurements:** step 5's report is with the operator and the decision is recorded under E5.

### Phase 23 risks

| Risk | Mitigation |
| ---- | ---------- |
| DiffusionGemma crash-loops on this card, as it did before: in June 2026, 23 of 34 router launches ended "not serving" 4-13 s after ready, and the diffusion-sampler flags crash-looped under the bench pattern | the gate (I30) before any router work, with chat mixed into reads at 4 clients and 10 counted launches; on a failure, E3's deferral, and the students go on alone; once served, the router's breaker stops a launch that keeps failing, and on `vllm-s1` stays open until the router restarts (503 `launch_breaker_open`, not retried) |
| open defects in 0.31.0's DiffusionGemma path: vllm#58035 (a shape mismatch under concurrent decode), vllm#50699 with its open fix vllm#51953 (a runtime OOM under concurrent decode), vllm#59992 (the sampler step runs eagerly) | at most 4 sequences, a served canvas of 64, constrained reads; the soak at 4 clients; a later version with the fixes is a new record under E2, not a silent bump |
| no published run on a 24 GB Blackwell card; every published structured-mode run used a nightly, a fork or a patched commit, not the 0.31.0 release | the gate measures the release on this card; no upstream number is assumed |
| FlashInfer without `flashinfer-cubin` (not on PyPI, so not installed, M3) needs a kernel it cannot JIT-compile on 0.31.0 | the gate's first launch starts with an empty FlashInfer cache and run-time cubin downloads disabled (`FLASHINFER_CUBINS_REPOSITORY`), and must answer a read and a chat request; a failure there is a deferral, and taking the package from flashinfer.ai's index would be the operator's decision under E3 |
| every swap costs minutes: the teacher's cold start is about 2 min (119-124 s, aiagent's measurement), and DiffusionGemma took 90-115 s to ready on the old image | the router answers 503 + `Retry-After` instead of holding requests; aiagent reads in a batch before it escalates (its side); step 1 records the new startup time |
| a System One reader and a chat client cancel each other's loading engine jobs | one direction only: a System One request never switches away from a holder that is pending, loading or waking (503 `gpu_busy`), so the teacher always reaches serving; a chat request still cancels a loading `vllm-s1` job (Phase 9 case 4), so under chat traffic denser than DiffusionGemma's load time it never serves. Chat winning is the operator's choice (2026-10-06); the exit criterion is checked on an otherwise idle router, and step 1 records the load time that sets the gap |
| host RAM runs short when a sleeper and a serving engine share it (62 GiB, swap already in use, research round 2) | the teacher never sleeps beside `vllm-s1`: a switch into it cancels the teacher's job, and its prolog unloads any sleeper; `vllm-s1` sleeping beside the teacher is read in step 3 before `sleep_mode` can change |
| the student service contends for the CPU with the LLM stack and with aiagent's own onnxruntime students | pinned to the E-cores; step 2 measures the teacher's decode rate and TTFT with and without it, and the operator decides under E4 |
| a second vLLM environment grows `devai-engines`: another torch 2.13.0 with its CUDA wheels (`vllm-devai`'s whole image was 9.76 GB) | measured at the build; its stage sits before vLLM's, so a vLLM rebuild does not rewrite it |
| answers are uncalibrated, and `confidence` means a different quantity per backend | the router passes fields through unmapped (I27); only students carry `answer_confidence`; aiagent computes its own calibration for any backend it uses (its gate) |
| zero-shot use of DiffusionGemma or of teacher reads in aiagent's gate | aiagent's policy needs its own tau, a shadow run and its pass test for any new model; nothing is switched on here |
| the vendored server's own behaviour: `ensure_ascii` for dict states, a 600 s upstream wait, upstream errors as 502 and 500, a `/health` that always answers ok | the router's own 503 while not serving; `devai-control`'s own engine check; states sent as `serialize_state(state)` (docs/systemone.md) |
| teacher reads crash or give wrong values under MTP (vllm#42592, vllm#44727) | measured in three launches (MTP off, on, off), idle and beside streaming decodes, in step 5; nothing is planned before the operator decides E5 |

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| every upstream bump becomes a rebuild here | all | accepted; the same as Ollama and vLLM today |
| security fixes in Debian arrive only when the digest is bumped | all | a deliberate periodic bump and rebuild, documented next to the pin |
| renaming the lab images (`-gpu` suffix dropped) breaks scripts and habits that name them | 3 | every reference in the tree changes in the same phase; the old names leave the store only in Phase 14 |
| per-engine staleness code touches probers, bench and router at once | 4 | one change with tests; the Ollama-only rebuild exit criterion |
| a job vanishes without unloading (its script killed, `devai-slurm` restarted) | 5 | the epilog's `unload`; the 60 s lease; the guard's idle check; a job lost with its node is never requeued (`JobRequeue=0`, C3) |
| a host reboot leaves the node DOWN, and nothing starts | 5 | `ReturnToService=2`: the node returns when slurmd registers |
| the bus is down longer than a lease, and engines unload under running jobs | 5 | supervisord restarts nats-server at once; the jobs end, and the router submits new ones |
| GPU accounting by sampling is coarser than Slurm's own | 5 | one GPU job at a time, so the whole card is the job's; 5 s samples plus NVML's energy counter |
| Slurm reports no GPU energy on NVIDIA | 5 | devai reads NVML's energy counter at the start and end of each job (I3) |
| the web login controls the host user, on the LAN | 6 | TLS, one htpasswd account, CSRF, no shell; login attempts logged; untrusted text shown as text only, under a strict Content-Security-Policy |
| each running lab publishes JupyterLab and dsh on the LAN, and a lab terminal runs as the host user | 12 | each behind its own per-launch token (JupyterLab's from lab-start, dsh's own); the token links are shown only behind the operator login; slurm.md Sec. 12 lists the ports |
| a sleeping model's VRAM leaves too little for the next load | 8 | for a vLLM load, `devai-control` lowers the utilization by the sleeper's `used_mib`, unless recovery flags set it (SGLang sizes from free memory; a pool below its context counts as a failure); otherwise it unloads the sleeper and loads again |
| a job signals or reads what it should not | 5 | jobs run as a subordinate uid: no key but their bus password, no signal to the host user's processes; no devai service shares that uid (I26) |
| an action outlives its job (its script killed, `devai-slurm` restarted) | 5, 10 | the action's lease: `devai-control` stops its process group 60 s after the last renewal |
| a recovery file rewritten by rename stays invisible | 8, 17, 18 | `${DEVAI_REPO}/deploy` mounted as a directory, and both files re-read at each launch (I25) |
| a job held after a prolog failure starts only after Slurm's requeue delay | 9 | the router cancels it; the next request submits a new job, never releasing a held one |
| a keep-warm engine job blocks a queued trainer, probe or GPU-action job forever | 9 | the router yields to any pending, unheld job other than an engine job that requests the `engine` license (a trainer, a probe or a GPU action, C8) |
| no fallback inside the router after the cut-over | 9 | Phase 8's live checks first; rollback by reverting Phase 9 (I10) |
| a sleeper nobody needs keeps host RAM and VRAM while the router is down | 9 | the router unloads it when it is back (I9) |
| an engine that reaches serving and then dies is relaunched without end (the 72-recreate loop) | 9 | every engine-job submit spends a breaker unit that only a served non-5xx response repays |
| a router restart loses the holds of running benches | 9 | holds are renewed every 30 s and expire after 90 s unrenewed; boot cancels pending engine jobs |
| model-generated bench code reaches the engines' control routes, a token or devai's files | 10 | bench-task containers on devai-lab-egress only, engine metrics through the router (C5), HumanEval checks with a minimal environment, the HF token only for GPQA |
| an interrupt with another model costs a cold start (about 95 s for Qwen3.8-27B) | 10 | the sleep fast path for vLLM and SGLang; the same-model path needs nothing |
| submitting a job can run any registered action, destructive ones included | 5-10 | the JWT key only in `devai-slurm`, the router and `devai-operator`; nothing published; bus users limited to their subjects; `job_submit.lua` accepts jobs from `devai` only, so a token signed for root gets no job |
| SGLang is kept or frozen on a measurement of the wrong thing | 15 | the instrument's dry run before the measured run; thresholds fixed before the runs, and a failed condition is a freeze, not a re-run (I16) |
| a recipe or a guessed parser misleads a probe | 17 | curated and card-derived values win; the probe is the authority; recipe flags reach the generated recovery file only after review, and never a C2 setting |
| alternating lanes thrash cold starts | 19 | opt-in; documented |
| a short-circuit answers a real request with a canned string | 16 | all markers must co-occur; off by default; every hit logged |
| an auto-fix makes a probe pass for the wrong reason | 18 | one retry; every applied flag recorded and attributed |
| a PDF larger than pipelock's 5 MiB response cap blocked or truncated | 20 | behaviour above the cap observed first in 20.1; the cap raised above the service's `max_fetch_bytes`; `too_large` decided by the service |
| a ban from arXiv (403) after a policy mistake | 20 | rate groups, the crawl delay, the arXiv breaker, persisted counters, the full-text cap until arXiv is contacted |
| GROBID does not build from source with Debian's JDK | 20 | checked first in 20.1; the PDF text then comes from pypdfium2 alone until it builds |
| vLLM 0.30.0 loses context or quality on the prepared Qwen3.8 rows | 4 | 4v's acceptance rule, fixed before the runs, with MTP on as served: the exact-context and load probes, and quality non-inferiority with stated margins; on a failure or an inconclusive result 0.28.0 stays and E1 goes back to the operator; 0.28.0's wheelhouse and image tag kept until 4g passes |
| a HyperQwen series that applies at `--fuzz 0` no longer means what it did (its own 0.29 -> 0.30 port shipped four such defects) | 4 | the newest commit that still pins 0.30.0; the built image is judged on loads, contexts and bench items served with MTP, not on the patch apply |
| a rollback to 0.28.0 keeps what 4v set for 0.30.0 (an adopted retention flag 0.28.0 has no argument for, a raised utilization) | 4 | 4v's rollback reverts both, and restores the vllm-devai probe cache from a backup whose manifest now holds it |
| a rollback of Phase 4 after 4d points an image variable back, and launches exec a `devai-engine` the old images lack | 4 | revert 4a-4h, then restore the probe caches and the ledger from the archive taken before 4e's merge; the old images stay until Phase 14 |
| 0.30.0's changed defaults (tool-parser plugin imports, prefix-cache retention 0) break or slow agent sessions | 4 | plugins ported with a fallback import and imported before any probe; retention measured on the teacher with long replies, where the settings can differ, `None` adopted only by a rule fixed before the runs and emitted by the argument builders (I1) |
| devai bumps laya before aiagent accepts 0.3.28: aiagent refuses every new artifact | 22 | aiagent's release, with both versions in `LAYA_COMPAT`, is checked in the rebuilt lab image first (E6) |
| laya 0.3.28's head attention changes the exported graph, and parity within 1e-3 under torch 2.14 is not yet shown | 22 | the trainer refuses such an artifact (exit 7); a CPU job on the real base shows it before any GPU window |
| a training recipe is adopted on noise | 22 | I31's rule, fixed before the runs: three seeds per arm, a Bonferroni interval, calibration not worse beyond a fixed +0.02 margin, by interval; otherwise today's recipe stays |
| DiffusionGemma does not load, serve or stay up on this card: its crash-loop history here, the open vllm#58035, vllm#50699 / #51953 and vllm#59992, no published run on a 24 GB Blackwell card | 23 | the gate (I30) before any router work; on a failure, E3's deferral, and the students go on alone |
| a System One read forces a GPU swap of minutes (the teacher's cold start about 2 min; DiffusionGemma 90-115 s to ready on the old image) | 23 | the router answers 503 + `Retry-After` and holds no request; aiagent reads in a batch before it escalates; a chat request still cancels a loading `vllm-s1` job, so DiffusionGemma serves only in a chat gap longer than its load |
| host RAM runs short while a sleeper and a serving engine share it | 23 | the teacher never sleeps beside `vllm-s1`: a switch into it cancels the teacher's job and its prolog unloads any sleeper; `vllm-s1` sleeping beside the teacher is read in step 3 before `sleep_mode` can change |
| the student service slows the LLM stack, or aiagent's own onnxruntime students, on the CPU | 23 | pinned to the E-cores; step 2 measures the teacher's decode rate and TTFT with and without it |
| the second vLLM environment grows `devai-engines` | 23 | measured at the build; its stage sits before vLLM's |
| System One answers are uncalibrated, and `confidence` means something different per backend | 23 | passed through unmapped (I27); only students carry `answer_confidence`; calibration is aiagent's, and zero-shot use needs aiagent's own tau and shadow run |

## Migration / rollback story

- **Phase 1:** MCP and Open WebUI come back by reverting the commit; the AMD overlay and sops/age by their `attic/` RESTORE.md.
- **Phases 2-3:** the compose and Makefile image variables can point back at the upstream images and the old lab images, which stay in the local store until Phase 14.
- **Phase 4:** `make backup-create` runs before 4v's probes, before 4e's cache merge and again before 4g. 4v rolls back on its own (4v, "Rollback"). From 4d on, every launch runs `devai-engine`, which only `devai-engines` contains, and `ENGINES_IMAGE` has replaced the per-engine image variables, so pointing an image variable back is not enough. The rest of Phase 4 rolls back by reverting its commits (4a-4h): that brings back the per-engine image variables, the `vllm-devai` backend and the digest stamps. Then `deploy/.ollama-reasoning-cache.json`, `deploy/.vllm-reasoning-cache.json`, `deploy/.sglang-reasoning-cache.json`, `deploy/.vllm-devai-reasoning-cache.json` and `deploy/.model-status.json` are restored from the archive taken before 4e's cache merge, the last one that holds the vllm-devai cache and the unmerged vllm cache, and older than every 4g re-probe, as 4v's rollback restores its one file. The old images (`devai-ollama`, the stock vLLM image, the `vllm-devai` image `:latest` names after 4v, the upstream SGLang image) stay in the store until Phase 14. Removing `vllm-devai` changes agent provider names (`router-vllm-devai`); the picker rewrites those sets at every launch.
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
- **Phase 22:** reverting the bump (requirements, lock, `contract.py`, the fixture) brings 0.3.20 back. Artifacts trained under 0.3.28 stay valid, since aiagent and the student service accept both versions (E6, I28). The evaluation installs nothing; its `eval/` directory in the laya store can be deleted.
- **Phase 23:** the System One port is off while `DEVAI_SYSTEMONE_PORT` is unset, and the router is then byte-identical to before (I27). The student service is one supervisord program; stopping it, or reverting it, leaves the trainer as it was. `vllm-s1` leaves by reverting its stage, its service, its prober and its catalog and recovery changes; nothing else in `devai-engines` depends on it.
- **Existing hosts:**
  - delete the registry mirror stanza from `~/.config/containers/registries.conf` (Phase 2);
  - create `/var/cache/devai/jobs`, a plain directory (Phase 5, E9);
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
| 22 | 2-3 days after aiagent's release, plus two GPU windows: `make laya-check` (about 3 min) and the recipe evaluation (about 20-25 min, with the teacher's cold start) |
| 23 | 5-8 days, plus GPU windows for steps 1, 3 and 5 |

Estimates, not measurements. Phases 5-7, 15-17, 20 and 21 run beside others, so the wall-clock total is shorter than the sum.

## References

- [docs/slurm.md](../slurm.md) -- the architecture this plan builds.
- Slurm documentation (slurm.schedmd.com):
  - slurm.conf: `Licenses`, `PrologEpilogTimeout`, `BatchStartTimeout`, `KillWait`, `PrologFlags`, `HealthCheckProgram`, `SlurmdParameters`, `JobRequeue`, `ReturnToService`;
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
- laya (Phase 22): github.com/NandhaKishorM/laya, tags v0.3.20 (23a1752) and v0.3.28 (a4a8921afe); the 0.3.28 sdist on PyPI (`laya/train.py`, `laya/common.py`, `laya/agent.py`, `laya/revisions.py`); gh#131, gh#462, gh#559, gh#741, gh#941; the checkpoints at huggingface.co/convaiinnovations/laya (commit 55cf4c4). aiagent: `src/aiagent/system1/artifacts.py` (`LAYA_COMPAT`), `tools/system1/make_fixture.py`.
- System One (Phase 23): vLLM v0.31.0 `examples/features/structured_diffusion/` (`structured_server.py` and its README; vllm#57250, vllm#58216); open vLLM PRs and issues vllm#59299, vllm#58429, vllm#42592, vllm#44727, vllm#58035, vllm#50699, vllm#51953, vllm#59992; docs/laya-trainer.md (the dataset and artifact contracts, the 2026-09-25 campaign's held-out split).
- In this repository:
  - the host-build pattern: scripts/build-vllm.sh, scripts/build-ollama.sh, deploy/Dockerfile.vllm, deploy/Dockerfile.ollama;
  - attic/README.md and attic/cluster-mode/RESTORE.md: the pattern for moving work to the attic;
  - gpu-arbiter/main.go (`ensureBackendRunning`, `stopOtherBackends`, `containerRecreate`, `computeLaunchConfig`);
  - scripts/_probe_hf_common.py, scripts/bench-sync.py, scripts/bench/clean_slate.py, laya-trainer/laya_trainer/controller.py;
  - docs/router.md, docs/bench-results.md "Run protocol", docs/laya-trainer.md.

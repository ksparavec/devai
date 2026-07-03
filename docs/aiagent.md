# aiagent (the "AIAgent (shell)" picker agent)

Operator/user reference for the `aiagent` integration in the devai lab.

`aiagent` (https://github.com/devitops-com/aiagent) is a programmatic DSPy
agent CLI -- prompt/pipeline optimization, goal-reaching loops, and autonomous
data processing over local LLMs. It is NOT a chat UI; basic chat is a minor
feature. It runs against any OpenAI-compatible endpoint, so in the lab it talks
to the devai router.

Unlike the other picker agents (Claude Code, Aider, Codex, OpenCode, LATE, Open
Interpreter), aiagent is a tool the user drives explicitly. The picker therefore
does NOT exec it. When you choose "AIAgent (shell)", the picker configures the
router endpoint + model in the environment and drops you into an interactive
bash shell. You then run `aiagent ...` yourself, for example:

```
aiagent doctor        # check config + reach the router
aiagent models list   # show configured aliases + router-advertised models
aiagent chat          # interactive chat with the configured model
aiagent run <skill>   # run a skill once
aiagent optimize <skill> --out compiled/<skill>.json
aiagent eval <skill> --compiled compiled/<skill>.json
aiagent --help        # full command surface
```

Type `exit` to leave the shell and return to the picker/host.

## How the picker wires it

aiagent has no devai-specific code; it adapts through config env-fallbacks
(`AIAGENT_API_BASE` -> `OPENAI_BASE_URL` -> `OLLAMA_HOST`). The picker's
`_build("aiagent", ...)` sets, from the chosen (model, backend):

| Env var            | Value                                             |
|--------------------|---------------------------------------------------|
| `AIAGENT_API_BASE` | `http://devai-router:<port>/v1` (INCLUDES `/v1`)  |
| `AIAGENT_API_KEY`  | `local` (local backends ignore it; client needs a value) |
| `AIAGENT_MODEL`    | router model string: bare Ollama tag, or `<name>@<ctx>` for vLLM |
| `OPENAI_BASE_URL`  | same base (setdefault -- never clobbers a preset) |
| `OPENAI_API_KEY`   | `local` (setdefault)                              |

`<port>` is the chosen backend's router port (Ollama 11434, vLLM 11435). The
shell launcher (`scripts/aiagent-launcher.sh`, installed as
`/usr/local/bin/aiagent-shell`) prints a one-line hint banner before
`exec bash -i`.

It deliberately does NOT set `AIAGENT_CONTEXT`. aiagent turns that into a
`<model>@<ctx>` control-surface suffix, which is redundant and can double up:
Ollama's `/v1` surface uses the router's global `OLLAMA_CONTEXT_LENGTH`
(per-request ctx is ignored), and vLLM's ctx already rides in `AIAGENT_MODEL`'s
own `@<ctx>` (composed by the picker) -- so also setting `AIAGENT_CONTEXT` would
risk `<name>@<ctx>@<ctx>`. (aiagent <= v0.1.1 additionally composed `@<ctx>`
before `::<reasoning>`, which the router mis-parsed into an Ollama
`invalid model name`; fixed in v0.1.2, devitops-com/aiagent#3.)

aiagent resolves config as env > TOML > devai-env > defaults; verify with
`aiagent config show`.

## GPU policy (DEVAI_AIAGENT_GPU)

The router already holds the served model on the GPU (a 27B Q4 model uses
~17-19 GB of a 24 GB card, leaving little headroom). aiagent can, in principle,
run its own CUDA code independent of the router, which would contend for that
headroom and risk an OOM on either side.

The aiagent shell therefore has a GPU-sharing toggle, `DEVAI_AIAGENT_GPU`:

- `router-only` (default) -- the launcher exports `CUDA_VISIBLE_DEVICES=""`, so
  aiagent cannot touch the GPU directly; all its compute flows through the
  router's OpenAI endpoint. Safe against VRAM contention.
- `share` -- the GPU stays visible; aiagent may run its own CUDA code, accepting
  the OOM risk against the router-loaded model.

The picker shows a sub-modal (Router-only / Share GPU) when you pick the aiagent
shell. A pre-set `DEVAI_AIAGENT_GPU` env value wins and suppresses the prompt --
e.g. `DEVAI_AIAGENT_GPU=share devai-agent --agent aiagent`.

Note: the aiagent bundle ships numpy/tokenizers/tiktoken but NOT torch (and its
python is isolated under /opt/aiagent), so it does not use the GPU directly
today -- the toggle is a forward-looking guard for when local embeddings/RAG land.

## Install

aiagent ships as a self-extracting makeself bundle (~63 MB) that carries its own
CPython 3.13 -- no Python or zstd is needed in the lab image to run it.

- `make fetch-cli` downloads the bundle to
  `/var/cache/devai/pip/bin/aiagent-install.sh` (ETag-stamped; the etag feeds
  `BIN_HASH`, so a new release busts the image's binary-install layer).
- `deploy/Dockerfile.lab` extracts it with `AIAGENT_PREFIX=/opt/aiagent` -- an
  ISOLATED prefix so aiagent's bundled interpreter
  (`/opt/aiagent/bin/python3.13`) stays OFF the system PATH. This is
  load-bearing: the lab's own modules install into the `/usr/local` python via
  `uv pip install --system`, and nothing ever lands in aiagent's tree. Only the
  `aiagent` launcher is symlinked onto PATH
  (`/usr/local/bin/aiagent -> /opt/aiagent/bin/aiagent`); the bundled python is
  never exposed. The install runs AFTER all `uv pip install --system` steps.
  (Installing under `/usr/local` previously put aiagent's `python3.13` on PATH,
  so `uv --system` resolved to it and bled torch/jupyterlab/a transitive typer
  into the bundle, overwriting its pinned deps and breaking the CLI.)

There are two interpreters in the lab image, and they never mix: the lab's
`/usr/local/python/current` (uv-managed; holds torch/jupyterlab/etc.) and
aiagent's own `/opt/aiagent/bin/python3.13` (bundle-only; holds aiagent's deps).

Linux x86_64 only; the upstream bootstrap refuses other platforms.

## Verified example (qwen3.6:27b-q4_K_M)

End-to-end in the built `devai-lab-gpu` image (aiagent v0.1.2):

- `aiagent` and `aiagent-shell` are on PATH; `aiagent` resolves to
  `/opt/aiagent/.../aiagent`; there is no `/usr/local/bin/python3.13` shadow.
- The two interpreters are cleanly separated: `/opt/aiagent/bin/python3.13` has
  `typer 0.26.8` and NO torch/jupyterlab; `/usr/local/python/current` has them.
- `aiagent doctor` (online) -> `status: ok`, and lists the router's models.
- `aiagent config show` resolves `api_base=http://devai-router:11434/v1`.
- Generation works: `aiagent run chat --text "..."` (no `--model` needed --
  the `default` alias follows `AIAGENT_MODEL`) cold-loads
  `qwen3.6:27b-q4_K_M` via the router and returns the answer (`DEVAI_OK`).

## Notes and caveats

Five upstream bugs were found and fixed during this integration; the lab image
bakes the fixed **v0.1.2** bundle:

- **#1 / #2** (v0.1.0): a Typer/Click `make_metavar` incompatibility that crashed
  `run` / online `doctor`, and a skill loader that failed with
  `entry module skill.py not found` in the sourceless bundle. Fixed in v0.1.1.
- **#3** (<= v0.1.1): the control-surface string was composed as
  `<model>@<ctx>::<reasoning>`, but the router strips `@<ctx>` only when it is
  last, so a set context reached Ollama as `invalid model name`. Fixed in v0.1.2.
- **#4** (<= v0.1.1): `aiagent run`/`chat` used a baked `default` alias instead of
  `AIAGENT_MODEL`. Fixed in v0.1.2 -- the `default` alias now follows
  `AIAGENT_MODEL`, so no `--model` override is needed.
- **#5** (<= v0.1.1): `aiagent version` printed `0.1.0` regardless of the bundle.
  Fixed in v0.1.2 (`aiagent version` -> `0.1.2`).

The launcher still omits `AIAGENT_CONTEXT` (see above) -- not because of #3 (now
fixed) but because the context is already conveyed per backend and a duplicate
would risk `@<ctx>@<ctx>` on vLLM.

## Files

- `scripts/aiagent-launcher.sh` -- shell launcher (installed as `aiagent-shell`).
- `scripts/model-picker.py` -- `_AGENTS` entry, `_build` branch,
  `_resolve_aiagent_gpu_mode` / `_apply_aiagent_gpu` GPU sub-modal.
- `deploy/Dockerfile.lab` -- bundle install + launcher COPY + PATH symlink.
- `Makefile` (`fetch-cli`) -- bundle download (ETag-stamped).
- `tests/python/test_aiagent_picker.py` -- picker wiring + launcher GPU policy.

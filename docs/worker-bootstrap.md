# Worker bootstrap image

The `devai-worker-bootstrap` image is the minimal container that
SkyPilot launches on cloud VMs (and on-prem secondary hosts) to
participate in a devai cluster fleet. It is **not** the lab image --
it carries the gpu-arbiter binary and the three backend container
images, nothing user-facing.

This is the operator reference. Architecture rationale and SkyPilot
integration live in:

- [docs/plans/gpu-arbiter-cluster-mode.md](plans/gpu-arbiter-cluster-mode.md)
  decision 11 (folded into Phase 1)
- [docs/plans/skypilot-fleet-provisioner.md](plans/skypilot-fleet-provisioner.md)
  Phase 1 (consumer)

## Build

```bash
make build-router            # builds gpu-arbiter binary first
make build-worker-bootstrap  # then assembles the bootstrap image
```

Image size target: under 5 GiB. The big-ticket items are:

- CUDA 13.0 runtime base image (~3 GiB)
- Pre-pulled vllm-openai image (~2.5 GiB compressed)
- Pre-pulled lmsysorg/sglang image (~2 GiB compressed)
- Pre-pulled ollama/ollama image (~700 MiB compressed)

Pre-pull only the images the target GPU class can serve to keep size
manageable on smaller cards.

## Env-var contract

Required:

| Variable                  | Purpose                                      |
| ------------------------- | -------------------------------------------- |
| `DEVAI_HEAD_URL`          | Full URL of the head (`http://head.lan:11444`) |
| `DEVAI_WORKER_TOKEN_FILE` | Path to bearer token (default `/run/devai/cluster-token`) |

Recommended:

| Variable                    | Default                | Purpose                                 |
| --------------------------- | ---------------------- | --------------------------------------- |
| `DEVAI_WORKER_NAME`         | `$(hostname)`          | Unique-per-fleet identifier             |
| `DEVAI_LIFECYCLE`           | `ephemeral`            | `ephemeral` or `persistent`             |
| `GPU_MEMORY_GB`             | `24`                   | VRAM the head should advertise          |
| `DEVAI_GPU_TYPE`            | `unknown`              | Short label for routing                 |
| `DEVAI_BACKENDS`            | `ollama,vllm,sglang`   | Comma-separated subset                  |
| `DEVAI_WORKER_INBOUND_PORT` | `11444`                | TCP port for forwarded requests         |
| `DEVAI_WORKER_HOST`         | `localhost`            | Hostname the head should use            |

The full per-variable reference lives in
[docs/cluster-env.md](cluster-env.md).

## Cloud-init shape

`deploy/worker-cloud-init.sh` is the entrypoint baked into the
image. It validates env vars, normalises the lifecycle string, and
execs the arbiter in `--mode=worker`. SkyPilot task specs typically:

```yaml
# In the SkyPilot task YAML
resources:
  cloud: runpod
  accelerators: 3090:1
  ports: 11444
file_mounts:
  /run/devai/cluster-token: ~/.config/devai/cluster-token
envs:
  DEVAI_HEAD_URL: http://${HEAD_PUBLIC_IP}:11444
  DEVAI_WORKER_NAME: ${SKYPILOT_TASK_NAME}-${SKYPILOT_NODE_RANK}
  DEVAI_LIFECYCLE: ephemeral
  DEVAI_GPU_TYPE: RTX3090
  GPU_MEMORY_GB: "24"
run: |
  docker run --rm --gpus all \
    -e DEVAI_HEAD_URL="$DEVAI_HEAD_URL" \
    -e DEVAI_WORKER_TOKEN_FILE=/run/devai/cluster-token \
    -e DEVAI_WORKER_NAME="$DEVAI_WORKER_NAME" \
    -e DEVAI_LIFECYCLE="$DEVAI_LIFECYCLE" \
    -e DEVAI_GPU_TYPE="$DEVAI_GPU_TYPE" \
    -e GPU_MEMORY_GB="$GPU_MEMORY_GB" \
    -v /run/devai/cluster-token:/run/devai/cluster-token:ro \
    -p 11444:11444 \
    devai-worker-bootstrap
```

The bootstrap image listens on `:11444` for the head's forwarded
requests; SkyPilot's `ports: 11444` opens that port on the cloud
provider's network rules.

## Layout inside the image

```
/usr/local/bin/
  gpu-arbiter                  # the same Go binary used in single mode
  worker-cloud-init.sh         # entrypoint
/var/lib/containers/storage/   # pre-pulled vllm/sglang/ollama images
```

No JupyterLab, no model picker, no agent CLIs (`claude`, `codex`,
`gemini`, `aider`), no Open WebUI, no code-server. The bootstrap
image is intentionally lean.

## Verifying

A clean smoke test against a stub head (per cluster-mode Phase 1.5):

```bash
podman run --rm --network host \
  -e DEVAI_HEAD_URL=http://localhost:18080 \
  -e DEVAI_WORKER_TOKEN_FILE=/run/devai/cluster-token \
  -e DEVAI_WORKER_NAME=test-1 \
  -v /tmp/cluster-token:/run/devai/cluster-token:ro \
  devai-worker-bootstrap
```

The stub head should log a successful registration within ~1
second and a heartbeat within 10s.

## Upgrades

The worker image follows the gpu-arbiter version. To upgrade:

1. `make build-router` (rebuilds the binary).
2. `make build-worker-bootstrap` (re-assembles the image).
3. Push to whatever registry your SkyPilot tasks pull from.
4. Existing in-flight worker VMs continue to run their old binary;
   new VMs use the new one. There is no in-place upgrade -- the
   ephemeral lifecycle assumes "drop and re-launch."

## References

- Plan: [docs/plans/gpu-arbiter-cluster-mode.md](plans/gpu-arbiter-cluster-mode.md) (decision 11)
- Cluster-mode docs: [docs/cluster-mode.md](cluster-mode.md)
- Env contract: [docs/cluster-env.md](cluster-env.md)
- SkyPilot consumer plan:
  [docs/plans/skypilot-fleet-provisioner.md](plans/skypilot-fleet-provisioner.md)

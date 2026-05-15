# SkyPilot in the devai lab

The devai lab image bundles the [SkyPilot](https://github.com/skypilot-org/skypilot)
CLI plus its broad cloud-extras set
(`aws,gcp,azure,kubernetes,slurm,runpod,lambda`). With cloud
credentials in place, you can ask any of the bundled CLI agents
(Claude Code, Gemini CLI, Codex CLI) to provision GPU jobs on your
own cloud accounts via natural-language instructions -- the
[SkyPilot Agent Skill](https://blog.skypilot.co/agent-skill/) plugin
teaches each agent how to drive the `sky` CLI.

This is the user-facing reference. The architecture rationale and
phase split live in
[docs/plans/skypilot-agent-skill.md](plans/skypilot-agent-skill.md).
The system-side fleet provisioner (gpu-arbiter cloud-burst) is the
**sibling** plan
[docs/plans/skypilot-fleet-provisioner.md](plans/skypilot-fleet-provisioner.md);
this guide is independent of it.

## What you get

| Surface             | Behaviour                                              |
| ------------------- | ------------------------------------------------------ |
| `sky` CLI           | All standard commands (launch/exec/status/down/etc.)   |
| Cloud extras        | aws, gcp, azure, kubernetes, slurm, runpod, lambda     |
| Agent Skill         | Pre-installed for the agents that support it           |
| Credentials         | Read from $HOME/{.aws,.config/gcloud,.config/sky,...}  |
| Cost path           | Your cloud account; devai never sees a bill            |

## First-time setup

1. Mount cloud credentials into your `$HOME` before launching the
   lab. devai's existing `$HOME` mount already covers this -- nothing
   special required if you already use the AWS / gcloud / sky CLIs
   on your host.

2. Inside the lab container:
   ```bash
   bash /usr/local/bin/sky-setup.sh
   # or
   sky check
   ```

3. Confirm at least one cloud is enabled in the output. If not, set
   up credentials per the relevant section below.

## Per-cloud credential setup

### RunPod (cheapest spot 3090s and 4090s)

1. Sign up at runpod.io, create an API key.
2. On the host:
   ```bash
   pip install --user runpod
   runpod config
   ```
   Paste the API key when prompted. Writes `~/.runpod/config.toml`.
3. Re-launch the lab; `sky check` should show RunPod enabled.

### Lambda Labs

1. Sign up at lambdalabs.com, create an API key.
2. On the host:
   ```bash
   mkdir -p ~/.lambda_cloud
   echo 'api_key = your_key_here' > ~/.lambda_cloud/lambda_keys
   ```
3. Re-launch the lab; `sky check` should show Lambda Labs enabled.

### AWS / GCP / Azure

Use the standard `aws configure`, `gcloud auth login`,
`az login` flow on the host. SkyPilot reads the same credentials.

### Kubernetes / Slurm

If you have a kubeconfig at `~/.kube/config` or a Slurm cluster
already accessible (SSH config in `~/.ssh/config`), SkyPilot picks
them up automatically.

## Hello-world

Inside the lab container:

```bash
sky launch --cloud runpod --gpus 3090:1 --use-spot -- 'echo hello'
sky down --all -y
```

Total cost: ~$0.05 if you remember to tear down. Lifetime cap:
`sky cost-report` shows running spend across all your clouds.

## Driving SkyPilot from an AI agent

The SkyPilot Agent Skill plugin is pre-installed for supported CLI
agents. From inside the lab:

- **Claude Code**: `claude` -- "Spin up a single 3090 on the
  cheapest cloud, run `train.py` from this directory, copy results
  back, then shut it down."
- **Gemini CLI**: `gemini` -- equivalent.
- **Codex CLI**: `codex` -- equivalent.

The agent will translate the request into the appropriate
`sky launch` / `sky exec` / `sky down` sequence and run it for you.

## Cost guidance

GPU instances are not free. Some examples (spot pricing, late-2025):

| GPU class    | Cloud   | $/hr (spot) | $/hr (on-demand) |
| ------------ | ------- | ----------- | ----------------- |
| RTX 3090     | RunPod  | $0.20       | $0.44             |
| A100 (40GB)  | Lambda  | $1.10       | $1.29             |
| H100         | RunPod  | $1.99       | $4.69             |

A 24-hour fine-tune on a single A100 is ~$30. A 24-hour H100 run
that you forget to shut down is ~$120. **Use `sky cost-report`
weekly** and set up budget alerts on each cloud account.

## Plugin install per agent (if needed)

The plugin should be pre-installed by the lab image; if it's missing
for your agent, install manually:

- **Claude Code**:
  ```bash
  claude plugin install skypilot-org/skypilot
  ```
- **Gemini CLI**: managed via Gemini's plugin command (varies by
  version; see the agent's `--help`).
- **Codex CLI**: same idea -- check Codex's plugin command.

If your agent doesn't list the SkyPilot tools after install,
restart the agent process so it re-reads the plugin manifest.

## Troubleshooting

### `sky check` shows zero enabled clouds

The credential files for at least one cloud need to be mounted into
the lab container. Verify with:

```bash
ls -la ~/.aws/credentials ~/.config/gcloud/ ~/.runpod/ ~/.lambda_cloud/ 2>/dev/null
```

If empty, follow the per-cloud setup section above on the host (not
inside the container) and re-launch the lab.

### `sky launch` rejects with "no available capacity"

Spot instances are scarce in popular regions. Try:
- `--use-spot=false` to fall back to on-demand pricing.
- A different region with `--region us-east-1`.
- A different cloud with `--cloud lambda` instead of `--cloud aws`.

### Image build complains "skypilot wheels not found"

The wheel cache is populated by `make fetch-cli` on the build host.
Re-run that, then `make build-gpu` (or `make build-cpu`).

If the wheel cache is empty for a network-firewalled environment,
the lab image will still build but without `sky` -- install
post-launch with:

```bash
uv pip install --system 'skypilot[aws,gcp,azure,kubernetes,slurm,runpod,lambda]'
```

## References

- Plan: [docs/plans/skypilot-agent-skill.md](plans/skypilot-agent-skill.md)
- Sibling system-side plan:
  [docs/plans/skypilot-fleet-provisioner.md](plans/skypilot-fleet-provisioner.md)
- SkyPilot docs: docs.skypilot.co
- SkyPilot GitHub: github.com/skypilot-org/skypilot
- Agent Skill: blog.skypilot.co/agent-skill/

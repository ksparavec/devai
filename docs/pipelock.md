# pipelock -- egress firewall for the lab (whole-lab fail-closed)

Operator reference for `devai-pipelock`, the MITM egress proxy that makes the
interactive lab container network **fail-closed**: the lab has NO route to the
internet except through pipelock, which decrypts and inspects outbound TLS,
DLP-blocks accidental secret egress, and logs every decision.

This is the source of truth for the pipelock integration. See also
`docs/secrets.md` (the sops/age scaffold the CA key uses) and `docs/mcp.md`
(the peer opt-in service whose conventions pipelock mirrors).

## Why

An untrusted local LLM (run through opencode or any agent in the lab) can leak
secrets over the network. A cooperative `HTTPS_PROXY` is not enough -- the agent
can shell out to `curl`/`git`/`pip` and bypass it. So the enforcement lives in
the network topology, not in any application's goodwill.

Threat model is **buggy-leak**, not hostile: a cooperative-but-fallible model
that might *accidentally* include a secret in an otherwise-honest request. The
posture therefore blocks secret egress (DLP) and audits everything, while
keeping the adversarial-evasion scanners in warn-only to avoid false positives.

## Architecture

```
lab container (devai-lab-gpu/cpu)              devai-net (has internet)
  joined ONLY to devai-lab-egress                ^
  - no internet route, no host-gateway           | internet leg
  - HTTPS_PROXY=http://devai-pipelock:8888        |
  - NO_PROXY=devai-router,apt-cache,...    -->  devai-pipelock (dual-homed)
  - pipelock CA in the system trust store         - MITM + DLP block + audit
        |                                          - cap_drop:ALL, no-new-privs
        | devai-lab-egress (internal: true, NO NAT)
        v
  devai-router / apt-cache / registry-cache   (dual-homed: devai-net + egress;
                                               they do NOT IP-forward -> no leak)
```

The lab joins only the internal `devai-lab-egress` network. pipelock is the only
member of that network that is also on `devai-net` (the internet leg), so it is
the lab's sole path out. The router and caches are dual-homed so inference and
package installs still work; containers do not forward between their interfaces,
so they cannot be used as a back-door route.

## Why fail-closed holds

1. An `internal: true` podman network gets a bridge + gateway IP but netavark
   installs **no masquerade / forward rules** for it -- a container whose only
   interface is on that bridge has no route off-host. (The gateway IP existing
   is not a leak; the absence of NAT is what matters.)
2. Dual-homed infra (`router`, `apt-cache`, `registry-cache`) does not enable
   `ip_forward` and has no `NET_ADMIN`, so the lab cannot route through them.
3. The lab is launched with no `--add-host=host.containers.internal:host-gateway`
   (that would be an off-host route) and `cap_drop`-style restraint, so nothing
   inside can reconfigure its way out. The only egress interface lives in
   pipelock's separate namespace, unreachable from the lab.

Verified: from inside the egress net, `curl https://1.1.1.1` (bypassing the
proxy) fails to connect, and external DNS does not resolve.

## One-time setup (operator)

The pipelock MITM CA is **operator-generated and per-host**: the private key
lives on the host and is never committed. Bootstrap is three commands:

```
make pipelock-ca-init   # CA cert -> deploy/pipelock-ca.crt (local build input, gitignored)
                        # CA key  -> ~/.config/devai/pipelock-ca-key.pem (0600, host-only)
make build-gpu          # or build-cpu -- bakes the CA cert into the lab trust store
make cache-up           # mounts the host key into devai-pipelock automatically
```

No sops, no tmpfs, no `.env` edit. `cache-up` looks for the key at
`$(PIPELOCK_CA_KEY)` (default `~/.config/devai/pipelock-ca-key.pem`, override
via the Make variable) and mounts it; if it is missing, cache-up prints a note
and pipelock stays unhealthy until you run `make pipelock-ca-init`. Nothing
about the CA is committed: the key is host-only and the cert is just a local,
gitignored build input the lab image bakes in (it must sit in the build context,
so it lives in `deploy/`, not the repo history). Each host generates its own CA
-- regenerate + rebuild the lab image per host; a CA is not shared across
machines. The build refuses to run if the cert is missing, pointing you at
`make pipelock-ca-init`.

## Bring-up

pipelock is an always-on infra service (no profile) started by `make cache-up`
alongside the router and caches. It requires the host CA key (mounted
automatically by cache-up); without it pipelock fails its healthcheck and
`bin/devai-agent` refuses to launch the lab (fail-closed, not fail-open).

```
make cache-up         # starts devai-pipelock (+ creates devai-lab-egress --internal)
make pipelock-logs    # tail decisions (also at /var/cache/devai/logs/devai-pipelock.log)
```

`bin/devai-agent` and `make lab-gpu/shell-gpu` launch the lab on
`devai-lab-egress` with the proxy + CA env forced. To bypass the lock for
debugging (UNSAFE -- no egress control), set `DEVAI_NO_EGRESS_LOCK=1` (launcher
only); this restores the open `devai-net` path.

## Known breakages and mitigations

- **External git over SSH breaks.** SSH cannot traverse an HTTP CONNECT proxy
  and there is no L3 route. Use HTTPS remotes (proxied + CA-trusted). The
  `~/.ssh` mount still works for internal hosts and commit signing.
- **Cloud LLM APIs are TLS-spliced, not inspected.** `api.anthropic.com` and
  `api.openai.com` are in `passthrough_domains` so cert-pinning SDKs don't break
  -- a documented DLP blind spot. The common path is the local router, so most
  installs never hit these. Remove them from passthrough to force inspection
  (and accept possible breakage).
- **External DNS does not resolve in the lab.** Intended: with the proxy set,
  clients send the hostname via CONNECT and pipelock resolves on its devai-net
  leg. Tools must honor the proxy; `NO_PROXY` lists every internal alias.
- **Dev installs go through pipelock.** pip/uv/npm/git-over-HTTPS work because
  they honor the proxy and trust the baked CA; their destinations are in the
  allowlist (`pypi.org`, `registry.npmjs.org`, `github.com`, ...). apt uses the
  internal `apt-cache` and never touches pipelock.
- **Trusted harnesses are not locked.** `make test-agents`, `bench-*`, and
  `probe*` stay on `devai-net` -- they run trusted code and need direct access.
  Only the interactive agent lab is egress-locked.
- **DLP false-positives on signed URLs.** The `JWT Token` (and similar) pattern
  matches signed tokens embedded in legitimate URLs -- e.g. a GET to
  `release-assets.githubusercontent.com` is blocked as "JWT Token (high)", so
  some legitimate fetches fail. Tuning (scope request-body scanning to outbound
  bodies, or exempt known CDN hosts) is a backlog item; see TODO.md.

## Verification

```
# Topology
podman network inspect devai-lab-egress --format '{{.Internal}}'   # -> true
podman inspect devai-pipelock --format '{{.State.Health.Status}}'  # -> healthy

# From inside the egress net (must FAIL):
#   curl --noproxy '*' --max-time 6 https://1.1.1.1        -> no route
#   getent hosts github.com                                 -> no resolution
# Through the proxy (must SUCCEED, cert issued by O=devai-pipelock):
#   curl -sSI https://github.com                            -> 200, SSL verify ok
# DLP (must FAIL=403):
#   curl -X POST https://github.com/x -d 'sk-ant-FAKE...'   -> 403 blocked
```

## Buggy-leak config (deploy/pipelock.yaml)

Generated from `pipelock generate config --preset balanced`, then:

- `forward_proxy.enabled: true` -- required for the lab's CONNECT proxy.
- `tls_interception.enabled: true` + `ca_cert: /config/ca.crt`,
  `ca_key: /config/ca-key.pem`; cloud APIs in `passthrough_domains`.
- `request_body_scanning.action: block` -- the accidental-secret guard.
- `response_scanning.action: warn`, `tool_chain_detection.enabled: false` --
  adversarial scanners stay warn/off (buggy-leak, not hostile).
- `api_allowlist` extended with the dev toolchain destinations.

Validate after edits: `pipelock check --config deploy/pipelock.yaml` (with the
CA mounted at /config/ca.crt + /config/ca-key.pem).

## Agent web search

Only **opencode** has working web search against local models, via its built-in
Exa tool, enabled in the lab image with `ENV OPENCODE_ENABLE_EXA=1`
(`deploy/Dockerfile.lab`). Its queries egress through pipelock like everything
else (and may trip the DLP false-positive above on query content). The other
agents' built-in search is cloud-provider-coupled and does not function against
the local router; giving them search uniformly is a backlog item (the MCP
gateway -- see TODO.md).

## Backlog

Pipelock-related open items live in TODO.md under "Open Items -> pipelock egress
lockdown": MCP search for all agents, making the MCP gateway reachable from the
locked lab, fixing web fetch for the Node-runtime agents (Gemini CLI / Claude
Code), the DLP signed-URL false-positive, and the systemd boot network-create
gap.

## Logging

pipelock logs every decision (allow/block, scanner, MITRE technique, redacted
URL) as structured JSON to stdout, captured automatically by the `devai-logger`
sidecar to `/var/cache/devai/logs/devai-pipelock.log`. The secret value itself
is never logged (redaction on). The signed flight-recorder receipts are a
separate, deferred enhancement (not required for the audit log).

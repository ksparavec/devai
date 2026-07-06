# Security CI

devai ships two GitHub Actions workflows plus Dependabot config. One
gates merges, one never does. This is the operator reference for both,
plus the local commands that run the same checks before you push.

**Verification status:** the workflow YAML is validated locally
(actionlint, golangci-lint, gofmt against the actual `gpu-arbiter/` and
`devai-tools/` modules -- see "Local validation" below). This session
did not push a branch or open a live PR against GitHub Actions, so the
exact check names in "Branch protection" below are read directly from
the job/matrix names in the workflow YAML, not confirmed against a
real Actions run. Confirm them against your own first PR's checks
before wiring branch protection, and correct this doc if GitHub
renders them differently than expected.

## Blocking vs advisory

| Check | Gate | Tool | Catches |
|---|---|---|---|
| gitleaks | blocking | `gitleaks/gitleaks` CLI (installed from a pinned release tag) | committed secrets, scanned against `.gitleaks.toml`'s allowlist |
| hadolint (x4) | blocking | `hadolint/hadolint-action` | Dockerfile anti-patterns, one job per Dockerfile (`Dockerfile.base`, `.lab`, `.router`, `.worker-bootstrap`) |
| golangci-lint (x2) | blocking | `golangci/golangci-lint-action` | Go static analysis (errcheck, staticcheck, govet, ...), one job per module (`gpu-arbiter`, `devai-tools`); `only-new-issues: true` scopes gpu-arbiter's pre-existing findings out so the gate doesn't require a historical cleanup first |
| dependency-review | blocking | `actions/dependency-review-action` | new vulnerable/license-incompatible dependencies introduced by a PR, against both modules' `go.sum` |
| govulncheck (x2) | advisory | `golang.org/x/vuln/cmd/govulncheck` | known Go vulnerabilities reachable from actual call graphs, one job per module |
| CodeQL (x2) | advisory | `github/codeql-action` | semantic code-scanning, `go` and `python` (the `scripts/` tree) |
| Trivy | advisory | `aquasecurity/trivy-action` | OS package + dependency CVEs in the built `devai-router` image |

Blocking checks live in `.github/workflows/security-blocking.yml`,
triggered on `pull_request` against `main`. Advisory checks live in
`.github/workflows/security-advisory.yml`, triggered on both
`pull_request` and `push` to `main`, every job carrying
`continue-on-error: true` so a finding never fails the workflow run --
it shows up in the Actions tab (and CodeQL's results in the repo's
Security tab) without blocking anything.

## Dependabot

`.github/dependabot.yml` covers:

- `gomod` for `/gpu-arbiter` and `/devai-tools`
- `github-actions` for `/` (the workflow files themselves)
- `docker` for `/deploy` (the four Dockerfiles' base images)

**No `pip` ecosystem entry.** devai has no `pyproject.toml` or
`requirements.txt` at a location Dependabot's pip ecosystem recognizes
-- only `requirements-base.txt` (image-build-only, non-standard name,
resolved via `uv` rather than pip directly). Rather than force a
`package-ecosystem: pip` entry that would silently fail to resolve
correctly, this gap is left visible here: Python dependency bumps in
`requirements-base.txt` are a manual review item, not automated.

## Branch protection

Once the workflows have run at least once against a real PR (so
GitHub has recorded the check names), wire the blocking ones as
required status checks:

1. Repo Settings -> Branches -> add/edit a rule for `main`.
2. Enable "Require status checks to pass before merging".
3. Search for and add each blocking check. Based on the job/matrix
   names in `security-blocking.yml`, expect to see:
   - `gitleaks (secret scan)`
   - `hadolint (deploy/Dockerfile.base)`, `hadolint (deploy/Dockerfile.lab)`, `hadolint (deploy/Dockerfile.router)`, `hadolint (deploy/Dockerfile.worker-bootstrap)`
   - `golangci-lint (gpu-arbiter)`, `golangci-lint (devai-tools)`
   - `dependency-review`
4. Do **not** add the advisory workflow's jobs here -- they are
   `continue-on-error` by design and would otherwise misleadingly
   read as "required".

## Local pre-push validation

Containerized, mirrors what runs in CI (adjust `podman` to `docker` if
that's your `CONTAINER_RUNTIME`):

```bash
# Secret scan
podman run --rm -v "$(pwd):/repo:z" -w /repo zricethezav/gitleaks:latest \
    detect --source=/repo --config=/repo/.gitleaks.toml --no-git -v

# Dockerfile lint, one per Dockerfile
podman run --rm -i hadolint/hadolint < deploy/Dockerfile.base
podman run --rm -i hadolint/hadolint < deploy/Dockerfile.lab
podman run --rm -i hadolint/hadolint < deploy/Dockerfile.router
podman run --rm -i hadolint/hadolint < deploy/Dockerfile.worker-bootstrap

# Go lint, one per module
podman run --rm -v "$(pwd)/gpu-arbiter:/src:z" -w /src golangci/golangci-lint:latest golangci-lint run ./...
podman run --rm -v "$(pwd)/devai-tools:/src:z" -w /src golangci/golangci-lint:latest golangci-lint run ./...

# Workflow YAML lint
podman run --rm -v "$(pwd):/repo:z" rhysd/actionlint:latest -color /repo/.github/workflows/*.yml
```

If you have Go installed locally (no container needed), the same
tools install directly:

```bash
go install github.com/rhysd/actionlint/cmd/actionlint@latest
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest
$(go env GOPATH)/bin/actionlint .github/workflows/*.yml
(cd gpu-arbiter && $(go env GOPATH)/bin/golangci-lint run ./...)
(cd devai-tools && $(go env GOPATH)/bin/golangci-lint run ./...)
```

`gpu-arbiter` currently reports pre-existing errcheck/staticcheck
findings under golangci-lint's default rules -- expected, and why CI's
gate uses `only-new-issues: true` (see the table above). `devai-tools`
is lint-clean.

## Reading advisory results

- **govulncheck / Trivy**: check the Actions tab -> the
  `security-advisory` workflow run -> the relevant job's log. Both
  print human-readable findings even though the job is marked
  successful (`continue-on-error`).
- **CodeQL**: findings appear under the repo's Security tab ->
  "Code scanning alerts", not just in the workflow log -- CodeQL
  uploads SARIF regardless of the job's pass/fail status.

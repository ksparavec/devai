# library service (devai side of the digital library)

_devai runs a library service. aiagent decides what to search, what to fetch and what to keep; the service makes every external request, politely, stores what aiagent accepts as plain files, and keeps search indexes that can be rebuilt from those files._

## Status

Draft, 2026-09-26. The owner's decisions for devai's side are locked (below), including the answers to the plan's own open questions (D24-D28). The field-level API contract is aiagent's `docs/design/library-service-needs.md` (in the aiagent repo, a draft for the owner's review, revised on 2026-09-26 for the decisions below). This plan adopts that document as the contract, with three explicit adjustments ("API"), and records devai's answers to its questions for devai. It was reviewed adversarially before the owner saw it (48 confirmed findings, all addressed here). Nothing is implemented.

## Owner decisions (2026-09-25 and 2026-09-26)

Given in devai's session unless marked otherwise.

| # | Decision |
| --- | --- |
| D1 | **The split.** aiagent owns everything that thinks: search planning and queries, classification (laya students), the multi-round loop between encoder and LLM, translation, verdicts, answers with citations, and the versioned research profile. devai owns the library service: all external network I/O, the store, the index, credentials, and model serving. |
| D2 | **A devai service with a small API**, like the laya trainer, not aiagent's own files. |
| D3 | **Consumers:** aiagent only, for now. An MCP server may be added later. |
| D4 | **arXiv: the PDF plus arXiv's HTML rendering, no LaTeX source.** Chosen after the research showed that `arxiv.org/robots.txt` disallows `/e-print` and `/src` for robots. It replaces the first answer ("PDF plus LaTeX source"). |
| D5 | **robots.txt and documented APIs.** A documented API follows its provider's API terms and limits. robots.txt governs every page and file fetch. (`export.arxiv.org/robots.txt` says `Disallow: /`, while arXiv's API terms sanction `/api/query` there; Wikipedia's robots.txt disallows `/w/` and `/api/`, while its bot policy documents those APIs.) |
| D6 | **Sources, in this order:** arXiv; free, independent sources (open-access journals, author pages; SSRN see D8); GitHub repositories that papers link to; carefully selected blogs, forums and Wikipedia. "Independent" means not corporate or government owned. aiagent's O4, answered by the owner in aiagent's session, reads this as who publishes the fetched document: company and government publications are out, and the authors' affiliations are information only. |
| D7 | **Free sources only for now.** Paid sources may be added later. "Well-known" is a metric (for example the h-index), not a curated list. |
| D8 | **Sources whose terms forbid automation or storage** (SSRN, Substack, Kaggle's web pages, Stack Exchange for AI use, Reddit, Drive-hosted Colab notebooks) are used only through an official API whose terms allow it, or as a link. The service first looks for the same work at another open location. |
| D9 | **Logins** (Kaggle, GitHub, Medium, Substack, Reddit, Stack Exchange, Colab) are all wanted eventually. Honour robots.txt and every limit; never get banned. |
| D10 | **Code: no clones, no datasets.** Keep short, relevant excerpts only, each with a permanent link to its exact commit (repository, file, lines) and the repository's licence. Verification level (a), metadata and content checks, is required now but not sufficient. Level (b), sandboxed execution compared against the paper's results, comes later and is the project's edge. |
| D11 | **Relevance has two bounds:** the research profile is the outer bound (CS/AI, computational finance, analysis of PDEs, numerical analysis, probability); each query is the inner bound. |
| D12 | **Rejected items keep a verdict record and no artifacts.** "A negative answer is as valuable as a positive one." |
| D13 | **Verdict stamps.** Every verdict records the profile version and the model or student that decided. Nothing is re-judged automatically; an item is re-judged on request, or when it comes up again. Accepted items are never deleted silently. Some verdicts will go stale, or turn wrong, over time. How "comes up again" works is narrowed by aiagent's O1 (below): a resurfaced item is judged again without an owner override only when its current verdict is an `off_topic` reject for another topic; any other re-judge is an owner-requested override (O8). |
| D14 | **Latest version only**, for papers and for code: a stored repository's excerpts stay pinned to the commit they were taken from; a refresh to a newer commit happens on request, and afterwards only the latest accepted commit's excerpts are kept. |
| D15 | **Storage: one configurable directory**, configured once and used everywhere. Its type, backups and exports are out of this project's scope. Plain files, organised in a hierarchy like git's object store, fast for standard Linux tools, planned for millions of files. |
| D16 | **English only**, for everything stored. A non-English source is machine-translated first (by aiagent), marked as such with the model named; the original is kept as a link, and citations point to the original. |
| D17 | **Search** is keyword plus vector, with an English embedding model on the CPU. Indexes are derived and rebuildable. |
| D18 | **Citations:** section, paragraph and equation, plus the PDF page. PDF pages alone are good enough. |
| D19 | **The research profile** lives in aiagent as a versioned file; the library records which version each verdict used. |
| D20 | **Acquisition runs as background jobs**, and aiagent must show progress and a time estimate. A prompt that stays silent for minutes is unacceptable. |
| D21 | **Contact identity:** a dedicated address, which the owner will create; a placeholder until then. |
| D22 | **Blogs are selected, not crawled.** A blog qualifies when it strictly covers the topics and its author also publishes cited articles or implements algorithms on the platforms above (ideally both). |
| D23 | **The first acceptance test:** the topic "Automatic Adjoint Differentiation method in computational finance". Not arXiv only: other sources are used where feasible. |
| D24 | **The embedding model gets its own volume**: a new store, `/var/cache/devai/embed`, on its own LV (per the mount-point convention), filled only through `scripts/select-models.py` from a pinned catalog row in a new `deploy/embed-models.yaml`. |
| D25 | **Other paths to the sources stay open.** The lab can reach arXiv and other sources directly (through pipelock, the MCP gateway's arXiv, Wikipedia and fetch servers, and trusted harnesses on devai-net); the shared per-operator budget is accepted and documented, and nothing is closed. |
| D26 | **GROBID** is used for PDFs without arXiv HTML (about 4 GB of RAM, its own container). |
| D27 | **The anonymous arXiv PDF mirror** (Cornell's public GCS bucket) is checked in Phase 1 and used only if its terms and robots.txt allow automated downloads. |
| D28 | **Revoked items keep their artifacts for 30 days** in `trash/`, then lose them: a stated exception to D12, as a safety net against a faulty client. |

**Relayed from aiagent's session.** The owner answered aiagent's O1-O14 there on 2026-09-26 with "go with recommendations" (aiagent's session reported it; the owner has not said it in devai's session). Those that shape devai's rules: an `off_topic` reject is scoped to its topic, so the same item may be judged again for another topic, while `out_of_bounds`, `low_quality` and `duplicate` rejects stay final (O1); a new arXiv version is reported as `latest_seen` and refreshed only on request, and the old version's text units are kept while its PDF and HTML are dropped, which is a small, owner-approved exception to D14 so that old citations still resolve (O2); only the owner overrides a verdict (O8); every metadata-only rejection keeps a record (O9); the staging TTL for fetched items that were never judged is 30 days (O11). O13's original recommendation (arXiv only first) is superseded by D23.

**devai's own safety rules** (not owner decisions): the service makes no external request while the contact address is unset; it refuses to run on a library directory that was not initialised explicitly.

## Dependencies

- **aiagent's requirements document** (`docs/design/library-service-needs.md` in devitops-com/aiagent), the field-level contract, and aiagent's client implementation.
- **Owner actions, each blocking the phase named:**
  - the dedicated contact address (D21) on the company's own domain: blocks Phase 1's live exit (no external request without it), and every key application below;
  - an OpenAlex API key: optional, raises the daily budget from $0.10 to $1 (Phase 2);
  - a GitHub fine-grained token (Phase 4); a Hugging Face token (Phase 4); a Semantic Scholar key, requested only when Phase 5 starts because unused keys are pruned after about 60 days; an OpenReview service account (Phase 5);
  - contacting arXiv before the library passes about 1,000 full-text downloads ("Politeness engine").
- **The sops/age secret scaffold** (`docs/secrets.md`, status Non-functional). The library is its first working consumer: from Phase 2 when an OpenAlex key exists, otherwise from Phase 4.
- **pipelock** (`docs/pipelock.md`): the service's outbound traffic goes through it for DLP.
- **The lab launchers** (`bin/devai-agent`, the Makefile), which gain an optional read-only `/library` mount.

## Enables / Unblocks

- The owner's three use cases: a ranked list of important recent articles with critical opinions and verifiable implementations; explanations built only from the library, with every claim cited and a confidence level; software architectures built from published, tested algorithms, with their sources.
- A later sandboxed code-execution job runner (verification level (b)), which reuses the job model.
- A later MCP server over the library, for other agents (D3).

## Out of scope

- **Judgement of any kind.** The service never ranks against the profile or a topic, calls no LLM and runs no classifier (aiagent's principle 1).
- **Translation.** aiagent translates; the service stores the English it submits.
- **The storage medium, backups and exports** (D15).
- **Repository clones and datasets** (D10).
- **Verification level (b)**, sandboxed execution: a separate plan, later.
- **Paid sources** (D7) and **web-search APIs** (corporate, credentialed).
- **Automated access to sources whose terms forbid it** (D8): no scraping, no captcha or Cloudflare workarounds.
- **An MCP server** (D3), for now.

## Verified source facts (2026-09-25/26)

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
| Wikipedia | open, User-Agent with contact | At most 200 requests/min per client (target about 2/s), one in-flight Action API request. API paths allowed under D5. |
| SSRN | link only (D8) | Terms forbid automated queries "of any sort"; Cloudflare challenges every PDF; TDM rights reserved. The same work is looked up by DOI (prefix 10.2139) at other open locations. |
| Substack, Stack Exchange, Reddit, Drive-hosted Colab | link only (D8) | Substack's terms forbid crawling and storing any significant portion (a Substack publication may sit on a custom domain: its feed's generator, hosting headers and CNAME give it away); Stack Exchange's policy bans gathering for AI tools; Reddit disallows everything in robots.txt and needs approval and local deletion sync; Drive-hosted Colab stays link only until the Drive API with the user's consent is researched. GitHub-hosted Colab notebooks go through the GitHub adapter. |
| Kaggle | official API only (D8) | Its terms forbid scraping and storing significant portions. Only through the official API, only notebooks the user names, and stored only under a stated open licence (as D10 requires for code); otherwise a link. |
| Medium | not decided | Not on the owner's restricted list. Its official API is unsupported (archived 2023) and never offered post content; public feeds are allowed by robots.txt; its terms could not be read (Cloudflare). Before Phase 5: a manual read of the terms, then feed-only, on demand, for owner-approved authors. |

## Architecture

### Components

- **`devai-library`**: the service. Python 3.14 (the base image's), a stdlib HTTP server and worker threads in the laya trainer's style, hash-locked dependencies with permissive licences only: `protego` (BSD), `lxml` (BSD), `onnxruntime` (MIT), `tokenizers` (Apache), SQLite (public domain), and a language detector whose model ships inside the package (for example `lingua-language-detector`, Apache-2.0; the choice and licence are confirmed in Phase 1), since the download rule forbids fetching models at run time. No AGPL or GPL code: no PyMuPDF, poppler or pandoc. It never parses a fetched PDF or HTML page itself.
- **`devai-library-extract`**: the extractor. It turns PDF and HTML bytes into units: `pypdfium2` (Apache/BSD) and `pypdf` (BSD) for PDFs, `lxml` for HTML. It has no secrets, no mount of the store and no network except an internal one; bytes come in over HTTP and JSON goes out, and the service validates the JSON before writing anything.
- **`devai-grobid`**: GROBID (Apache-2.0), the CRF image, for PDF structure (sections, paragraphs, formulas, references with DOI and arXiv ids, page coordinates). About 4 GB of RAM. Called with `consolidateHeader=0` and `consolidateCitations=0`, so it never contacts CrossRef or anything else.
- **The embedding model**: `bge-small-en-v1.5` (MIT, 384 dimensions, 512-token input), run with ONNX Runtime on the CPU inside the service. Downloaded only through `scripts/select-models.py`, from a pinned catalog row, into its own store (D24).

None of them is a router backend: none needs the GPU, so the router's GPU exclusivity does not apply.

### Network and egress

- **Networks.** `devai-library` sits on `devai-net` and `devai-lab-egress`, like the MCP gateway, with no host port; the lab reaches it as `http://devai-library:8080`, and the name joins the lab's `NO_PROXY` lists (Makefile `PIPELOCK_NO_PROXY`, `bin/devai-agent` `NO_PROXY_HOSTS`). A third network, `devai-library-internal`, created with `--internal` by `make cache-up` and declared external in compose like the others, is the only network of `devai-library-extract` and `devai-grobid`; only `devai-library` joins it too.
- **No URL ever comes from aiagent** (agreed with aiagent, its L11). Requests name a vetted source and a query, an identifier the service knows how to resolve (`arxiv:`, `doi:`), or an id the service issued (item, version, link). A link found in fetched content or in third-party metadata comes back as a link record, and the service alone decides whether it is fetchable.
- **Every hop is checked.** Automatic redirects are off; each redirect is checked like a first request (host list, robots.txt, the host's limits). IP-literal URLs, internal service names, and loopback, private, link-local and unique-local addresses are refused. A host that is not in the host list is not fetched: admission is default-deny.
- **Egress through pipelock**, so its DLP scans what leaves, including query text that aiagent wrote. The service trusts pipelock's CA. pipelock's header DLP will match real credentials (GitHub tokens, API keys): each is exempted with a per-pattern `exempt_domains` limited to its issuer's API host, in the phase that first sends it; keys travel in headers, never in URLs.
- **pipelock's own answers are not upstream answers.** A pipelock block (DLP) comes back as a 403, and its connect failures as 5xx, on what looks like the upstream connection. The service recognises pipelock's responses (its block body or marker, confirmed in Phase 1) and never counts them as the source's answer, so a DLP hit on one query cannot open a source's breaker: a DLP block is `egress_blocked`, final for that request and never resent unchanged; a pipelock connect failure or 5xx is `egress_unavailable`, a wait retried within the item's retry budget.
- **pipelock does not restrict destinations.** Verified 2026-09-26 from a lab container: `example.com`, `arxiv.org` and `api.openalex.org` all answer 200 through pipelock, although only dev-tool domains are in its `api_allowlist`. So the service enforces its own host list; and other lab tools can still reach arXiv directly, spending the budget arXiv counts per operator, which the owner accepts (D25).

### Politeness engine

Scheduling is keyed on **rate groups**, defined in a versioned file, `deploy/library-sources.yaml`. A group spans the hosts that share one budget: `arxiv-legacy-api` (`export.arxiv.org` and `oaipmh.arxiv.org`: one connection, at least 3 s apart), `arxiv-files` (`arxiv.org`: robots' 15-second crawl delay, one connection), one group per other host. All jobs share a group's queue, first come first served in v1. Each group has:

- kind: `api` (governed by the provider's API terms) or `pages` (governed by robots.txt), D5;
- minimum interval, maximum connections, daily budget (OpenAlex's in dollars, each call priced from its parameters and reconciled against the response's cost and budget headers);
- authentication: none, or a named secret;
- two kinds of breaker, keyed on the provider. A **denial** (an origin 403 from any host) opens a breaker that stays open until an operator clears it; for arXiv it stops all arXiv traffic and raises an alert, since arXiv treats continued requests after a 403 as an attack. **Overload** (sustained origin 429s or 5xx despite waiting, a configured count within a window, for example 5 within 10 minutes) pauses the provider for at least 15 minutes and then clears by itself. A single 429 or 503 is never a breaker event: arXiv sends 429 even at compliant rates.

Waiting on a single 429 or 503: `Retry-After` when given (wait reason `retry_after`), otherwise exponential backoff with jitter (`backoff`), both within the item's retry budget (5 attempts or 30 minutes, aiagent's section 3.1). Only failed attempts count against the budget, and its 30 minutes exclude time spent waiting for a rate slot, an overload pause or a `quota` reset, so a wait never fails an item. Rate-limit headers (OpenAlex's `X-RateLimit-*`, GitHub's, Hugging Face's) update the budgets at runtime. Every request carries `User-Agent: devai-library/<version> (+mailto:<contact>)`.

robots.txt is parsed with Protego and fetched by the service's own wrapper, following RFC 9309 with a stricter 4xx rule: an explicit User-Agent, a timeout, at most 5 redirects (each checked as above), a 500 KiB cap, a 24-hour cache. Outcomes and their reason codes: the rules forbid the path (`robots_disallowed`); robots.txt itself answers 401, 403 or a challenge (for example `cf-mitigated: challenge`), treated as disallow (`robots_denied`); robots.txt answers 429, 5xx or fails on the network, treated as disallow for now, with the cached copy kept and a retry later (`robots_unreachable`); 404, 410 and other 4xx mean "allow".

**Stops, and how they show.** Three conditions stop a source until someone acts: an open denial breaker (`breaker_open`), the missing contact address (`contact_not_configured`, all sources), and the arXiv full-text cap (`arxiv_fulltext_cap`: a configured count below 1,000 papers, each paper counted once whatever its formats; it stops arXiv PDF and HTML downloads only, not search or OAI-PMH metadata, until the owner records that arXiv was contacted). An overload pause and an exhausted daily budget are not stops: they end by themselves, so jobs wait for them (`backoff`, or `quota` until the budget's reset time).

- `/health` and `/v1/info` show a source `down` with its cause when all of its traffic is stopped (a missing contact, an open breaker), and `limited` with its cause when part of it is (the arXiv cap stops downloads, not search); the top-level status is `degraded` whenever any stop holds.
- A job-creating request whose every named source is stopped answers 503 `unavailable` with `error.details.cause`.
- A request naming a stopped source and a working one is accepted (202): the stopped source reports a per-source `error` in the search results, and fetch items that need a stopped source end promptly as `failed` with `upstream_unavailable`, still fetchable. The cause is always in `error.details.cause`, on 503 answers, per-source errors and per-item failures alike.
- A running job whose source stops meanwhile gives `eta: {basis: unknown, reason: source_down}`, and its remaining items for that source end the same way.
- Clearing a denial breaker, or recording the arXiv contact, is an explicit operator command, recorded under `state/`. Counters that must survive a restart (daily budgets, the arXiv total, breaker state) live under `state/`.

### Configuration

- **`LIBRARY_DIR` in `.env`**, set once (D15). Compose and the Makefile read it from `.env`. `bin/devai-agent` does not read `.env` by design, so `make install` stages it as `~/.devai/library`, a symlink to the directory, exactly as it stages the probe caches; the launcher mounts it and prints a warning naming `make install` when it is missing. A test checks that compose, the Makefile and the launcher resolve the same path.
- **Optional everywhere except in the service.** The lab mounts `/library` (read-only) only when the library is configured, as it mounts `/laya` only when that exists. `devai-library`, `devai-library-extract` and `devai-grobid` are in a compose profile, `library`, which `make cache-up` enables only when `LIBRARY_DIR` is set (as it already skips `laya-trainer` without its image); compose gets an inert default for the variable (the `${MCP_SECRETS_FILE:-/dev/null}` pattern), so an unset key never breaks other compose commands. A host that starts the stack without `.env` (the systemd unit, where installed) simply runs without the library.
- **Explicit initialisation.** `make library-init` writes `LAYOUT.json` into an empty directory, recording the layout version and the filesystem's id. The service refuses to start on a directory without `LAYOUT.json`, or whose filesystem id differs, so an unmounted volume or a mistyped path cannot silently become a new, empty library on the root filesystem. Under `/var/cache/devai/` the mount-point convention applies (CLAUDE.md): a new top-level directory there needs its own volume.
- **`LIBRARY_CONTACT`** in `.env`: the contact address (D21). Unset, the service answers but makes no external request.

### The store (`LIBRARY_DIR`)

The service owns the layout; aiagent uses only paths the API returns (its principle 8).

```
<LIBRARY_DIR>/
  LAYOUT.json                    layout version and filesystem id, written by make library-init
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
  trash/<date>/                  artifacts of revoked items, kept 30 days (D28)
  index/                         derived only: alias index, keyword index, embeddings, topic listings
```

- **The plain files are the only source of truth.** An item's current state is the fold of its `records/`; everything under `index/` is derived and can be deleted and rebuilt (aiagent's acceptance check 6).
- **Fan-out.** Two levels of two hex digits, as in git's object store but one level deeper: 65,536 leaf directories, so ten million items or objects leave about 150 entries per directory. XFS and ext4 both index large directories; the fan-out keeps `ls`, `find` and shell globs fast.
- **Writes are atomic** (write to a temporary name, then rename); files are 0444 and directories 0555 once final, as on the laya store. Standard tools work directly: `find items -name key.json`, `jq` over `records/`, `grep` over `units.jsonl`.
- **English only (D16) in what stays.** Permanent records hold English only: for a non-English item, aiagent submits English metadata (`metadata_en`) with its verdict, and English renderings of evidence quotes (`quote_en`); the service stores those and keeps the original only as a link with its sha256. Staging, which is temporary, holds the original-language text until the verdict or the TTL.
- **Rejects and purges.** A reject deletes the item's staged artifacts and texts and writes tombstone records; unit addresses of a purged text answer 410 `text_purged` with the tombstone. A purge after the staging TTL is done by a maintenance job, which emits `item.purged` events like any other job.
- **Revoked items** (an accept turned into a reject) move their artifacts to `trash/<date>/`, where a maintenance job deletes them after 30 days (D28); their record says where they are until then.

### Text extraction

- **Versions are fixed first.** The fetch job resolves the latest version `vN` once, then fetches `/pdf/<id>vN` and `/html/<id>vN`, and the `arXivRaw` record (full version list, withdrawal marker, licence) as one paced request on the legacy-API group. Accept only copies the licence from that stored record; it makes no external request.
- **Format checks before hashing.** A PDF must start with `%PDF-`. A valid HTML page is 200, `text/html`, carries the LaTeXML generator marker, and names version `vN`. A definite "no HTML" is a 404 or 410, or a 200 without the LaTeXML marker: PDF-only staging at once, not a failure, with `quality.warnings` saying why. A 429, 5xx, network failure, or a 200 with the marker but the wrong content type or version is an error, retried within the item's retry budget like the PDF, and PDF-only staging with a warning if the budget runs out. HTML for a PDF-only item is fetched later only on request. Nothing that fails a check is stored or becomes a `sha256:` alias.
- **arXiv HTML is the primary text when it exists** (`origin: html`, `role: primary`): sections and paragraphs from LaTeXML's ids (`anchor.html_id` carries the unit's own id, for example `S3.SS2.p2`; `anchor.paragraph` comes from it; `anchor.block` is null), display equations as TeX with their printed numbers (`equation_number_method: html`; `equation_labels` empty, since arXiv's HTML drops the authors' labels; `refs.labels` point to LaTeXML ids), inline math as `$TeX$` with `math_spans` (offsets including the `$` delimiters), theorem-like environments (`anchor.env`: `{name, label, number}`), captions, and bibliography entries with parsed identifiers (`refs.cites` entries: `{html_id, printed, ids: {doi, arxiv, url}}`); a `heading` unit carries its section's `html_id`. `html_id`, `env`, `refs.cites` and `refs.labels` are v1 for HTML texts and null for PDF texts; section and paragraph anchors come from GROBID for PDF texts. The PDF page of each unit comes from aligning its text with the PDF's text layer (`page_method: text_align`), and the PDF text becomes a `fallback` text (`fallback_text_id`).
- **Licences of non-arXiv works** are recorded at fetch time from the open-access location the copy came from (OpenAlex's `locations[].license`, Unpaywall's `best_oa_location.license`), with `stated_by` naming that source (`stated_by` is one of `arxiv`, `openalex`, `unpaywall`); when none is given, `license.status` is `none_stated`. Accept never makes an external request.
- **Otherwise the PDF through GROBID** (`origin: pdf`): sections, paragraphs, formulas, references and page coordinates (`page_method: pdf_native`), with pypdfium2 as the fallback. A PDF without a text layer fails the fetch with `no_text_layer` (O7: OCR later).
- **Units** follow aiagent's rules (its section 3): structural kinds, at most 1,300 characters, split only when oversized, at sentence boundaries outside math and citations, no overlap, `prev`/`next` links, link records for every link found, and `lang` per unit from the language detector. The service counts characters only.

### Index

- **Keyword:** SQLite FTS5 with the porter/unicode61 tokenizer, a derived file under `index/`, updated incrementally.
- **Vector:** embeddings stored as derived files keyed by (unit text sha256, model sha256), so a rebuild does not re-embed. Search starts as exact search over memory-mapped matrices (float32, later int8); faiss-cpu (MIT) is the next step when measurements call for it. Units longer than the model's 512 tokens are chunked internally; a hit always reports the unit.
- **Hybrid search** fuses the two lists mechanically (reciprocal-rank fusion) and returns the fused score (v1), later the component scores too.
- **Rebuild times** are measured, not promised, and reported by `/v1/info`.

### Jobs and progress

As aiagent's section 8 specifies, in the laya trainer's style: every job-starting request returns 202 at once; `jobs/<id>/job.json` is rewritten atomically at every event, at most once per second, and `events.jsonl` is append-only. Waits carry their reason (`rate_limit`, `crawl_delay`, `retry_after`, `backoff`, `quota`, `concurrency`) and an `until` time. The first ETA comes within 2 s, computed from the schedule (requests left per rate group times the group's interval, plus measured transfer and extraction times), and a heartbeat at least every 5 s. `eta_inputs` and `/v1/info` report per rate group (`per_group`), since one source can span groups with different intervals (arXiv: 3 s and 15 s). After a restart a job resumes: every item step is idempotent, and unfinished items are queued again. For scale: 25 arXiv papers as PDF and HTML are about 50 files at arXiv's 15-second crawl delay, so about 12-13 minutes of waiting (the PDF mirror, if D27's check allows it, would halve it).

### API

aiagent's section 9.6 lists the endpoints and marks what v1 needs; this plan adopts it with three adjustments, which aiagent's document reflects:

1. arXiv artifacts are `pdf` and `html`, never `eprint` (D4).
2. Sources restricted by D8 come back as candidates or link records with a not-fetchable reason (`terms_disallow_automation`) and their link.
3. **HTML-primary text in v1**, as described under "Text extraction", instead of PDF-only text: `origin: html`, `page_method: text_align`, paragraph anchors from LaTeXML, `math_spans`, printed equation numbers. This serves the reason for D4 (exact formulas for PDE and numerical analysis work).

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

### devai's answers to aiagent's questions (O-D1 to O-D13)

| # | Answer |
| --- | --- |
| O-D1 | A plain compose container, not a router backend: `http://devai-library:8080` on devai-net and devai-lab-egress. |
| O-D2 | The library root is mounted read-only at `/library` in the lab when configured (later `inbox/` read-write), from `LIBRARY_DIR` (staged for `bin/devai-agent` by `make install`). Secrets never live in the library directory. `state/` and `index/` live under the root; aiagent must not depend on their files. |
| O-D3, O-D11 | Replaced by arXiv HTML (D4): paragraph and equation anchors and printed numbers come from LaTeXML's output; the authors' `\label` keys are not available. PDFs without HTML go through GROBID. |
| O-D4 | Dropped: the service counts no tokens. |
| O-D5 | `bge-small-en-v1.5`; embeddings cached by (unit text sha256, model sha256); exact search first; rebuild times measured. |
| O-D6 | SQLite FTS5, porter/unicode61, derived. |
| O-D7 | The arXiv legacy APIs (search and OAI-PMH) share one queue: one request per 3 s on one connection. arXiv files: robots' 15-second crawl delay, one queue. No e-print (D4). |
| O-D8 | OpenAlex (discovery, open-access locations, citations, advisory author metrics), Unpaywall by DOI; SSRN link only (D8); Semantic Scholar later, once a key is granted. |
| O-D9 | Jobs resume after a restart; per-group queues are first come, first served across jobs in v1. |
| O-D10 | Yes: revoked items' artifacts stay in `trash/` for 30 days (D28, an owner-approved exception to D12). |
| O-D12 | `devai-library/<version> (+mailto:<contact>)`; robots.txt as described above; candidates report `access.robots_allowed` with the reason. |
| O-D13 | Later: author pages from OpenAlex open-access locations, references parsed by GROBID and LaTeXML, and ORCID; blogs from an owner-approved list (D22). No web-search API. |

## Phases

### Phase 1 -- store, service core, politeness, arXiv

**Goal:** a running service that searches arXiv and fetches PDFs and HTML politely, stages English text units, and records verdicts, with jobs, progress and ETA.

**Deliverables:**
- Configuration: `LIBRARY_DIR` and `LIBRARY_CONTACT` in `.env.example`; the compose profile with the inert default; `make install` staging `~/.devai/library`; the optional lab mount in the Makefile and `bin/devai-agent`; `make library-init`.
- Images and services: `deploy/Dockerfile.library` and `deploy/Dockerfile.library-extract` (hash-locked), `devai-library`, `devai-library-extract`, `devai-grobid`, the `devai-library-internal` network; `make build-library`, `make test-library`.
- `deploy/library-sources.yaml` with the arXiv rate groups; the politeness engine, the RFC 9309 wrapper with the stricter 4xx rule, the redirect and address checks, the breakers, the stops, recognition of pipelock's responses.
- The store: layout, append-only records and their fold, alias resolution with a single writer, tombstones, the maintenance job (staging TTL, and the 30-day trash of revoked items, D28).
- The arXiv adapter: search (GET, cached 24 h), version resolution, PDF, HTML, `arXivRaw` in the fetch job; format checks. The check of the anonymous PDF mirror's terms and robots.txt (D27); the mirror is wired in only if both allow it.
- Extraction: HTML-primary units with text-aligned pages, GROBID for PDFs without HTML, the language detector.
- The v1 endpoints except enrichments and library search: verdicts with evidence checks, the one-current-verdict rule with O1's scoped exception, topic assessments, tombstones and 410.
- Jobs: events, waits, ETA per rate group, heartbeat, cancel, resume.
- Docs: `docs/library.md` (reference), CLAUDE.md, `docs/pipelock.md` (the finding that it does not restrict destinations).

**Exit criteria:**
- Unit tests with a fake upstream: intervals and crawl delays per rate group (search and OAI-PMH never overlap); 429 and 503 with and without `Retry-After` (a wait, never a breaker); sustained 429s (the self-clearing overload pause) and an origin 403 (the denial breaker, cleared only by the operator); a fake pipelock block that must not open arXiv's breaker; robots outcomes (`robots_disallowed`, `robots_denied`, `robots_unreachable`, 404 as allow) and the D5 exemption; the stops' 503 and per-source error shapes; a cross-host redirect and a redirect to an internal name, both refused; every per-item outcome in aiagent's section 3.1; HTML present, absent and broken; verdict rules and the fold; the stops as `/health` and ETA show them; `LIBRARY_DIR` unset, set, and set but uninitialised.
- A layout test at scale: synthetic items and objects in the hundreds of thousands, timing `find` and a full fold.
- A live smoke run, once the contact address exists: one small arXiv query, a fetch of two papers (one with HTML, one without), units with anchors and pages, a reject that purges and answers 410, an accept that keeps, all within arXiv's limits.

### Phase 2 -- OpenAlex, open-access locations, index

**Goal:** sources beyond arXiv (D23), and search over the library.

**Entry:** a manual read of OpenAlex's Terms of Service in an ordinary browser.

**Deliverables:**
- The OpenAlex adapter: discovery (structural filters where they fit, text matching priced as search), open-access locations, `cited_by_count` and citing works, advisory author metrics with provenance (source, as-of date, match method), the dollar budget; `POST /v1/enrichments` and `GET /v1/candidates/{id}`.
- Unpaywall by DOI. Fetchable link records for open-access PDFs on allowed hosts (each host added to `library-sources.yaml` with its own robots and limits); "known, not held" link records otherwise, SSRN included (D8).
- The secret path, if an OpenAlex key exists: sops/age, tmpfs, mounted into `devai-library` only, the per-pattern DLP exemption for OpenAlex's host. Without a key, Phases 2-3 run keyless at $0.10/day, and the acceptance test is sized to that.
- The keyword and vector indexes, library search; the embedding model's store (D24: `/var/cache/devai/embed` on its own LV, set up with `deploy/setup-logs-volume.sh`), its catalog row in `deploy/embed-models.yaml` and its pull through `scripts/select-models.py`.
- The rebuild command: delete `index/` and rebuild it from the plain files.

**Exit criteria:** candidates from OpenAlex merge with arXiv records by identifier; a non-arXiv open-access PDF is fetched from an allowed host; an SSRN work comes back as a link with its reason; after a rebuild from the plain files the topic listing is byte-identical and every alias resolves as before (aiagent's check 6).

### Phase 3 -- the acceptance test, with aiagent

**Goal:** the owner's test (D23) end to end, as aiagent's section 10 walks it.

**Exit criteria:** aiagent's checks 1-7 pass on the AAD topic: check 2 as revised (a PDF, plus the arXiv HTML when the fetch staged it, otherwise a `quality.warnings` reason, and a license field, which may be `none_stated` for old arXiv papers), and check 7 (the run searches OpenAlex as well as arXiv, and its results hold at least one non-arXiv work, either an open-access copy fetched from an allowed host or a "known, not held" link with its reason). Progress never goes silent for more than 5 s. The measured times are recorded here.

### Phase 4 -- code (verification level (a))

**Goal:** GitHub repositories that papers link to (D6, D10).

**Deliverables:** the GitHub adapter (fine-grained token, the documented retry ladder, commit pinning, licence detection, the tree with dataset-like files refused, fetches of chosen files only); code links from Hugging Face's papers API (token, `githubRepoAddedBy` recorded with each link, so user-set links are not treated as official) and from the papers themselves; excerpts stored as aiagent's section 3.6 specifies (at most 80 lines each, 10 per repository, licence required, O3), refreshed per D14; the level (a) checks as data; per-pattern DLP exemptions for GitHub's and Hugging Face's API hosts.

### Phase 5 -- selected community sources

**Goal:** the remaining sources, within their terms (D8, D9, D22).

**Deliverables:** the Wikipedia API (revision-pinned, cited by `oldid`); owner-approved blogs through their feeds, each blog's hosting platform classified first, so a Substack-hosted blog on its own domain inherits Substack's D8 status; Semantic Scholar citation contexts (key requested at the start of this phase; storage of each field only under its licence); OpenReview reviews (a service account); Kaggle's API for notebooks the user names (stored only under a stated open licence); GitHub-hosted Colab notebooks; Medium after the terms are read. Link records for everything else.

### Later, separate plan -- verification level (b)

Sandboxed execution of chosen code, compared against the paper's reported results, as a devai job runner behind the router (it needs the GPU, so it takes the laya trainer's hold). It reuses the job model.

## Open questions

None for the owner: the plan's five open questions were answered on 2026-09-26 (D24-D28). Phase 1 still checks the arXiv PDF mirror's terms (D27) and confirms the language detector's licence.

## Combined risk register

| Risk | Phase | Mitigation |
| --- | --- | --- |
| A ban from arXiv (403) after a policy mistake. | 1 | Rate groups, the crawl delay, the arXiv breaker, persisted counters, the full-text cap until arXiv is contacted. |
| A pipelock DLP block mistaken for a source's denial. | 1 | pipelock's responses recognised and recorded as `egress_blocked`; a fake-block test. |
| Other lab paths spend arXiv's per-operator budget. | 1 | Accepted by the owner (D25); documented in `docs/library.md`; the breakers stop the library quickly if arXiv pushes back. |
| Limits change without notice (OpenAlex changed its model in 2026; arXiv's docs disagree with each other). | all | Every limit is configuration; runtime headers win; the first 429 from any host is logged and alerted. |
| Parser exploits in PDFs or HTML. | 1 | Parsing only in `devai-library-extract` and GROBID: no secrets, no store, internal network only; the service validates their JSON. |
| The service used to reach internal hosts (SSRF). | 1 | No URLs from aiagent; every redirect hop checked; internal names and private addresses refused; default-deny host admission. |
| arXiv HTML missing or broken for a paper. | 1 | Format checks; GROBID over the PDF; `quality.warnings` in `text.json`. |
| An unmounted volume becomes an empty library. | 1 | `make library-init`, `LAYOUT.json` and the filesystem id checked at every start. |
| Real credentials blocked by pipelock's DLP. | 2, 4, 5 | Per-pattern exemptions limited to each issuer's API host, added in the phase that sends the credential. |
| Author metrics wrong (split or merged profiles). | 2 | Advisory only, with source, as-of date and match method (O5). |
| Millions of files slow the tools. | 1 | Two-level fan-out, measured in the layout test; indexes derived. |
| Licence contamination of the service. | 1 | Permissive dependencies only. |
| Credentials leaking into the lab. | 2, 4, 5 | Secrets only in `devai-library`, from tmpfs; never in the library directory. |

## Estimated effort

| Phase | Scope | Estimate |
| --- | --- | --- |
| 1 | configuration, store, core, politeness, arXiv, extraction, jobs | 5-6 days |
| 2 | OpenAlex, Unpaywall, enrichments, index, rebuild | 2-3 days |
| 3 | acceptance test with aiagent | 1 day (plus aiagent's side) |
| 4 | GitHub, code links, excerpts, secrets | 2-3 days |
| 5 | community sources | 3-5 days |

## References

- aiagent: `docs/design/library-service-needs.md` (devitops-com/aiagent), the field-level contract.
- arXiv: API terms https://info.arxiv.org/help/api/tou.html; user manual https://info.arxiv.org/help/api/user-manual.html; https://arxiv.org/robots.txt and https://export.arxiv.org/robots.txt; robots guidance https://info.arxiv.org/help/robots.html; per-article downloads https://info.arxiv.org/help/ir.html; bulk data https://info.arxiv.org/help/bulk_data.html; OAI-PMH https://info.arxiv.org/help/oa/index.html; identifiers https://info.arxiv.org/help/arxiv_identifier_for_services.html; media types https://info.arxiv.org/help/mimetypes.html; the 2025-11 API replacement https://groups.google.com/a/arxiv.org/g/api/c/-WpHNxbaxU0.
- OpenAlex: authentication https://help.openalex.org/api/authentication/; pricing https://help.openalex.org/access/pricing/; searching https://help.openalex.org/api/searching/; semantic search https://help.openalex.org/api/semantic-search/; deprecations https://help.openalex.org/guides/deprecations.
- Unpaywall: https://unpaywall.org/products/api and https://unpaywall.org/legal/terms-of-service.
- Semantic Scholar: https://www.semanticscholar.org/product/api, its licence https://www.semanticscholar.org/product/api/license, release notes https://github.com/allenai/s2-folks/blob/main/API_RELEASE_NOTES.md. OpenReview: https://docs.openreview.net/getting-started/using-the-api and https://openreview.net/legal/terms.
- SSRN: https://www.ssrn.com/index.cfm/en/terms-of-use/ and https://papers.ssrn.com/robots.txt.
- Wikimedia: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy, https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits, https://wikitech.wikimedia.org/wiki/Robot_policy.
- Tooling: RFC 9309 https://www.rfc-editor.org/rfc/rfc9309.txt; Protego https://github.com/scrapy/protego; pypdfium2 https://github.com/pypdfium2-team/pypdfium2; GROBID https://grobid.readthedocs.io/en/latest/Grobid-docker/ and https://github.com/kermitt2/grobid/blob/master/doc/Coordinates-in-PDF.md; LaTeXML https://github.com/brucemiller/LaTeXML.

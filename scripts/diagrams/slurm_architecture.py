#!/usr/bin/env python3
"""Draw docs/slurm-architecture.svg: the proposed Slurm gatekeeper architecture.

Every component and every interaction; the numbers on the arrows are the rows
of the Interactions table in docs/slurm.md Sec. 2. Coordinates are explicit
(a hand layout) because Graphviz scattered this many cross-cluster edges.

    python3 scripts/diagrams/slurm_architecture.py            # writes the SVG
    python3 scripts/diagrams/slurm_architecture.py -o out.svg

Stdlib only; output is plain SVG 1.1 with ASCII text.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1600, 1140
FONT = "Helvetica, Arial, sans-serif"

# Interaction colours (also the legend).
BLUE, ORANGE, PURPLE, GREEN, GREY, RED = "#2F6FD1", "#D98A1C", "#7A52B3", "#2E8B3E", "#7A7A7A", "#C73A3A"
COLOURS = {"data": BLUE, "control": ORANGE, "slurm": PURPLE, "gpu": GREEN, "storage": GREY, "kill": RED}

out: list[str] = []


def text(x, y, s, size=11, weight="normal", colour="#222", anchor="start", style=""):
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
               f'fill="{colour}" text-anchor="{anchor}"{style}>{escape(s)}</text>')


def zone(x, y, w, h, label, fill, stroke, dashed=False, label_colour="#333", label_dx=14):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.4"{dash}/>')
    text(x + label_dx, y + 20, label, size=12, weight="bold", colour=label_colour)


def box(x, y, w, h, title, lines=(), fill="#FFFFFF", stroke="#666", width=1.4, subtitle=""):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
    cy = y + 19
    text(x + 10, cy, title, size=12.5, weight="bold")
    if subtitle:
        text(x + w - 10, cy, subtitle, size=11, colour="#555", anchor="end")
    for line in lines:
        cy += 14.5
        text(x + 10, cy, line, size=10.8, colour="#333")


def folder(x, y, w, h, title, lines=(), width=1.4):
    tab = f'M{x} {y + 8} v-6 a2 2 0 0 1 2 -2 h{min(70, w // 3)} l8 8 z'
    out.append(f'<path d="{tab}" fill="#FFFFFF" stroke="#777" stroke-width="{width}"/>')
    box(x, y + 8, w, h - 8, title, lines, stroke="#777", width=width)


def cylinder(x, y, w, h, title, lines=()):
    ry = 8
    out.append(f'<path d="M{x} {y + ry} v{h - 2 * ry} a{w / 2} {ry} 0 0 0 {w} 0 v{-(h - 2 * ry)}" fill="#EFE7F8" stroke="{PURPLE}" stroke-width="1.4"/>')
    out.append(f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="#EFE7F8" stroke="{PURPLE}" stroke-width="1.4"/>')
    cy = y + ry + 22
    text(x + 12, cy, title, size=12.5, weight="bold")
    for line in lines:
        cy += 14.5
        text(x + 12, cy, line, size=10.8, colour="#333")


def gpu_box(x, y, w, h, title, lines):
    d = 10
    out.append(f'<path d="M{x} {y + d} l{d} {-d} h{w} v{h} l{-d} {d} z" fill="#CFEACF" stroke="{GREEN}" stroke-width="2"/>')
    box(x, y + d, w, h, title, lines, fill="#E3F4E3", stroke=GREEN, width=2)


def arrow(points, kind, n=None, badge_at=None, label="", label_at=None, anchor="start",
          both=False, dashed=None, width=1.8):
    colour = COLOURS[kind]
    if dashed is None:
        dashed = kind in ("storage", "kill")
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    d = "M" + " L".join(f"{px} {py}" for px, py in points)
    start = f' marker-start="url(#s-{kind})"' if both else ""
    out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"{dash} '
               f'marker-end="url(#e-{kind})"{start}/>')
    if n is not None:
        bx, by = badge_at if badge_at else points[len(points) // 2]
        out.append(f'<circle cx="{bx}" cy="{by}" r="9.5" fill="#FFFFFF" stroke="{colour}" stroke-width="1.6"/>')
        text(bx, by + 3.8, str(n), size=10.5, weight="bold", colour=colour, anchor="middle")
    if label:
        lx, ly = label_at
        for i, part in enumerate(label.split("\n")):
            text(lx, ly + i * 13, part, size=10.5, colour=colour, anchor=anchor)


def markers():
    out.append("<defs>")
    for kind, colour in COLOURS.items():
        out.append(f'<marker id="e-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                   f'<path d="M0 0 L10 5 L0 10 z" fill="{colour}"/></marker>')
        out.append(f'<marker id="s-{kind}" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                   f'<path d="M10 0 L0 5 L10 10 z" fill="{colour}"/></marker>')
    out.append("</defs>")


def draw() -> str:
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    markers()
    out.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    text(W / 2, 30, "devai Slurm gatekeeper -- proposed architecture", size=20, weight="bold", anchor="middle")
    text(W / 2, 50, "single host, rootless podman, Slurm 26.05; one GPU shown (one license per GPU). "
         "Numbered arrows = Interactions table, docs/slurm.md Sec. 2", size=12, colour="#555", anchor="middle")

    # ---------------------------------------------------------- row A: clients
    box(40, 104, 250, 64, "Browser (LAN)", ["Open WebUI over HTTPS :8443"], fill="#F4F4F4")
    zone(330, 72, 520, 128, "devai-lab-egress  (internal network, no internet)", "#F4F8FD", "#8AA4C8")
    box(350, 100, 480, 86, "Lab containers", [
        "agents: claude, codex, opencode, pi, dsh, dstui, aiagent",
        "talk only to the router; no GPU device (convention D1)",
        "aiagent: fine-tuning jobs through :11438"], fill="#E8F0FC", stroke=BLUE)
    # legend
    zone(880, 72, 420, 128, "Legend", "#FFFFFF", "#BBBBBB")
    for i, (kind, what) in enumerate([("data", "inference data path"), ("control", "control: router / operator -> Slurm"),
                                      ("slurm", "Slurm internal RPC / accounting"), ("gpu", "GPU use (CUDA, NVML)"),
                                      ("storage", "volume mount / file write"), ("kill", "GPU guard kills a stray holder")]):
        cx, cy = 900 + (i // 3) * 200, 108 + (i % 3) * 30
        arrow([(cx, cy), (cx + 36, cy)], kind)
        text(cx + 44, cy + 4, what, size=10.2, colour="#333")
    box(1330, 96, 230, 92, "Operator (host shell)", [
        "devai-jobs: history, queue,", "suspend / resume, interrupt", "make: cache-up, bench, probe"], fill="#F4F4F4")
    text(1445, 212, "10: REST, scontrol, result files", size=10.2, colour=ORANGE, anchor="middle")

    # -------------------------------------------------------- row B: devai-net
    zone(20, 225, 1560, 690, "devai-net  10.89.0.0/24   (Slurm addresses fixed: .200-.203 control plane, .210-.220 nodes)",
         "#FCFCF8", "#B9B08A", label_dx=600)
    box(40, 262, 250, 56, "devai-webui-proxy", ["-> devai-open-webui"], fill="#E8F0FC", stroke=BLUE)
    box(420, 248, 560, 124, "devai-router  (gpu-arbiter)", [
        ":11434 ollama  :11435 vllm  :11436 sglang  :11437 vllm-devai  :11438 laya-trainer",
        "request path (kept): rewrites, admission, drain, SSE keepalive, breaker",
        "launch layer (new): Slurm client -- REST + self-signed JWT",
        "engine state read from Slurm; holds for workload jobs",
        "fine-tuning API on :11438 -> trainer jobs"], fill="#FFF3DC", stroke=ORANGE, width=2.2)
    box(1060, 262, 230, 56, "devai-logger", ["podman logs of every devai-* container"], fill="#F4F4F4")
    box(1320, 262, 240, 56, "devai-model-status (MCP)", ["get_job_history (planned)"], fill="#F4F4F4")

    # nodes
    zone(40, 420, 960, 410, "Slurm nodes = backend engine image + slurmd, configless  (privileged, --pid=host, GPU via CDI)",
         "#EEF7EE", "#4F9A5A")
    gx = [60, 365, 670]
    node = dict(fill="#E0F1E0", stroke="#4F9A5A")
    box(gx[0], 470, 290, 84, "devai-ollama", ["engine: ollama serve  :11434", "image: devai-ollama (Debian trixie)"], subtitle=".210", **node)
    box(gx[1], 470, 290, 84, "devai-vllm", ["engine: vLLM 0.22.1  :11434", "image: vllm-openai (Ubuntu 24.04)",
                                          "+ one node per image override"], subtitle=".211", **node)
    box(gx[2], 470, 290, 84, "devai-vllm-devai", ["engine: vLLM 0.28 + HyperQwen  :11434", "image: devai/vllm-devai (Debian trixie)"],
        subtitle=".212", **node)
    box(gx[0], 574, 290, 84, "devai-sglang", ["engine: SGLang 0.5.16  :11434", "image: lmsysorg/sglang (Ubuntu 24.04)"], subtitle=".213", **node)
    box(gx[1], 574, 290, 84, "devai-laya-trainer", ["trainer job: laya fine-tune", "image: devai-laya-trainer (Ubuntu 24.04)"],
        subtitle=".214", **node)
    box(gx[2], 574, 290, 84, "devai-slurm-work", ["no GPU; workload jobs: bench, probe, tests", "image: devai-lab-cpu (Debian trixie)"],
        subtitle=".220", fill="#FFFFFF", stroke="#4F9A5A")
    box(60, 690, 440, 118, "GPU guard  (prolog / epilog of every GPU job)", [
        "prolog: card idle? else SIGTERM, then SIGKILL the holder;",
        "   still busy -> fail: node drained, job held, router resubmits",
        "prolog: wipe vLLM compile cache; record NVML energy",
        "epilog: this job's leftovers only; record NVML energy",
        "(a global idle check here would race the next engine)"], fill="#FFFBE6", stroke="#B8A04A")
    box(520, 690, 440, 118, "Job contract", [
        "engine job: exec <engine argv> built from the router's",
        "   launch config (flags, env, parsers, MTP, recovery flags)",
        "environment: /run/devai/node-env.sh + job-specific vars",
        "engine nodes: gres/gpu:1; license gpu0:1 is cluster-wide",
        "log, results, GPU energy -> jobs/results/<jobid>/"], fill="#FFFFFF", stroke="#4F9A5A")

    # control plane
    zone(1100, 420, 460, 480, "Slurm control plane  (SchedMD images)", "#F5F0FB", PURPLE)
    cp = dict(fill="#EFE7F8", stroke=PURPLE)
    box(1130, 460, 400, 64, "devai-slurmrestd", ["REST API v0.0.44, auth/jwt"], subtitle=".202:6820", **cp)
    box(1130, 560, 400, 84, "devai-slurmctld", ["queue, priorities, license gpu0:1, holds",
                                                 "configless config for the nodes",
                                                 "state -> jobs/slurmctld/"], subtitle=".200:6817", width=2.2, **cp)
    box(1130, 684, 400, 64, "devai-slurmdbd", ["accounting: TRES incl. gres/gpumem, gres/gpuutil"], subtitle=".201:6819", **cp)
    cylinder(1130, 790, 400, 90, "devai-slurm-db  (MariaDB 11.8)", [".203:3306; data -> jobs/mariadb/"])

    # -------------------------------------------------- row C: GPU and storage
    gpu_box(40, 990, 380, 70, "GPU 0", ["RTX PRO 4000 Blackwell, 24 GB", "one GPU job at a time (license gpu0:1)"])
    out.append(f'<path d="M455 1000 h110 l18 18 v34 l-18 18 h-110 l-18 -18 v-34 z" fill="#FDECEC" stroke="{RED}" '
               f'stroke-width="1.6" stroke-dasharray="5 3"/>')
    text(510, 1030, "GPU process", size=11, weight="bold", colour=RED, anchor="middle")
    text(510, 1045, "outside Slurm", size=11, weight="bold", colour=RED, anchor="middle")
    text(510, 1060, "(not allowed)", size=10, colour=RED, anchor="middle")
    zone(640, 958, 940, 166, "/var/cache/devai   (host volumes)", "#F6F6F6", "#999999")
    folder(660, 988, 220, 76, "model stores", ["ollama/ vllm/ sglang/ laya/", "+ vLLM parser plugins"])
    folder(900, 988, 220, 76, "engine-cache volumes", ["FlashInfer, SGLang: kept", "vLLM compile cache: wiped"])
    folder(1140, 988, 420, 120, "jobs/   (new volume)", [
        "results/<jobid>/   result.json, gpu.json,",
        "                   engine / job logs, artifacts",
        "mariadb/           history database (slurmdbd)",
        "slurmctld/         scheduler state",
        "backed up by devai-backup"], width=2.2)

    # ----------------------------------------------------------------- arrows
    # 1 inference
    arrow([(165, 168), (165, 262)], "data", 1, (165, 215))
    arrow([(590, 186), (590, 248)], "data", 1, (590, 212),
          "inference API: OpenAI, Anthropic, Ollama", (606, 216))
    arrow([(290, 290), (420, 290)], "data", 1, (355, 290))
    # 2 proxied requests
    arrow([(560, 372), (560, 420)], "data", 2, (560, 396), "proxied requests to <node>:11434", (546, 400), anchor="end", width=2.4)
    # 3 router -> slurmrestd
    arrow([(980, 340), (1330, 340), (1330, 460)], "control", 3, (1060, 340),
          "submit / cancel / signal / state (REST + JWT)", (1078, 334), width=2.2)
    # 4 restd -> ctld
    arrow([(1330, 524), (1330, 560)], "slurm", 4, (1330, 542), "RPC (auth/slurm)", (1346, 546))
    # 5 ctld <-> nodes
    arrow([(1130, 602), (1000, 602)], "slurm", 5, (1050, 602), both=True, width=2.2)
    text(1050, 628, "launch, signal,", size=10, colour=PURPLE, anchor="middle")
    text(1050, 641, "kill, config,", size=10, colour=PURPLE, anchor="middle")
    text(1050, 654, "node health", size=10, colour=PURPLE, anchor="middle")
    # 6, 7 accounting
    arrow([(1330, 644), (1330, 684)], "slurm", 6, (1330, 664), "accounting", (1346, 668))
    arrow([(1330, 748), (1330, 790)], "slurm", 7, (1330, 769), "SQL", (1346, 773))
    # 8 GPU
    arrow([(220, 830), (220, 990)], "gpu", 8, (220, 905), "CUDA (engine job);\nNVML (guard, accounting)", (234, 945), width=2.4)
    # 9 workload -> router
    arrow([(960, 616), (985, 616), (985, 400), (940, 400), (940, 372)], "data", 9, (985, 500), dashed=True)
    text(930, 394, "workload requests; hold / release", size=10.5, colour=BLUE, anchor="end")
    # 10 operator
    arrow([(1560, 142), (1572, 142), (1572, 492), (1530, 492)], "control", 10, (1572, 380))
    arrow([(1572, 492), (1572, 602), (1530, 602)], "control", 10, (1572, 560), dashed=True)
    arrow([(1572, 602), (1572, 1060), (1560, 1060)], "storage", 10, (1572, 1000))
    # 11 MCP history
    arrow([(1440, 318), (1440, 460)], "slurm", 11, (1440, 392), "history", (1456, 396), dashed=True)
    # 12 guard kills stray holders
    arrow([(480, 808), (480, 995)], "kill", 12, (480, 900), "SIGTERM, SIGKILL\n(else drain the node)", (496, 896))
    # 13 mounts
    arrow([(770, 988), (770, 830)], "storage", 13, (770, 905), "mounts", (786, 909))
    arrow([(980, 988), (980, 830)], "storage", 13, (980, 905))
    # 14 results
    arrow([(1000, 815), (1080, 815), (1080, 940), (1220, 940), (1220, 996)], "storage", 14, (1080, 880),
          "results, logs, GPU energy", (1096, 934))
    arrow([(1330, 900), (1330, 996)], "storage", 15, (1330, 925), "DB files, scheduler state", (1346, 929))

    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", default=str(Path(__file__).resolve().parents[2] / "docs" / "slurm-architecture.svg"))
    args = ap.parse_args()
    Path(args.output).write_text(draw(), encoding="ascii")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

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

W, H = 1500, 1090
FONT = "Helvetica, Arial, sans-serif"

# Interaction colours (also the legend).
BLUE, ORANGE, PURPLE, GREEN, GREY, RED = "#2F6FD1", "#D98A1C", "#7A52B3", "#2E8B3E", "#7A7A7A", "#C73A3A"
COLOURS = {"data": BLUE, "control": ORANGE, "slurm": PURPLE, "gpu": GREEN, "storage": GREY, "kill": RED}

out: list[str] = []


def text(x, y, s, size=11, weight="normal", colour="#222", anchor="start", style=""):
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
               f'fill="{colour}" text-anchor="{anchor}"{style}>{escape(s)}</text>')


def zone(x, y, w, h, label, fill, stroke, dashed=False, label_colour="#333", label_dx=14, label_bottom=False):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.4"{dash}/>')
    text(x + label_dx, y + h - 10 if label_bottom else y + 20, label, size=12, weight="bold", colour=label_colour)


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
    text(W / 2, 50, "single host, rootless podman, Slurm 26.05, one GPU; no Slurm in the backend containers. "
         "Numbered arrows = the table in docs/slurm.md Sec. 2", size=12, colour="#555", anchor="middle")

    # ---------------------------------------------------------- row A
    box(20, 92, 310, 88, "Operator (host shell)", [
        "devai-jobs: history, queue, suspend / resume,", "interrupt, result files",
        "make: build, cache-up, bench, probe"], fill="#F4F4F4")
    zone(360, 72, 430, 120, "Legend", "#FFFFFF", "#BBBBBB")
    for i, (kind, what) in enumerate([("data", "inference data path"), ("control", "control: router / operator -> Slurm"),
                                      ("slurm", "Slurm internal / podman exec"), ("gpu", "GPU use (CUDA)"),
                                      ("storage", "volume mount / file write"), ("kill", "GPU guard kills a holder")]):
        cx, cy = 378 + (i // 3) * 214, 108 + (i % 3) * 28
        arrow([(cx, cy), (cx + 34, cy)], kind)
        text(cx + 42, cy + 4, what, size=10.2, colour="#333")
    zone(820, 72, 650, 120, "devai-lab-egress  (internal network, no internet)", "#F4F8FD", "#8AA4C8")
    box(840, 98, 610, 80, "Lab containers  (lab image)", [
        "agents: claude, codex, opencode, pi, dsh, dstui, aiagent",
        "no GPU device; they talk only to the router (D1)"], fill="#E8F0FC", stroke=BLUE)

    # -------------------------------------------------------- devai-net
    zone(20, 215, 1450, 625, "devai-net", "#FCFCF8", "#B9B08A")
    box(560, 240, 640, 90, "devai-router  (own unprivileged container, small image)", [
        ":11434 ollama    :11435 vllm    :11436 sglang    :11438 laya-trainer",
        "request path unchanged; launch layer = Slurm client (REST + self-signed JWT)",
        "reads the GPU holder from Slurm; holds for workload jobs; no podman socket"],
        fill="#FFF3DC", stroke=ORANGE, width=2.2)

    # devai-slurm: all of Slurm, one image, one container
    zone(40, 360, 460, 470, "devai-slurm  (one image, one container)", "#F5F0FB", PURPLE, label_dx=160)
    text(200, 397, "privileged, --pid=host, GPU (NVML only), podman socket", size=10.2, colour=PURPLE)
    cp = dict(fill="#EFE7F8", stroke=PURPLE)
    box(60, 410, 420, 42, "slurmrestd  :6820", ["REST API v0.0.44, auth/jwt"], **cp)
    box(60, 475, 420, 60, "slurmctld", ["queue, priorities, holds; state -> jobs/slurmctld/"], width=2.2, **cp)
    box(60, 560, 420, 40, "slurmdbd", ["accounting, job scripts, comments"], **cp)
    cylinder(60, 620, 420, 70, "MariaDB  (Debian package)", ["data -> jobs/mariadb/"])
    box(60, 712, 420, 108, "slurmd  --  the only node  (Gres=gpu:1)", [
        "one GPU job at a time; runs each job script",
        "job = podman exec <container> devai-run <jobid> ...",
        "guard (prolog / epilog): idle card, kill holder, or drain;",
        "GPU sampler (NVML) -> results/<jobid>/gpu.json"], fill="#EDE3F7", stroke=PURPLE, width=2.2)

    # backend containers, no Slurm
    zone(540, 360, 460, 240, "devai-engines  (image + container; no Slurm)", "#EEF7EE", "#4F9A5A")
    job = dict(fill="#E0F1E0", stroke="#4F9A5A")
    box(560, 392, 420, 44, "ollama serve  :11434", ["Ollama, compiled here"], **job)
    box(560, 444, 420, 44, "vLLM 0.28 + HyperQwen  :11435", ["own env; CUDA 13.1, compiled here"], **job)
    box(560, 496, 420, 44, "SGLang 0.5.16  :11436", ["own env; sglang-kernel compiled for sm120"], **job)
    text(570, 562, "idle until a job runs one engine with devai-run;", size=10.5, colour="#2E6A2E")
    text(570, 577, "devai-kill stops it (TERM, then KILL)", size=10.5, colour="#2E6A2E")
    zone(1020, 360, 210, 240, "devai-laya-trainer", "#EEF7EE", "#4F9A5A")
    text(1034, 397, "own image; no Slurm", size=10.5, colour="#2E6A2E")
    box(1034, 420, 182, 70, "trainer runs", ["one fine-tuning job", "per trainer job"], **job)
    zone(1250, 360, 200, 240, "devai-workload", "#FFFFFF", "#4F9A5A")
    text(1264, 397, "lab image; no GPU", size=10.5, colour="#2E6A2E")
    box(1264, 420, 172, 70, "clients", ["bench and test", "workload jobs"], fill="#FFFFFF", stroke="#4F9A5A")

    # -------------------------------------------------- below devai-net
    out.append(f'<path d="M58 870 h116 l18 18 v30 l-18 18 h-116 l-18 -18 v-30 z" fill="#FDECEC" stroke="{RED}" '
               f'stroke-width="1.6" stroke-dasharray="5 3"/>')
    text(116, 895, "GPU process", size=11, weight="bold", colour=RED, anchor="middle")
    text(116, 909, "outside Slurm", size=11, weight="bold", colour=RED, anchor="middle")
    text(116, 922, "(not allowed)", size=10, colour=RED, anchor="middle")
    box(240, 868, 280, 68, "podman service", ["host user, rootless", "reached through its socket"], fill="#F4F4F4")
    gpu_box(1020, 872, 430, 58, "GPU 0", ["RTX PRO 4000 Blackwell, 24 GB"])
    zone(20, 958, 1450, 120, "/var/cache/devai   (host volumes)", "#F6F6F6", "#999999", label_dx=1060)
    folder(40, 985, 480, 80, "jobs/   (new volume)", [
        "results/<jobid>/  result.json, gpu.json, logs", "mariadb/  history database    slurmctld/  state"], width=2.2)
    folder(560, 985, 240, 80, "model stores", ["ollama/ vllm/ sglang/ laya/", "+ vLLM parser plugins"])
    folder(820, 985, 180, 80, "engine caches", ["FlashInfer, SGLang", "(named volumes)"])

    # ----------------------------------------------------------------- arrows
    arrow([(1000, 178), (1000, 240)], "data", 1, (1000, 207), "inference API; fine-tuning jobs on :11438", (1016, 211))
    arrow([(900, 330), (900, 360)], "data", 2, (900, 345), "proxied requests to devai-engines:<port>", (888, 350),
          anchor="end", width=2.4)
    arrow([(560, 290), (520, 290), (520, 431), (480, 431)], "control", 3, (520, 360),
          "submit / cancel / state (REST + JWT)", (548, 282), anchor="end", width=2.2)
    arrow([(270, 452), (270, 475)], "slurm", 4, (270, 463))
    arrow([(60, 505), (48, 505), (48, 766), (60, 766)], "slurm", 5, (48, 640), both=True, width=2.2)
    arrow([(270, 535), (270, 560)], "slurm", 6, (270, 547))
    arrow([(270, 600), (270, 620)], "slurm")
    arrow([(380, 820), (380, 868)], "slurm", 7, (380, 846), "podman socket (rw)", (396, 850), width=2.2)
    arrow([(520, 900), (530, 900), (530, 615), (770, 615), (770, 600)], "slurm", 7, (530, 760), width=2.0)
    arrow([(770, 615), (1125, 615), (1125, 600)], "slurm", width=2.0)
    arrow([(1125, 615), (1350, 615), (1350, 600)], "slurm", width=2.0)
    text(1190, 632, "exec devai-run / devai-kill <jobid>", size=10.5, colour=PURPLE)
    arrow([(980, 600), (980, 850), (1060, 850), (1060, 872)], "gpu", 8, (980, 780), width=2.4)
    arrow([(1180, 600), (1180, 872)], "gpu", 8, (1180, 780), "CUDA (engines, trainer)", (1196, 784), width=2.4)
    arrow([(1350, 360), (1350, 300), (1200, 300)], "data", 9, (1350, 330), dashed=True)
    text(1338, 322, "requests; hold / release", size=10.5, colour=BLUE, anchor="end")
    arrow([(175, 180), (175, 410)], "control", 10, (175, 300), "REST; scontrol", (191, 304), width=2.0)
    arrow([(125, 820), (125, 870)], "kill", 11, (125, 845))
    arrow([(680, 985), (680, 602)], "storage", 12, (680, 800), "mounts", (696, 804))
    arrow([(910, 985), (910, 602)], "storage", 12, (910, 800))
    arrow([(225, 820), (225, 993)], "storage", 13, (225, 950), "results, DB, state", (160, 972), anchor="end")

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

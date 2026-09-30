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

W, H = 1500, 1130
FONT = "Helvetica, Arial, sans-serif"

# Interaction colours (also the legend).
BLUE, ORANGE, PURPLE, GREEN, GREY, RED, TEAL = "#2F6FD1", "#D98A1C", "#7A52B3", "#2E8B3E", "#7A7A7A", "#C73A3A", "#178A8A"
COLOURS = {"data": BLUE, "control": ORANGE, "slurm": PURPLE, "bus": TEAL, "gpu": GREEN, "storage": GREY, "kill": RED}

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
    text(W / 2, 50, "single host, rootless podman; engines always run, with no Slurm in them; the GPU is optional. "
         "Numbered arrows = the table in docs/slurm.md Sec. 2", size=12, colour="#555", anchor="middle")

    # ---------------------------------------------------------- row A
    box(20, 100, 220, 70, "Browser (LAN)", ["devai-operator over HTTPS", "(login)"], fill="#F4F4F4")
    box(260, 92, 260, 88, "Host shell", ["development (make)", "one-time root setup:", "logs volume, tmpfs, installer"], fill="#FAFAFA")
    zone(540, 72, 430, 120, "Legend", "#FFFFFF", "#BBBBBB")
    for i, (kind, what) in enumerate([("data", "inference data path"), ("slurm", "Slurm internal"),
                                      ("bus", "bus (NATS): commands, state"), ("gpu", "GPU use (CUDA)"),
                                      ("control", "control -> Slurm (REST)"), ("storage", "volume mount / file write"),
                                      ("kill", "guard kills a stray holder")]):
        cx, cy = 558 + (i // 4) * 214, 104 + (i % 4) * 24
        arrow([(cx, cy), (cx + 34, cy)], kind)
        text(cx + 42, cy + 4, what, size=10.2, colour="#333")
    zone(990, 72, 480, 120, "devai-lab-egress  (internal, no internet)", "#F4F8FD", "#8AA4C8")
    box(1010, 98, 440, 80, "Lab containers  (devai-lab)", [
        "agents: claude, codex, opencode, pi, dsh, dstui, aiagent",
        "no GPU device; they talk only to the router (D1)"], fill="#E8F0FC", stroke=BLUE)

    # -------------------------------------------------------- devai-net
    zone(20, 215, 1450, 665, "", "#FCFCF8", "#B9B08A")
    text(1456, 234, "devai-net", size=12, weight="bold", colour="#333", anchor="end")
    box(40, 240, 460, 100, "devai-operator  (image + container)", [
        "web service: Apache + mod_wsgi, login over TLS",
        "devai-control: runs registered actions for jobs",
        "podman socket (its scripts); GPU when present (probes)"],
        fill="#FDF6EC", stroke=ORANGE, width=2.2)
    box(620, 240, 640, 90, "devai-router  (own unprivileged container, small image)", [
        ":11434 ollama    :11435 vllm    :11436 sglang    :11438 laya-trainer",
        "request path unchanged; launch layer = Slurm client (REST + self-signed JWT)",
        "engine state from the bus; blocks engine control routes; no podman socket"],
        fill="#FFF3DC", stroke=ORANGE, width=2.2)

    # devai-slurm: all of Slurm and the bus, one image, one container
    zone(40, 360, 460, 470, "devai-slurm  (one image, one container)", "#F5F0FB", PURPLE)
    text(54, 397, "--pid=host; GPU (NVML) if present; no podman socket", size=10.2, colour=PURPLE)
    cp = dict(fill="#EFE7F8", stroke=PURPLE)
    bus = dict(fill="#E3F3F3", stroke=TEAL)
    box(60, 410, 420, 42, "slurmrestd  :6820", ["REST API v0.0.42, auth/jwt  (Slurm 24.11, Debian)"], **cp)
    box(60, 470, 420, 42, "slurmctld", ["queue, priorities, holds; state -> jobs/slurmctld/"], width=2.2, **cp)
    box(60, 530, 420, 40, "slurmdbd", ["accounting, job scripts, comments"], **cp)
    cylinder(60, 586, 420, 60, "MariaDB  (Debian package)", ["data -> jobs/mariadb/"])
    box(60, 662, 420, 56, "nats-server  :4222  --  the bus", [
        "commands and state between containers",
        "Debian 2.10.27; each user limited to its subjects"], width=2.2, **bus)
    box(60, 734, 420, 86, "slurmd  --  the only node;  license engine:1", [
        "one engine / trainer / probe job at a time",
        "job = load, lease, unload through the bus (devai-bus)",
        "GPU host: guard (unload, kill stray holder, or drain)",
        "   and sampler (NVML) -> results/<jobid>/gpu.json"], fill="#EDE3F7", stroke=PURPLE, width=2.2)

    # backend containers: always running, no Slurm
    zone(580, 360, 440, 240, "devai-engines  (image + container; no Slurm)", "#EEF7EE", "#4F9A5A")
    eng = dict(fill="#E0F1E0", stroke="#4F9A5A")
    box(600, 390, 400, 44, "ollama serve  :11434", ["always running; loads and unloads on a job's command"], **eng)
    box(600, 440, 400, 44, "vLLM 0.28 + HyperQwen  :11435", ["one model process while loaded; sleep = fast path"], **eng)
    box(600, 490, 400, 44, "SGLang 0.5.16  :11436", ["one model process while loaded; sleep = fast path"], **eng)
    box(600, 540, 400, 48, "supervisord + devai-control", ["keeps the engines running; takes commands from the bus"], **bus)
    zone(1040, 360, 410, 240, "devai-laya-trainer  (own image; no Slurm)", "#EEF7EE", "#4F9A5A")
    box(1060, 400, 370, 64, "trainer runs", ["one fine-tuning run per trainer job", "started and stopped by devai-control"], **eng)
    box(1060, 540, 370, 48, "supervisord + devai-control", ["keeps the trainer running; bus commands"], **bus)

    # the bus, drawn as a bar every devai-control hangs off
    out.append(f'<path d="M30 855 H1440" fill="none" stroke="{TEAL}" stroke-width="5"/>')
    out.append(f'<path d="M480 690 H490 V855" fill="none" stroke="{TEAL}" stroke-width="3"/>')
    text(240, 874, "the bus: NATS in devai-slurm, :4222", size=11, weight="bold", colour=TEAL)

    # -------------------------------------------------- below devai-net
    out.append(f'<path d="M58 905 h116 l18 18 v30 l-18 18 h-116 l-18 -18 v-30 z" fill="#FDECEC" stroke="{RED}" '
               f'stroke-width="1.6" stroke-dasharray="5 3"/>')
    text(116, 930, "GPU process", size=11, weight="bold", colour=RED, anchor="middle")
    text(116, 944, "outside devai", size=11, weight="bold", colour=RED, anchor="middle")
    text(116, 957, "(not allowed)", size=10, colour=RED, anchor="middle")
    box(240, 903, 280, 68, "podman service", ["host user, rootless", "used only by devai-operator (15)"], fill="#F4F4F4")
    gpu_box(960, 905, 480, 58, "GPU 0  (optional)", ["RTX PRO 4000 Blackwell, 24 GB; without one: Ollama, trainer on CPU"])
    zone(20, 990, 1450, 125, "/var/cache/devai   (host volumes)", "#F6F6F6", "#999999", label_dx=1060)
    folder(40, 1017, 480, 80, "jobs/   (new volume)", [
        "results/<jobid>/  result.json, gpu.json, logs",
        "mariadb/  history    slurmctld/  state    bus/  bus log"], width=2.2)
    folder(580, 1017, 240, 80, "model stores", ["ollama/ vllm/ sglang/ laya/", "+ vLLM parser plugins"])
    folder(840, 1017, 180, 80, "engine caches", ["FlashInfer, SGLang", "(named volumes)"])

    # ----------------------------------------------------------------- arrows
    arrow([(1100, 178), (1100, 240)], "data", 1, (1100, 207), "inference API; fine-tuning jobs on :11438", (1116, 211))
    arrow([(1000, 330), (1000, 360)], "data", 2, (1000, 345), "proxied requests to devai-engines:<port>", (988, 350),
          anchor="end", width=2.4)
    arrow([(620, 300), (540, 300), (540, 431), (480, 431)], "control", 3, (540, 380), width=2.2)
    arrow([(270, 452), (270, 470)], "slurm", 4, (270, 461))
    arrow([(60, 491), (48, 491), (48, 777), (60, 777)], "slurm", 5, (48, 640), both=True, width=2.2)
    arrow([(270, 512), (270, 530)], "slurm", 6, (270, 521))
    arrow([(270, 570), (270, 586)], "slurm")
    arrow([(380, 734), (380, 718)], "bus", width=2.2)
    arrow([(800, 855), (800, 588)], "bus", 7, (800, 720), "7, 17 in; 16 out", (812, 724), both=True, width=2.2)
    arrow([(1150, 855), (1150, 588)], "bus", 7, (1150, 720), both=True, width=2.2)
    arrow([(1030, 855), (1030, 330)], "bus", 16, (1030, 650), width=2.2)
    arrow([(30, 855), (30, 300), (40, 300)], "bus", 17, (30, 560), both=True, width=2.2)
    arrow([(980, 600), (980, 905)], "gpu", 8, (980, 760), width=2.4)
    arrow([(1245, 600), (1245, 905)], "gpu", 8, (1245, 760), "CUDA (engines, trainer)", (1261, 764), width=2.4)
    arrow([(500, 262), (620, 262)], "data", 9, (560, 262), dashed=True)
    text(506, 236, "9: bench requests, holds", size=9.8, colour=BLUE)
    arrow([(400, 340), (400, 410)], "control", 10, (400, 375), "REST", (416, 379), width=2.0)
    arrow([(125, 820), (125, 905)], "kill", 11, (125, 885))
    arrow([(680, 1017), (680, 602)], "storage", 12, (680, 800), "mounts", (696, 804))
    arrow([(930, 1017), (930, 602)], "storage", 12, (930, 800))
    arrow([(225, 820), (225, 1025)], "storage", 13, (225, 945), "results, DB, state, bus log", (212, 1006), anchor="end")
    arrow([(130, 170), (130, 240)], "data", 14, (130, 206), "HTTPS", (146, 210), width=2.0)
    text(40, 355, "15: podman calls from its scripts (not drawn)", size=9.8, colour=GREY)

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

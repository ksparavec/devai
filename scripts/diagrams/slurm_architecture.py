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

W, H = 1500, 1030
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
    text(W / 2, 50, "single host, rootless podman, Slurm 26.05, one GPU. "
         "Numbered arrows = the table in docs/slurm.md Sec. 2", size=12, colour="#555", anchor="middle")

    # ---------------------------------------------------------- row A: clients
    zone(20, 72, 650, 120, "devai-lab-egress  (internal network, no internet)", "#F4F8FD", "#8AA4C8")
    box(40, 98, 610, 80, "Lab containers  (devai-lab image)", [
        "agents: claude, codex, opencode, pi, dsh, dstui, aiagent",
        "no GPU device; they talk only to the router (D1)"], fill="#E8F0FC", stroke=BLUE)
    zone(690, 72, 430, 120, "Legend", "#FFFFFF", "#BBBBBB")
    for i, (kind, what) in enumerate([("data", "inference data path"), ("control", "control: router / operator -> Slurm"),
                                      ("slurm", "Slurm internal"), ("gpu", "GPU use (CUDA, NVML)"),
                                      ("storage", "volume mount / file write"), ("kill", "GPU guard kills a holder")]):
        cx, cy = 708 + (i // 3) * 214, 108 + (i % 3) * 28
        arrow([(cx, cy), (cx + 34, cy)], kind)
        text(cx + 42, cy + 4, what, size=10.2, colour="#333")
    box(1150, 92, 310, 86, "Operator (host shell)", [
        "devai-jobs: history, queue, suspend / resume,", "interrupt", "make: build, cache-up, bench, probe"],
        fill="#F4F4F4")

    # -------------------------------------------------------- row B: devai-net
    zone(20, 215, 1450, 590, "devai-net", "#FCFCF8", "#B9B08A")
    box(300, 240, 640, 100, "devai-router  (own unprivileged container, small image)", [
        ":11434 ollama    :11435 vllm    :11436 sglang    :11438 laya-trainer",
        "request path unchanged; launch layer = Slurm client (REST + self-signed JWT)",
        "reads the GPU holder from Slurm; holds for workload jobs"], fill="#FFF3DC", stroke=ORANGE, width=2.2)

    # the engines container
    zone(40, 372, 1410, 420, "devai-engines  (one image, one container)", "#F3F9F3", "#3E8E4E")
    text(1436, 392, "Debian trixie; privileged, --pid=host, GPU via CDI", size=11, colour="#2E6A2E", anchor="end")
    zone(60, 405, 900, 370, "slurmd  --  node 'engines'  (Gres=gpu:1)", "#E6F3E6", "#4F9A5A")
    zone(80, 440, 420, 320, "GPU jobs  (one at a time; each exec's its server)", "#FFFFFF", "#4F9A5A")
    job = dict(fill="#E0F1E0", stroke="#4F9A5A")
    for i, (title, line) in enumerate([
            ("engine: ollama serve  :11434", "Ollama, our build"),
            ("engine: vLLM 0.28 + HyperQwen  :11435", "venv /opt/vllm, our build"),
            ("engine: SGLang 0.5.16  :11436", "venv /opt/sglang"),
            ("trainer: laya fine-tune", "venv /opt/laya; API on the router's :11438"),
            ("probe: engine + prober client", "make probe-*")]):
        box(100, 472 + i * 56, 380, 48, title, [line], **job)
    box(530, 440, 410, 70, "workload jobs  (no GPU)", [
        "bench and test clients, venv /opt/workload",
        "each takes a hold on its engine via the router"], fill="#FFFFFF", stroke="#4F9A5A")
    box(530, 530, 410, 120, "GPU guard  (prolog / epilog of every GPU job)", [
        "prolog: card idle? else SIGTERM, then SIGKILL the holder;",
        "   still busy -> fail: node drained, job held",
        "prolog: wipe vLLM compile cache",
        "prolog + epilog: NVML energy -> results/<id>/gpu.json",
        "each job: cgroup, signals, GPU accounting"], fill="#FFFBE6", stroke="#B8A04A")

    zone(1000, 405, 430, 370, "supervisord: control plane, same container", "#F5F0FB", PURPLE)
    cp = dict(fill="#EFE7F8", stroke=PURPLE)
    box(1020, 440, 390, 48, "slurmrestd  :6820", ["REST API v0.0.44, auth/jwt"], **cp)
    box(1020, 520, 390, 64, "slurmctld", ["queue, priorities, holds", "state -> jobs/slurmctld/"], width=2.2, **cp)
    box(1020, 614, 390, 50, "slurmdbd", ["accounting: TRES incl. gres/gpumem, gpuutil"], **cp)
    cylinder(1020, 690, 390, 58, "MariaDB  (Debian package)", ["data -> jobs/mariadb/"])

    # -------------------------------------------------- row C: GPU and storage
    gpu_box(60, 862, 380, 64, "GPU 0", ["RTX PRO 4000 Blackwell, 24 GB"])
    out.append(f'<path d="M538 867 h128 l18 18 v34 l-18 18 h-128 l-18 -18 v-34 z" fill="#FDECEC" stroke="{RED}" '
               f'stroke-width="1.6" stroke-dasharray="5 3"/>')
    text(602, 894, "GPU process", size=11, weight="bold", colour=RED, anchor="middle")
    text(602, 909, "outside Slurm", size=11, weight="bold", colour=RED, anchor="middle")
    text(602, 923, "(not allowed)", size=10, colour=RED, anchor="middle")
    zone(740, 842, 730, 178, "/var/cache/devai   (host volumes)", "#F6F6F6", "#999999", label_bottom=True)
    folder(760, 858, 220, 76, "model stores", ["ollama/ vllm/ sglang/ laya/", "+ vLLM parser plugins"])
    folder(1000, 858, 180, 76, "engine caches", ["FlashInfer, SGLang", "(named volumes)"])
    folder(1200, 858, 255, 130, "jobs/   (new volume)", [
        "results/<jobid>/  result.json,", "   gpu.json, logs, artifacts",
        "mariadb/   history database", "slurmctld/  scheduler state"], width=2.2)

    # ----------------------------------------------------------------- arrows
    arrow([(450, 178), (450, 240)], "data", 1, (450, 207), "inference API; fine-tuning jobs on :11438", (466, 211))
    arrow([(420, 340), (420, 440)], "data", 2, (420, 372), "proxied requests to devai-engines:<port>", (408, 362),
          anchor="end", width=2.4)
    arrow([(940, 290), (985, 290), (985, 464), (1020, 464)], "control", 3, (985, 340),
          "submit / cancel / signal / state (REST + JWT)", (955, 282), width=2.2)
    arrow([(1215, 488), (1215, 520)], "slurm", 4, (1215, 504), "RPC", (1232, 508))
    arrow([(1020, 552), (960, 552)], "slurm", 5, (990, 552), both=True, width=2.2)
    text(990, 578, "launch,", size=10, colour=PURPLE, anchor="middle")
    text(990, 591, "signal,", size=10, colour=PURPLE, anchor="middle")
    text(990, 604, "kill", size=10, colour=PURPLE, anchor="middle")
    arrow([(1215, 584), (1215, 614)], "slurm", 6, (1215, 599), "accounting", (1232, 603))
    arrow([(1215, 664), (1215, 690)], "slurm")
    arrow([(290, 760), (290, 862)], "gpu", 7, (290, 815), "CUDA; NVML (guard, accounting)", (306, 840), width=2.4)
    arrow([(735, 440), (735, 340)], "data", 8, (735, 388), "workload requests; hold / release", (751, 392), dashed=True)
    arrow([(1460, 135), (1478, 135), (1478, 464), (1410, 464)], "control", 9, (1478, 300))
    arrow([(1478, 464), (1478, 552), (1410, 552)], "control", 9, (1478, 510), dashed=True)
    arrow([(1478, 552), (1478, 935), (1455, 935)], "storage", 9, (1478, 760))
    text(1305, 206, "9: REST, scontrol suspend / resume, result files", size=10.2, colour=ORANGE, anchor="middle")
    arrow([(600, 650), (600, 867)], "kill", 10, (600, 740), "SIGTERM, SIGKILL\n(else drain the node)", (616, 736))
    arrow([(870, 858), (870, 792)], "storage", 11, (870, 822), "mounts", (886, 826))
    arrow([(1090, 858), (1090, 792)], "storage", 11, (1090, 822))
    arrow([(1330, 792), (1330, 866)], "storage", 12, (1330, 822), "results, DB, state", (1346, 826))

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

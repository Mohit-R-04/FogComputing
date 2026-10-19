# -*- coding: utf-8 -*-
"""Write the Review 3 diagrams as draw.io files and export them to PNG.

    python3 make_diagrams.py            (from details/Review 3/build)

fig01  system block diagram   (the project's colour block diagram, title strip removed)
fig02  BBN structure          (the project's colour BBN figure)
fig03  per-epoch control loop
fig04  data and experiment pipeline
fig05  train / held-out / closed-loop split grid
fig06  implementation stack
fig07  reactive vs threshold vs BBN timeline   (presentation)
demo_* terminal panels holding captured output of `bash run_demo.sh --no-pause`
"""

import html
import os
import re
import sys

import figlib as F
from figlib import Drawio, esc

DETAILS = os.path.normpath(os.path.join(F.HERE, "..", ".."))
DEMO_TXT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(F.HERE, "demo_out.txt")


def title_body(title, body, tsize=15, bsize=12.5, tcolor=F.INK, bcolor=F.MUTED):
    return ('<div style="font-weight:700;font-size:%spx;color:%s;">%s</div>'
            '<div style="font-size:%spx;color:%s;margin-top:4px;line-height:1.3;">%s</div>'
            % (tsize, tcolor, title, bsize, bcolor, body))


# --------------------------------------------------------------------------- #
# fig01 / fig02: reuse the project's colour diagrams
# --------------------------------------------------------------------------- #

def strip_cells(xml, needles):
    out = xml
    for cell in re.findall(r'<mxCell [^>]*?(?:/>|>.*?</mxCell>)', xml, flags=re.S):
        val = html.unescape(re.search(r'value="([^"]*)"', cell).group(1)) if 'value="' in cell else ""
        if any(n in val for n in needles):
            out = out.replace(cell, "")
    return out


def fig01_02():
    src = open(os.path.join(DETAILS, "Project_Block_Diagram_colour.drawio"), encoding="utf-8").read()
    src = strip_cells(src, ["BBN-Driven Proactive Fault Tolerance for Fog Computing",
                            "System block diagram", "SSN College of Engineering"])
    p = os.path.join(F.FIG, "fig01_block_diagram.drawio")
    open(p, "w", encoding="utf-8").write(src)
    F.export_png(p, 2)

    src = open(os.path.join(DETAILS, "BBN_Structure_colour.drawio"), encoding="utf-8").read()
    p = os.path.join(F.FIG, "fig02_bbn_structure.drawio")
    open(p, "w", encoding="utf-8").write(src)
    F.export_png(p, 2)


# --------------------------------------------------------------------------- #
# fig03: control loop of one decision epoch
# --------------------------------------------------------------------------- #

def fig03():
    d = Drawio("Control loop", 1340, 640)
    W, H = 285, 128
    xs = [40, 365, 690, 1015]
    y1, y2 = 70, 300
    d.text(40, 18, 900, 30, "<b>One decision epoch</b> &nbsp;(every 100 time units, one time unit after all reports are in)",
           font=16, color=F.INK)

    b1 = d.box(xs[0], y1, W, H, title_body("Telemetry reports",
               "every live device reports 13 channels; a channel that did not report stays missing"),
               fill=F.F_GREY, stroke=F.SLATE)
    b2 = d.box(xs[1], y1, W, H, title_body("N1 &nbsp;BBN inference",
               "P(fail within 2 epochs | e) for every fog node by Eq. (2), and the condition behind it"),
               fill=F.F_BLUE, stroke=F.BLUE)
    b3 = d.box(xs[2], y1, W, H, title_body("Online learning",
               "reports whose horizon has passed update the node_failure table (Dirichlet); acted-on nodes are censored"),
               fill=F.F_GREEN, stroke=F.GREEN)
    b4 = d.box(xs[3], y1, W, H, title_body("N2 &nbsp;Expected-loss migration",
               "move a session when P(F<sub>n</sub>) &middot; C<sub>fail</sub> &gt; C<sub>mig</sub>, i.e. above its own P* = C<sub>mig</sub>/C<sub>fail</sub> &nbsp;(Eq. 5)"),
               fill=F.F_BLUE, stroke=F.BLUE)
    b5 = d.box(xs[3], y2, W, H, title_body("N3 &nbsp;Selective replication",
               "per session, the tier with least overhead + P &middot; &kappa; &middot; outage: none, checkpoint or warm standby (&le; 20%) &nbsp;(Eq. 7)"),
               fill=F.F_BLUE, stroke=F.BLUE)
    b6 = d.box(xs[2], y2, W, H, title_body("Actuation in iFogSim",
               "stop-and-copy migrations; replicas provisioned or released; the simulation continues"),
               fill=F.F_ORANGE, stroke=F.ORANGE)
    b7 = d.box(xs[1], y2, W, H, title_body("Between epochs: detect and recover",
               "heartbeat timeout 30 ms; outage by tier: warm 35 ms, checkpoint &asymp; 60&ndash;80 ms, cold restart 330 ms"),
               fill=F.F_RED, stroke=F.RED)
    b8 = d.box(xs[0], y2, W, H, title_body("Outcomes recorded",
               "failures, migrations, outages and loop delays; reports wait two epochs for their label"),
               fill=F.F_GREY, stroke=F.SLATE)

    for i, (b, c) in enumerate([(b1, F.SLATE), (b2, F.BLUE), (b3, F.GREEN), (b4, F.BLUE),
                                (b5, F.BLUE), (b6, F.ORANGE), (b7, F.RED), (b8, F.SLATE)]):
        x = xs[[0, 1, 2, 3, 3, 2, 1, 0][i]]
        y = y1 if i < 4 else y2
        d.badge(x + 2, y + 2, str(i + 1), fill=c, r=14, font=13)

    d.edge(b1, b2); d.edge(b2, b3); d.edge(b3, b4)
    d.edge(b4, b5, label="sessions")
    d.edge(b5, b6); d.edge(b6, b7); d.edge(b7, b8)
    d.edge(b8, b1, label="next epoch", color=F.GREEN, dashed=True)

    # placement on demand
    pl = d.box(365, 495, 935, 92, title_body("N2 &nbsp;Risk-aware placement, on demand &nbsp;(Eq. 4)",
               "used whenever a session needs a node: at start-up, after every mobility step, as a migration target "
               "and for recovery. &nbsp;U = &alpha;(1&minus;L) + &beta;(1&minus;E) + &gamma;(1&minus;P), weights set by the patient's criticality; "
               "a user moves for utility only if U improves by more than 0.10"),
               fill=F.F_BLUE2, stroke=F.BLUE, dashed=True, align="left")
    d.edge(b4, pl, color=F.BLUE, dashed=True, exit=(1, 0.75), entry=(1, 0.5),
           points=[(1325, y1 + 0.75 * H), (1325, 541)])
    d.edge(b7, pl, color=F.BLUE, dashed=True, exit=(0.5, 1), entry=(0.1, 0))
    d.save("fig03_control_loop", scale=2)


# --------------------------------------------------------------------------- #
# fig04: data and experiment pipeline
# --------------------------------------------------------------------------- #

def fig04():
    d = Drawio("Pipeline", 1420, 470)
    W, H, G = 248, 150, 40
    y1, y2 = 40, 280
    xs = [20 + i * (W + G) for i in range(5)]
    rows1 = [
        ("Inputs", "Melbourne CBD edge resources: 1 cloud, 118 fog gateways; mobility traces; cardiac-monitoring application", F.F_GREY, F.SLATE),
        ("Simulation", "iFogSim2 + degradation fault model; 12 operating regimes &times; seeds; 31 runs of 10,000 time units", F.F_GREY, F.SLATE),
        ("Raw telemetry", "one row per device per epoch: 13 channels, device class and the label <i>fails within 200 time units</i>", F.F_GREY, F.SLATE),
        ("Split by regime <u>and</u> seed", "train: 7 regimes &times; 3 seeds = 21 runs, 314,323 rows<br>held out: 5 regimes &times; 2 seeds = 10 runs, 160,823 rows", F.F_GREEN, F.GREEN),
        ("Model selection", "24 configurations, four scenario folds of the training runs only; CV PR-AUC 0.041 &rarr; 0.759", F.F_GREEN, F.GREEN),
    ]
    ids = []
    for (t, b, fill, st), x in zip(rows1, xs):
        ids.append(d.box(x, y1, W, H, title_body(t, b), fill=fill, stroke=st))
    for a, b in zip(ids, ids[1:]):
        d.edge(a, b)

    W2 = 372
    xs2 = [20, 20 + W2 + 112, 20 + 2 * (W2 + 112)]
    rows2 = [
        ("Training", "supervised bins (&le; 8 per channel and device class); learned ranked-node latent tables; "
                     "MAP-EM, 60 iterations, &asymp; 10 s &rarr; <b>bbn_trained.json</b>", F.F_GREEN, F.GREEN),
        ("Prediction evaluation", "BBN against 8 alternatives on 159,813 held-out fog and edge snapshots, scored once; "
                                  "bootstrap intervals over nodes; MCAR / MNAR telemetry loss", F.F_ORANGE, F.ORANGE),
        ("Closed-loop evaluation", "7 policies &times; 25 held-out runs (5 regimes &times; 5 seeds) inside iFogSim with common random numbers; "
                                   "paired Wilcoxon tests, ablation, cost sensitivity", F.F_ORANGE, F.ORANGE),
    ]
    ids2 = []
    for (t, b, fill, st), x in zip(rows2, xs2):
        ids2.append(d.box(x, y2, W2, H, title_body(t, b), fill=fill, stroke=st))
    d.edge(ids[-1], ids2[0], points=[(xs[-1] + W / 2, 245), (206, 245)], label="selected configuration")
    d.edge(ids2[0], ids2[1])
    d.edge(ids2[1], ids2[2], label="exports ML and<br>FTAPA models")
    d.save("fig04_pipeline", scale=2)


# --------------------------------------------------------------------------- #
# fig05: split grid
# --------------------------------------------------------------------------- #

REGIMES = [(0, "baseline", "T"), (1, "idle-fleet", "H"), (2, "chatty-sensors", "T"), (3, "thin-fog", "T"),
           (4, "starved-fog", "T"), (5, "fat-fog", "T"), (6, "weak-edge", "T"), (7, "dense-edge", "H"),
           (8, "slow-backhaul", "H"), (9, "fog-heavy", "H"), (10, "edge-heavy", "H"), (11, "saturated", "T")]


def fig05():
    d = Drawio("Split", 820, 560)
    seeds = [101, 102, 103, 201, 202, 203, 204, 205]
    x0, y0, cw, ch, lw = 230, 92, 62, 30, 210
    d.text(x0, 22, 3 * cw, 22, "training seeds", font=13, color=F.GREEN, align="center", bold=True)
    d.text(x0 + 3 * cw, 22, 2 * cw, 22, "held-out seeds", font=13, color=F.ORANGE, align="center", bold=True)
    d.text(x0 + 5 * cw, 22, 3 * cw, 22, "closed-loop only", font=13, color=F.BLUE, align="center", bold=True)
    d.line([(x0 + 3 * cw - 1, 20), (x0 + 3 * cw - 1, y0 + 12 * ch + 4)], color=F.FAINT, sw=1, dashed=True)
    d.line([(x0 + 5 * cw - 1, 20), (x0 + 5 * cw - 1, y0 + 12 * ch + 4)], color=F.FAINT, sw=1, dashed=True)
    for j, s in enumerate(seeds):
        d.text(x0 + j * cw, 56, cw, 26, str(s), font=13, color=F.INK, align="center")
    d.text(20, 56, lw, 26, "<b>Regime</b>", font=13, color=F.INK)
    for i, (rid, name, split) in enumerate(REGIMES):
        y = y0 + i * ch
        d.text(20, y, lw, ch, "%2d &nbsp;%s" % (rid, name), font=13,
               color=F.INK if split == "H" else F.MUTED, bold=(split == "H"))
        for j, s in enumerate(seeds):
            if split == "T" and s < 200:
                fill, lab, fc = F.GREEN, "T", F.WHITE
            elif split == "H" and s in (201, 202):
                fill, lab, fc = F.ORANGE, "H+C", F.WHITE
            elif split == "H" and s > 202:
                fill, lab, fc = F.BLUE, "C", F.WHITE
            else:
                fill, lab, fc = "#eef0f3", "", F.FAINT
            d.box(x0 + j * cw + 3, y + 3, cw - 6, ch - 6, lab, fill=fill, stroke="none", font=11,
                  color=fc, bold=True, arc=4)
    ly = y0 + 12 * ch + 22
    items = [(F.GREEN, "T", "training run (21)"), (F.ORANGE, "H+C", "held-out telemetry and closed loop (10)"),
             (F.BLUE, "C", "closed loop only (15)")]
    x = 20
    for fill, lab, txt in items:
        d.box(x, ly, 46, 22, lab, fill=fill, stroke="none", font=11, color=F.WHITE, bold=True, arc=4)
        d.text(x + 54, ly, 250, 22, txt, font=13, color=F.INK)
        x += 54 + {0: 150, 1: 285, 2: 0}[items.index((fill, lab, txt))] + 20
    d.save("fig05_split", scale=2.5)


# --------------------------------------------------------------------------- #
# fig06: implementation stack
# --------------------------------------------------------------------------- #

def code_list(title, items, color):
    li = "".join('<div style="font-family:Menlo,monospace;font-size:12px;color:%s;margin-top:3px;">%s</div>'
                 % (F.INK, it) for it in items)
    return ('<div style="font-weight:700;font-size:14px;color:%s;margin-bottom:4px;">%s</div>%s'
            % (color, title, li))


def fig06():
    d = Drawio("Implementation", 1460, 620)
    R1, R2, JH = 92, 282, 150
    jw = 270
    # Java panel
    d.box(20, 20, 600, 460, "", fill=F.F_BLUE2, stroke=F.BLUE, sw=1.4, arc=10)
    d.text(40, 30, 560, 46, '<b style="color:%s;font-size:16px;">Java &middot; iFogSim 2 on CloudSim</b><br>'
           '<span style="color:%s;font-size:12.5px;">new package <span style="font-family:Menlo,monospace;">org.fog.ft</span> '
           '(31 classes); upstream source unmodified</span>' % (F.BLUE, F.MUTED), font=13, valign="top")
    j = {}
    def sub(x, y, title, items, col):
        return d.box(x, y, jw, JH, code_list(title, items, col), fill=F.WHITE, stroke=col, align="left", valign="top")
    j["sim"] = sub(40, R1, "Simulation", ["MelbCBDFaultToleranceSim", "ScenarioSweep, ScenarioSpec",
                   "Topology, MobilityDriver", "FtExperiment (CLI)"], F.SLATE)
    j["dev"] = sub(330, R1, "Devices and faults", ["InstrumentedFogDevice", "FaultInjector, FtRandom",
                   "NodeTelemetry, TelemetryCollector", "InstrumentedActuator"], F.SLATE)
    j["ctl"] = sub(40, R2, "Controller", ["FaultToleranceController", "FtPolicy (7 policies)",
                   "FtCostModel, Criticality", "ServiceSession, SessionRegistry"], F.BLUE)
    j["pred"] = sub(330, R2, "Predictors", ["BbnEngine, BbnPredictor", "MlPredictor",
                    "HealthIndexPredictor (FTAPA)", "OraclePredictor"], F.BLUE)
    # Python panel
    d.box(840, 20, 600, 460, "", fill=F.F_GREEN, stroke=F.GREEN, sw=1.4, arc=10)
    d.text(860, 30, 560, 46, '<b style="color:%s;font-size:16px;">Python 3.12 &middot; model/</b><br>'
           '<span style="color:%s;font-size:12.5px;">NumPy, pandas, SciPy, scikit-learn, Rich; pytest; pgmpy for cross-checks</span>'
           % (F.GREEN, F.MUTED), font=13, valign="top")
    p = {}
    p["data"] = sub(860, R1, "Dataset", ["generate_telemetry", "javarun (parallel JVMs)",
                    "build_dataset", "design (split, seeds)"], F.GREEN)
    p["learn"] = sub(1150, R1, "Learning", ["discretise (exact DP)", "ranked_node, bbn_fast",
                     "select_bbn", "train_bbn (MAP-EM)"], F.GREEN)
    p["eval"] = sub(860, R2, "Evaluation", ["predict, verify", "comparison/run_comparison",
                    "export_models", "run_ft_experiments"], F.ORANGE)
    p["test"] = sub(1150, R2, "Tests and demo", ["test_inference (29)", "test_pipeline (18)",
                    "test_java_parity (3)", "crosscheck_pgmpy, demo"], F.ORANGE)
    # shared files (centre column)
    def note(y, a, b):
        return d.box(650, y, 160, 64, '<div style="font-family:Menlo,monospace;font-size:11.5px;font-weight:700;color:%s;">%s</div>'
                     '<div style="font-size:11.5px;color:%s;margin-top:3px;">%s</div>' % (F.INK, a, F.MUTED, b),
                     fill=F.WHITE, stroke=F.FAINT, extra="shape=note;size=14;")
    f_csv = note(R1 + 5, "data/raw/*.csv", "telemetry, 31 runs")
    f_bbn = note(230, "bbn_trained.json", "the trained BBN")
    f_ml = note(R2 + 40, "ml_best.json<br>ftapa.json", "baseline predictors")
    f_run = note(398, "results/ft/runs/", "450 closed-loop runs")
    yc = lambda y0: y0 + 32
    d.edge(j["dev"], f_csv, exit=(1, (yc(R1 + 5) - R1) / JH), entry=(0, 0.5))
    d.edge(f_csv, p["data"], exit=(1, 0.5), entry=(0, (yc(R1 + 5) - R1) / JH))
    d.edge(p["learn"], f_bbn, exit=(0.5, 1), entry=(1, 0.5), points=[(1285, 262)])
    d.edge(f_bbn, j["pred"], exit=(0, 0.5), entry=(1, 0.12), points=[(630, 262), (630, R2 + 18)])
    d.edge(p["eval"], f_ml, exit=(0, (yc(R2 + 40) - R2) / JH), entry=(1, 0.5))
    d.edge(f_ml, j["pred"], exit=(0, 0.5), entry=(1, (yc(R2 + 40) - R2) / JH))
    d.edge(j["ctl"], f_run, exit=(0.5, 1), entry=(0, 0.5), points=[(175, 462), (630, 462), (630, 430)])
    d.edge(f_run, p["eval"], exit=(1, 0.5), entry=(0, (430 - R2) / JH))

    d.box(20, 505, 1420, 80, '<b>One model file, two engines.</b> &nbsp;<span style="font-family:Menlo,monospace;">bbn_trained.json</span> '
          'drives both the Python evaluation and the Java controller inside iFogSim; their posteriors agree to within 1e-15 '
          '(the parity test requires 1e-9), and the Python engine matches pgmpy. &nbsp;<b>One command each:</b> '
          '<span style="font-family:Menlo,monospace;">bash run_all.sh</span> regenerates every result (&asymp; 45 min); '
          '<span style="font-family:Menlo,monospace;">bash run_demo.sh</span> runs the nine-stage demonstration (&asymp; 35 s).',
          fill=F.F_GREY, stroke=F.SLATE, font=13.5, align="left")
    d.save("fig06_implementation", scale=2)


# --------------------------------------------------------------------------- #
# fig07: reactive vs threshold vs BBN timeline (presentation)
# --------------------------------------------------------------------------- #

def fig07():
    d = Drawio("Timeline", 1260, 470)
    X0, X1, Y0, Y1 = 110, 1040, 60, 300      # plot box: health 1 at Y0, 0 at Y1
    d.line([(X0, Y0 - 10), (X0, Y1)], color=F.INK, sw=1.4)
    d.line([(X0, Y1), (X1 + 60, Y1)], color=F.INK, sw=1.4, end="block")
    d.text(X1 + 30, Y1 + 8, 120, 22, "time", font=14, color=F.MUTED)
    d.text(10, Y0 - 12, 95, 40, "hidden<br>health", font=13, color=F.MUTED, align="right")
    # degradation window shading
    xw0, xf = 470, 900
    d.rect(xw0, Y0 - 10, xf - xw0, Y1 - Y0 + 10, F.F_BLUE2, extra="opacity=100;")
    d.text(xw0 + 8, Y0 - 6, 330, 22, "<b>degradation window</b> &nbsp;(fail-slow symptoms)", font=13, color=F.BLUE)
    # health curve
    pts = []
    import math
    for k in range(0, 41):
        t = k / 40.0
        x = X0 + 20 + t * (xf - X0 - 20)
        h = 1.0 - 0.06 * t if t < 0.38 else 0.977 - 0.98 * ((t - 0.38) / 0.62) ** 1.7
        pts.append((x, Y0 + (1 - max(h, 0)) * (Y1 - Y0)))
    d.line(pts, color=F.INK, sw=2.6)
    d.line([(xf, Y0 - 10), (xf, Y1 + 10)], color=F.RED, sw=2, dashed=True)
    d.text(xf + 8, Y0 - 6, 140, 22, "<b>node fails</b>", font=14, color=F.RED)
    # markers
    def marker(x, y, color, lab, ty, align="center", w=220):
        d.ellipse(x - 8, y - 8, 16, 16, color)
        d.line([(x, y + 10), (x, ty - 4)], color=color, sw=1.2, dashed=True)
        d.text(x - w / 2, ty, w, 52, lab, font=13, color=color, align=align, valign="top")
    hy = lambda x: [p for p in pts if p[0] >= x][0][1]
    marker(560, hy(560), F.GREEN, "<b>BBN</b>: P(fail) rises above the session's P*; migrate or add a standby", Y1 + 24, w=240)
    marker(805, hy(805), F.GOLD, "<b>fixed threshold</b> on one metric fires late (or on a harmless spike)", Y1 + 24, w=230)
    # reactive
    d.rect(xf, Y1 - 46, 130, 30, F.F_RED, stroke=F.RED)
    d.text(xf, Y1 - 46, 130, 30, "detect + recover", font=12.5, color=F.RED, align="center", bold=True)
    d.text(xf + 10, Y1 - 128, 230, 70, "<b>reactive</b>: recovery starts only after the failure, so the outage is paid in full",
           font=13, color=F.RED, align="left", valign="bottom")
    d.save("fig07_reactive_vs_proactive", scale=2.5)


# --------------------------------------------------------------------------- #
# demo terminal panels
# --------------------------------------------------------------------------- #

TERM_BG = "#1b2230"
TERM_FG = "#e6e9ef"
T_GREEN = "#7bd88f"
T_RED = "#ff7a7a"
T_ORANGE = "#ffb86b"
T_BLUE = "#7ab8ff"
T_DIM = "#9aa5b8"


def colourise(line):
    s = esc(line).replace(" ", "&nbsp;")
    rules = [(r"MIGRATE", T_GREEN), (r"FAIL&nbsp;", T_RED), (r"RECOVER", T_ORANGE), (r"USER&nbsp;", T_DIM),
             (r"✓", T_GREEN), (r"Proposed&nbsp;BBN&nbsp;framework", T_BLUE), (r"BBN&nbsp;\(proposed\)", T_BLUE),
             (r"cause:&nbsp;[a-z&nbsp;]+stress", T_ORANGE), (r"EM&nbsp;━+", T_GREEN)]
    for pat, col in rules:
        s = re.sub(pat, lambda m: '<span style="color:%s;font-weight:700;">%s</span>' % (col, m.group(0)), s)
    return s


def terminal(stem, title, lines, width=1180, font=13.5):
    lh = font * 1.45
    h = 46 + len(lines) * lh + 24
    d = Drawio(stem, width, int(h))
    d.box(0, 0, width, h, "", fill=TERM_BG, stroke="#0d1117", sw=1, arc=12, shadow=False)
    for k, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        d.ellipse(18 + k * 22, 13, 13, 13, c)
    d.text(100, 6, width - 200, 26, title, font=13, color=T_DIM, align="center")
    d.line([(0, 36), (width, 36)], color="#2e3747", sw=1)
    body = "<br>".join(colourise(l) for l in lines)
    d.text(20, 46, width - 40, len(lines) * lh + 6,
           '<div style="font-family:Menlo,Courier New,monospace;font-size:%spx;line-height:%spx;color:%s;'
           'white-space:nowrap;">%s</div>' % (font, lh, TERM_FG, body), font=font, color=TERM_FG, valign="top",
           extra="fontFamily=Menlo;")
    d.save(stem, scale=2)


def demo_panels():
    raw = open(DEMO_TXT, encoding="utf-8").read().splitlines()

    def block(start, stop):
        i = next(k for k, l in enumerate(raw) if l.startswith(start))
        j = next(k for k in range(i + 1, len(raw)) if raw[k].startswith(stop))
        return raw[i:j]

    # stage 5: training (re-join lines the 104-column terminal wrapped)
    s5 = block("5  Training the BBN", "6  How well")
    train = [
        "5  Training the BBN on the devices' telemetry",
        "node_failure  P(fail within 2 epochs | evidence)   <- 4 latent conditions + reliability prior",
        "├── computational_stress  <- cpu_util, ram_util, queue_depth, cpu_trend, queue_trend",
        "├── network_degradation   <- uplink_latency, jitter_pktloss, handover_rate, latency_trend",
        "├── energy_stress         <- residual_energy, energy_drain",
        "├── thermal_stress        <- temperature, temp_trend",
        "└── reliability_prior     (device class: cloud / fog / edge)",
        "",
    ]
    em = [l for l in s5 if l.strip().startswith("EM ")][0]
    train.append(em.rstrip())
    for l in s5:
        if "distinct discretised patterns" in l:
            train.append(l.rstrip())
    train += ["", "Learned: P(fail within horizon | latent conditions), fog node"]
    k = next(i for i, l in enumerate(s5) if l.startswith("Learned:"))
    train += [l.rstrip() for l in s5[k + 1:k + 9] if l.strip()]
    terminal("demo_training", "bash run_demo.sh  —  stage 5 of 9: live BBN training", train, width=1010)

    # stage 7: live run
    s7 = block("7  LIVE", "8  Every")
    joined = []
    for l in s7:
        core = l.strip("│ ").rstrip()
        if not core or core.startswith("╭") or core.startswith("╰") or set(core) <= set("─ "):
            continue
        if joined and re.match(r"^[a-z]", core) and joined[-1].endswith("cause:"):
            joined[-1] += " " + core
        else:
            joined.append(core)
    live = [l for l in joined if l.startswith(("t =", "proactive", "t=", "online"))]
    live = [re.sub(r"\s{2,}cause:", "  cause:", l) for l in live]
    fin = [l for l in s7 if "run finished" in l][0].strip()
    fin2 = s7[s7.index([l for l in s7 if "run finished" in l][0]) + 1].strip()
    full = fin + " " + fin2
    cut = full.index(", 0 users")
    live += ["", full[:cut + 1], "  " + full[cut + 2:]]
    terminal("demo_live", "stage 7 of 9: the trained BBN running fault tolerance inside iFogSim  ·  scenario 7 (dense-edge, unseen) · seed 201",
             live, width=1340, font=13)

    # stage 8: policy race over all held-out runs
    s8 = block("Over all held-out runs", "9  Verdict")
    race = ["Policy                           Failures+migr./user  BBN vs this  Wilcoxon p  Users hit  Warm standby  Crit. misses"]
    for l in s8:
        m = re.match(r"^ (No fault|Cloud-only|Reactive|Threshold|Expected-loss|Proposed|Oracle)", l)
        if m:
            parts = re.split(r"\s{2,}", l.strip())
            name = parts[0]
            vals = parts[1:]
            if len(vals) == 4:      # the BBN row has no comparison columns
                vals = [vals[0], "", "", vals[1], vals[2], vals[3]]
            race.append("%-32s %10s %13s %11s %10s %13s %13s" % tuple([name] + vals))
    race += ["", "25 held-out runs per policy (5 unseen regimes x 5 unseen seeds), identical failure luck for every policy"]
    terminal("demo_policies", "stage 8 of 9: every fault-tolerance policy on the same held-out runs", race, width=1060)


if __name__ == "__main__":
    which = sys.argv[2:] or ["01", "03", "04", "05", "06", "07", "demo"]
    if "01" in which:
        fig01_02()
    if "03" in which:
        fig03()
    if "04" in which:
        fig04()
    if "05" in which:
        fig05()
    if "06" in which:
        fig06()
    if "07" in which:
        fig07()
    if "demo" in which:
        demo_panels()
    print("done")

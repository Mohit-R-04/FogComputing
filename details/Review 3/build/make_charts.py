# -*- coding: utf-8 -*-
"""Write the Review 3 result charts as draw.io files and export them to PNG.

Every value is copied from the project's own reports (model/comparison/results/model_comparison.md,
model/results/ft/policy_comparison.md, model/MODEL_IMPROVEMENTS.md); the reliability diagram is
binned from model/comparison/results/test_predictions.npz. Run with the project's Python:

    ../../../model/.venv/bin/python make_charts.py
"""

import math
import os

import numpy as np

import figlib as F
from figlib import Drawio

ROOT = os.path.normpath(os.path.join(F.HERE, "..", "..", ".."))
FS = 17            # base font size of chart text
BBN_C = F.BLUE
OTHER_C = "#8a97a8"


def fmt(v, nd=3):
    return ("%%.%df" % nd) % v


def axis_x(d, x0, x1, y, lo, hi, ticks, labfmt, title="", grid_top=None, font=FS - 2):
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    d.line([(x0, y), (x1, y)], color=F.INK, sw=1.2)
    for t in ticks:
        X = sx(t)
        if grid_top is not None:
            d.line([(X, grid_top), (X, y)], color=F.GRID, sw=1)
        d.line([(X, y), (X, y + 5)], color=F.INK, sw=1.2)
        d.text(X - 40, y + 8, 80, 20, labfmt(t), font=font, color=F.MUTED, align="center")
    if title:
        d.text(x0 - 200, y + 32, x1 - x0 + 400, 22, title, font=font + 1, color=F.INK, align="center")
    return sx


def axis_y(d, x, y0, y1, lo, hi, ticks, labfmt, title="", grid_right=None, font=FS - 2, log=False):
    """y0 = bottom pixel, y1 = top pixel."""
    if log:
        sy = lambda v: y0 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (y0 - y1)
    else:
        sy = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    d.line([(x, y0), (x, y1)], color=F.INK, sw=1.2)
    for t in ticks:
        Y = sy(t)
        if grid_right is not None:
            d.line([(x, Y), (grid_right, Y)], color=F.GRID, sw=1)
        d.line([(x - 5, Y), (x, Y)], color=F.INK, sw=1.2)
        d.text(x - 78, Y - 10, 70, 20, labfmt(t), font=font, color=F.MUTED, align="right")
    if title:
        d.text(x - 120, (y0 + y1) / 2 - 11, 40, 22, title, font=font + 1, color=F.INK, align="center",
               rotation=-90, extra="")
    return sy


def legend(d, x, y, items, font=FS - 1, dx=None, vertical=False):
    for k, (col, lab, kind) in enumerate(items):
        xx, yy = (x, y + k * 26) if vertical else (x + k * (dx or 160), y)
        if kind == "line":
            d.line([(xx, yy + 10), (xx + 28, yy + 10)], color=col, sw=3)
            d.ellipse(xx + 9, yy + 5, 10, 10, col)
        elif kind == "dash":
            d.line([(xx, yy + 10), (xx + 28, yy + 10)], color=col, sw=2, dashed=True)
        else:
            d.rect(xx, yy + 2, 22, 16, col)
        d.text(xx + 36, yy, 520, 20, lab, font=font, color=F.INK)


# --------------------------------------------------------------------------- #
# C1 discrimination: PR-AUC and ROC-AUC with 95% bootstrap intervals
# --------------------------------------------------------------------------- #

MODELS = [  # name, PR, lo, hi, ROC, lo, hi
    ("Random forest", .651, .592, .708, .846, .802, .886),
    ("BBN (proposed)", .639, .585, .695, .866, .829, .895),
    ("Gradient boosting", .629, .569, .690, .851, .809, .886),
    ("Naive Bayes", .522, .465, .578, .862, .822, .895),
    ("Logistic regression", .469, .419, .521, .840, .803, .876),
    ("FTAPA health index", .376, .324, .433, .860, .822, .888),
    ("Neural network (MLP)", .239, .210, .282, .837, .799, .870),
    ("CPU threshold", .130, .114, .150, .861, .827, .888),
    ("BBN, elicited only", .015, .010, .021, .750, .717, .777),
]


def c1():
    d = Drawio("Discrimination", 1000, 520)
    top, rh = 40, 40
    n = len(MODELS)
    bottom = top + n * rh
    lab_w = 215
    ax0, ax1 = lab_w + 20, lab_w + 20 + 360
    bx0, bx1 = ax1 + 70, ax1 + 70 + 300
    d.text(ax0, 6, ax1 - ax0, 24, "<b>(a)</b> PR-AUC (chance = 0.0026)", font=FS, align="center")
    d.text(bx0, 6, bx1 - bx0, 24, "<b>(b)</b> ROC-AUC (axis starts at 0.70)", font=FS, align="center")
    sa = axis_x(d, ax0, ax1, bottom, 0, 0.8, [0, .2, .4, .6, .8], lambda v: "%.1f" % v, "PR-AUC", grid_top=top)
    sb = axis_x(d, bx0, bx1, bottom, 0.70, 0.90, [.70, .75, .80, .85, .90], lambda v: "%.2f" % v, "ROC-AUC", grid_top=top)
    for i, (nm, pr, plo, phi, roc, rlo, rhi) in enumerate(MODELS):
        y = top + i * rh + 8
        bbn = nm.startswith("BBN (proposed)")
        col = BBN_C if bbn else ("#9cc0ef" if nm.startswith("BBN") else OTHER_C)
        d.text(0, y, lab_w, rh - 16, nm, font=FS - 1, color=F.INK, align="right", bold=bbn)
        for s, v, lo, hi, x0, base in [(sa, pr, plo, phi, ax0, 0), (sb, roc, rlo, rhi, bx0, 0.70)]:
            d.rect(x0, y, max(s(v) - x0, 1), rh - 16, col)
            d.line([(s(lo), y + (rh - 16) / 2), (s(hi), y + (rh - 16) / 2)], color=F.INK, sw=1.4)
            d.line([(s(lo), y + 3), (s(lo), y + rh - 19)], color=F.INK, sw=1.4)
            d.line([(s(hi), y + 3), (s(hi), y + rh - 19)], color=F.INK, sw=1.4)
            d.text(s(hi) + 6, y, 60, rh - 16, fmt(v), font=FS - 2, color=F.INK, bold=bbn)
    # ceilings
    d.line([(sa(.712), top - 4), (sa(.712), bottom)], color=F.RED, sw=1.6, dashed=True)
    d.text(sa(.712) - 70, bottom + 56, 140, 20, "ceiling 0.712", font=FS - 2, color=F.RED, align="center")
    d.line([(sb(.872), top - 4), (sb(.872), bottom)], color=F.RED, sw=1.6, dashed=True)
    d.text(sb(.872) - 70, bottom + 56, 140, 20, "ceiling 0.872", font=FS - 2, color=F.RED, align="center")
    d.save("chart_c1_discrimination")


# --------------------------------------------------------------------------- #
# C2 reliability diagram (log-log), binned from the held-out predictions
# --------------------------------------------------------------------------- #

def c2():
    z = np.load(os.path.join(ROOT, "model/comparison/results/test_predictions.npz"), allow_pickle=True)
    y = z["y"]
    names = [str(n) for n in z["names"]]
    edges = [0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0001]
    d = Drawio("Reliability", 980, 640)
    X0, X1, Y0, Y1 = 130, 600, 560, 40
    lo, hi = 3e-5, 1.0
    sx = lambda v: X0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (X1 - X0)
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    lf = lambda v: {1e-4: "0.0001", 1e-3: "0.001", 1e-2: "0.01", 1e-1: "0.1", 1: "1"}[v]
    d.line([(X0, Y0), (X1, Y0)], color=F.INK, sw=1.2)
    for t in ticks:
        d.line([(sx(t), Y1), (sx(t), Y0)], color=F.GRID, sw=1)
        d.text(sx(t) - 40, Y0 + 8, 80, 20, lf(t), font=FS - 2, color=F.MUTED, align="center")
    d.text(X0, Y0 + 34, X1 - X0, 22, "mean predicted P(fail) in the bin", font=FS - 1, align="center")
    sy = axis_y(d, X0, Y0, Y1, lo, hi, ticks, lf, grid_right=X1, log=True)
    d.text(-60, (Y0 + Y1) / 2 - 15, 220, 30, "observed failure rate", font=FS - 1, align="center", rotation=-90)
    d.line([(sx(lo), sy(lo)), (sx(hi), sy(hi))], color=F.FAINT, sw=1.4, dashed=True)
    d.text(sx(0.12) - 120, sy(0.35) - 10, 110, 20, "perfect calibration", font=FS - 3, color=F.FAINT, align="right")
    series = [("Random forest", OTHER_C, "RF"), ("Gradient boosting", F.ORANGE, "GB"), ("BBN (proposed)", BBN_C, "BBN")]
    for nm, col, _ in series:
        p = z["p%d" % names.index(nm)]
        pts = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (p >= a) & (p < b)
            if m.sum() >= 50 and y[m].mean() > 0:
                pts.append((sx(p[m].mean()), sy(y[m].mean())))
        d.line(pts, color=col, sw=3 if nm.startswith("BBN") else 2.2)
        for (px, py) in pts:
            r = 7 if nm.startswith("BBN") else 6
            d.ellipse(px - r, py - r, 2 * r, 2 * r, col, stroke=F.WHITE, sw=1.5)
    # legend and scores
    lx = 640
    d.text(lx, 60, 250, 24, "<b>Held-out pool</b>", font=FS)
    rows = [("BBN (proposed)", BBN_C, "0.00128", "0.0005"), ("Gradient boosting", F.ORANGE, "0.00134", "0.0006"),
            ("Random forest", OTHER_C, "0.00157", "0.0084")]
    d.text(lx + 170, 96, 80, 20, "Brier", font=FS - 2, color=F.MUTED, align="right")
    d.text(lx + 255, 96, 70, 20, "ECE", font=FS - 2, color=F.MUTED, align="right")
    for k, (nm, col, br, ece) in enumerate(rows):
        yy = 124 + k * 34
        d.line([(lx, yy + 10), (lx + 26, yy + 10)], color=col, sw=3)
        d.ellipse(lx + 7, yy + 4, 12, 12, col)
        d.text(lx + 34, yy, 120, 20, nm.replace(" (proposed)", ""), font=FS - 2, bold=nm.startswith("BBN"))
        d.text(lx + 170, yy, 80, 20, br, font=FS - 2, align="right")
        d.text(lx + 255, yy, 70, 20, ece, font=FS - 2, align="right")
    d.text(lx, 240, 320, 140, "Bins by predicted probability (bins with &ge; 50 snapshots). Between 0.01 and 0.3 the BBN "
           "over-predicts by about a factor of two; the random forest puts 22,476 snapshots near 0.04 "
           "that fail at 0.0006, a seventy-fold gap.",
           font=FS - 3, color=F.MUTED, valign="top")
    d.save("chart_c2_reliability")


# --------------------------------------------------------------------------- #
# C3 robustness under degradation-correlated (MNAR) telemetry loss
# --------------------------------------------------------------------------- #

LOSS = [0, 10, 20, 30, 40, 50, 60]
PR_MNAR = {
    "BBN (proposed)": [0.639, 0.534, 0.440, 0.282, 0.123, 0.019, 0.016],
    "Random forest": [0.651, 0.563, 0.478, 0.294, 0.128, 0.020, 0.016],
    "Gradient boosting": [0.629, 0.414, 0.241, 0.078, 0.023, 0.008, 0.005],
    "Naive Bayes": [0.522, 0.413, 0.345, 0.225, 0.099, 0.021, 0.027],
    "FTAPA health index": [0.376, 0.194, 0.144, 0.110, 0.068, 0.011, 0.012],
}
LL_MNAR = {
    "BBN (proposed)": [0.0082, 0.0098, 0.0116, 0.0141, 0.0167, 0.0187, 0.0189],
    "Random forest": [0.0159, 0.0162, 0.0164, 0.0174, 0.0198, 0.0252, 0.0243],
    "Gradient boosting": [0.0085, 0.0119, 0.0149, 0.0175, 0.0188, 0.0196, 0.0197],
    "Logistic regression": [0.0113, 0.0169, 0.0203, 0.0227, 0.0240, 0.0237, 0.0222],
}
COLS = {"BBN (proposed)": BBN_C, "Random forest": OTHER_C, "Gradient boosting": F.ORANGE,
        "Naive Bayes": F.GREEN, "FTAPA health index": F.GOLD, "Logistic regression": F.PURPLE}


def c3():
    d = Drawio("Robustness", 1100, 560)
    panels = [(110, 480, PR_MNAR, 0, 0.7, [0, .1, .2, .3, .4, .5, .6, .7], lambda v: "%.1f" % v,
               "<b>(a)</b> PR-AUC (higher is better)"),
              (650, 1020, LL_MNAR, 0.005, 0.027, [0.005, 0.01, 0.015, 0.02, 0.025], lambda v: "%.3f" % v,
               "<b>(b)</b> log-loss (lower is better)")]
    Y0, Y1 = 400, 50
    for x0, x1, data, lo, hi, ticks, lf, title in panels:
        d.text(x0, 12, x1 - x0, 24, title, font=FS, align="center")
        sx = axis_x(d, x0, x1, Y0, 0, 60, LOSS, lambda v: "%d%%" % v, "channels missing, more often on degrading nodes")
        sy = axis_y(d, x0, Y0, Y1, lo, hi, ticks, lf, grid_right=x1)
        for nm, vals in data.items():
            col = COLS[nm]
            pts = [(sx(a), sy(v)) for a, v in zip(LOSS, vals)]
            d.line(pts, color=col, sw=3.2 if nm.startswith("BBN") else 2)
            for px, py in pts:
                d.ellipse(px - 5, py - 5, 10, 10, col)
    items = [(COLS[k], k, "line") for k in ["BBN (proposed)", "Random forest", "Gradient boosting", "Naive Bayes",
                                             "FTAPA health index", "Logistic regression"]]
    for k, it in enumerate(items):
        legend(d, 110 + (k % 3) * 300, 470 + (k // 3) * 28, [it])
    d.save("chart_c3_robustness")


# --------------------------------------------------------------------------- #
# C4 closed-loop downtime per user
# --------------------------------------------------------------------------- #

POLICIES = [  # name, fault (failures + migrations), all downtime, warm standby %, critical deadline misses %
    ("No fault tolerance", 517.2, 603.4, "0.0", "3.82"),
    ("Reactive checkpoint + backup", 110.9, 201.2, "0.0", "4.39"),
    ("Threshold migration (FTAPA)", 37.7, 132.4, "100.0", "5.46"),
    ("Expected-loss controller + ML", 18.8, 86.2, "0.3", "0.22"),
    ("Proposed BBN framework", 20.0, 87.6, "0.2", "0.13"),
    ("Oracle predictor (bound)", 14.5, 81.5, "0.4", "0.12"),
    ("Cloud-only placement", 2.3, 2.3, "0.0", "100.00"),
]


def c4():
    d = Drawio("Downtime", 1180, 470)
    lab_w, top, rh = 280, 60, 44
    x0, x1 = lab_w + 20, lab_w + 20 + 560
    bottom = top + len(POLICIES) * rh
    sx = axis_x(d, x0, x1, bottom, 0, 650, [0, 100, 200, 300, 400, 500, 600], lambda v: "%d" % v,
                "downtime per user over a run (ms)", grid_top=top)
    cw = x1 + 20
    d.text(cw + 140, 8, 120, 46, "warm<br>standby", font=FS - 2, color=F.MUTED, align="right")
    d.text(cw + 260, 8, 130, 46, "critical<br>deadline misses", font=FS - 2, color=F.MUTED, align="right")
    for i, (nm, fault, alld, warm, miss) in enumerate(POLICIES):
        y = top + i * rh + 8
        bbn = nm.startswith("Proposed")
        dark = BBN_C if bbn else F.SLATE
        light = "#a9c7f2" if bbn else "#c9d1db"
        d.text(0, y, lab_w, rh - 16, nm, font=FS - 1, align="right", bold=bbn)
        d.rect(x0, y, max(sx(fault) - x0, 1.5), rh - 16, dark)
        if alld > fault:
            d.rect(sx(fault), y, sx(alld) - sx(fault), rh - 16, light)
        d.text(sx(alld) + 8, y, 160, rh - 16, "<b>%.1f</b> / %.1f" % (fault, alld), font=FS - 2,
               color=F.INK)
        d.text(cw + 140, y, 120, rh - 16, warm + "%", font=FS - 2, align="right", bold=bbn)
        mc = F.RED if float(miss) > 50 else F.INK
        d.text(cw + 260, y, 130, rh - 16, miss + "%", font=FS - 2, align="right", bold=bbn, color=mc)
    legend(d, x0, bottom + 64, [(F.SLATE, "failures + migrations (what fault tolerance controls)", "box"),
                                ("#c9d1db", "mobility handovers", "box")], dx=470)
    d.save("chart_c4_downtime")


# --------------------------------------------------------------------------- #
# C5 ablation
# --------------------------------------------------------------------------- #

def c5():
    rows = [("without the risk term in placement (N2)", 71.0, "p = 0.004", True),
            ("without selective replication (N3)", 57.7, "p = 0.020", True),
            ("without online CPT learning (N1)", 10.9, "p = 0.63, n.s.", False),
            ("without proactive migration (N2)", -7.2, "p = 0.12, n.s.", False)]
    d = Drawio("Ablation", 1000, 330)
    lab_w, top, rh = 360, 30, 50
    x0, x1 = lab_w + 30, lab_w + 30 + 470
    bottom = top + len(rows) * rh
    sx = axis_x(d, x0, x1, bottom, -20, 80, [-20, 0, 20, 40, 60, 80], lambda v: "%+d%%" % v if v else "0",
                "change in failures + migrations downtime against the full framework", grid_top=top)
    d.line([(sx(0), top - 6), (sx(0), bottom)], color=F.INK, sw=1.4)
    for i, (nm, v, p, sig) in enumerate(rows):
        y = top + i * rh + 10
        col = F.ORANGE if sig else "#b8c2ce"
        d.text(0, y, lab_w, rh - 20, nm, font=FS - 1, align="right")
        a, b = sorted([sx(0), sx(v)])
        d.rect(a, y, b - a, rh - 20, col)
        tx = sx(v) + 8 if v > 0 else sx(0) + 8
        d.text(tx, y, 220, rh - 20, "<b>%+.1f%%</b> &nbsp;<span style='color:%s'>%s</span>" % (v, F.MUTED, p),
               font=FS - 2)
    d.save("chart_c5_ablation")


# --------------------------------------------------------------------------- #
# C6 cost sensitivity
# --------------------------------------------------------------------------- #

def c6():
    mult = ["0.25", "0.5", "1", "2", "4"]
    mig = [104.1, 63.5, 34.7, 19.8, 10.8]
    mig_deg = [59.9, 43.4, 30.0, 18.4, 10.6]
    dt = [21.1, 21.2, 20.0, 22.4, 23.3]
    sd = [14.5, 14.8, 14.7, 18.3, 19.9]
    d = Drawio("Sensitivity", 1100, 470)
    Y0, Y1 = 360, 50
    # (a) migrations
    x0, x1 = 110, 480
    d.text(x0, 10, x1 - x0, 24, "<b>(a)</b> proactive migrations per run", font=FS, align="center")
    sy = axis_y(d, x0, Y0, Y1, 0, 120, [0, 20, 40, 60, 80, 100, 120], lambda v: "%d" % v, grid_right=x1)
    bw = (x1 - x0) / 5
    for k, (m, a, b) in enumerate(zip(mult, mig, mig_deg)):
        xx = x0 + k * bw + bw * 0.18
        d.rect(xx, sy(a), bw * 0.64, Y0 - sy(a), "#a9c7f2")
        d.rect(xx, sy(b), bw * 0.64, Y0 - sy(b), BBN_C)
        d.text(xx - 10, sy(a) - 22, bw * 0.64 + 20, 20, "%.1f" % a, font=FS - 2, align="center")
        d.text(xx - 10, Y0 + 8, bw * 0.64 + 20, 20, m + "&times;", font=FS - 2, color=F.MUTED, align="center")
    d.line([(x0, Y0), (x1, Y0)], color=F.INK, sw=1.2)
    d.text(x0, Y0 + 34, x1 - x0, 22, "migration cost multiplier", font=FS - 1, align="center")
    legend(d, x0, Y0 + 70, [("#a9c7f2", "all migrations", "box"), (BBN_C, "from genuinely worn nodes", "box")], dx=180)
    # (b) downtime
    x0, x1 = 660, 1030
    d.text(x0, 10, x1 - x0, 24, "<b>(b)</b> failures + migrations downtime / user", font=FS, align="center")
    sy = axis_y(d, x0, Y0, Y1, 0, 50, [0, 10, 20, 30, 40, 50], lambda v: "%d ms" % v, grid_right=x1)
    pts = []
    for k, (m, v, s) in enumerate(zip(mult, dt, sd)):
        xx = x0 + k * bw + bw / 2
        d.line([(xx, sy(v - s)), (xx, sy(v + s))], color=F.FAINT, sw=1.4)
        d.line([(xx - 6, sy(v - s)), (xx + 6, sy(v - s))], color=F.FAINT, sw=1.4)
        d.line([(xx - 6, sy(v + s)), (xx + 6, sy(v + s))], color=F.FAINT, sw=1.4)
        pts.append((xx, sy(v)))
        d.text(xx - 30, Y0 + 8, 60, 20, m + "&times;", font=FS - 2, color=F.MUTED, align="center")
    d.line(pts, color=BBN_C, sw=3)
    for (px, py), v in zip(pts, dt):
        d.ellipse(px - 6, py - 6, 12, 12, BBN_C)
        d.text(px + 8, py - 26, 60, 20, "%.1f" % v, font=FS - 2)
    d.line([(x0, Y0), (x1, Y0)], color=F.INK, sw=1.2)
    d.text(x0, Y0 + 34, x1 - x0, 22, "migration cost multiplier", font=FS - 1, align="center")
    d.text(x0, Y0 + 70, x1 - x0, 22, "mean over 25 held-out runs; whiskers &plusmn; 1 s.d.", font=FS - 2,
           color=F.MUTED, align="center")
    d.save("chart_c6_sensitivity")


# --------------------------------------------------------------------------- #
# C7 model-improvement ladder (cross-validated on training scenarios)
# --------------------------------------------------------------------------- #

def c7():
    steps = [("0  paper's Fig. 2 graph, tertile bins", 0.041, "0.0255"),
             ("1  supervised cut points (3 bins)", 0.434, "0.0154"),
             ("2  + five trend channels", 0.562, "0.0138"),
             ("3  + learned ranked-node tables, 8 bins", 0.653, "0.0131"),
             ("4  + bins per device class", 0.753, "0.0114"),
             ("5  Dirichlet strength 5 → 1 (selected)", 0.759, "0.0106")]
    d = Drawio("Ladder", 1000, 380)
    lab_w, top, rh = 370, 40, 46
    x0, x1 = lab_w + 25, lab_w + 25 + 430
    bottom = top + len(steps) * rh
    sx = axis_x(d, x0, x1, bottom, 0, 0.8, [0, .2, .4, .6, .8], lambda v: "%.1f" % v,
                "cross-validated PR-AUC (four scenario folds of the training runs)", grid_top=top)
    d.text(x1 + 10, 8, 120, 26, "CV log-loss", font=FS - 2, color=F.MUTED, align="right")
    for i, (nm, v, ll) in enumerate(steps):
        y = top + i * rh + 9
        last = i == len(steps) - 1
        d.text(0, y, lab_w, rh - 18, nm, font=FS - 1, align="left", bold=last)
        d.rect(x0, y, sx(v) - x0, rh - 18, BBN_C if last else "#7fa9e6")
        d.text(sx(v) + 8, y, 70, rh - 18, "%.3f" % v, font=FS - 2, bold=last)
        d.text(x1 + 10, y, 120, rh - 18, ll, font=FS - 2, align="right")
    d.save("chart_c7_ladder")


if __name__ == "__main__":
    for f in (c1, c2, c3, c4, c5, c6, c7):
        f()
    print("done")

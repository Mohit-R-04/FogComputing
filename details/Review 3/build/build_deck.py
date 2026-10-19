# -*- coding: utf-8 -*-
"""Build Review3_Presentation.pptx on the SSN template (details/SSN_Template.pptx).

The template's master, title slide and Thank You slide are kept; content slides reuse the
template's blue title bar. Every image is a draw.io export from ../figures.

    python build_deck.py          (python-pptx; from details/Review 3/build)
"""

import copy
import os
import subprocess
import sys

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
R3 = os.path.normpath(os.path.join(HERE, ".."))
FIG = os.path.join(R3, "figures")
TEMPLATE = os.path.normpath(os.path.join(R3, "..", "SSN_Template.pptx"))
OUT = os.path.join(R3, "Review3_Presentation.pptx")

FONT = "Times New Roman"
NAVY = RGBColor(0x15, 0x43, 0x60)
BLUE = RGBColor(0x00, 0x79, 0xBC)
LINK = RGBColor(0x00, 0x66, 0xCC)
INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x52, 0x60, 0x6D)
ORANGE = RGBColor(0xC2, 0x55, 0x1C)
GREEN = RGBColor(0x0F, 0x8A, 0x6A)
RED = RGBColor(0xB8, 0x32, 0x32)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
F_BLUE = RGBColor(0xE8, 0xF2, 0xFA)
F_GREY = RGBColor(0xF4, 0xF6, 0xF8)
F_GREEN = RGBColor(0xEE, 0xF8, 0xF3)
F_ORANGE = RGBColor(0xFD, 0xF4, 0xEC)
LINE = RGBColor(0xC9, 0xD1, 0xDB)

X0, X1 = 0.55, 12.78          # content area (inches)
Y0, Y1 = 1.32, 6.55


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #

def runs_from(text):
    """Split '**bold**' markup into (text, bold) runs."""
    out, bold = [], False
    for k, part in enumerate(text.split("**")):
        if part:
            out.append((part, bold))
        bold = not bold
    return out


def style_run(r, size, bold=False, color=INK, italic=False, font=FONT):
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    rPr = r._r.get_or_add_rPr()
    for tag in ("a:latin", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", font)


def set_bullet(p, char="•", color=BLUE, level=0, indent=0.26):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent * (level + 1)))))
    pPr.set("indent", str(int(-Inches(indent))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum", "a:buClr", "a:buFont"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    clr = etree.SubElement(pPr, qn("a:buClr"))
    s = etree.SubElement(clr, qn("a:srgbClr"))
    s.set("val", str(color))
    f = etree.SubElement(pPr, qn("a:buFont"))
    f.set("typeface", "Arial")
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", char)


def no_bullet(p):
    pPr = p._p.get_or_add_pPr()
    for el in pPr.findall(qn("a:buChar")):
        pPr.remove(el)
    if pPr.find(qn("a:buNone")) is None:
        etree.SubElement(pPr, qn("a:buNone"))


def textbox(slide, x, y, w, h, items, size=16, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            space=6, margin=0.05, line=1.05):
    """items: list of str or (str, dict). dict keys: bullet, bold, size, color, italic, level, align, space."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, "margin_" + side, Inches(margin))
    first = True
    for it in items:
        text, o = (it, {}) if isinstance(it, str) else it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = o.get("align", align)
        p.space_after = Pt(o.get("space", space))
        p.line_spacing = o.get("line", line)
        sz = o.get("size", size)
        col = o.get("color", color)
        for t, b in runs_from(text):
            r = p.add_run()
            r.text = t
            style_run(r, sz, bold=b or o.get("bold", False), color=col, italic=o.get("italic", False))
        if o.get("bullet"):
            set_bullet(p, o.get("char", "•"), o.get("bcolor", BLUE), o.get("level", 0))
        else:
            no_bullet(p)
    return tb


def B(text, **kw):
    kw.setdefault("bullet", True)
    return (text, kw)


def H(text, **kw):
    kw.setdefault("bold", True)
    kw.setdefault("color", NAVY)
    kw.setdefault("size", 18)
    kw.setdefault("space", 4)
    return (text, kw)


def card(slide, x, y, w, h, fill=F_BLUE, line=BLUE, radius=0.06, width=1.25):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line
    s.line.width = Pt(width)
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def badge(slide, x, y, label, fill=BLUE, d=0.46, size=15):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, "margin_" + side, 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    style_run(r, size, bold=True, color=WHITE)
    return s


def picture(slide, name, x, y, w=None, h=None, center_w=None):
    path = os.path.join(FIG, name)
    iw, ih = Image.open(path).size
    if w and h:
        if iw / ih > w / h:
            h = w * ih / iw
        else:
            w = h * iw / ih
    elif w:
        h = w * ih / iw
    else:
        w = h * iw / ih
    if center_w is not None:
        x = x + (center_w - w) / 2
    slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    return x, y, w, h


def table(slide, x, y, w, rows, widths, size=13, header_fill=BLUE, row_h=0.36, bold_rows=(), align=None,
          zebra=True):
    nr, nc = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gs.table
    tot = sum(widths)
    for ci, cw in enumerate(widths):
        t.columns[ci].width = Inches(w * cw / tot)
    for ri in range(nr):
        t.rows[ri].height = Inches(row_h)
        for ci in range(nc):
            c = t.cell(ri, ci)
            c.margin_left = c.margin_right = Inches(0.07)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if ri == 0:
                c.fill.fore_color.rgb = header_fill
            elif ri in bold_rows:
                c.fill.fore_color.rgb = RGBColor(0xD6, 0xE8, 0xF7)
            else:
                c.fill.fore_color.rgb = F_GREY if (zebra and ri % 2 == 0) else WHITE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (align[ci] if align else (PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER))
            for txt, b in runs_from(str(rows[ri][ci])):
                r = p.add_run()
                r.text = txt
                style_run(r, size, bold=(ri == 0 or ri in bold_rows or b),
                          color=WHITE if ri == 0 else INK)
    # plain table style (no banding from the theme)
    tblPr = gs._element.graphic.graphicData.tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    return gs


def stat(slide, x, y, w, value, label, color=BLUE, vsize=30, lsize=13, h=1.05):
    card(slide, x, y, w, h, fill=WHITE, line=LINE, width=1)
    textbox(slide, x + 0.08, y + 0.06, w - 0.16, 0.55, [(value, {"bold": True, "size": vsize, "color": color,
                                                               "align": PP_ALIGN.CENTER, "space": 0})])
    textbox(slide, x + 0.08, y + 0.58, w - 0.16, h - 0.62, [(label, {"size": lsize, "color": MUTED,
                                                                     "align": PP_ALIGN.CENTER, "space": 0,
                                                                     "line": 1.0})])


def caption(slide, text, y=6.2, x=X0, w=X1 - X0, size=12):
    textbox(slide, x, y, w, 0.34, [(text, {"size": size, "color": MUTED, "italic": True,
                                           "align": PP_ALIGN.CENTER, "space": 0})])


# --------------------------------------------------------------------------- #
# deck plumbing on the template
# --------------------------------------------------------------------------- #

class Deck:
    def __init__(self):
        self.prs = Presentation(TEMPLATE)
        s = self.prs.slides
        self.title_slide = s[0]
        self.thanks = s[5]
        self.title_bar = copy.deepcopy(s[1].shapes.title._element)
        self.layout = s[1].slide_layout
        # drop template slides 2-5
        ids = self.prs.slides._sldIdLst
        for sld in list(ids)[1:5]:
            rid = sld.get(qn("r:id"))
            self.prs.part.drop_rel(rid)
            ids.remove(sld)

    def slide(self, title, notes=""):
        s = self.prs.slides.add_slide(self.layout)
        for ph in list(s.placeholders):
            ph._element.getparent().remove(ph._element)
        bar = copy.deepcopy(self.title_bar)
        s.shapes._spTree.append(bar)
        tf = s.shapes.title.text_frame
        p = tf.paragraphs[0]
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        r = p.add_run()
        r.text = title
        style_run(r, 28, bold=True, color=WHITE)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def finish(self):
        ids = self.prs.slides._sldIdLst
        thanks_id = list(ids)[1]
        ids.remove(thanks_id)
        ids.append(thanks_id)
        # new slides were numbered from the shortened list and can collide with the template's slide6.xml
        self.prs.part.rename_slide_parts([sld.get(qn("r:id")) for sld in ids])
        self.prs.save(OUT)
        print("wrote", OUT)


def fill_title_slide(s):
    shapes = {sh.shape_id: sh for sh in s.shapes}
    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text
        if t.startswith("Course Name"):
            r = sh.text_frame.paragraphs[0].runs[0]
            r.text = "UG Project Work – Phase I – Review III"
        elif t.startswith("Project Title"):
            p = sh.text_frame.paragraphs[0]
            p.runs[0].text = ("Bayesian Belief Network-Driven Proactive Fault Tolerance for Microservice "
                              "Placement in Fog Computing Environments")
            p.runs[0].font.size = Pt(26)
            sh.top = Emu(int(Inches(3.02)))
            sh.left = Emu(int(Inches(1.17)))
            sh.width = Emu(int(Inches(11.0)))
        elif t.startswith("Student Name"):
            lines = ["Rahul Akshath Rajendran – 3122235002095", "Mohit R – 3122235002072", "",
                     "Project Guide: Dr. Sabireen H, Assistant Professor"]
            for p, txt in zip(sh.text_frame.paragraphs, lines):
                if p.runs:
                    p.runs[0].text = txt
                    for r in p.runs[1:]:
                        r.text = ""
            last = sh.text_frame.paragraphs[3].runs[0]
            last.font.color.rgb = NAVY
            sh.top = Emu(int(Inches(4.62)))


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #

def build():
    D = Deck()
    fill_title_slide(D.title_slide)
    D.title_slide.notes_slide.notes_text_frame.text = (
        "Good morning. We are presenting the third review of our project on Bayesian Belief Network-driven proactive "
        "fault tolerance for microservice placement in fog computing. Since Review 2 the design has been implemented "
        "inside iFogSim2 and evaluated end to end; today we show the implementation, a live demonstration and the results.")

    # 2 Outline ---------------------------------------------------------------
    s = D.slide("Outline", "A quick map of the talk: we restate the problem and the proposal briefly, spend most of the "
                "time on what is new since Review 2 — the dataset, the implementation, the demonstration and the results "
                "— and close with what the results do and do not show.")
    items = ["Problem Statement", "Motivation and Research Gap", "Proposed Work", "Methodology",
             "Dataset Description", "Implementation", "Demonstration", "Experiments and Results",
             "Inferences", "Conclusion and Future Work"]
    for k, it in enumerate(items):
        col, row = k // 5, k % 5
        x, y = 1.2 + col * 5.75, 1.75 + row * 0.9
        badge(s, x, y, str(k + 1), fill=BLUE if k < 4 else (GREEN if k < 8 else NAVY), d=0.55, size=17)
        textbox(s, x + 0.75, y + 0.02, 4.6, 0.55, [(it, {"size": 21, "color": INK, "space": 0})],
                anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, 1.2, 6.15, 10.9, 0.35, [("New since Review 2: dataset, implementation, demonstration and results (items 5–8)",
                                        {"size": 13, "color": GREEN, "italic": True, "space": 0})])

    # 3 Problem statement -----------------------------------------------------
    s = D.slide("Problem Statement", "Fog nodes fail far more often than data-centre machines, and they usually degrade "
                "before they stop. Existing fog fault tolerance reacts after the failure, or uses a fixed threshold on one "
                "metric. Neither puts a probability on failure, so neither can weigh the risk against the cost of acting. "
                "The figure shows the opportunity: the degradation window before a failure.")
    card(s, X0, Y0 + 0.05, X1 - X0, 1.55, fill=F_BLUE, line=BLUE)
    textbox(s, X0 + 0.25, Y0 + 0.15, X1 - X0 - 0.5, 1.4, [
        ("Fog nodes are fragile and usually **degrade before they fail**, yet fog fault tolerance is "
         "**reactive** (recovery starts after the failure) or **threshold-based** (a fixed cut-off on one metric). "
         "No existing mechanism estimates **the probability that a node fails within a given horizon**, so placement, "
         "migration and redundancy cannot be matched to the actual risk — and a silent telemetry channel, likely on a "
         "failing node, simply blinds them.", {"size": 18, "line": 1.1, "space": 0})], anchor=MSO_ANCHOR.MIDDLE)
    picture(s, "fig07_reactive_vs_proactive.png", X0, 3.12, w=X1 - X0, h=3.0)
    caption(s, "A proactive mechanism must act inside the degradation window, before the node fails", y=6.18)

    # 4 Motivation and gap ----------------------------------------------------
    s = D.slide("Motivation and Research Gap", "From the survey: three of five systems are reactive, the two proactive "
                "ones rely on fixed thresholds, and all reserve redundancy uniformly. None models node health as a hidden "
                "condition with uncertainty, which is exactly what a Bayesian network provides.")
    rows = [["Work", "Mechanism", "What it lacks"],
            ["Rajab & Younis 2021", "CPU-threshold checkpointing, primary-backup", "fixed threshold; backup for every task"],
            ["Syed et al. 2022", "SDN priority routing, ACK retransmission", "purely reactive; no replication"],
            ["Naim et al. 2016", "heartbeat polling, backup failover", "detection-driven; single backup"],
            ["Reyana et al. 2023", "PSO-ABC placement optimisation", "reliability only a static fitness term"],
            ["Gupta & Singh 2025 (FTAPA)", "health-index threshold migration", "threshold hard to set; standby for all"]]
    table(s, X0, Y0 + 0.08, 7.6, rows, [2.1, 2.9, 2.6], size=13, row_h=0.62,
          align=[PP_ALIGN.LEFT] * 3)
    card(s, 8.45, Y0 + 0.08, X1 - 8.45, 3.72, fill=F_ORANGE, line=ORANGE)
    textbox(s, 8.65, Y0 + 0.2, X1 - 8.85, 3.5, [
        H("The gap", color=ORANGE),
        B("Node health is never treated as a **latent, uncertain** quantity", bcolor=ORANGE, size=15),
        B("No mechanism reasons when **telemetry is incomplete**", bcolor=ORANGE, size=15),
        B("No **calibrated probability** to weigh against the cost of acting", bcolor=ORANGE, size=15),
        B("Redundancy is **uniform**, not proportional to risk", bcolor=ORANGE, size=15)])
    card(s, X0, 5.35, X1 - X0, 1.08, fill=F_GREEN, line=GREEN)
    textbox(s, X0 + 0.25, 5.4, X1 - X0 - 0.5, 1.0, [
        ("**Why a Bayesian Belief Network:** it needs no large failure corpus (tables start from expert priors), sums out "
         "missing channels instead of imputing them, outputs a probability that can be multiplied by a cost, and names "
         "the condition behind each alarm.", {"size": 16, "line": 1.05, "space": 0})], anchor=MSO_ANCHOR.MIDDLE)

    # 5 Proposed work -----------------------------------------------------------
    s = D.slide("Proposed Work", "Three novelties, one per objective, all now implemented. N1 predicts, N2 and N3 act on "
                "the prediction. Below them, the four things that changed between the Review 2 design and the system we "
                "evaluated.")
    cards = [("N1", "BBN failure prediction", BLUE, F_BLUE,
              ["P(node fails within 2 epochs | telemetry)", "missing channels summed out, never imputed",
               "names the condition behind the risk", "learned by MAP-EM, refined online"]),
             ("N2", "Expected-loss migration and risk-aware placement", GREEN, F_GREEN,
              ["migrate when P(fail) · C_fail > C_mig", "every session gets its own threshold",
               "placement utility over latency, energy and risk", "weights follow patient criticality"]),
             ("N3", "Criticality-aware selective replication", ORANGE, F_ORANGE,
              ["tier = least overhead + expected loss", "none · checkpoint · warm standby",
               "warm standbys capped at 20% of sessions", "recomputed every epoch"])]
    cw = (X1 - X0 - 0.5) / 3
    for k, (tag, ttl, col, fill, pts) in enumerate(cards):
        x = X0 + k * (cw + 0.25)
        card(s, x, Y0 + 0.08, cw, 3.25, fill=fill, line=col)
        badge(s, x + 0.2, Y0 + 0.25, tag, fill=col, d=0.62, size=16)
        textbox(s, x + 0.95, Y0 + 0.2, cw - 1.1, 0.75, [(ttl, {"bold": True, "size": 17, "color": col, "space": 0,
                                                                "line": 1.0})], anchor=MSO_ANCHOR.MIDDLE)
        textbox(s, x + 0.2, Y0 + 1.05, cw - 0.35, 2.2, [B(p, size=15, bcolor=col, space=5) for p in pts])
    textbox(s, X0, 4.82, X1 - X0, 0.4, [H("Since Review 2", size=17)])
    chg = [("Valid data", "degradation fault model; leak fixed (92.1% → 2.0%)"),
           ("Stronger BBN", "trend channels, device class, supervised bins, ranked nodes"),
           ("Built in iFogSim2", "controller with N1–N3 and six comparison policies"),
           ("Evaluated", "unseen regimes and seeds; paired statistical tests")]
    for k, (a, b) in enumerate(chg):
        x = X0 + k * (X1 - X0) / 4
        textbox(s, x, 5.22, (X1 - X0) / 4 - 0.15, 1.2, [(a, {"bold": True, "size": 15, "color": NAVY, "space": 2}),
                                                         (b, {"size": 13.5, "color": MUTED, "space": 0, "line": 1.0})])

    # 6 Block diagram ---------------------------------------------------------
    s = D.slide("Methodology — System Block Diagram", "Four blocks. A is the simulated fog environment with telemetry "
                "agents and a fault injector. B learns the BBN offline from 31 runs. C is the controller inside the "
                "simulator with the three novelties. D evaluates everything on unseen runs. The numbered flows go from "
                "telemetry to dataset, to trained model, to live inference, to decisions, to actuation, and back as online "
                "learning.")
    picture(s, "fig01_block_diagram.png", X0, Y0 - 0.02, w=X1 - X0, h=5.25, center_w=X1 - X0)

    # 7 BBN ---------------------------------------------------------------------
    s = D.slide("Methodology — The Bayesian Belief Network", "Thirteen evidence channels feed four hidden conditions, "
                "which feed the failure target together with the device class. Because each channel feeds only one "
                "condition, inference is exact and takes about 3.6 microseconds. A missing channel is summed out. The "
                "tables are ranked nodes: one learned point per parent state instead of an exponential table, which is "
                "what made eight bins per channel learnable.")
    picture(s, "fig02_bbn_structure.png", X0, Y0 + 0.1, w=6.9, h=4.6)
    textbox(s, 7.75, Y0 + 0.05, X1 - 7.75, 5.1, [
        H("Structure"),
        B("13 evidence channels (8 from Review 2 + 5 trends) → 4 latent conditions → node_failure", size=15),
        B("Device class conditions every channel and condition", size=15),
        H("Learning"),
        B("**Supervised bins:** ≤ 8 per channel and class, placed by exact DP on mutual information", size=15),
        B("**Ranked-node tables:** 40 numbers per class instead of 32,768 rows", size=15),
        B("**MAP-EM:** latent conditions never observed; 43,374 patterns, ≈ 10 s", size=15),
        H("Inference"),
        B("Exact; a dark channel is **summed out**, never imputed", size=15),
        B("Cause = condition whose P(high) rose most above its prior", size=15)], space=4)

    # 8 Decision rules ----------------------------------------------------------
    s = D.slide("Methodology — Decision Rules", "Placement scores each node with a utility in which risk is one term and "
                "the weights depend on the patient. Migration compares expected loss with the cost of moving, so each "
                "session derives its own threshold: 0.020 for a critical patient, 0.049 for best-effort. Replication picks "
                "the tier with least overhead plus expected loss.")
    rules = [("Placement  (Eq. 4)", BLUE, F_BLUE,
              "U(m, n) = α(1 − L̂) + β(1 − Ê) + γ(1 − P(Fₙ = fail | e))",
              ["L̂: predicted loop latency / 150 ms deadline (with M/M/1 queueing)",
               "Ê: access distance / 5 km (radio energy)",
               "(α, β, γ) = (0.30, 0.10, 0.60) critical … (0.60, 0.25, 0.15) best-effort"]),
             ("Migration  (Eq. 5)", GREEN, F_GREEN,
              "P(Fₙ = fail | e) · C_fail(m)  >  C_mig(m, n → n′)",
              ["each session’s own threshold P* = C_mig / C_fail",
               "unprotected: P* = 0.020 critical, 0.022 standard, 0.049 best-effort",
               "no global threshold to tune"]),
             ("Replication  (Eqs. 6–7)", ORANGE, F_ORANGE,
              "τ*(m) = argmin [ O(τ) + R(m) · D(τ) ]  ;  R(m) = P(Fₙ = fail | e) · κ(m)",
              ["outage: none 330 ms · checkpoint ≈ 60–80 ms · warm 35 ms",
               "critical patient: P = 0.001 → none; 0.05 → checkpoint; 0.30 → warm",
               "warm standby for at most 20% of sessions"])]
    for k, (ttl, col, fill, eq, pts) in enumerate(rules):
        y = Y0 + 0.05 + k * 1.75
        card(s, X0, y, X1 - X0, 1.6, fill=fill, line=col)
        textbox(s, X0 + 0.2, y + 0.1, 3.1, 0.5, [(ttl, {"bold": True, "size": 18, "color": col, "space": 0})])
        textbox(s, X0 + 0.2, y + 0.58, 5.75, 0.9, [(e, {"size": 17, "italic": True, "color": INK, "space": 2,
                                                       "line": 1.0}) for e in eq.split("  ;  ")],
                anchor=MSO_ANCHOR.MIDDLE)
        textbox(s, 6.6, y + 0.12, X1 - 6.75, 1.4, [B(p, size=14, bcolor=col, space=3) for p in pts],
                anchor=MSO_ANCHOR.MIDDLE)

    # 9 Control loop ------------------------------------------------------------
    s = D.slide("Methodology — Execution Flow per Epoch", "Every 100 time units the controller reads the reports, infers "
                "risk, learns from matured outcomes, migrates where the expected loss justifies it, re-tiers redundancy and "
                "lets the simulation run. Between epochs failures are detected after a 30 ms heartbeat timeout and "
                "recovered according to each session's tier. Placement is called whenever a session needs a node.")
    picture(s, "fig03_control_loop.png", X0, Y0, w=X1 - X0, h=5.2, center_w=X1 - X0)

    # 10 Dataset 1 ------------------------------------------------------------
    s = D.slide("Dataset Description — Sources and Telemetry", "There is no public failure trace for fog, so the "
                "dataset is generated by our instrumented iFogSim2 from real inputs shipped with the simulator: the "
                "Melbourne CBD topology and mobility traces, and a cardiac-monitoring application. Every device reports "
                "thirteen channels every epoch; an empty value means the channel did not report.")
    textbox(s, X0, Y0 + 0.02, 5.6, 5.1, [
        H("Real inputs (shipped with iFogSim2)"),
        B("**Melbourne CBD edge resources:** 1 cloud + 118 fog gateways at real coordinates", size=15),
        B("**Mobility traces:** 20–80 patients walk and hand over between gateways", size=15),
        B("**Cardiac monitoring:** vitals → ECG analyser on the fog node → alarm; deadline 150 ms", size=15),
        B("Patients: 25% critical, 45% standard, 30% best-effort", size=15),
        H("One record"),
        B("One device at one epoch (100 time units); 13 channels + device class", size=15),
        B("**Label:** does the node fail within the next 200 time units?", size=15),
        B("Hidden health and failure cause stored, **never used as features**", size=15)], space=4)
    rows = [["Channel", "Feeds", "From iFogSim2"],
            ["cpu_util, ram_util, queue_depth", "computational", "host work, RAM, queues"],
            ["cpu_trend, queue_trend", "computational", "agent’s last 3 readings"],
            ["uplink_latency, jitter_pktloss", "network", "link delay + load; arrivals"],
            ["handover_rate, latency_trend", "network", "mobility; trend"],
            ["residual_energy, energy_drain", "energy", "power model"],
            ["temperature, temp_trend", "thermal", "fog/cloud only"]]
    table(s, 6.45, Y0 + 0.1, X1 - 6.45, rows, [2.8, 1.45, 2.1], size=13, row_h=0.58,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
    textbox(s, 6.45, 5.6, X1 - 6.45, 0.8, [("Missing values stay missing: temperature on edge devices, jitter with "
                                            "fewer than three tuples, trends without history.",
                                            {"size": 13, "italic": True, "color": MUTED, "space": 0})])

    # 11 Dataset 2 ------------------------------------------------------------
    s = D.slide("Dataset Description — Fault Model and Split", "iFogSim has no failures, so we added a degradation "
                "model: hidden health wears faster under stress and shows fail-slow symptoms that the telemetry can see. "
                "One failure in five is sudden. Whole regimes and seeds are held out, and the leak audit confirms "
                "independence: 2 percent of shared failures coincide, which is chance.")
    picture(s, "fig05_split.png", X0, Y0 + 0.05, w=5.9, h=4.0)
    textbox(s, 6.75, Y0 + 0.02, X1 - 6.75, 2.6, [
        H("Fault model"),
        B("Hidden health wears faster under load, backlog, heat, energy depletion and poor links", size=15),
        B("**Fail-slow symptoms** inside iFogSim: more CPU per tuple, slower links, more power and heat", size=15),
        B("Wear-out, battery and **sudden** failures (≈ 1 in 5, no precursor)", size=15)], space=4)
    rows = [["Pool", "Runs", "Snapshots", "Failing", "Prevalence"],
            ["Training", "21", "314,323", "1,508", "0.48%"],
            ["Held out", "10", "160,823", "422", "0.26%"]]
    table(s, 6.75, 4.2, X1 - 6.75, rows, [1.3, 0.8, 1.3, 1.1, 1.3], size=14, row_h=0.42)
    card(s, 6.75, 5.65, X1 - 6.75, 0.78, fill=F_GREEN, line=GREEN)
    textbox(s, 6.9, 5.68, X1 - 7.05, 0.72, [("**Leak audit:** shared failures at the same epoch 92.1% (Review 2 "
                                             "injector) → **2.0%** now, i.e. chance level", {"size": 14, "space": 0})],
            anchor=MSO_ANCHOR.MIDDLE)
    caption(s, "12 operating regimes; held-out runs use unseen regimes and unseen seeds", y=5.6, x=X0, w=5.9)

    # 12 Implementation ---------------------------------------------------------
    s = D.slide("Implementation — Technology Stack and Code", "The Java side is one new package of 31 classes inside an "
                "unmodified iFogSim2; the Python side builds the data, trains and evaluates. They share one model file, and "
                "the Java and Python engines agree to 1e-15. Fifty tests and a pgmpy cross-check guard correctness, and two "
                "scripts regenerate everything or run the demo.")
    picture(s, "fig06_implementation.png", X0, Y0 - 0.02, w=X1 - X0, h=5.25, center_w=X1 - X0)

    # 13 Demo 1 -----------------------------------------------------------------
    s = D.slide("Demonstration — bash run_demo.sh (1/3)", "The demo runs in about 35 seconds and has nine stages. Here is "
                "stage five: the BBN is retrained live from the devices' telemetry, from 314 thousand snapshots compressed "
                "to 43 thousand patterns, in about ten seconds, and the learned failure table is printed. The retrained "
                "model is identical to the one behind every reported result.")
    picture(s, "demo_training.png", X0, Y0 + 0.05, w=8.45, h=3.6)
    facts = [("314,323 → 43,374", "snapshots merged into distinct patterns"),
             ("10.6 s", "60 MAP-EM iterations, live"),
             ("identical", "retrained model = the one behind every result")]
    for k, (a, b) in enumerate(facts):
        x = X0 + k * 2.88
        stat(s, x, 5.0, 2.73, a, b, color=GREEN if k < 2 else NAVY, vsize=21, lsize=12.5, h=1.3)
    stages = ["Build the iFogSim extension", "The simulated system", "Telemetry from the devices",
              "Dataset and leak audit", "**Live BBN training**", "Prediction comparison",
              "**Live BBN inside iFogSim**", "**All policies, same run**", "Verdict"]
    textbox(s, 9.3, Y0 + 0.05, X1 - 9.3, 0.45, [H("Nine stages, ≈ 35 s")])
    for k, st in enumerate(stages):
        y = Y0 + 0.62 + k * 0.53
        badge(s, 9.3, y, str(k + 1), fill=GREEN if "**" in st else NAVY, d=0.4, size=12)
        textbox(s, 9.82, y - 0.02, X1 - 9.82, 0.44, [(st, {"size": 15, "space": 0})], anchor=MSO_ANCHOR.MIDDLE)

    # 14 Demo 2 -----------------------------------------------------------------
    s = D.slide("Demonstration — BBN Controlling iFogSim Live (2/3)", "Scenario 7 with seed 201, both unseen in "
                "training. fog-61 is wearing out: the BBN puts its failure probability at 0.41 and blames energy stress, "
                "and each session moves when that exceeds its own threshold. fog-38 fails suddenly, which no predictor can "
                "see; its four users are detected and recovered by cold restart. The controller needs 2.24 ms per epoch.")
    picture(s, "demo_live.png", X0, Y0 + 0.02, w=X1 - X0, h=3.9, center_w=X1 - X0)
    notes3 = [("Proactive migration", "sessions leave fog-61 at P(fail) = 0.41 > P* = 0.272 — cause: energy stress", GREEN),
              ("Own thresholds", "same node, different P*: 0.111 standard vs 0.273 best-effort", BLUE),
              ("Sudden failure", "fog-38 has no precursor; 4 users recovered by cold restart (330 ms)", RED)]
    for k, (a, b, col) in enumerate(notes3):
        x = X0 + k * (X1 - X0) / 3
        textbox(s, x, 5.3, (X1 - X0) / 3 - 0.2, 1.2, [(a, {"bold": True, "size": 16, "color": col, "space": 2}),
                                                       (b, {"size": 14, "color": INK, "space": 0, "line": 1.0})])

    # 15 Demo 3 -----------------------------------------------------------------
    s = D.slide("Demonstration — Every Policy on the Same Runs (3/3)", "Stage eight runs all seven policies with "
                "identical failure luck and then shows the summary over all 25 held-out runs with paired Wilcoxon tests. "
                "Against FTAPA the BBN framework cuts fault downtime by 55 percent with a standby for 0.2 percent of "
                "sessions instead of all of them. Against the ML-driven controller the difference is not significant with "
                "complete telemetry.")
    picture(s, "demo_policies.png", X0, Y0 + 0.05, w=X1 - X0, h=3.5, center_w=X1 - X0)
    stat(s, X0, 5.15, 2.9, "−97%", "vs no fault tolerance", color=GREEN)
    stat(s, X0 + 3.1, 5.15, 2.9, "−55%", "vs FTAPA threshold migration", color=GREEN)
    stat(s, X0 + 6.2, 5.15, 2.9, "0.2%", "warm standby (FTAPA: 100%)", color=BLUE)
    stat(s, X0 + 9.3, 5.15, X1 - X0 - 9.3, "+18%", "vs ML controller, p = 0.12 (n.s.)", color=MUTED)

    # 16 Experimental setup ----------------------------------------------------
    s = D.slide("Experiments — Protocol and Performance Metrics", "Every modelling choice is made on folds of the "
                "training runs; the held-out pool is scored once. Prediction is judged by ranking and by probability "
                "quality, because the controller multiplies the probability by a cost. The closed loop is judged by "
                "downtime split into the part fault tolerance controls and the part it cannot.")
    picture(s, "fig04_pipeline.png", X0, Y0, w=X1 - X0, h=2.55, center_w=X1 - X0)
    cols = [("Protocol", NAVY, ["choices on 4 scenario folds of training runs", "held-out pool scored once",
                                "bootstrap CIs over nodes", "paired Wilcoxon tests, 25 runs"]),
            ("Prediction metrics", BLUE, ["ROC-AUC, PR-AUC (chance 0.0026)", "Brier, log-loss, ECE (calibration)",
                                          "precision, recall, F1; early warning", "robustness to MCAR / MNAR loss"]),
            ("Service metrics", GREEN, ["downtime per user: failures + migrations", "users hit / evacuated; availability",
                                        "critical deadline misses (150 ms), p95", "warm-standby overhead"]),
            ("Baselines", ORANGE, ["predictors: RF, GB, NB, LR, MLP, FTAPA, CPU", "policies: none, cloud-only, reactive",
                                   "FTAPA threshold migration", "ML-driven controller; oracle bound"])]
    cw = (X1 - X0 - 0.45) / 4
    for k, (ttl, col, pts) in enumerate(cols):
        x = X0 + k * (cw + 0.15)
        card(s, x, 4.05, cw, 2.4, fill=WHITE, line=col)
        textbox(s, x + 0.12, 4.1, cw - 0.24, 2.3, [(ttl, {"bold": True, "size": 16, "color": col, "space": 4})] +
                [B(p, size=13, bcolor=col, space=3) for p in pts])

    # 17 Model selection ------------------------------------------------------
    s = D.slide("Results — Improving the BBN (Training Folds Only)", "The Review 2 graph with tertile bins was almost "
                "blind: cross-validated PR-AUC 0.041. Supervised bins were the biggest single gain, then trends, ranked "
                "nodes with eight bins, and per-class bins. All of this was decided on training scenarios; the held-out "
                "runs were untouched until the end.")
    picture(s, "chart_c7_ladder.png", X0, Y0 + 0.2, w=8.0, h=4.6)
    card(s, X0, 4.85, 8.0, 1.25, fill=F_BLUE, line=BLUE)
    textbox(s, X0 + 0.2, 4.9, 7.6, 1.15, [
        ("**Then, once, on the held-out pool:** ROC-AUC 0.866 · PR-AUC 0.639 · Brier 0.00128 · ECE 0.0005 "
         "— not directly comparable with the CV score: the training runs fail more often (0.48% vs 0.26%), and PR-AUC rises with prevalence",
         {"size": 15, "space": 0, "line": 1.05})], anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, 8.85, Y0 + 0.15, X1 - 8.85, 5.0, [
        H("What moved accuracy"),
        B("Tertiles hide the tail where failures live: **0.041**", size=15),
        B("Supervised cut points: **0.434**", size=15),
        B("Trend channels catch drift: **0.562**", size=15),
        B("Ranked nodes make 8 bins learnable: **0.653**", size=15),
        B("Per-class bins: **0.753 → 0.759**", size=15),
        ("24 configurations, 4 scenario folds; the held-out pool was scored once, afterwards.",
         {"size": 13, "italic": True, "color": MUTED, "space": 0})], space=6)

    # 18 Prediction results -----------------------------------------------------
    s = D.slide("Results — Failure Prediction against Eight Models", "On 160 thousand held-out snapshots the BBN has the "
                "best ROC-AUC and a PR-AUC tied with the random forest. Because 29 percent of failures are sudden, no model "
                "can exceed 0.872 and 0.712; the BBN reaches 99 and 90 percent of that. The random forest is slightly "
                "ahead on F1 and early detection, and we say so.")
    picture(s, "chart_c1_discrimination.png", X0, Y0 + 0.05, w=8.3, h=4.9)
    stat(s, 9.15, Y0 + 0.05, X1 - 9.15, "0.866", "ROC-AUC — best of 9 models", color=BLUE)
    stat(s, 9.15, Y0 + 1.25, X1 - 9.15, "0.639", "PR-AUC — tied with RF (0.651)", color=BLUE)
    stat(s, 9.15, Y0 + 2.45, X1 - 9.15, "99% / 90%", "of the achievable ROC / PR-AUC", color=GREEN, vsize=26)
    stat(s, 9.15, Y0 + 3.65, X1 - 9.15, "63%", "failing nodes flagged in time (RF 68%)", color=ORANGE)
    caption(s, "159,813 held-out snapshots, 422 failing (prevalence 0.26%); 95% bootstrap intervals over nodes", y=6.2)

    # 19 Calibration & robustness ---------------------------------------------
    s = D.slide("Results — Calibration and Missing Telemetry", "Left: the random forest ranks well but is badly "
                "calibrated, with 22 thousand snapshots at 0.04 that fail at 0.0006; the BBN is within about a factor of "
                "two. Right: when channels go missing more often on degrading nodes, the BBN keeps the best probabilities "
                "at every loss rate. Naive Bayes ranks best under heavy loss.")
    picture(s, "chart_c2_reliability.png", X0, Y0 + 0.05, w=5.7, h=4.05)
    picture(s, "chart_c3_robustness.png", 6.55, Y0 + 0.35, w=X1 - 6.55, h=3.6)
    textbox(s, X0, 5.5, 5.7, 0.95, [B("Best **Brier 0.00128** and **ECE 0.0005**: probabilities mean what they say — "
                                      "they are multiplied by costs", size=15)])
    textbox(s, 6.55, 5.5, X1 - 6.55, 0.95, [B("Lowest **log-loss at every loss rate**; under heavy loss Naive Bayes "
                                              "ranks best (ROC 0.821 vs 0.700)", size=15)])

    # 20 Closed loop ------------------------------------------------------------
    s = D.slide("Results — Closed-Loop Fault Tolerance in iFogSim", "Seven policies on the same 25 held-out runs. The "
                "dark part of each bar is what fault tolerance controls. The BBN framework cuts it by 97, 85 and 55 percent "
                "against no fault tolerance, reactive checkpointing and FTAPA, and misses the 150 ms deadline on 0.13 "
                "percent of critical alarms. Cloud-only looks good on downtime but misses every deadline.")
    picture(s, "chart_c4_downtime.png", X0, Y0 + 0.15, w=8.6, h=4.2)
    stat(s, 9.45, Y0 + 0.05, X1 - 9.45, "−97 / −85 / −55%", "fault downtime vs none, reactive, FTAPA",
         color=GREEN, vsize=22)
    stat(s, 9.45, Y0 + 1.25, X1 - 9.45, "4.5", "users hit per run (FTAPA 54.1)", color=BLUE)
    stat(s, 9.45, Y0 + 2.45, X1 - 9.45, "0.13%", "critical alarms past 150 ms", color=BLUE)
    stat(s, 9.45, Y0 + 3.65, X1 - 9.45, "98.9%", "of the oracle’s downtime reduction", color=NAVY)
    caption(s, "Cloud-only placement: least downtime, but its 310 ms p95 loop delay misses every critical deadline", y=6.2)

    # 21 Degraded telemetry + ablation -----------------------------------------
    s = D.slide("Results — Degraded Telemetry and Ablation", "With 30 percent of channels missing on degrading nodes, "
                "the BBN-driven controller beats the ML-driven one by 29 percent: imputing a healthy median misleads, "
                "summing out does not. FTAPA, with a standby for everyone, is the most robust here. The ablation shows "
                "that risk-aware placement and selective replication carry the gain; migration alone is not significant.")
    textbox(s, X0, Y0 + 0.02, 5.8, 0.45, [H("30% of channels missing on degrading nodes")])
    rows = [["Policy", "Fault downtime / user", "Total downtime / user"],
            ["FTAPA threshold", "33.9 ms", "133.7 ms"],
            ["Controller + ML", "90.6 ms", "156.2 ms"],
            ["**BBN framework**", "65.9 ms", "131.8 ms"]]
    table(s, X0, Y0 + 0.55, 5.8, rows, [2.2, 1.8, 1.8], size=15, row_h=0.6, bold_rows=(3,))
    textbox(s, X0, 4.35, 5.8, 2.2, [
        B("BBN vs ML controller: **−29.2%** fault downtime (p = 0.006)", size=16),
        B("Summing out a dark channel beats imputing a healthy-looking median", size=16),
        B("FTAPA is most robust on fault downtime (p = 0.004) — bought with a standby for **every** session", size=16)])
    textbox(s, 6.75, Y0 + 0.02, X1 - 6.75, 0.45, [H("Ablation: remove one component")])
    picture(s, "chart_c5_ablation.png", 6.75, Y0 + 0.75, w=X1 - 6.75, h=2.6)
    textbox(s, 6.75, 4.35, X1 - 6.75, 2.2, [
        B("Risk-aware placement (+71%) and selective replication (+58%) are significant", size=16),
        B("Online learning and migration alone: not significant once placement and replication are risk-aware", size=16)])

    # 22 Sensitivity + explanations -------------------------------------------
    s = D.slide("Results — Cost Sensitivity and Explanations", "Multiplying every migration cost by 0.25 to 4 changes "
                "the number of migrations tenfold but downtime stays at 20 to 23 ms per user: no threshold needs tuning. "
                "The named causes match how nodes failed: battery and wear-out failures are blamed on energy stress, and "
                "for half of the sudden failures nothing is raised, as it should be.")
    picture(s, "chart_c6_sensitivity.png", X0, Y0 + 0.1, w=7.2, h=3.6)
    textbox(s, X0, 5.0, 7.2, 1.5, [B("Migrations fall **104 → 11** per run as the cost rises 16-fold", size=16),
                                    B("Fault downtime stays **20.0–23.3 ms** per user: the rule calibrates itself",
                                      size=16)])
    textbox(s, 8.05, Y0 + 0.02, X1 - 8.05, 0.45, [H("Does the named cause match?")])
    rows = [["Actual cause", "Energy", "Network", "None"],
            ["Battery (22)", "**0.86**", "0.14", "0.00"],
            ["Wear-out (278)", "**0.92**", "0.04", "0.00"],
            ["Sudden (122)", "0.16", "0.30", "**0.52**"]]
    table(s, 8.05, Y0 + 0.55, X1 - 8.05, rows, [1.7, 1.0, 1.0, 0.9], size=14, row_h=0.5)
    textbox(s, 8.05, 4.45, X1 - 8.05, 2.0, [
        B("Energy drain is the earliest wear symptom in this simulator", size=14),
        B("Sudden failures have no precursor: mostly **no condition raised**", size=14),
        B("Elicited tables alone (no failure data): ROC-AUC **0.750**", size=14)])

    # 23 Inferences ---------------------------------------------------------------
    s = D.slide("Inferences — Why a BBN, and Where It Is Not Ahead", "The honest summary. The BBN matches the best black "
                "box at ranking and is clearly the best at calibration, which is what decisions need. Its real advantage "
                "shows when telemetry fails with the node. It is not ahead everywhere: the random forest has slightly "
                "higher PR-AUC and F1, and nothing predicts sudden failures.")
    card(s, X0, Y0 + 0.05, 6.0, 5.05, fill=F_GREEN, line=GREEN)
    textbox(s, X0 + 0.2, Y0 + 0.15, 5.6, 4.9, [
        H("Supported by the results", color=GREEN),
        B("Ranks failures as well as the best of 8 models (ROC-AUC best; PR-AUC tied)", bcolor=GREEN, size=17),
        B("**Best-calibrated** probabilities — what cost-based rules need", bcolor=GREEN, size=17),
        B("Summing out missing telemetry: **−29%** fault downtime vs ML under MNAR loss", bcolor=GREEN, size=17),
        B("Beats thresholds with **0.2%** standby instead of 100%", bcolor=GREEN, size=17),
        B("Usable before any failure is seen (elicited ROC-AUC 0.750)", bcolor=GREEN, size=17),
        B("Explains every alarm with a named condition", bcolor=GREEN, size=17)], space=9)
    card(s, 6.78, Y0 + 0.05, X1 - 6.78, 5.05, fill=F_ORANGE, line=ORANGE)
    textbox(s, 6.98, Y0 + 0.15, X1 - 7.18, 4.9, [
        H("Where it is not ahead", color=ORANGE),
        B("Random forest: slightly higher PR-AUC (0.651), F1 (0.687), early detection (68%)", bcolor=ORANGE, size=17),
        B("Complete telemetry: ML-driven controller equal (+18%, p = 0.12)", bcolor=ORANGE, size=17),
        B("Heavy telemetry loss: every model collapses; Naive Bayes ranks best", bcolor=ORANGE, size=17),
        B("Sudden failures (29%) cannot be predicted — only redundancy helps", bcolor=ORANGE, size=17),
        B("Simulation study: a synthetic fault model, accelerated wear", bcolor=ORANGE, size=17)], space=5)

    # 24 Conclusion -------------------------------------------------------------
    s = D.slide("Conclusion and Future Work", "We built and evaluated the framework end to end. The BBN gives calibrated, "
                "explainable failure probabilities that drive placement, migration and replication, and the framework "
                "removes most of the fault-related downtime with very little standby. Next steps are real traces, a "
                "dynamic Bayesian network, and distributed inference.")
    textbox(s, X0, Y0 + 0.05, 7.2, 5.1, [
        H("Conclusion"),
        B("A causal BBN, extended and learned by MAP-EM, predicts fog-node failure from incomplete telemetry and "
          "explains each alarm", size=17),
        B("Inside iFogSim2 it drives placement, expected-loss migration and selective replication", size=17),
        B("On unseen regimes and seeds: ROC-AUC 0.866 with the best calibration; fault downtime −97% / −85% / −55% "
          "vs none / reactive / FTAPA, with 0.2% warm standby", size=17),
        B("Advantage over ML appears when telemetry fails with the node", size=17)], space=12)
    card(s, 8.1, Y0 + 0.05, X1 - 8.1, 5.05, fill=F_BLUE, line=BLUE)
    textbox(s, 8.3, Y0 + 0.15, X1 - 8.5, 4.9, [
        H("Future work"),
        B("**Real failure traces** from fog / edge deployments, to test whether the learned precursors transfer", size=17),
        B("**Dynamic Bayesian network** to model degradation over time and extend the horizon", size=17),
        B("**Distributed inference** across fog clusters — no single controller to fail", size=17)], space=14)

    # 25 References -------------------------------------------------------------
    s = D.slide("References", "")
    refs = [
        "[1] R. Mahmud, S. Pallewatta, M. Goudarzi, R. Buyya, “iFogSim2: An extended iFogSim simulator for mobility, clustering, and microservice management,” J. Systems and Software, vol. 190, 2022.",
        "[2] H. T. Rajab, M. F. Younis, “Dynamic fault tolerance aware scheduling for healthcare system on fog computing,” Iraqi J. Science, pp. 308–318, 2021.",
        "[3] S. A. Syed et al., “QoS aware and fault tolerance based software-defined vehicular networks using cloud-fog computing,” Sensors, vol. 22, no. 1, 2022.",
        "[4] M. H. Naim, J. M. Zain, K. A. Jalil, “Fault tolerance mechanism for software application through fog computing as middleware,” IJCDS, vol. 5, no. 3, 2016.",
        "[5] A. Reyana et al., “Opportunities of IoT in fog computing for high fault tolerance and sustainable energy optimization,” Sustainability, vol. 15, no. 11, 2023.",
        "[6] S. Gupta, N. Singh, “Optimized fault-tolerant placement of microservices in distributed fog networks,” IJDIIC, vol. 4, no. 4, pp. 52–69, 2025.",
        "[7] J. Pearl, Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann, 1988.",
        "[8] N. E. Fenton, M. Neil, J. G. Caballero, “Using ranked nodes to model qualitative judgments in Bayesian networks,” IEEE TKDE, vol. 19, no. 10, 2007.",
        "[9] A. P. Dempster, N. M. Laird, D. B. Rubin, “Maximum likelihood from incomplete data via the EM algorithm,” J. Royal Stat. Soc. B, vol. 39, 1977.",
        "[10] P. Weber, G. Medina-Oliva, C. Simon, B. Iung, “Overview on Bayesian networks applications for dependability, risk analysis and maintenance,” EAAI, vol. 25, no. 4, 2012.",
        "[11] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, “On calibration of modern neural networks,” ICML, 2017.",
        "[12] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. 1, no. 6, 1945.",
    ]
    textbox(s, X0, Y0 + 0.02, X1 - X0, 5.15, [(r, {"size": 12.5, "space": 5, "line": 1.0}) for r in refs])

    D.finish()


def to_pdf():
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", R3, OUT],
                   check=True, capture_output=True, timeout=300)
    print("wrote", os.path.splitext(OUT)[0] + ".pdf")


if __name__ == "__main__":
    build()
    if "--no-pdf" not in sys.argv:
        to_pdf()

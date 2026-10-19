# -*- coding: utf-8 -*-
"""Tiny helper for writing draw.io (.drawio) files and exporting them to PNG.

Every figure of the Review 3 paper and presentation is a draw.io diagram: the
scripts in this folder write the .drawio XML, and the draw.io desktop CLI renders it.
"""

import os
import subprocess
from xml.sax.saxutils import quoteattr

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
DRAWIO = "/opt/homebrew/bin/drawio"

# palette shared with details/Project_Block_Diagram_colour.drawio
INK = "#1f2933"
MUTED = "#52606d"
FAINT = "#7b8794"
GRID = "#d9dee4"
SLATE = "#4a5d73"
GREEN = "#0f8a6a"
BLUE = "#1f6fd1"
ORANGE = "#c2551c"
RED = "#b83232"
GOLD = "#b7891f"
PURPLE = "#6b4fbb"
F_BLUE = "#e3eefc"
F_BLUE2 = "#eef4fd"
F_GREEN = "#eef8f3"
F_ORANGE = "#fdf4ec"
F_GREY = "#f4f6f8"
F_RED = "#fbeaea"
F_PURPLE = "#f1edfa"
WHITE = "#ffffff"

FONT = "Helvetica"


def style(**kw):
    return ";".join("%s=%s" % (k, v) for k, v in kw.items()) + ";"


class Drawio:
    def __init__(self, name, width=1600, height=1000):
        self.name = name
        self.width = width
        self.height = height
        self.cells = []
        self.n = 1

    def _id(self):
        self.n += 1
        return "c%d" % self.n

    # -- vertices ------------------------------------------------------------
    def vertex(self, x, y, w, h, value="", st="", cid=None):
        cid = cid or self._id()
        self.cells.append(
            '<mxCell id=%s value=%s style=%s vertex="1" parent="1">'
            '<mxGeometry x="%.2f" y="%.2f" width="%.2f" height="%.2f" as="geometry"/></mxCell>'
            % (quoteattr(cid), quoteattr(value), quoteattr(st), x, y, w, h))
        return cid

    def box(self, x, y, w, h, value="", fill=WHITE, stroke=SLATE, sw=1.4, font=14,
            color=INK, rounded=True, arc=8, bold=False, align="center", valign="middle",
            dashed=False, extra="", shadow=False):
        st = style(rounded=1 if rounded else 0, arcSize=arc, absoluteArcSize=1, whiteSpace="wrap",
                   html=1, fillColor=fill, strokeColor=stroke, strokeWidth=sw,
                   fontFamily=FONT, fontSize=font, fontColor=color, align=align,
                   verticalAlign=valign, fontStyle=1 if bold else 0,
                   dashed=1 if dashed else 0, shadow=1 if shadow else 0,
                   spacingLeft=6, spacingRight=6) + extra
        return self.vertex(x, y, w, h, value, st)

    def rect(self, x, y, w, h, fill, stroke="none", sw=1, extra=""):
        st = style(rounded=0, html=1, fillColor=fill, strokeColor=stroke, strokeWidth=sw) + extra
        return self.vertex(x, y, w, h, "", st)

    def ellipse(self, x, y, w, h, fill, stroke="none", sw=1, value="", font=11, color=WHITE):
        st = style(shape="ellipse", perimeter="ellipsePerimeter", html=1, fillColor=fill, strokeColor=stroke, strokeWidth=sw,
                   fontFamily=FONT, fontSize=font, fontColor=color, fontStyle=1,
                   align="center", verticalAlign="middle")
        return self.vertex(x, y, w, h, value, st)

    def text(self, x, y, w, h, value, font=13, color=INK, align="left", valign="middle",
             bold=False, italic=False, rotation=0, family=FONT, extra=""):
        fs = (1 if bold else 0) + (2 if italic else 0)
        st = style(text=1, html=1, whiteSpace="wrap", fontFamily=family, fontSize=font,
                   fontColor=color, align=align, verticalAlign=valign, fontStyle=fs,
                   spacing=0, rotation=rotation, strokeColor="none", fillColor="none") + extra
        return self.vertex(x, y, w, h, value, st)

    def badge(self, cx, cy, label, fill=SLATE, r=13, font=12):
        return self.ellipse(cx - r, cy - r, 2 * r, 2 * r, fill, value=label, font=font)

    # -- edges ---------------------------------------------------------------
    def edge(self, src, tgt, color=SLATE, sw=1.6, dashed=False, end="block", start="none",
             label="", font=12, points=None, ortho=True, extra="", exit=None, entry=None):
        st = style(html=1, rounded=1, endArrow=end, endFill=1, startArrow=start, startFill=1,
                   strokeColor=color, strokeWidth=sw, dashed=1 if dashed else 0,
                   fontFamily=FONT, fontSize=font, fontColor=MUTED, labelBackgroundColor=WHITE)
        if ortho:
            st += "edgeStyle=orthogonalEdgeStyle;"
        if exit:
            st += "exitX=%s;exitY=%s;exitDx=0;exitDy=0;" % exit
        if entry:
            st += "entryX=%s;entryY=%s;entryDx=0;entryDy=0;" % entry
        st += extra
        pts = ""
        if points:
            pts = '<Array as="points">%s</Array>' % "".join(
                '<mxPoint x="%.2f" y="%.2f"/>' % p for p in points)
        cid = self._id()
        self.cells.append(
            '<mxCell id=%s value=%s style=%s edge="1" parent="1" source=%s target=%s>'
            '<mxGeometry relative="1" as="geometry">%s</mxGeometry></mxCell>'
            % (quoteattr(cid), quoteattr(label), quoteattr(st), quoteattr(src), quoteattr(tgt), pts))
        return cid

    def line(self, pts, color=INK, sw=1.2, dashed=False, end="none", start="none", extra=""):
        """Free polyline through absolute points (no source/target cells)."""
        st = style(html=1, endArrow=end, endFill=1, startArrow=start, startFill=1,
                   strokeColor=color, strokeWidth=sw, dashed=1 if dashed else 0,
                   rounded=0) + extra
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        mids = ""
        if len(pts) > 2:
            mids = '<Array as="points">%s</Array>' % "".join(
                '<mxPoint x="%.2f" y="%.2f"/>' % p for p in pts[1:-1])
        cid = self._id()
        self.cells.append(
            '<mxCell id=%s value="" style=%s edge="1" parent="1">'
            '<mxGeometry relative="1" as="geometry">'
            '<mxPoint x="%.2f" y="%.2f" as="sourcePoint"/><mxPoint x="%.2f" y="%.2f" as="targetPoint"/>%s'
            '</mxGeometry></mxCell>' % (quoteattr(cid), quoteattr(st), x0, y0, x1, y1, mids))
        return cid

    # -- output --------------------------------------------------------------
    def xml(self):
        return ('<mxfile host="drawio" type="device"><diagram name=%s id=%s>'
                '<mxGraphModel dx="%d" dy="%d" grid="0" gridSize="10" guides="1" tooltips="1" '
                'connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="%d" '
                'pageHeight="%d" math="0" shadow="0" background="#ffffff"><root>'
                '<mxCell id="0"/><mxCell id="1" parent="0"/>%s</root></mxGraphModel>'
                '</diagram></mxfile>' % (quoteattr(self.name), quoteattr(self.name.replace(" ", "_")),
                                         self.width, self.height, self.width, self.height,
                                         "".join(self.cells)))

    def save(self, stem, scale=2.5, export=True):
        os.makedirs(FIG, exist_ok=True)
        path = os.path.join(FIG, stem + ".drawio")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.xml())
        if export:
            export_png(path, scale)
        return path


def export_png(path, scale=2.5, border=12):
    png = os.path.splitext(path)[0] + ".png"
    subprocess.run([DRAWIO, "-x", "-f", "png", "-s", str(scale), "-b", str(border),
                    "-o", png, path], check=True, capture_output=True)
    return png


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

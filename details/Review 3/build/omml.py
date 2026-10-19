# -*- coding: utf-8 -*-
"""A very small builder for Office Math (OMML), so the paper's equations are native
Word equations rather than pictures or tab-aligned italics."""

from xml.sax.saxutils import escape

M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" ' \
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def R(text, plain=False, bold=False):
    """A math run. Single letters are italic by default; plain=True sets upright text."""
    sty = ""
    if plain and bold:
        sty = '<m:rPr><m:sty m:val="b"/></m:rPr>'
    elif plain:
        sty = '<m:rPr><m:sty m:val="p"/></m:rPr>'
    elif bold:
        sty = '<m:rPr><m:sty m:val="bi"/></m:rPr>'
    return ('<m:r>%s<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'
            '<m:t xml:space="preserve">%s</m:t></m:r>' % (sty, escape(text)))


def T(text):
    """Upright text (operators, words such as 'fail')."""
    return R(text, plain=True)


def cat(*parts):
    return "".join(parts)


def _e(x):
    return "<m:e>%s</m:e>" % x


def S(base, sub):
    return "<m:sSub><m:e>%s</m:e><m:sub>%s</m:sub></m:sSub>" % (base, sub)


def P(base, sup):
    return "<m:sSup><m:e>%s</m:e><m:sup>%s</m:sup></m:sSup>" % (base, sup)


def SP(base, sub, sup):
    return ("<m:sSubSup><m:e>%s</m:e><m:sub>%s</m:sub><m:sup>%s</m:sup></m:sSubSup>"
            % (base, sub, sup))


def FR(num, den):
    return "<m:f><m:num>%s</m:num><m:den>%s</m:den></m:f>" % (num, den)


def NA(chr_, sub, sup, body):
    hide = ""
    if not sup:
        hide += '<m:supHide m:val="1"/>'
    if not sub:
        hide += '<m:subHide m:val="1"/>'
    return ('<m:nary><m:naryPr><m:chr m:val="%s"/><m:limLoc m:val="undOvr"/>%s</m:naryPr>'
            '<m:sub>%s</m:sub><m:sup>%s</m:sup><m:e>%s</m:e></m:nary>'
            % (chr_, hide, sub, sup, body))


def SUM(sub, body, sup=""):
    return NA("∑", sub, sup, body)


def PROD(sub, body, sup=""):
    return NA("∏", sub, sup, body)


def D(body, beg="(", end=")"):
    return ('<m:d><m:dPr><m:begChr m:val="%s"/><m:endChr m:val="%s"/></m:dPr><m:e>%s</m:e></m:d>'
            % (escape(beg), escape(end), body))


def HAT(base):
    return '<m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e>%s</m:e></m:acc>' % base


def BAR(base):
    return '<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>%s</m:e></m:bar>' % base


def LIM(func, below):
    """argmin-style operator with a limit underneath."""
    return "<m:limLow><m:e>%s</m:e><m:lim>%s</m:lim></m:limLow>" % (func, below)


def omath(body):
    return '<m:oMath %s>%s</m:oMath>' % (M, body)

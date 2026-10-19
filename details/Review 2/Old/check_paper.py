# -*- coding: utf-8 -*-
"""Verify that the five revisions the guide asked for are actually in the built .docx.

Nothing on this machine renders .docx, so this re-opens the generated file and
asserts structurally.  Run:  python3 check_paper.py
"""

import os
import re
import sys
import unicodedata

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_content as C

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "Review2_ResearchPaper.docx")
EMU_IN = 914400.0

fails = []
notes = []


def check(cond, msg):
    (notes if cond else fails).append(("PASS" if cond else "FAIL", msg))


def flatten(d):
    """Yield ('p', text) and ('tbl', [[cell,...]]) in document order."""
    for child in d.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield "p", Paragraph(child, d)
        elif child.tag == qn("w:tbl"):
            yield "tbl", Table(child, d)


d = docx.Document(DOC)
items = list(flatten(d))
paras = [p for k, p in items if k == "p"]
tables = [t for k, t in items if k == "tbl"]
ptexts = [p.text for p in paras]

ref_start = next(i for i, t in enumerate(ptexts) if t.strip() == "References")
body_texts = ptexts[:ref_start]
ref_texts = [t for t in ptexts[ref_start + 1:] if t.strip()]
table_texts = [c.text for t in tables for r in t.rows for c in r.cells]
all_body = " ".join(body_texts + table_texts)

# --- 1. single column, A4, 1in margins -------------------------------------- #
for i, sec in enumerate(d.sections):
    cols = sec._sectPr.find(qn("w:cols"))
    num = cols.get(qn("w:num")) if cols is not None else "1"
    check(num in (None, "1"), "section %d is single-column (w:cols num=%s)" % (i, num))
    check(abs(sec.page_width / EMU_IN - 8.27) < 0.02 and
          abs(sec.page_height / EMU_IN - 11.69) < 0.02, "section %d page size is A4" % i)
    check(all(abs(getattr(sec, m) / EMU_IN - 1.0) < 0.02
              for m in ("left_margin", "right_margin", "top_margin", "bottom_margin")),
          "section %d margins are 1in" % i)

# --- 2. Chicago citations ---------------------------------------------------- #
stray = re.findall(r"\[\d+\]", all_body)
check(not stray, "no IEEE-style [n] markers remain in body or tables (found %d)" % len(stray))

check(len(ref_texts) == len(C.REFERENCES),
      "reference list has %d entries (expected %d)" % (len(ref_texts), len(C.REFERENCES)))


def sortkey(s):
    # strip accents so 'Ferná' etc. sort the way Word displays them
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


check(ref_texts == sorted(ref_texts, key=sortkey), "reference list is alphabetical by surname")

for key, ref in C.REFERENCES.items():
    inline = "%s (%d)" % (ref["short"], ref["year"])
    paren = "%s %d" % (ref["short"], ref["year"])
    check(inline in all_body or paren in all_body,
          "reference %-12s is cited in the text" % key)
    check(any(r.startswith(ref["sort"][0]) for r in ref_texts),
          "reference %-12s appears in the reference list" % key)

# every author-date marker in the text resolves to a listed reference
known = {"%s %d" % (r["short"], r["year"]) for r in C.REFERENCES.values()}
known |= {"%s (%d)" % (r["short"], r["year"]) for r in C.REFERENCES.values()}
for m in re.finditer(r"\(([A-Z][A-Za-z.,'’\- ]{2,60}? \d{4}(?:; [A-Z][A-Za-z.,'’\- ]{2,60}? \d{4})*)\)",
                     " ".join(body_texts)):
    for one in m.group(1).split("; "):
        check(one in known, "in-text citation %r resolves to a listed reference" % one)

# --- 3. removals -------------------------------------------------------------- #
check(len(tables) == 2, "document has exactly 2 tables (found %d)" % len(tables))
for gone in ("TABLE III", "TABLE IV", "Device Instantiation", "ApplicationConfig.json",
             "edgeResources-melbCBD", "Sharma", "Khan"):
    check(gone not in all_body, "removed content absent: %r" % gone)
sec4 = [t for t in body_texts if re.match(r"^[A-G]\.  ", t)]
after_iv = body_texts[body_texts.index("IV.  Experimental Setup and Dataset Description"):]
iv_subs = [t.split("  ")[0] for t in after_iv if re.match(r"^[A-G]\.  ", t)]
check(iv_subs == ["A.", "B.", "C."], "Section IV subheadings renumbered to A, B, C (got %s)" % iv_subs)

# --- 4. formulas explained ---------------------------------------------------- #
eq_idx = [i for i, t in enumerate(body_texts) if re.search(r"\t\(\d\)$", t)]
check(len(eq_idx) == 4, "4 numbered equations found (got %d)" % len(eq_idx))
for i in eq_idx:
    check(body_texts[i + 1].strip() == "where",
          "equation %r is followed by a 'where' block" % body_texts[i].strip()[:30])

GREEK = "αβγκτΔΣΠ"
for blk in [b for b in C.BODY if b[0] == "eq"]:
    expr, num = blk[1], blk[2]
    wh = next(w for w in C.BODY[C.BODY.index(blk):] if w[0] == "where")
    defined = " ".join(s + " " + t for s, t in wh[1])
    atoms = set(re.findall(r"[A-Za-z]̂?_\[[A-Za-z]+\]", expr))          # C_[fail]
    atoms |= set(re.findall(r"(?<![A-Za-z])(pa|[A-Za-z]̂)(?![A-Za-z])", expr))
    atoms |= {a for a in re.findall(r"(?<![A-Za-ẑ])([A-Za-z])(?![A-Za-ẑ_])", expr)}
    atoms |= {c for c in expr if c in GREEK}
    atoms -= {"f", "a", "i", "l"}                                            # from the word 'fail'
    missing = sorted(a for a in atoms if a not in defined)
    check(not missing, "every symbol in %s is defined in its where block (missing %s)"
          % (num, missing))

# --- 5. figures ---------------------------------------------------------------- #
xml = d.element.body.xml
extents = [(round(int(a) / EMU_IN, 2), round(int(b) / EMU_IN, 2))
           for a, b in re.findall(r'<wp:extent cx="(\d+)" cy="(\d+)"', xml)]
check(len(extents) == 3, "3 images embedded (found %d)" % len(extents))
check(extents == [(3.6, 5.43), (5.6, 4.31), (4.8, 4.83)],
      "image sizes match the single-column layout: %s" % (extents,))
caps = [t for t in body_texts if re.match(r"^Fig\. \d\.", t)]
check(len(caps) == 3, "3 figure captions (found %d)" % len(caps))
check(all(w <= 6.27 for w, _ in extents), "no image is wider than the 6.27in text measure")

# --- report -------------------------------------------------------------------- #
for status, msg in notes:
    print("  %s  %s" % (status, msg))
print()
for status, msg in fails:
    print("  %s  %s" % (status, msg))
print("\n%d passed, %d failed" % (len(notes), len(fails)))
sys.exit(1 if fails else 0)

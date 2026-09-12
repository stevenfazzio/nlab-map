"""Stage 2: parse the raw mirror into data/corpus.parquet, one row per page we will map.

Reads  data/raw/nlab-content/pages/**/{content.md,name}
       data/raw/nlab-content-html/pages/**/content.html   (only for the "Last revised on" / "Created on" footer)
Writes data/corpus.parquet

Pages are classified into a page type from the wiki's own `category:` tags plus heuristics for the
untagged cases (people pages without the tag, "X -- table" fragments, "X > history" subpages), and
everything except concept and reference pages is dropped. Each kept page then gets an embedding text
made of its title plus the section where the page describes itself: the Idea-like section, else the
Definition, else the Statement, else the preamble before the first heading. Pages with none of those
are dropped too. The tier a page landed in is kept as a column, since it is a colormap in the plot.

The link graph over ALL pages (including dropped ones) gives each kept page an in-degree.
"""

import collections
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from common import RAW, validate_stage_output, write_parquet_safely

MD_ROOT = RAW / "nlab-content" / "pages"
HTML_ROOT = RAW / "nlab-content-html" / "pages"
OUT = Path("data/corpus.parquet")

EMBED_WORD_CAP = 400
SUMMARY_WORD_CAP = 60
SECTION_MIN_WORDS = 5  # an Idea/Definition/Statement shorter than this is a stub pointer, not a description
PREAMBLE_MIN_WORDS = 25

# --- page-source syntax -------------------------------------------------------------------------
SIDEBAR_RE = re.compile(r"\+--\s*\{:\s*\.rightHandSide\}(.*?)\n=--\s*\n=--", re.S)
CONTEXT_HEAD_RE = re.compile(r"^####\s*(.+?)\s*#*\s*$", re.M)
INCLUDE_RE = re.compile(r"\[\[!include\s+[^\]]+\]\]")
REDIRECT_RE = re.compile(r"\[\[!redirects\s+([^\]]+)\]\]")
CATEGORY_RE = re.compile(r"^\s*category:\s*(.+?)\s*$", re.M)
LINK_RE = re.compile(r"\[\[(?!!)([^\]|#]+?)(?:[#|][^\]]*)?\]\]")
TOC_RE = re.compile(r"\\tableofcontents|^#?\s*Contents\s*#?\s*$|^\* table of contents\s*$|^\{:\s*toc\}\s*$", re.M)
HEADING_RE = re.compile(
    r"^\s*(?:(#{1,5})\s*|\\(section|subsection|subsubsection)\{)([^}\n#]+?)\}?\s*#*\s*(?:\{#[^}]*\})?\s*$", re.M
)
TEX_LEVEL = {"section": 2, "subsection": 3, "subsubsection": 4}
IDEA_RE = re.compile(
    r"^(idea|overview|introduction|summary|motivation|general idea|the idea|description|about)\b", re.I
)
DEFN_RE = re.compile(r"^(definition|definitions|defintion|definiton|the definition)\b", re.I)
STMT_RE = re.compile(r"^(statement|statements|theorem|the theorem)\b", re.I)
NAV_NAME_RE = re.compile(r"\s-+\s*(contents|references)\s*$|contents$", re.I)
FRAGMENT_NAME_RE = re.compile(r"\s--?-?\s*(table|section|chapter|subsection)\s*$", re.I)
# untagged-people heuristics
SELECTED_WRITINGS_RE = re.compile(
    r"^\s*(?:#{1,4}\s*|\\section\{)\s*selected\s+(writings|works|publications|papers)", re.I | re.M
)
BIO_RE = re.compile(
    r"\b(is|was) an? (?:[\w-]+ )?(mathematician|physicist|philosopher|logician|computer scientist|professor|"
    r"researcher|scientist|topologist|geometer|algebraist)\b",
    re.I,
)
HOMEPAGE_RE = re.compile(
    r"\b(homepage|home page|webpage|personal page|website|MathGenealogy|arXiv author|Google ?Scholar|ORCID|"
    r"Wikipedia)\b",
    re.I,
)
CAPNAME_RE = re.compile(r"[A-ZÀ-Ý][\w'.\-]+(?: [A-ZÀ-Ý][\w'.\-]*\.?)+")
LAST_REVISED_RE = re.compile(r"(?:Last revised|Created) on\s+(\w+\s+\d{1,2},\s+\d{4})\s+at\s+(\d{1,2}:\d{2}:\d{2})")


def read_pages(root):
    for name_file in sorted(root.rglob("name"), key=lambda p: int(p.parent.name)):
        d = name_file.parent
        md = d / "content.md"
        yield (
            int(d.name),
            name_file.read_text(encoding="utf-8", errors="replace").strip(),
            (md.read_text(encoding="utf-8", errors="replace") if md.exists() else ""),
        )


def body_text(text):
    """Page source minus the navigation furniture: sidebar, includes, redirects, category line, TOC."""
    b = SIDEBAR_RE.sub("", text)
    b = INCLUDE_RE.sub("", b)
    b = REDIRECT_RE.sub("", b)
    b = CATEGORY_RE.sub("", b)
    b = TOC_RE.sub("", b)
    return b.strip()


def parse_sections(body):
    """Hierarchical split: a section runs until the next heading of the same or higher level."""
    ms = list(HEADING_RE.finditer(body))
    secs = []
    for i, m in enumerate(ms):
        lvl = len(m.group(1)) if m.group(1) else TEX_LEVEL[m.group(2)]
        end = len(body)
        for m2 in ms[i + 1 :]:
            l2 = len(m2.group(1)) if m2.group(1) else TEX_LEVEL[m2.group(2)]
            if l2 <= lvl:
                end = m2.start()
                break
        secs.append((m.group(3).strip(), body[m.end() : end].strip()))
    preamble = body[: ms[0].start()].strip() if ms else body
    return secs, preamble


MATH_SPAN_RE = re.compile(r"(\$\$.*?\$\$|\$[^$\n]*\$)", re.S)


def _clean_prose(t):
    """Markup rules that must not run inside math (they would eat subscripts and stars)."""
    t = re.sub(r"\[\[([^\]|]+?)\|([^\]]+?)\]\]", r"\2", t)  # [[target|shown]] -> shown
    t = re.sub(r"\[\[([^\]]+?)\]\]", r"\1", t)  # [[target]] -> target
    t = re.sub(r"\[([^\]]*?)\]\([^)]*\)", r"\1", t)  # [text](url) -> text
    t = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", t, flags=re.S | re.I)
    t = re.sub(r"<[^>\n]{1,200}>", " ", t)  # remaining HTML tags
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t, flags=re.S)
    t = re.sub(r"(?<![\w\\])[_*](\S.*?\S|\S)[_*](?!\w)", r"\1", t)
    t = re.sub(r"^\s*[*+-]\s+", "", t, flags=re.M)  # list bullets
    return t


def clean(md):
    """Markdown+itex -> plain-ish text. Math stays as LaTeX; wiki links and markup become their text."""
    t = re.sub(r"<svg\b.*?</svg>", " ", md, flags=re.S | re.I)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"\(\s*(?:\.\.\.|…)\s*\)|^\s*(?:\.\.\.|…)\s*$", " ", t, flags=re.M)  # "(...)" = section not yet written
    t = re.sub(r"^\s*(?:\+--|=--).*$", "", t, flags=re.M)  # nLab's +-- {: .num_defn} ... =-- block markers
    t = re.sub(r"\\(begin|end)\{[^}]*\}", " ", t)  # theorem/proof environments: keep the content
    t = re.sub(r"^\s*\\(?:sub)*section\{([^}]*)\}.*$", r"\1", t, flags=re.M)  # \section{X} -> X
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)  # nested markdown heading markers -> text
    t = re.sub(r"\{[:#][^}]*\}", " ", t)  # {: .class} and {#anchor} attribute lists
    parts = MATH_SPAN_RE.split(t)
    t = "".join(part if i % 2 else _clean_prose(part) for i, part in enumerate(parts))
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def strip_heading_lines(md):
    """Drop sub-heading lines (e.g. "General", "Definition 2.1") so they do not lead the hovercard summary.
    Applied to the summary only: embed_text must stay byte-identical to what was embedded."""
    return re.sub(r"^\s*(?:#{1,6}|\\(?:sub)*section\{)[^\n]*$", "", md, flags=re.M).strip()


def truncate_words(text, cap):
    words = text.split()
    return " ".join(words[:cap])


def first_sentences(text, cap):
    """Leading sentences of a cleaned text, up to about `cap` words, never cutting inside $...$."""
    out = []
    n = 0
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[$])", text.replace("\n", " ")):
        if n and n + len(sent.split()) > cap:
            break
        out.append(sent)
        n += len(sent.split())
        if n >= cap:
            break
    s = " ".join(out)
    if s.count("$") % 2:  # unbalanced inline math from a hard cut: drop the tail
        s = s[: s.rfind("$")].rstrip()
    return truncate_words(s, cap + 20)


def page_url(name):
    return "https://ncatlab.org/nlab/show/" + quote(name.replace(" ", "+"), safe="+()',:-!")


def classify(name, text, cats):
    c = cats.lower()
    if text.lstrip().startswith("<svg") or "svg" in c:
        return "svg"
    if " > " in name:
        return "subpage"
    if NAV_NAME_RE.search(name):
        return "navigation"
    if "people" in c:
        return "person"
    if "reference" in c:
        return "reference"
    if "disambiguation" in c:
        return "disambiguation"
    if "empty" in c or not text.strip():
        return "empty"
    if "meta" in c:
        return "meta"
    if FRAGMENT_NAME_RE.search(name):
        return "fragment"
    likely_person = bool(SELECTED_WRITINGS_RE.search(text)) or bool(BIO_RE.search(text))
    likely_person = likely_person or (bool(CAPNAME_RE.fullmatch(name)) and bool(HOMEPAGE_RE.search(text[:600])))
    if likely_person:
        return "person (untagged)"
    return "concept"


def last_revised_dates():
    dates = {}
    if not HTML_ROOT.exists():
        print("WARNING: HTML mirror missing; last_revised will be null")
        return dates
    for f in HTML_ROOT.rglob("content.html"):
        m = LAST_REVISED_RE.search(f.read_text(encoding="utf-8", errors="replace"))
        if m:
            stamp = re.sub(r"\s+", " ", f"{m.group(1)} {m.group(2)}")
            dates[int(f.parent.name)] = datetime.strptime(stamp, "%B %d, %Y %H:%M:%S")
    return dates


def main():
    pages = list(read_pages(MD_ROOT))
    print(f"read {len(pages)} pages from {MD_ROOT}")
    names = {name for _, name, _ in pages}

    # --- pass 1: redirects and link graph over ALL pages
    redirect_to = {}
    redirects_of = collections.defaultdict(list)
    out_links = {}
    for pid, name, text in pages:
        for r in REDIRECT_RE.findall(text):
            r = r.strip()
            if r and r != name:
                redirect_to[r] = name
                redirects_of[name].append(r)
        out_links[name] = {t.strip() for t in LINK_RE.findall(text)}
    in_degree = collections.Counter()
    out_degree = {}
    for src, targets in out_links.items():
        resolved = {t if t in names else redirect_to.get(t) for t in targets} - {None, src}
        out_degree[src] = len(resolved)
        for t in resolved:
            in_degree[t] += 1
    print(f"link graph: {sum(out_degree.values())} resolved links, {len(redirect_to)} redirect names")

    revised = last_revised_dates()
    print(f"last-revised dates for {len(revised)} pages")

    # --- pass 2: classify, tier, compose
    rows = []
    drops = collections.Counter()
    for pid, name, text in pages:
        cats = ";".join(c.strip() for c in CATEGORY_RE.findall(text))
        ptype = classify(name, text, cats)
        if ptype not in ("concept", "reference"):
            drops[f"page type: {ptype}"] += 1
            continue
        body = body_text(text)
        secs, preamble = parse_sections(body)
        idea = next((s for h, s in secs if IDEA_RE.match(h)), "")
        defn = next((s for h, s in secs if DEFN_RE.match(h)), "")
        stmt = next((s for h, s in secs if STMT_RE.match(h)), "")
        idea_c, defn_c, stmt_c, pre_c = clean(idea), clean(defn), clean(stmt), clean(preamble)
        if len(idea_c.split()) >= SECTION_MIN_WORDS:
            tier, lead, raw_lead = "idea", idea_c, idea
        elif len(defn_c.split()) >= SECTION_MIN_WORDS:
            tier, lead, raw_lead = "definition", defn_c, defn
        elif len(stmt_c.split()) >= SECTION_MIN_WORDS:
            tier, lead, raw_lead = "statement", stmt_c, stmt
        elif len(pre_c.split()) >= PREAMBLE_MIN_WORDS:
            tier, lead, raw_lead = "preamble", pre_c, preamble
        else:
            drops["no self-description (no Idea/Definition/Statement section or preamble)"] += 1
            continue
        parts = [lead] + [p for p in (defn_c, stmt_c) if p and p is not lead]
        embed_text = truncate_words(f"{name}. " + "\n".join(parts), EMBED_WORD_CAP)
        sidebar = SIDEBAR_RE.search(text)
        contexts = [re.sub(r"\s+", " ", h) for h in CONTEXT_HEAD_RE.findall(sidebar.group(1))] if sidebar else []
        rows.append(
            dict(
                pid=pid,
                name=name,
                url=page_url(name),
                page_type=ptype,
                tier=tier,
                embed_text=embed_text,
                summary=first_sentences(clean(strip_heading_lines(raw_lead)), SUMMARY_WORD_CAP),
                context_primary=contexts[0] if contexts else "None",
                context_all=";".join(contexts),
                category_tags=cats,
                redirects=";".join(redirects_of.get(name, [])),
                in_degree=in_degree.get(name, 0),
                out_degree=out_degree.get(name, 0),
                n_words=len(clean(body).split()),
                last_revised=revised.get(pid),
            )
        )

    df = pd.DataFrame(rows)
    print(f"\nclassify+tier: {len(pages)} in -> {len(df)} out ({len(pages) - len(df)} dropped):")
    for reason, n in drops.most_common():
        print(f"  {n:6d}  {reason}")
    print("\nkept by page_type:", df.page_type.value_counts().to_dict())
    print("kept by tier:", df.tier.value_counts().to_dict())
    print("embed_text words:", df.embed_text.str.split().str.len().describe().round().to_dict())
    print("last_revised null:", df.last_revised.isna().sum(), " context None:", (df.context_primary == "None").sum())
    assert df.name.is_unique
    validate_stage_output(
        df,
        "build_corpus",
        ["pid", "name", "url", "page_type", "tier", "embed_text", "summary", "context_primary", "in_degree"],
    )
    write_parquet_safely(df, OUT)


if __name__ == "__main__":
    main()

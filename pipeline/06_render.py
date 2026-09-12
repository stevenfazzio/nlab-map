"""Stage 6: render the interactive map with DataMapPlot into docs/ (served by GitHub Pages).

Reads  data/corpus.parquet, data/coords.parquet, data/labels.parquet, data/raw/mirror_commits.json
Writes docs/index.html and docs/data/nlab_*.zip (point/label/meta data, externalised so the page
       itself stays small enough for link previews and phones)

Surfaces:
  hovercard   name, summary, primary Context, page structure, in-degree, last revised
  search      name + redirect names + summary + all Context headings
  click       opens the page on ncatlab.org
  size        log in-degree (how often other pages link here)
  colormaps   page structure (tier), primary Context (top-N), last-revised year
"""

import json
import re

import glasbey
import numpy as np
import pandas as pd
from common import DATA, RAW
from datamapplot import create_interactive_plot

DOCS = DATA.parent / "docs"
OUT_HTML = DOCS / "index.html"
DATA_PREFIX = DOCS / "data" / "nlab"
SITE_URL = "https://stevenfazzio.github.io/nlab-map/"
N_CONTEXTS = 20
TIER_LABELS = {
    "idea": "Concept: Idea section",
    "definition": "Concept: Definition only",
    "statement": "Concept: Statement only",
    "preamble": "Concept: preamble only",
}

HOVER_TEMPLATE = """
<div style="max-width:420px">
  <div style="font-weight:600;font-size:1.05em;margin-bottom:4px">{hover_text}</div>
  <div style="font-size:0.9em;line-height:1.35;margin-bottom:6px">{summary}</div>
  <div style="font-size:0.8em;color:#666">
    {context_primary} &middot; {structure} &middot; {in_degree} inbound links &middot; revised {revised}
  </div>
</div>
"""


def categorical_palette(n_colors):
    return glasbey.create_palette(
        palette_size=n_colors, colorblind_safe=True, cvd_severity=100.0, lightness_bounds=(25, 75)
    )


def load_label_layers(labels):
    """Return label arrays finest-first (what DataMapPlot wants), whatever order the file used."""
    cols = sorted((c for c in labels.columns if c.startswith("label_layer_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    layers = [labels[c].fillna("Unlabelled").to_numpy() for c in cols]
    layers.sort(key=lambda a: -pd.Series(a).nunique())  # most unique names = finest
    assert pd.Series(layers[0]).nunique() >= pd.Series(layers[-1]).nunique()
    return layers


def main():
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    coords = pd.read_parquet(DATA / "coords.parquet")
    labels = pd.read_parquet(DATA / "labels.parquet")
    assert (coords.pid.to_numpy() == corpus.pid.to_numpy()).all()
    assert (labels.pid.to_numpy() == corpus.pid.to_numpy()).all()
    mirror = json.loads((RAW / "mirror_commits.json").read_text())
    snapshot = mirror["nlab-content"]["commit_date"][:10]
    n = len(corpus)
    print(f"{n} pages, snapshot {snapshot}")

    layers = load_label_layers(labels)
    print("label layers (finest-first):", [int(pd.Series(a).nunique()) for a in layers])

    # --- point fields
    structure = np.where(corpus.page_type == "reference", "Reference page", corpus.tier.map(TIER_LABELS))
    ctx = corpus.context_primary.copy()
    top_ctx = ctx[ctx != "None"].value_counts().head(N_CONTEXTS).index
    ctx_grouped = np.where(ctx.isin(top_ctx), ctx, np.where(ctx == "None", "No Context sidebar", "Other"))
    revised = corpus.last_revised.dt.strftime("%Y-%m-%d").fillna("unknown")
    year = corpus.last_revised.dt.year.astype("float32").fillna(np.nan)
    extra = pd.DataFrame(
        {
            "summary": corpus.summary.fillna("").str.replace("<", "&lt;", regex=False),
            "context_primary": ctx.replace("None", "no Context sidebar"),
            "structure": structure,
            "in_degree": corpus.in_degree.astype(int),
            "revised": revised,
            "url": corpus.url,
            "search": (
                corpus.name
                + " | "
                + corpus.redirects.str.replace(";", " | ")
                + " | "
                + corpus.summary.fillna("")
                + " | "
                + corpus.context_all.str.replace(";", " | ")
            ),
        }
    )
    marker_size = np.log1p(corpus.in_degree.to_numpy(dtype=np.float32))
    marker_size = 0.6 + 2.0 * marker_size / marker_size.max()

    # --- colormaps
    def cat(field, description, values):
        cats = sorted(pd.unique(values))
        return np.asarray(values, dtype=object), {
            "field": field,
            "description": description,
            "kind": "categorical",
            "colors": categorical_palette(len(cats)),
        }

    rawdata, metadata = [], []
    for arr, meta in (
        cat("structure", "Page structure (which section the embedding text came from)", structure),
        cat("context", f"Primary Context sidebar heading (top {N_CONTEXTS}, else Other)", ctx_grouped),
    ):
        rawdata.append(arr)
        metadata.append(meta)
    rawdata.append(year.to_numpy())
    metadata.append({"field": "revised", "description": "Year last revised", "kind": "continuous", "cmap": "viridis"})

    DATA_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    fig = create_interactive_plot(
        coords[["x", "y"]].to_numpy(),
        *layers,
        hover_text=corpus.name.to_numpy(),
        extra_point_data=extra,
        hover_text_html_template=HOVER_TEMPLATE,
        on_click="window.open(`{url}`)",
        enable_search=True,
        search_field="search",
        marker_size_array=marker_size,
        colormap_rawdata=rawdata,
        colormap_metadata=metadata,
        title="A map of the nLab",
        sub_title=f"{n:,} concept pages from the nLab wiki, snapshot {snapshot}. Hover for a summary, click to open.",
        font_family="Roboto",
        cvd_safer=True,
        enable_topic_tree=False,
        inline_data=False,
        offline_data_path=str(DATA_PREFIX),
        initial_zoom_fraction=0.95,
        minify_deps=True,
    )
    html = fig._html_str if hasattr(fig, "_html_str") else str(fig)
    og = "\n".join(
        f'<meta property="{k}" content="{v}">'
        for k, v in {
            "og:title": "A map of the nLab",
            "og:description": f"{n:,} nLab concept pages, embedded and laid out as an interactive map.",
            "og:type": "website",
            "og:url": SITE_URL,
        }.items()
    )
    html = re.sub(r"<head>", "<head>\n" + og, html, count=1)
    OUT_HTML.write_text(html)
    sizes = {p.name: p.stat().st_size for p in sorted(DOCS.glob("**/*")) if p.is_file()}
    print("output sizes (bytes):")
    for name, size in sizes.items():
        print(f"  {size:>12,}  {name}")
    b64 = len(re.findall(r"base64,", html))
    print(f"base64 blobs still inlined in index.html: {b64}")


if __name__ == "__main__":
    main()

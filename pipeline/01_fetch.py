"""Stage 1: fetch the nLab content mirrors into data/raw/.

Two GitHub repos mirror the wiki hourly:
  ncatlab/nlab-content       markdown+itex source, one dir per page: content.md + name
  ncatlab/nlab-content-html  rendered HTML, one dir per page: content.html + name + revision_id
                             (used only for the "Last revised on" date in each page's footer)

Both are shallow-cloned. An existing clone is left untouched so the corpus stays pinned to
one snapshot; pass --update to fast-forward to the mirrors' current HEAD. The commit SHA and
date of each clone are recorded in data/raw/mirror_commits.json.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RAW = Path("data/raw")
MIRRORS = {
    "nlab-content": "https://github.com/ncatlab/nlab-content.git",
    "nlab-content-html": "https://github.com/ncatlab/nlab-content-html.git",
}


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch(name, url, update):
    dest = RAW / name
    if not dest.exists():
        print(f"{name}: cloning (shallow) ...", flush=True)
        git("clone", "--depth", "1", "--quiet", url, str(dest))
    elif update:
        print(f"{name}: updating to origin HEAD ...", flush=True)
        git("fetch", "--depth", "1", "--quiet", "origin", cwd=dest)
        git("reset", "--hard", "--quiet", "origin/HEAD", cwd=dest)
    else:
        print(f"{name}: already present, leaving pinned (use --update to refresh)")
    sha = git("rev-parse", "HEAD", cwd=dest)
    date = git("log", "-1", "--format=%cI", cwd=dest)
    n_pages = sum(1 for _ in (dest / "pages").rglob("name"))
    print(f"{name}: {sha[:12]} committed {date}, {n_pages} pages")
    return {"sha": sha, "commit_date": date, "n_pages": n_pages, "url": url}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="fast-forward existing clones to the mirrors' HEAD")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    info = {name: fetch(name, url, args.update) for name, url in MIRRORS.items()}
    info["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = RAW / "mirror_commits.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=2) + "\n")
    tmp.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

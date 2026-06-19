#!/usr/bin/env python3
"""
build_pdf.py — render the article markdown into a professional, wide (landscape,
two-column) self-contained PDF.

Pipeline: markdown -> HTML (python-markdown) -> inline every local image as a
base64 data URI -> wrap in a whitepaper CSS template -> headless Chrome
--print-to-pdf. No LaTeX / pandoc needed.

Usage:  python article/build_pdf.py
Out:    article/automatic-feature-ideation-with-llms.{html,pdf}
"""

import base64
import os
import re
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "automatic-feature-ideation-with-llms.md"
HTML = HERE / "automatic-feature-ideation-with-llms.html"
PDF = HERE / "automatic-feature-ideation-with-llms.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
:root{
  --green:#1D9E75; --green-d:#147a59; --purple:#534AB7; --ink:#1a1d21;
  --muted:#5b6470; --line:#d9dee3; --bg-soft:#f4f7f6;
  --serif:Georgia,"Times New Roman",serif;
  --sans:"Segoe UI",-apple-system,"Helvetica Neue",Arial,sans-serif;
  --mono:"Cascadia Code","Consolas","Courier New",monospace;
}
@page{ size:A4 landscape; margin:13mm 15mm 14mm; }
*{ box-sizing:border-box; }
html{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }
body{ font-family:var(--serif); color:var(--ink); font-size:9.6pt;
  line-height:1.5; margin:0; }

/* ---- title block (full width) ---- */
.cover{ border-top:5px solid var(--green); padding-top:9px; margin-bottom:14px; }
.kicker{ font-family:var(--sans); font-weight:700; letter-spacing:.16em;
  font-size:8pt; color:var(--green-d); text-transform:uppercase; }
h1{ font-family:var(--sans); font-size:25pt; line-height:1.08; margin:5px 0 4px;
  color:var(--ink); font-weight:800; letter-spacing:-.01em; }
.subtitle{ font-family:var(--sans); font-size:12.5pt; color:var(--muted);
  font-weight:500; margin:0 0 8px; max-width:230mm; }
.byline{ font-family:var(--sans); font-size:8.6pt; color:var(--muted);
  border-top:1px solid var(--line); border-bottom:1px solid var(--line);
  padding:6px 0; margin-bottom:11px; }
.byline b{ color:var(--ink); }
.abstract{ background:var(--bg-soft); border-left:3px solid var(--green);
  padding:9px 13px; border-radius:0 4px 4px 0; }
.abstract p{ margin:0; }
.abstract .lbl{ font-family:var(--sans); font-weight:700; font-size:8pt;
  letter-spacing:.1em; text-transform:uppercase; color:var(--green-d);
  display:block; margin-bottom:3px; }

/* ---- two-column body ---- */
.body{ column-count:2; column-gap:11mm; column-fill:auto; text-align:justify;
  hyphens:auto; -webkit-hyphens:auto; }
.body > h2, .body > h3{ break-after:avoid; }
.body > figure, .body > table, .body > pre, .body > .callout,
.body > .cover{ break-inside:avoid; }

h2{ font-family:var(--sans); column-span:all; font-size:14pt; font-weight:750;
  margin:15px 0 7px; padding-bottom:4px; border-bottom:2px solid var(--green);
  color:var(--ink); }
h2 .num{ color:var(--green); font-weight:800; margin-right:8px; }
h3{ font-family:var(--sans); font-size:10.5pt; font-weight:700; color:var(--purple);
  margin:11px 0 3px; }
p{ margin:0 0 7px; }
a{ color:var(--green-d); text-decoration:none; border-bottom:1px solid #b7e0d2; }
strong{ color:#000; }
ul,ol{ margin:0 0 7px; padding-left:16px; }
li{ margin:0 0 3px; }
code{ font-family:var(--mono); font-size:8.4pt; background:var(--bg-soft);
  padding:.5px 3px; border-radius:3px; color:#0b3d2e; }

/* ---- figures & tables span both columns ---- */
figure{ column-span:all; margin:9px 0 11px; text-align:center;
  background:#fff; border:1px solid var(--line); border-radius:6px; padding:9px; }
figure img{ max-width:100%; max-height:115mm; height:auto; }
figcaption{ font-family:var(--sans); font-size:8pt; color:var(--muted);
  margin-top:6px; text-align:center; }
figcaption b{ color:var(--green-d); }

table{ column-span:all; border-collapse:collapse; width:100%; margin:8px 0 12px;
  font-family:var(--sans); font-size:7.5pt; }
thead th{ background:var(--green); color:#fff; text-align:left; padding:4px 6px;
  font-weight:600; border:1px solid var(--green); }
tbody td{ padding:3px 6px; border:1px solid var(--line); vertical-align:top; }
tbody tr:nth-child(even){ background:var(--bg-soft); }

pre{ column-span:all; background:#0f1a16; color:#e6f3ee; font-family:var(--mono);
  font-size:8pt; line-height:1.45; padding:10px 13px; border-radius:6px;
  overflow-x:auto; margin:8px 0 12px; }
pre code{ background:none; color:inherit; padding:0; }

/* blockquote -> callout box */
blockquote{ break-inside:avoid; margin:8px 0 11px; padding:9px 13px;
  background:#f6f4fc; border-left:3px solid var(--purple); border-radius:0 4px 4px 0;
  color:#2c2747; }
blockquote p{ margin:0 0 4px; }
blockquote p:last-child{ margin:0; }
hr{ border:none; border-top:1px solid var(--line); margin:12px 0; }
"""


def md_to_html(text):
    return markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists", "md_in_html"],
    )


def inline_images(html):
    """Replace <img src="local"> with base64 data URIs (resolved vs the md dir)."""
    def repl(m):
        src = m.group("src")
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        p = (HERE / src).resolve()
        if not p.exists():
            print(f"  ! missing image: {src}", file=sys.stderr)
            return m.group(0)
        ext = p.suffix.lstrip(".").lower().replace("jpg", "jpeg")
        b64 = base64.b64encode(p.read_bytes()).decode()
        return m.group(0).replace(src, f"data:image/{ext};base64,{b64}")
    return re.sub(r'<img[^>]*\ssrc="(?P<src>[^"]+)"[^>]*>', repl, html)


def wrap_figures(html):
    """<p><img alt="cap"></p> -> <figure><img><figcaption>cap</figcaption></figure>."""
    def repl(m):
        img, alt = m.group("img"), m.group("alt") or ""
        cap = f"<figcaption>{alt}</figcaption>" if alt.strip() else ""
        return f"<figure>{img}{cap}</figure>"
    return re.sub(r'<p>(?P<img><img[^>]*\salt="(?P<alt>[^"]*)"[^>]*>)</p>', repl, html)


def number_sections(html):
    """Prefix each <h2> with an auto-incrementing green number."""
    n = [0]
    def repl(m):
        n[0] += 1
        return f'<h2><span class="num">{n[0]}</span>{m.group(1)}'
    return re.sub(r"<h2>(.*?)(?=</h2>)", repl, html, flags=re.S)


def find_chrome():
    for c in CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
    sys.exit("No Chrome/Edge found for PDF rendering.")


def inline_md(text):
    """Render inline markdown (bold/links) and strip the wrapping <p>."""
    h = md_to_html(text.strip())
    return re.sub(r"^<p>|</p>$", "", h.strip())


def build_cover(head):
    """Construct the styled title block from standard markdown in the head:
    `# Title`, an `*italic subtitle*` line, and a `> blockquote abstract`."""
    title = re.search(r"(?m)^#\s+(.+?)\s*$", head)
    sub = re.search(r"(?m)^\*([^*\n].+?)\*\s*$", head)
    quote = re.findall(r"(?m)^>\s?(.*)$", head)
    title = inline_md(title.group(1)) if title else "Untitled"
    subtitle = inline_md(sub.group(1)) if sub else ""
    abstract = inline_md(" ".join(q for q in quote)) if quote else ""
    byline = ('A worked, closed-loop example &nbsp;·&nbsp; <b>mezcal-researcher</b> '
              '&nbsp;·&nbsp; github.com/Simonomer/mezcal-researcher')
    return f"""<div class="cover">
<div class="kicker">Automated EDA &nbsp;·&nbsp; Method</div>
<h1>{title}</h1>
<div class="subtitle">{subtitle}</div>
<div class="byline">{byline}</div>
<div class="abstract"><span class="lbl">Abstract</span><p>{abstract}</p></div>
</div>"""


def main():
    raw = SRC.read_text(encoding="utf-8")

    # Everything before the first H2 is the cover; the rest is the two-column body.
    parts = re.split(r"(?m)^##\s", raw, maxsplit=1)
    head, rest = parts[0], ("## " + parts[1] if len(parts) > 1 else "")
    cover_html = inline_images(build_cover(head))
    body_html = number_sections(wrap_figures(inline_images(md_to_html(rest))))

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Automatic feature ideation with LLMs</title><style>{CSS}</style></head>
<body>{cover_html}
<div class="body">{body_html}</div></body></html>"""
    HTML.write_text(doc, encoding="utf-8")
    print(f"wrote {HTML}")

    chrome = find_chrome()
    subprocess.run([chrome, "--headless=new", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={PDF}",
                    HTML.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {PDF}  ({PDF.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()

# the built pages as a local website: every docs/build/<Page>.md becomes docs/preview/<Page>.html
# with a sidebar of the chapters in the order of MATLAB's documentation (the .toc files of
# ../frameSymmetry/doc), the status of each page from remaining-pages.md, a link to its MATLAB
# page and the page as a cell script to download; the figures stay in docs/build. Run
# docs/make.py first. Usage:
#   .venv/bin/python docs/preview.py            write the preview and serve it on localhost
#   .venv/bin/python docs/preview.py --no-serve write it only
import functools
import html
import http.server
import importlib.util
import os
import re
import sys
import webbrowser

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

from mtex.io.opendoc import cellScript

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
buildDir = os.path.join(root, "docs", "build")
outDir = os.path.join(root, "docs", "preview")
tocDir = os.path.join(os.path.dirname(root), "frameSymmetry", "doc")
MATLAB = "https://mtex-toolbox.github.io/{}.html"

# the status words of site.py, read from the same table
_spec = importlib.util.spec_from_file_location("docsite", os.path.join(root, "docs", "site.py"))
docsite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(docsite)

COLORS = {"Checked": "#2e7d32", "Adapted": "#1565c0", "New": "#6a1b9a", "Partial": "#ef6c00",
          "Written": "#9e9e9e", "Failed": "#c62828"}


# the entries of a .toc file: (name, title), the title the name when the file gives none
def readToc(path):
  if not os.path.exists(path):
    return []
  entries = []
  for line in open(path, encoding="utf-8"):
    parts = line.strip().split(None, 1)
    if parts:
      entries.append((parts[0], parts[1].strip() if len(parts) > 1 else parts[0]))
  return entries


# whether MATLAB's documentation has the page
def hasMatlab(page, chapter):
  return os.path.exists(os.path.join(tocDir, chapter, page + ".m"))


# the chapters and their pages in MATLAB's order; a page without a MATLAB source, the
# introductions this port adds, comes first in its chapter, a MATLAB page missing from the
# toc last, a chapter MATLAB lacks at the end
def outline(chapterOf):
  chapters = readToc(os.path.join(tocDir, "Documentation.toc"))
  known = {c for c, _ in chapters}
  chapters += [(c, c) for c in sorted(set(chapterOf.values()) - known)]
  result = []
  for chapter, title in chapters:
    mine = sorted(p for p, c in chapterOf.items() if c == chapter)
    order = [p for p, _ in readToc(os.path.join(tocDir, chapter, chapter + ".toc"))]
    pages = ([p for p in mine if not hasMatlab(p, chapter)] + [p for p in order if p in mine]
             + [p for p in mine if hasMatlab(p, chapter) and p not in order])
    if pages:
      result.append((chapter, title, pages))
  return result


def highlight(code, lang, attrs):
  try:
    from pygments import highlight as pyg
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonLexer
  except ImportError:
    return ""
  return pyg(code, PythonLexer(), HtmlFormatter(nowrap=True)) if lang == "python" else ""


def renderer():
  md = MarkdownIt("commonmark", {"html": True, "highlight": highlight}).enable("table")
  math = lambda content, opts: (f"\\[{html.escape(content)}\\]" if opts.get("display_mode")
                                else f"\\({html.escape(content)}\\)")
  return md.use(dollarmath_plugin, allow_digits=False, double_inline=True, renderer=math)


# the figures and 3D pages of a page point into the build directory, links to pages stay
def relink(body, pages):
  def one(m):
    attr, target = m.group(1), m.group(2)
    if target.startswith(("http:", "https:", "#", "mailto:")):
      return m.group(0)
    name = target.split("#")[0][:-5] if target.split("#")[0].endswith(".html") else None
    if name in pages:
      return m.group(0)
    if os.path.exists(os.path.join(buildDir, target.split("#")[0])):
      return f'{attr}="../build/{target}"'
    # a page this port lacks, as the function reference, is MATLAB's on the website
    if attr == "href" and name is not None:
      return f'href="{MATLAB.format(name)}{target[len(name) + 5:]}"'
    return m.group(0)
  return re.sub(r'(src|href)="([^"]+)"', one, body)


def sidebar(tree, current, status):
  out = ['<nav><a class="home" href="index.html">MTEX for Python</a>']
  for chapter, title, pages in tree:
    isOpen = " open" if current in pages else ""
    out.append(f"<details{isOpen}><summary>{html.escape(title)}</summary><ul>")
    for p in pages:
      cls = ' class="current"' if p == current else ""
      dot = f'<span class="dot" style="background:{COLORS.get(status[p], "#9e9e9e")}" title="{status[p]}"></span>'
      out.append(f'<li{cls}>{dot}<a href="{p}.html">{html.escape(pageTitle(p))}</a></li>')
    out.append("</ul></details>")
  out.append("</nav>")
  return "\n".join(out)


@functools.cache
def pageTitle(page):
  first = open(os.path.join(buildDir, page + ".md"), encoding="utf-8").readline()
  return first[2:].strip() if first.startswith("# ") else page


def frame(title, nav, body):
  return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
<script>window.MathJax = {{tex: {{inlineMath: [['\\\\(', '\\\\)']]}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head><body>
{nav}
<main>{body}</main>
<script>document.querySelector('nav li.current')?.scrollIntoView({{block: 'center'}});</script>
</body></html>
"""


CSS = """
:root { --fg: #1d1d1f; --muted: #6e6e73; --bg: #fff; --side: #f5f5f7; --line: #d2d2d7;
        --code: #f6f8fa; --out: #fbfbfb; --accent: #0b5cad; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e8e8ed; --muted: #a1a1a6; --bg: #1c1c1e; --side: #232326; --line: #3a3a3c;
          --code: #26262a; --out: #202023; --accent: #6cb2ff; }
  main img { background: #fff; }
}
* { box-sizing: border-box; }
body { margin: 0; font: 16px/1.55 system-ui, sans-serif; color: var(--fg); background: var(--bg); }
nav { position: fixed; top: 0; bottom: 0; left: 0; width: 290px; overflow-y: auto; padding: 16px 12px;
      background: var(--side); border-right: 1px solid var(--line); font-size: 14px; }
nav .home { display: block; font-weight: 600; font-size: 16px; margin-bottom: 12px; color: var(--fg); }
nav summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
nav ul { list-style: none; margin: 0 0 8px; padding-left: 8px; }
nav li { padding: 2px 4px; border-radius: 4px; }
nav li.current { background: var(--line); }
nav a { color: var(--fg); text-decoration: none; }
nav a:hover { text-decoration: underline; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; vertical-align: 1px; }
main { margin-left: 290px; padding: 24px 40px 80px; max-width: 1040px; }
main a { color: var(--accent); }
main img { max-width: 100%; display: block; margin: 12px auto; }
pre { background: var(--code); padding: 10px 14px; border-radius: 6px; overflow-x: auto; font-size: 13.5px; }
pre.output, pre:has(code.language-text) { background: var(--out); border-left: 3px solid var(--line); }
table { border-collapse: collapse; margin: 12px 0; }
th, td { border: 1px solid var(--line); padding: 4px 10px; text-align: left; vertical-align: top; }
.bar { display: flex; gap: 16px; align-items: center; color: var(--muted); font-size: 14px; margin-bottom: 8px; }
.badge { color: #fff; padding: 1px 8px; border-radius: 10px; font-size: 12px; }
.legend span { margin-right: 14px; }
@media (max-width: 800px) { nav { position: static; width: auto; height: auto; }
  main { margin-left: 0; padding: 16px; } }
.k, .kn, .kc { color: #cf222e; } .s, .s1, .s2, .sa { color: #0a3069; } .c1, .c { color: #6e7781; }
.mi, .mf { color: #0550ae; } .nf, .nc { color: #8250df; }
@media (prefers-color-scheme: dark) { .k, .kn, .kc { color: #ff7b72; } .s, .s1, .s2, .sa { color: #a5d6ff; }
  .c1, .c { color: #8b949e; } .mi, .mf { color: #79c0ff; } .nf, .nc { color: #d2a8ff; } }
"""


def write():
  chapterOf = {p: c for p, c in docsite.chapters().items() if os.path.exists(os.path.join(buildDir, p + ".md"))}
  words = docsite.statuses()
  matlabPages = {p for p, c in chapterOf.items() if hasMatlab(p, c)}
  status = {}
  for p in chapterOf:
    stopped = "**The page stopped here.**" in open(os.path.join(buildDir, p + ".md"), encoding="utf-8").read()
    status[p] = "Failed" if stopped else words.get(p) or ("Written" if p in matlabPages else "New")
  tree = outline(chapterOf)
  md = renderer()
  os.makedirs(os.path.join(outDir, "scripts"), exist_ok=True)
  for chapter, chapterTitle, pages in tree:
    for p in pages:
      src = open(os.path.join(buildDir, p + ".md"), encoding="utf-8").read()
      body = relink(md.render(src), chapterOf)
      links = [f'<span class="badge" style="background:{COLORS.get(status[p], "#9e9e9e")}">{status[p]}</span>',
               html.escape(chapterTitle)]
      if p in matlabPages:
        links.append(f'<a href="{MATLAB.format(p)}">MATLAB page</a>')
      links.append(f'<a href="scripts/{p}.py" download>download as script</a>')
      links.append(f'<a href="../pages/{chapter}/{p}.py">source</a>')
      src = open(os.path.join(root, "docs", "pages", chapter, p + ".py"), encoding="utf-8").read()
      open(os.path.join(outDir, "scripts", p + ".py"), "w", encoding="utf-8").write(cellScript(src, chapterOf))
      bar = '<div class="bar">' + " · ".join(links) + "</div>"
      open(os.path.join(outDir, p + ".html"), "w", encoding="utf-8").write(
        frame(pageTitle(p), sidebar(tree, p, status), bar + body))
  # the start page: the legend and the count of every status
  counts = {}
  for s in status.values():
    counts[s] = counts.get(s, 0) + 1
  legend = "".join(f'<span><span class="dot" style="background:{COLORS[s]}"></span>{s} {counts[s]}</span>'
                   for s in COLORS if s in counts)
  rows = "".join(f"<tr><td>{html.escape(t)}</td><td>{len(ps)}</td><td>"
                 + ", ".join(f'<a href="{p}.html">{html.escape(pageTitle(p))}</a>' for p in ps) + "</td></tr>"
                 for _, t, ps in tree)
  index = (f"<h1>MTEX for Python: documentation preview</h1><p class='legend'>{legend}</p>"
           f"<table><tr><th>Chapter</th><th>Pages</th><th></th></tr>{rows}</table>")
  open(os.path.join(outDir, "index.html"), "w", encoding="utf-8").write(frame("MTEX documentation preview", sidebar(tree, None, status), index))
  print(f"{len(chapterOf)} pages written to {outDir}")


# serve docs/ so that the pages reach the figures in docs/build
def serve(port=8000):
  handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.join(root, "docs"))
  handler.log_message = lambda *a: None
  for p in range(port, port + 20):
    try:
      server = http.server.ThreadingHTTPServer(("127.0.0.1", p), handler)
      break
    except OSError:
      continue
  url = f"http://127.0.0.1:{server.server_port}/preview/index.html"
  print(f"serving {url}, Ctrl+C to stop")
  webbrowser.open(url)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  write()
  if "--no-serve" not in sys.argv[1:]:
    serve()

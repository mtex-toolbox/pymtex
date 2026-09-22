# run the MATLAB script of a documentation page through the bridge on the frameSymmetry
# checkout and keep what it printed and drew in docs/matlab/<Page>/: diary.txt with a
# "#### cell k" marker before every cell's output, <Page>-<k>-<f>.png for every figure a
# cell left open. Usage: .venv/bin/python docs/matlabrun.py Vectors/VectorDefinition [...]
import os
import re
import subprocess
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mtexRoot = os.path.dirname(root)
docRoot = os.path.join(mtexRoot, "frameSymmetry", "doc")
bridge = os.path.join(mtexRoot, "frameSymmetry", "docs", "agents", "matlab-bridge")
out = os.path.join(root, "docs", "matlab")
preamble = f"warning('off', 'MATLAB:rmpath:DirNotFound'); rmpath(genpath('{os.path.join(mtexRoot, 'master')}')); warning('on', 'MATLAB:rmpath:DirNotFound'); clear classes; rehash; close all;"


# the cells of a publish-style script: everything from one "%%" line to the next
def cells(src):
  parts = re.split(r"^(?=%%)", src, flags=re.M)
  return [p for p in parts if p.strip()]


# the script that runs every cell under try/catch, saving and closing its figures
def script(page, cs, pageDir):
  name = page.split("/")[-1]
  lines = [f"diary off; cd('{docRoot}'); close all; clear mtexDocSaveFigures;"]
  # a page's local functions are hoisted out of the cells to the end of the script
  functions = []
  for k, c in enumerate(cs, 1):
    body = "\n".join(l for l in c.split("\n") if not l.startswith("%"))
    if re.match(r"\s*function\b", body):
      functions.append(body.rstrip())
      continue
    lines.append(f"fprintf('\\n#### cell {k}\\n');")
    lines.append("try")
    lines.append(c.rstrip())
    lines.append(f"catch mtexDocErr, fprintf('#### error: %s\\n', mtexDocErr.message); end")
    lines.append(f"mtexDocSaveFigures('{pageDir}', '{name}', {k});")
  lines.append("")
  lines += functions
  lines.append("")
  # the figures live on across the cells; one is saved after the cell that made or changed it
  lines.append("function mtexDocSaveFigures(d, name, k)")
  lines.append("  persistent last")
  lines.append("  if isempty(last), last = containers.Map('KeyType', 'double', 'ValueType', 'any'); end")
  lines.append("  figs = findobj('type', 'figure');")
  lines.append("  if isempty(figs), return; end")
  lines.append("  [~, order] = sort([figs.Number]); figs = figs(order);")
  lines.append("  for f = 1:numel(figs)")
  lines.append("    tmp = fullfile(d, 'tmp.png');")
  lines.append("    print(figs(f), tmp, '-dpng', '-r120');")
  lines.append("    bytes = fileread(tmp);")
  lines.append("    num = figs(f).Number;")
  lines.append("    if ~isKey(last, num) || ~isequal(last(num), bytes)")
  lines.append("      last(num) = bytes;")
  lines.append("      movefile(tmp, fullfile(d, sprintf('%s-%d-%d.png', name, k, f)));")
  lines.append("    else")
  lines.append("      delete(tmp);")
  lines.append("    end")
  lines.append("  end")
  lines.append("end")
  return "\n".join(lines) + "\n"


def run(page):
  src = open(os.path.join(docRoot, page + ".m"), encoding="utf-8").read()
  pageDir = os.path.join(out, page.split("/")[-1])
  os.makedirs(pageDir, exist_ok=True)
  for f in os.listdir(pageDir):
    os.remove(os.path.join(pageDir, f))
  runFile = os.path.join(pageDir, "run.m")
  open(runFile, "w", encoding="utf-8").write(script(page, cells(src), pageDir))
  cmd = f"{preamble} run('{runFile}');"
  env = dict(os.environ, SESSION_NAME="mtexfs")
  r = subprocess.run([os.path.join(bridge, ".venv", "bin", "python"), os.path.join(bridge, "matlab_run.py"), cmd],
                     capture_output=True, text=True, env=env)
  # the anchors of MATLAB's linked displays carry nothing a comparison reads
  diary = re.sub(r"</?a\b[^>]*>", "", r.stdout)
  open(os.path.join(pageDir, "diary.txt"), "w", encoding="utf-8").write(diary)
  if r.returncode:
    print(f"FAILED {page}\n{r.stderr[-2000:]}")
  else:
    print(f"ran {page}: {len(os.listdir(pageDir)) - 2} figures")
  return r.returncode


if __name__ == "__main__":
  sys.exit(max(run(p) for p in sys.argv[1:]))

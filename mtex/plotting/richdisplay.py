# the displays of a notebook and of the IPython terminal: MATLAB's displayClass prints the class
# name bold and links every frame of the header to its own display. A notebook shows the same
# text with the name bold and each frame carrying its display as a tooltip; the terminal prints
# the name bold. repr() stays plain text, which the doc build and the tests read
import html
import sys

_BOLD, _PLAIN = "\033[1m", "\033[0m"


def _frames(obj):
  """the frames an object names in its header, by the label the header gives them"""
  from ..geometry.display import frameLabel
  from ..geometry.frames import referenceFrame
  out = {}
  for name in ("frameA", "frameB", "CS", "SS", "frame"):
    try:
      fr = getattr(obj, name, None)
    except Exception:
      continue
    if isinstance(fr, referenceFrame):
      out.setdefault(frameLabel(fr), fr)
  return out


def _split(obj):
  """the class name, what the parentheses of the header hold, and the lines below it"""
  text = repr(obj)
  head, _, rest = text.partition("\n")
  name, paren, info = head.partition(" (")
  return name, info[:-1] if paren and info.endswith(")") else "", rest


def reprHTML(obj):
  """the display as HTML: the text in a pre block, the name bold, each frame of the header with
  its own display as a tooltip"""
  name, info, rest = _split(obj)
  frames = _frames(obj) if info else {}
  parts = []
  for piece in info.replace(" → ", "\0 → \0").replace(", ", "\0, \0").split("\0"):
    fr = frames.get(piece)
    if fr is None or fr is obj:
      parts.append(html.escape(piece))
    else:
      parts.append(f'<span title="{html.escape(repr(fr), quote=True)}" style="text-decoration: underline dotted; cursor: help">'
                   f"{html.escape(piece)}</span>")
  head = f"<b>{html.escape(name)}</b>" + (f" ({''.join(parts)})" if info else "")
  body = "\n" + html.escape(rest) if rest else ""
  return f'<pre style="line-height: 1.25; margin: 0">{head}{body}</pre>'


def reprPretty(obj, p, cycle):
  """the display in the IPython terminal, the name bold"""
  name, info, rest = _split(obj)
  p.text(f"{_BOLD}{name}{_PLAIN}" + (f" ({info})" if info else "") + ("\n" + rest if rest else ""))


def registerDisplays():
  """the two displays given to IPython for every class of mtex with a display of its own; nothing
  happens outside IPython"""
  ipython = sys.modules.get("IPython")
  shell = ipython.get_ipython() if ipython is not None and hasattr(ipython, "get_ipython") else None
  if shell is None:
    return
  htmlFormatter = shell.display_formatter.formatters["text/html"]
  plainFormatter = shell.display_formatter.formatters["text/plain"]
  for cls in displayedClasses():
    htmlFormatter.for_type(cls, reprHTML)
    plainFormatter.for_type(cls, reprPretty)


def displayedClasses():
  """the classes of mtex that define a display, the formatter reaching their subclasses too"""
  seen = []
  for module in list(sys.modules.values()):
    if not getattr(module, "__name__", "").startswith("mtex."):
      continue
    for cls in vars(module).values():
      if isinstance(cls, type) and cls.__module__ == module.__name__ and "__repr__" in cls.__dict__ and cls not in seen:
        seen.append(cls)
  return seen


__all__ = []

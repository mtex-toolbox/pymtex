# the completion of the strings a call takes, in IPython: Jupyter, VS Code's interactive window
# and the terminal shell complete from the running kernel, which reads no Literal type, so
# inside the quotes of plottingConvention.default(" the kernel offers its magics and the files
# of the folder. This matcher reads the annotation of the argument under the cursor and offers
# the values of its Literal types, the vocabularies tools/staticapi.py writes
#
#   plottingConvention.default("     the eight conventions
#   crystalFrame('m                  the point groups starting with m
#   plot(v, color="re                the colours starting with re
#   grains.boundary["Fo", "          the phases of the boundaries, from _ipython_key_completions_
from __future__ import annotations

import ast
import builtins
import functools
import importlib
import importlib.util
import inspect
import re
import sys
import typing

# the modules whose Literal aliases an annotation may name
VOCABULARIES = ("mtex.geometry.vocabulary", "mtex.io.vocabulary", "mtex.plotting.vocabulary")

# the TypedDicts a **kw may unpack, by name, and the module that defines them for the editor only
KEYWORD_DICTS = {"PlotKeywords": "mtex.plotting.keywords"}

_CALLEE = re.compile(r"([A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*)\s*$")
_KEYWORD = re.compile(r"([A-Za-z_]\w*)\s*=\s*$")
_OPEN = {"(": ")", "[": "]", "{": "}"}


@functools.cache
def vocabularies():
  """the Literal aliases of the vocabularies, by name"""
  out = {}
  for name in VOCABULARIES:
    module = importlib.import_module(name)
    for alias, value in vars(module).items():
      if typing.get_origin(value) is typing.Literal:
        out[alias] = list(typing.get_args(value))
  return out


@functools.cache
def keywordFields(name):
  """the fields of a TypedDict declared under TYPE_CHECKING, as annotation strings, read off its source"""
  module = KEYWORD_DICTS.get(name)
  if module is None:
    return {}
  spec = importlib.util.find_spec(module)
  with open(spec.origin, encoding="utf-8") as f:
    tree = ast.parse(f.read())
  for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == name:
      return {s.target.id: ast.unparse(s.annotation) for s in node.body
              if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
  return {}


def literalValues(annotation):
  """the strings an annotation allows, in order, each once: its Literal values and the vocabularies
  it names; an annotation may be the object or, under postponed evaluation, its source"""
  out = []
  if isinstance(annotation, str):
    try:
      tree = ast.parse(annotation, mode="eval")
    except SyntaxError:
      return []
    vocab = vocabularies()
    for node in ast.walk(tree):
      if isinstance(node, ast.Name) and node.id in vocab:
        out += vocab[node.id]
      elif isinstance(node, ast.Subscript) and isinstance(node.value, (ast.Name, ast.Attribute)) \
              and ast.unparse(node.value).split(".")[-1] == "Literal":
        out += [n.value for n in ast.walk(node.slice) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
  elif typing.get_origin(annotation) is typing.Literal:
    out += [a for a in typing.get_args(annotation) if isinstance(a, str)]
  else:
    for a in typing.get_args(annotation):
      out += literalValues(a)
  return list(dict.fromkeys(out))


# the unpacked TypedDict of a **kw annotation, by name: Unpack[PlotKeywords] -> PlotKeywords
def _unpacked(annotation):
  text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", repr(annotation))
  m = re.fullmatch(r"\s*(?:\w+\.)*Unpack\[\s*(\w+)\s*\]\s*", text)
  return m.group(1) if m else None


def openStringArgument(text):
  """the string literal open at the end of text and the call or subscript it is an argument of:
  (callee, "(", position or keyword, the typed part) or (callee, "[", the elements before, the typed
  part), or None. A position is None after a starred argument, the elements before where they are
  not literals"""
  stack = []
  i, n = 0, len(text)
  while i < n:
    ch = text[i]
    if ch == "#":
      j = text.find("\n", i)
      if j < 0:
        return None
      i = j + 1
      continue
    if ch in "'\"":
      quote = text[i:i + 3] if text[i:i + 3] in ("'''", '"""') else ch
      j = text.find(quote, i + len(quote))
      while j > 0 and text[j - 1] == "\\" and len(quote) == 1:
        j = text.find(quote, j + 1)
      if j < 0:
        if len(quote) == 3 or "\n" in text[i:] or not stack or stack[-1][0] == "{":
          return None
        prefix = text[i + 1:]
        if "\\" in prefix or re.search(r"[A-Za-z]$", text[:i]):
          return None
        opener, start, argStart, position = stack[-1]
        before = text[argStart:i]
        m = _KEYWORD.search(before)
        if opener == "[":
          if before.strip():
            return None
          try:
            target = ast.literal_eval("(" + text[start + 1:argStart] + ")") if argStart > start + 1 else ()
          except (ValueError, SyntaxError):
            target = None
          target = target if isinstance(target, tuple) else None
        elif m and before[:m.start()].strip() == "":
          target = m.group(1)
        elif before.strip() == "":
          target = position
        else:
          return None
        callee = _CALLEE.search(text[:start])
        if callee is None or re.search(r"[)\]}'\"]\s*\.?\s*$", text[:callee.start()]):
          return None
        return re.sub(r"\s+", "", callee.group(1)), opener, target, prefix
      i = j + len(quote)
      continue
    if ch in _OPEN:
      stack.append([ch, i, i + 1, 0])
    elif ch in ")]}":
      if not stack:
        return None
      stack.pop()
    elif ch == "," and stack:
      top = stack[-1]
      if top[3] is not None:
        top[3] = None if text[top[2]:i].lstrip().startswith("*") else top[3] + 1
      top[2] = i + 1
    i += 1
  return None


def resolve(dotted, namespace):
  """the object a dotted name stands for in a namespace: names and attributes only, nothing called,
  the attributes read only off modules, classes and MTEX's objects"""
  head, *rest = dotted.split(".")
  if head in namespace:
    obj = namespace[head]
  elif hasattr(builtins, head):
    obj = getattr(builtins, head)
  else:
    return None
  for name in rest:
    if not (inspect.ismodule(obj) or inspect.isclass(obj) or type(obj).__module__.startswith("mtex")):
      return None
    try:
      obj = getattr(obj, name)
    except Exception:
      return None
  return obj


# the signatures a call of obj may bind to: its overloads, else its own
def _signatures(obj):
  out = []
  f = inspect.unwrap(obj) if inspect.isfunction(obj) else None
  if f is not None:
    for o in typing.get_overloads(f):
      try:
        out.append(inspect.signature(o))
      except (TypeError, ValueError):
        pass
  if not out:
    try:
      out.append(inspect.signature(obj))
    except (TypeError, ValueError):
      pass
  return out


# the annotation of the argument a call binds at a position or keyword, None when none applies
def _annotation(sig, target):
  params = list(sig.parameters.values())
  if isinstance(target, str):
    for p in params:
      if p.name == target and p.kind not in (p.POSITIONAL_ONLY, p.VAR_POSITIONAL, p.VAR_KEYWORD):
        return p.annotation
    for p in params:
      if p.kind == p.VAR_KEYWORD and (name := _unpacked(p.annotation)):
        return keywordFields(name).get(target)
    return None
  positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
  if target < len(positional):
    return positional[target].annotation
  star = [p for p in params if p.kind == p.VAR_POSITIONAL]
  return star[0].annotation if star else None


# the strings an MTEX object's subscript takes after the elements before: the keys of its
# _ipython_key_completions_, the earlier elements of a tuple key matched as its selection reads
# them, case aside and abbreviated
def _keys(obj, before):
  f = getattr(type(obj), "_ipython_key_completions_", None)
  if f is None or not type(obj).__module__.startswith("mtex"):
    return []
  out = []
  for key in f(obj):
    key = key if isinstance(key, tuple) else (key,)
    if len(key) > len(before) and isinstance(key[len(before)], str) and all(
        isinstance(b, str) and k.lower().startswith(b.lower()) for k, b in zip(key, before)):
      out.append(key[len(before)])
  return out


def stringCompletions(text, namespace):
  """the strings the argument whose quotes the text ends in takes, as (typed part, values): the
  values of the Literal types of its annotation, or of the keys of an MTEX object's subscript,
  that start with what is typed, case aside; None where the text does not end inside such an
  argument"""
  found = openStringArgument(text)
  if found is None:
    return None
  callee, opener, target, prefix = found
  if target is None:
    return None
  obj = resolve(callee, namespace)
  if obj is None:
    return None
  if opener == "[":
    values = _keys(obj, target)
  elif callable(obj):
    values = []
    for sig in _signatures(obj):
      annotation = _annotation(sig, target)
      if annotation is not None and annotation is not inspect.Parameter.empty:
        values += literalValues(annotation)
  else:
    return None
  values = list(dict.fromkeys(values))
  if not values:
    return None
  return prefix, [v for v in values if v.lower().startswith(prefix.lower())]


# the text of a completion context up to the cursor
def _untilCursor(context):
  lines = context.full_text.split("\n")
  return "\n".join(lines[:context.cursor_line] + [lines[context.cursor_line][:context.cursor_position]])


def registerCompleter(shell=None):
  """adds the matcher to the running IPython shell, a kernel or the terminal, once; nothing outside
  IPython, which is never imported here. mtex calls it on import"""
  if shell is None:
    ipython = sys.modules.get("IPython")
    shell = ipython.get_ipython() if ipython is not None and hasattr(ipython, "get_ipython") else None
  completer = getattr(shell, "Completer", None)
  if completer is None or any(getattr(m, "__name__", "") == "mtexStrings" for m in completer.custom_matchers):
    return
  from IPython.core.completer import SimpleCompletion, context_matcher

  @context_matcher(identifier="mtexStrings")
  def mtexStrings(context):
    try:
      found = stringCompletions(_untilCursor(context), shell.user_ns)
    except Exception:
      found = None
    if found is None:
      return {"completions": []}
    prefix, values = found
    return {"completions": [SimpleCompletion(v, type="value") for v in values], "matched_fragment": prefix,
            "suppress": bool(values), "ordered": True}

  completer.custom_matchers.append(mtexStrings)


__all__ = []

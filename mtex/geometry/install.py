# the methods a module attaches to classes defined elsewhere, MATLAB's methods of a class
# spread over the toolbox: plot(v) is v.plot(), KAM(ebsd) is ebsd.KAM(). Every attachment is
# recorded, and tools/staticapi.py declares the recorded ones in the class bodies, so that an
# editor completes them without running the package
import functools
import types

# (class, name, attribute, declared) of every attachment, in the order they were made
INSTALLED = []


def install(classes, f, name=None, declare=None):
  """attach f to every class of classes as name, f a function, a staticmethod, a classmethod
  or a property; a function becomes a method taking the object first. declare is the
  function an editor shows instead, the one a dispatcher like plot calls for these classes"""
  if isinstance(classes, type):
    classes = (classes,)
  inner = f.fget if isinstance(f, property) else getattr(f, "__func__", f)
  name = name or inner.__name__
  # a callable that is not a function, a partial say, binds no self: wrapped, it does
  if not isinstance(f, (staticmethod, classmethod, property, types.FunctionType)):
    f = functools.wraps(f)(lambda self, *args, **kwargs: inner(self, *args, **kwargs))
  for cls in classes:
    setattr(cls, name, f)
    INSTALLED.append((cls, name, f, declare))


__all__ = []

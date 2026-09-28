# the import wizard from a terminal: python -m mtex.wizard [folder or file], or mtex-wizard
import sys

from . import importWizard


def main():
  """the entry point of python -m mtex.wizard and mtex-wizard: the import wizard opened at the folder
  or file given on the command line"""
  # a process of its own on the Windows taskbar, so that it shows the wizard's icon, not Python's
  if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mtex.importWizard")
  importWizard(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
  main()

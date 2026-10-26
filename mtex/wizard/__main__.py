# the import wizard from a terminal: python -m mtex.wizard [folder or file] [--scheme light|dark],
# or mtex-wizard
import argparse
import sys

from . import importWizard


def main():
  """the entry point of python -m mtex.wizard and mtex-wizard: the import wizard opened at the folder
  or file given on the command line, in the desktop's colour scheme or the one --scheme names"""
  parser = argparse.ArgumentParser(prog="mtex-wizard", description="the MTEX import wizard")
  parser.add_argument("path", nargs="?", help="the folder to browse or the file to open")
  parser.add_argument("--scheme", choices=("light", "dark"), help="the colours of the window, by default the desktop's")
  args = parser.parse_args()
  # a process of its own on the Windows taskbar, so that it shows the wizard's icon, not Python's
  if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mtex.importWizard")
  importWizard(args.path, scheme=args.scheme)


if __name__ == "__main__":
  main()

# the import wizard from a terminal: python -m mtex.wizard [folder or file], or mtex-wizard
import sys

from . import importWizard


def main():
  importWizard(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
  main()

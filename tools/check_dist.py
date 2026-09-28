"""the wheel and the sdist installed fresh outside the checkout and checked; uv has to be on PATH

  python tools/check_dist.py --dist dist --python 3.12
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


# the smoke test of an installation, run with -I so that neither the checkout, the working
# directory nor PYTHONPATH supplies mtex
SMOKE = """
import ast
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import mtex
from mtex import (Miller, calcGrains, calcODF, crystalFrame, degree, equispacedS2Grid,
                  loadEBSD, openDoc, setMTEXpref, uniformODF)
from mtex.io.opendoc import pagesDir

installed = Path(mtex.__file__).resolve()
assert installed.is_relative_to(Path(sys.prefix).resolve()), installed
assert mtex.__version__ == version('mtex')
setMTEXpref('threads', 2)
fixture = Path(sys.argv[1]) / 'tests' / 'fixtures' / 'ebsd'
ebsd = loadEBSD(str(fixture / 'Forsterite_cut.ctf'))
grains = calcGrains(ebsd, angle=15 * degree)
assert len(grains) > 0 and grains.isIndexed.any()
assert ebsd.grainId.shape == ebsd.shape

# a small pole figure and its inversion, which reach the numerical dependencies
cs = crystalFrame('m-3m')
reference = uniformODF(cs)
grid = equispacedS2Grid(resolution=30 * degree, antipodal=True)
pf = reference.calcPoleFigure(Miller(1, 0, 0, cs), grid)
odf = calcODF(pf, resolution=30 * degree, iterMax=3)
assert np.all(np.isfinite(odf.calcPoleFigure(Miller(1, 0, 0, cs), grid).intensities))

assert Path(pagesDir()).resolve().is_relative_to(installed.parent)
assert len(openDoc()) > 200
page = openDoc('EBSDTutorial', folder=Path.cwd() / 'tutorial', edit=False)
ast.parse(Path(page).read_text(encoding='utf-8'))
print(f'installed mtex {mtex.__version__}: {installed}; {len(grains)} grains; ODF and bundled docs OK')
"""

# the HDF5 import and a plot written to a file, with the io and plot extras
PLOT_SMOKE = """
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from mtex import loadEBSD, plot

fixtures = Path(sys.argv[1]) / 'tests' / 'fixtures' / 'ebsd'
ebsd = loadEBSD(str(fixtures / 'h5' / 'edax_cut.h5'))
assert ebsd.isIndexed.any()
plot(ebsd)
output = Path('phase-map.png')
plt.savefig(output)
assert output.stat().st_size > 0
plt.close('all')
print('HDF5 import and PNG plot OK')
"""


# a command, only its last two lines printed when quiet and it succeeds
def run(*args, cwd, env, quiet=False):
  result = subprocess.run([str(a) for a in args], cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None)
  if quiet:
    print(result.stdout if result.returncode else '\n'.join(result.stdout.splitlines()[-2:]), flush=True)
  result.check_returncode()


# the sdist's files, then each distribution in a venv of its own
def check(dist, python):
  wheels, sources = list(dist.glob('*.whl')), list(dist.glob('*.tar.gz'))
  if len(wheels) != 1 or len(sources) != 1:
    raise ValueError('dist must contain exactly one wheel and one source distribution')
  env = dict(os.environ, MTEX_SETTINGS='', MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='2', NUMBA_NUM_THREADS='2')
  with tempfile.TemporaryDirectory(prefix='mtex-dist-') as temporary:
    root = Path(temporary)
    with tarfile.open(sources[0]) as archive:
      archive.extractall(root / 'source', filter='data')
    source, = (root / 'source').iterdir()
    for path in ('tools/staticapi.py', 'tests/test_staticapi.py', 'hatch_build.py',
                 'tests/fixtures/ebsd/Forsterite_cut.ctf', 'docs/pages/Tutorials/EBSDTutorial.py'):
      if not (source / path).is_file():
        raise AssertionError(f'sdist is missing {path}')
    for kind, artifact in (('wheel', wheels[0]), ('sdist', sources[0])):
      print(f'Checking {kind}: {artifact.name}', flush=True)
      work = root / kind
      work.mkdir()
      venv = work / 'venv'
      run('uv', 'venv', '--python', python, venv, cwd=work, env=env)
      executable = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
      # the core dependencies alone first, then the io and plot extras
      run('uv', 'pip', 'install', '--python', executable, artifact, cwd=work, env=env)
      run(executable, '-I', '-c', SMOKE, source, cwd=work, env=env)
      run('uv', 'pip', 'install', '--python', executable, f'{artifact}[io,plot]', 'pytest', cwd=work, env=env)
      run(executable, '-I', '-c', PLOT_SMOKE, source, cwd=work, env=env)
      # the shipped static API and page tests, and every shipped test collected for a missing helper
      run(executable, '-I', '-m', 'pytest', '-q', source / 'tests/test_staticapi.py',
          source / 'tests/test_opendoc.py', cwd=work, env=env)
      run(executable, '-I', '-m', 'pytest', '--collect-only', '-q', source / 'tests',
          cwd=work, env=env, quiet=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dist', type=Path, default=Path('dist'))
  parser.add_argument('--python', default='3.12')
  args = parser.parse_args()
  check(args.dist.resolve(), args.python)

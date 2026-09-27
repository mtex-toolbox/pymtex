# the NFFT of ../nfft (branch feature/mtex-kernels) as an optional backend of the SO(3) plans,
# through ctypes: its real valued transform (NFFT_REAL) and pruned FFT make the adjoint 15 to
# 50 % faster than finufft's type 1 (docs/research/benchmark-harmonics.md), the evaluation is
# level. finufft stays the dependency; the library is used where it is found and computes
# real plans, which NFFT releases before that branch do not
#
#   lib = nfftLibrary()        the loaded library, or None
#   p = NFFTPlan(N, nodes, real=, keep=)   p.fhat, p.f, p.trafo(), p.adjoint()
#
# The library is the preference 'NFFTLibrary', else the environment variable
# MTEX_NFFT_LIBRARY, else libnfft3_omp on the system path; build it from ../nfft with
#   cmake -S . -B build -DNFFT_ENABLE_OPENMP=ON && cmake --build build -j
# and point the preference at build/kernel/libnfft3_omp.so
from __future__ import annotations

import ctypes
import ctypes.util
import os

import numpy as np

from ..geometry.kernels import passiveOpenMP, threads
from ..geometry.session import getMTEXpref

PRE_PHI_HUT = 1
PRE_PSI = 1 << 4
MALLOC_X, MALLOC_F_HAT, MALLOC_F = 1 << 6, 1 << 7, 1 << 8
FFTW_INIT = 1 << 10
NFFT_OMP_BLOCKWISE_ADJOINT = 1 << 12
NFFT_PRUNED_FFT = 1 << 13
NFFT_REAL = 1 << 14
FFTW_ESTIMATE = 1 << 6


# the public head of nfft_plan in nfft3.h; the rest of the plan is the library's, so a plan
# is this head at the start of a buffer large enough for the whole struct
class _PlanHead(ctypes.Structure):
  _fields_ = [("N_total", ctypes.c_ssize_t), ("M_total", ctypes.c_ssize_t),
              ("f_hat", ctypes.c_void_p), ("f", ctypes.c_void_p),
              ("mv_trafo", ctypes.c_void_p), ("mv_adjoint", ctypes.c_void_p),
              ("d", ctypes.c_ssize_t), ("N", ctypes.c_void_p), ("sigma", ctypes.c_void_p),
              ("n", ctypes.c_void_p), ("n_total", ctypes.c_ssize_t), ("m", ctypes.c_ssize_t),
              ("b", ctypes.c_void_p), ("K", ctypes.c_ssize_t), ("flags", ctypes.c_uint),
              ("fftw_flags", ctypes.c_uint), ("x", ctypes.c_void_p)]


PLAN_BYTES = 1024
_LIB = None


def _load(path):
  passiveOpenMP()
  lib = ctypes.CDLL(path)
  P, I = ctypes.c_void_p, ctypes.c_int
  lib.nfft_init_guru.argtypes = [P, I, ctypes.POINTER(I), I, ctypes.POINTER(I), I, ctypes.c_uint, ctypes.c_uint]
  for name in ("nfft_trafo", "nfft_adjoint", "nfft_precompute_one_psi", "nfft_finalize"):
    getattr(lib, name).argtypes = [P]
  return lib


# a library computes real plans when a real plan of 4 x 4 holds half the coefficients
def _computesRealPlans(lib):
  p = _Plan(lib, (4, 4), 1, (8, 8), 2, NFFT_REAL)
  return p.head.N_total == 8


# the system's library, looked up once: find_library starts processes, a few ms each
_SYSTEM = []


def _systemLibrary():
  if not _SYSTEM:
    _SYSTEM.append(ctypes.util.find_library("nfft3_omp"))
  return _SYSTEM[0]


def nfftLibrary():
  """the loaded NFFT library, found by the preference NFFTLibrary, then MTEX_NFFT_LIBRARY, then the
  system's nfft3_omp; None when there is none or it computes no real plans"""
  global _LIB
  path = getMTEXpref("NFFTLibrary") or os.environ.get("MTEX_NFFT_LIBRARY") or _systemLibrary()
  if not path:
    return None
  if _LIB is None or _LIB[0] != path:
    try:
      lib = _load(path)
      _LIB = (path, lib if _computesRealPlans(lib) else None)
    except OSError:
      _LIB = (path, None)
    if _LIB[1] is not None:
      try:
        ctypes.CDLL(ctypes.util.find_library("gomp") or "libgomp.so.1").omp_set_num_threads(threads())
      except (OSError, AttributeError):
        pass
  return _LIB[1]


class _Plan:

  def __init__(self, lib, N, M, n, m, flags):
    d = len(N)
    self.lib, self.buffer = lib, ctypes.create_string_buffer(PLAN_BYTES)
    self.head = _PlanHead.from_buffer(self.buffer)
    ints = ctypes.c_int * d
    lib.nfft_init_guru(ctypes.addressof(self.buffer), d, ints(*N), int(M), ints(*n), int(m),
                       flags | MALLOC_X | MALLOC_F | MALLOC_F_HAT | FFTW_INIT, FFTW_ESTIMATE)

  def _array(self, pointer, count, dtype):
    return np.ctypeslib.as_array(ctypes.cast(pointer, ctypes.POINTER(ctypes.c_double)),
                                 shape=(count * np.dtype(dtype).itemsize // 8,)).view(dtype)

  def __del__(self):
    self.lib.nfft_finalize(ctypes.addressof(self.buffer))


def _fftLength(n):
  from scipy.fft import next_fast_len
  n = next_fast_len(int(n))
  return n + n % 2


class NFFTPlan:
  """the plan of one node set on the lattice N (even lengths, frequencies -N/2 .. N/2 - 1): the
  window and oversampling by MATLAB's nfftParameters, the long window where the nodes are
  fewer than a quarter of the lattice. nodes are the rows of x in [-1/2, 1/2)^d; a real plan
  holds the frequencies 0 .. N/2 - 1 of the last axis and evaluates 2 Re of the half sum; keep
  precomputes the window values at the nodes (PRE_PSI) for a plan run many times"""

  def __init__(self, N, nodes, real=False, keep=False):
    lib = nfftLibrary()
    N, M = tuple(int(v) for v in N), nodes.shape[0]
    m, sigma = (6, 1.25) if M < np.prod(N) / 4 and min(N) >= 32 else (4, 2.0)
    flags = PRE_PHI_HUT | NFFT_OMP_BLOCKWISE_ADJOINT | NFFT_PRUNED_FFT | (NFFT_REAL if real else 0) | (PRE_PSI if keep else 0)
    self.plan = _Plan(lib, N, M, tuple(_fftLength(sigma * v) for v in N), m, flags)
    self.shape = N[:-1] + (N[-1] // 2,) if real else N
    head = self.plan.head
    self.plan._array(head.x, M * len(N), np.float64)[:] = nodes.ravel()
    lib.nfft_precompute_one_psi(ctypes.addressof(self.plan.buffer))
    self.fhat = self.plan._array(head.f_hat, head.N_total, np.complex128).reshape(self.shape)
    self.f = self.plan._array(head.f, M, np.complex128)

  def trafo(self):
    """the values at the nodes of the coefficients in fhat, the plan's array f"""
    self.plan.lib.nfft_trafo(ctypes.addressof(self.plan.buffer))
    return self.f

  def adjoint(self):
    """the adjoint transform of the values in f, the plan's coefficient array fhat"""
    self.plan.lib.nfft_adjoint(ctypes.addressof(self.plan.buffer))
    return self.fhat


__all__ = []

# LSQR with MATLAB's stopping rules
import numpy as np

from mtex.functions.lsqr import lsqr

rng = np.random.default_rng(5)


def test_the_least_squares_solution_of_a_complex_system():
  M = rng.standard_normal((200, 50)) + 1j * rng.standard_normal((200, 50))
  b = rng.standard_normal(200)
  x, flag, relres, it = lsqr(lambda v: M @ v, lambda u: M.conj().T @ u, b, 50, tol=1e-12, maxit=500)
  xs = np.linalg.lstsq(M, b, rcond=None)[0]
  assert flag == 0 and it < 500
  assert np.linalg.norm(x - xs) < 1e-9 * np.linalg.norm(xs)
  assert np.isclose(relres, np.linalg.norm(M @ xs - b) / np.linalg.norm(b))


def test_the_minimum_norm_solution_of_an_underdetermined_system():
  M = rng.standard_normal((30, 80))
  b = rng.standard_normal(30)
  x, flag, relres, _ = lsqr(lambda v: M @ v, lambda u: M.T @ u, b, 80, tol=1e-14, maxit=500)
  assert flag == 0 and relres < 1e-13
  assert np.allclose(x, np.linalg.pinv(M) @ b)


def test_the_flag_of_iterations_run_out_and_a_zero_right_hand_side():
  M = rng.standard_normal((100, 60))
  b = rng.standard_normal(100)
  x, flag, relres, it = lsqr(lambda v: M @ v, lambda u: M.T @ u, b, 60, tol=1e-15, maxit=3)
  assert flag == 1 and it == 3 and 0 < relres < 1
  x, flag, relres, it = lsqr(lambda v: M @ v, lambda u: M.T @ u, np.zeros(100), 60)
  assert flag == 0 and it == 0 and not np.any(x)

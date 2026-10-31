"""CPU-only tests for the high-risk layout mapping and free-check guard."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('guard', ROOT / 'tools/run_lt_check.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def mapped_product(x, w, transposed):
    m, k, n = len(x), len(x[0]), len(w)
    a = ([w[row][col] for col in range(k) for row in range(n)] if transposed
         else [v for row in w for v in row])
    b = [v for row in x for v in row]
    # cuBLASLt D[N,M] column-major has the same bytes as Y[M,N] row-major.
    result = [0] * (m * n)
    for row in range(n):
        for col in range(m):
            result[row + col*n] = sum(
                (a[row + q*n] if transposed else a[q + row*k]) * b[q + col*k]
                for q in range(k))
    return [result[i*n:(i+1)*n] for i in range(m)]


class LayoutTests(unittest.TestCase):
    def test_column_major_identity_on_rectangular_shapes(self):
        # Unequal dimensions expose the transpose/leading-dimension mistakes.
        for m, n, k in ((1, 3, 5), (4, 7, 3), (7, 2, 9), (3, 8, 4)):
            x = [[((i*7+j*3) % 11)-5 for j in range(k)] for i in range(m)]
            w = [[((i*5+j*2) % 13)-6 for j in range(k)] for i in range(n)]
            expected = [[sum(x[i][q]*w[j][q] for q in range(k))
                         for j in range(n)] for i in range(m)]
            for transposed in (False, True):
                self.assertEqual(mapped_product(x, w, transposed), expected)

    def test_fresh_input_changes_result_and_previous_result_is_retained(self):
        x = [[1, 2, 3], [4, 5, 6]]
        w = [[2, -1, 4], [-3, 2, 5]]
        first = mapped_product(x, w, False)
        saved = [row[:] for row in first]
        x[0][0] += 9
        self.assertNotEqual(first, mapped_product(x, w, False))
        self.assertEqual(first, saved)


class GuardTests(unittest.TestCase):
    def test_before_reset_denied_even_if_quota_looks_available(self):
        now = datetime(2026, 10, 30, 23, 59, tzinfo=timezone.utc)
        self.assertFalse(guard.eligible(now, {'left': 20}, 0)[0])

    def test_reset_boundary_allowed(self):
        self.assertTrue(guard.eligible(guard.NOT_BEFORE, {'left': 20}, 0)[0])

    def test_zero_or_missing_quota_denied(self):
        for quota in ({'left': 0}, {}):
            self.assertFalse(guard.eligible(guard.NOT_BEFORE, quota, 0)[0])

    def test_local_cap_denied(self):
        self.assertFalse(guard.eligible(guard.NOT_BEFORE, {'left': 20}, 8)[0])

    def test_manifest_and_champion(self):
        guard.verify_champion()
        guard.candidate_path('candidates/lt/lt_original_h0.py')
        with self.assertRaises(RuntimeError):
            guard.candidate_path('submission.py')


if __name__ == '__main__':
    unittest.main()

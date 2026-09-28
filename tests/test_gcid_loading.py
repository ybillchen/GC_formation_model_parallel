"""Regression coverage for ID-only input to the parallel tidal workers."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

module = importlib.import_module('GC_formation_model_parallel.get_tid_parallel')


class InlinePool:
    def __init__(self, processes):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def starmap(self, function, tasks):
        return [function(*task) for task in tasks]


class ParticleIDLoadingTest(unittest.TestCase):
    def test_single_and_combined_catalogs(self):
        for mode in ['single', 'parameters', 'seeds']:
            for count in [1, 3]:
                with self.subTest(mode=mode, count=count), tempfile.TemporaryDirectory() as directory:
                    self.check_case(Path(directory), mode, count)

    def check_case(self, directory, mode, count):
        ids = np.array([1000000000001, 1000000000007, 1000000000009], dtype=np.int64)[:count]
        quality = np.array([2, 1, 0], dtype=np.int64)[:count]
        prefix = 'allcat_s-0_p2-18_p3-0.5' if mode == 'single' else 'combine'
        np.savetxt(directory / (prefix + '_gcid.txt'),
                   np.column_stack([ids, quality]) if mode == 'single' else ids, fmt='%d')
        np.savetxt(directory / (prefix + '_offset_root.txt'), [[42, 0, count]], fmt='%d')
        params = dict(resultspath=str(directory) + '/', allcat_base='allcat',
                      seed=0, p2=18, p3=0.5, subs=[42], full_snap=[50, 99],
                      redshift_snap=np.zeros(100), mpb_only=False, verbose=False)
        calls = []

        def fake_tidal_worker(i, received_ids, roots, begin, end, run_params):
            # Before the fix, single-run input is (N,2), or (2,) for one GC.
            self.assertEqual(received_ids.shape, (count,))
            self.assertEqual(received_ids.dtype, np.dtype('int64'))
            np.testing.assert_array_equal(received_ids, ids)
            np.testing.assert_array_equal(end - begin, [count])
            calls.append(i)
            out = directory / 'independent_tidal_outputs'
            out.mkdir(exist_ok=True)
            for suffix in ['tag', 'eig1', 'eig2', 'eig3']:
                np.savetxt(out / (prefix + '_tid' + suffix + '_i0.txt'), np.ones((count, 2)))

        with patch.object(module, 'Pool', InlinePool), \
             patch.object(module, 'get_tid_i', fake_tidal_worker), \
             patch.object(module, 'combine_gc'), patch.object(module, 'combine_gc_seed'), \
             patch.object(module, 'assign_eig'), patch.object(module, 'assign_eig_seed'):
            module.get_tid_parallel(params, Np=1, param_based=mode == 'parameters',
                                    seed_based=mode == 'seeds', nthreads=1)
        self.assertEqual(calls, [0])
        # Exercise the real output-combination path too: the one-GC case used
        # to allocate two rows because the quality column was counted as an ID.
        for suffix in ['tag', 'eig1', 'eig2', 'eig3']:
            result = np.loadtxt(directory / (prefix + '_tid' + suffix + '.txt'), ndmin=2)
            self.assertEqual(result.shape, (count, 2))
            np.testing.assert_array_equal(result, np.ones((count, 2)))


if __name__ == '__main__':
    unittest.main()

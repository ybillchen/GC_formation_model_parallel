# Licensed under BSD-3-Clause License - see LICENSE
"""Exercise real formation/assignment through the parallel job entry points."""
from copy import deepcopy
import importlib
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

runner = importlib.import_module('GC_formation_model_parallel.run_parallel')
loader = importlib.import_module('GC_formation_model.loader')


class SmallCosmo:
    def __init__(self, h, omega_baryon, omega_matter):
        self.h = h
        self.fb = omega_baryon / omega_matter
        self.omega_matter = omega_matter
    def E(self, z):
        return np.sqrt(self.omega_matter * (1 + z)**3 + 1 - self.omega_matter)
    def cosmicTime(self, z, units='Gyr'):
        z = np.asarray(z)
        ol = 1 - self.omega_matter
        t = 2 * (9.778 / self.h) / (3 * np.sqrt(ol)) * np.arcsinh(
            np.sqrt(ol / self.omega_matter) * (1 + z)**(-1.5))
        return t * (1000 if units == 'Myr' else 1)


def fixture_tree(base, halo, fields=None):
    node = halo * 100
    h = .6774
    tree = dict(SubhaloID=np.arange(node, node+5), SnapNum=np.array([2,1,0,1,0]),
                FirstProgenitorID=np.array([node+1,node+2,-1,node+4,-1]),
                NextProgenitorID=np.array([-1,node+3,-1,-1,-1]),
                DescendantID=np.array([-1,node,node+1,node,node+3]),
                MainLeafProgenitorID=np.array([node+2,node+2,node+2,node+4,node+4]),
                SubfindID=np.arange(halo,halo+5), SubhaloMass=np.array([7,1,.1,.3,.05])*h,
                SubhaloPos=np.zeros((5,3)))
    return tree if fields is None else {key:tree[key] for key in fields}


def fixture_particles(base, root, halo, snap, kind, fields=None):
    count = 2048
    data = dict(count=count, Coordinates=np.random.default_rng(17).normal(0,.02,(count,3)),
                ParticleIDs=np.arange(root*100000,root*100000+count),
                GFM_StellarFormationTime=np.full(count,.9999))
    return data if fields is None else dict(count=count, **{key:data[key] for key in fields})


def forbidden_global_draw(*args, **kwargs):
    raise AssertionError('parallel worker used global random draws')


def fixture_patches():
    return [patch.object(runner.astro_utils,'cosmo',SmallCosmo),
            patch.object(loader,'load_merger_tree',fixture_tree),
            patch.object(loader,'load_halo',fixture_particles),
            patch.object(np.random,'normal',forbidden_global_draw),
            patch.object(np.random,'permutation',forbidden_global_draw)]


def install_worker_fixture():
    # These process-local patches live for the lifetime of the worker.
    global worker_patches
    worker_patches = fixture_patches()
    for item in worker_patches:
        item.start()


class SpawnFixturePool:
    def __init__(self, processes):
        self.pool = mp.get_context('spawn').Pool(processes, initializer=install_worker_fixture)
    def __enter__(self):
        return self.pool
    def __exit__(self, exc_type, *args):
        if exc_type:
            self.pool.terminate()
        else:
            self.pool.close()
        self.pool.join()


def parameters(directory, halos, seed=7):
    return dict(verbose=False, seed=seed, seed_feh=0, p2=18, p3=.01, h100=.6774,
                Ob=.0486, Om=.3089, subs=halos, allcat_base='allcat',
                resultspath=str(directory)+'/', base_tree='', base_halo='',
                redshift_snap=np.array([3.,1.,0.]), full_snap=[0,1,2],
                log_Mmin=4., log_mc=7., log_Mhmin=8., regen_feh=False,
                gaussian_process=False, gauss_l=2., gauss_l_sm=2., sigma_mg=.3,
                sm_scat=True, mmr_slope=.3, mmr_pivot=9., mmr_evolution=1.,
                mmr0=-.5, max_feh=.3, tdep=.3, sigma_gas=.3, sigma_mc=0.,
                UVB_constraint='KM22', no_random_at_formation=False,
                form_nuclear_cluster=True, low_mass=False, low_mass_attempt_N=1,
                mpb_only=False, t_lag=.01, rmax_form=3., max_lag_ratio=.5)


def read_output(directory, seed):
    prefix = Path(directory)/('allcat_s-%d_p2-18_p3-0.01'%seed)
    catalog = np.loadtxt(str(prefix)+'.txt',ndmin=2)
    ids = np.loadtxt(str(prefix)+'_gcid.txt',ndmin=2,dtype=np.int64)
    return {halo:(catalog[catalog[:,0]==halo],ids[catalog[:,0]==halo])
            for halo in np.unique(catalog[:,0]).astype(int)}


class RNGJobTests(unittest.TestCase):
    def setUp(self):
        self.patches = fixture_patches()
        for item in self.patches:
            item.start()
    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
    def assert_catalogs_equal(self, a, b):
        self.assertEqual(set(a),set(b))
        for halo in a:
            self.assertGreater(len(a[halo][0]),0)
            for actual, expected in zip(a[halo],b[halo]):
                np.testing.assert_array_equal(actual,expected)

    def test_serial_jobs_reset_streams_and_do_not_mutate_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            result = []
            for i, halos in enumerate([[42,84],[84,42],[42]]):
                target = Path(directory)/str(i);target.mkdir()
                params = parameters(target,halos)
                for key in ['rng','rng_smhm','rng_feh']:
                    params[key] = np.random.default_rng(999)
                    params[key].normal(size=5)
                before = {key:deepcopy(params[key].bit_generator.state)
                          for key in ['rng','rng_smhm','rng_feh']}
                params['cosmo'] = object()
                runner.run_serial(params,i)
                self.assertNotIn('allcat_name',params)
                for key,state in before.items():
                    self.assertEqual(state,params[key].bit_generator.state)
                result.append(read_output(target,7))
            self.assert_catalogs_equal(result[0],result[1])
            self.assert_catalogs_equal({42:result[0][42]},result[2])

    def test_multi_seed_workers_match_serial_with_one_and_two_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            references = {}
            for seed in [7,9]:
                target = Path(directory)/('serial%d'%seed);target.mkdir()
                runner.run_serial(parameters(target,[42,84],seed),0)
                references[seed] = read_output(target,seed)
            # Verify different configured seeds really change the realization.
            self.assertFalse(np.array_equal(references[7][42][0],references[9][42][0]))
            for workers in [1,2]:
                target = Path(directory)/('parallel%d'%workers);target.mkdir()
                params = parameters(target,[84,42])
                params['seed_list'] = [9,7]
                # These cannot be serialized: they must be stripped before submission.
                params.update(rng=lambda:None,rng_smhm=lambda:None,rng_feh=lambda:None,cosmo=lambda:None)
                with patch.object(runner,'Pool',SpawnFixturePool):
                    runner.run_parallel(params,Np=workers,param_based=False,seed_based=True,to_tid=False)
                for seed in [7,9]:
                    self.assert_catalogs_equal(references[seed],read_output(target,seed))


if __name__ == '__main__':
    unittest.main()

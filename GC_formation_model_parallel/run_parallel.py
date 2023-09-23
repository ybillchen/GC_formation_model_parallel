# Licensed under BSD-3-Clause License - see LICENSE

from copy import copy
from multiprocessing import Pool

import numpy as np
# from mpi4py.futures import MPIPoolExecutor # NOTE: not used so far

from GC_formation_model import astro_utils
from GC_formation_model.form import form
from GC_formation_model.offset import offset
from GC_formation_model.assign import assign
# from GC_formation_model.evolve import evolve # NOTE: not used so far

from .get_tid_parallel import get_tid_parallel

__all__ = ['run_parallel']

def run_serial(params, p, to_form=True, to_offset=True, to_assign=True):

    if params['verbose']:
        print('Runing model on %d halo(s) at process %d.'%(len(params['subs']),p))

    allcat_name = params['allcat_base'] + '_s-%d_p2-%g_p3-%g.txt'%(
        params['seed'], params['p2'], params['p3'])

    run_params = params
    run_params['allcat_name'] = allcat_name

    run_params['cosmo'] = astro_utils.cosmo(h=run_params['h100'], 
        omega_baryon=run_params['Ob'], omega_matter=run_params['Om'])

    if to_form:
        form(run_params)
    if to_offset:
        offset(run_params)
    if to_assign:
        assign(run_params)

    if params['verbose']:
        print('\nModel was run on %d halo(s) at process %d.\n'%(len(params['subs']),p))

def run_parallel(params, Np=32, param_based=True, seed_based=False, 
    to_form=True, to_offset=True, to_assign=True, to_tid=True, skip=None, checkj=False):
    assert not (param_based and seed_based)

    if to_form or to_offset or to_assign:
        run_params = copy(params)

        para_list = []
        p = 0

        if param_based:
            for p2 in params['p2_arr']:
                for p3 in params['p3_arr']:
                    run_params['p2'] = p2
                    run_params['p3'] = p3
                    para_list.append((copy(run_params), p, to_form, to_offset, to_assign))
                    p += 1
        elif seed_based:
            for s in params['seed_list']:
                run_params['seed'] = s
                para_list.append((copy(run_params), p, to_form, to_offset, to_assign))
                p += 1
        else:
            para_list.append((copy(run_params), p, to_form, to_offset, to_assign))

        with Pool(Np) as p:
            p.starmap(run_serial, para_list)

        # TODO: Maybe try MPI to enable running on multiple modes? But not for now
        # TODO: Should be run with: `mpiexec -n 1 -usize 16 python xxx.py`
        # executor = MPIPoolExecutor(max_workers=Np)
        # executor.starmap(run_serial, para_list)

    if to_tid:
        get_tid_parallel(params, Np, file_prefix = 'combine', 
            param_based=param_based, seed_based=seed_based, skip=skip, checkj=checkj)
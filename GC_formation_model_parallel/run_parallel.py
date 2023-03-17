# Licensed under BSD-3-Clause License - see LICENSE

from multiprocessing import Pool
import copy
import numpy as np

from GC_formation_model import astro_utils
from GC_formation_model.form import form
from GC_formation_model.offset import offset
from GC_formation_model.assign import assign
from GC_formation_model.get_tid import get_tid
from GC_formation_model.evolve import evolve

from .get_tid_parallel import get_tid_parallel

__all__ = ['run_parallel']

def run_serial(params, p):

    if params['verbose']:
        print('Runing model on %d halo(s) at process %d.'%(len(params['subs']),p))

    allcat_name = params['allcat_base'] + '_s-%d_p2-%g_p3-%g.txt'%(
        params['seed'], params['p2'], params['p3'])

    run_params = params
    run_params['allcat_name'] = allcat_name

    run_params['cosmo'] = astro_utils.cosmo(h=run_params['h100'], 
        omega_baryon=run_params['Ob'], omega_matter=run_params['Om'])

    form(run_params)
    offset(run_params)
    assign(run_params)

    if params['verbose']:
        print('\nModel was run on %d halo(s) at process %d.\n'%(len(params['subs']),p))

def run_parallel(params, Np=32, seed_based=False):
    run_params = copy.copy(params)

    para_list = []
    p = 0

    if seed_based:
        for s in params['seed_list']:
            run_params['seed'] = s
            para_list.append((copy.copy(run_params), p))
            p += 1
    else:
        for p2 in params['p2_arr']:
            for p3 in params['p3_arr']:
                run_params['p2'] = p2
                run_params['p3'] = p3
                para_list.append((copy.copy(run_params), p))
                p += 1
        for p in para_list:
            print(p[0]['p2'], p[0]['p3'])

    with Pool(Np) as p:
        p.starmap(run_serial, para_list)

    get_tid_parallel(params, Np, file_prefix = 'combine', seed_based=seed_based)
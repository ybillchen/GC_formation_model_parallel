# Licensed under BSD-3-Clause License - see LICENSE

import os
from copy import copy
from multiprocessing import Pool

import numpy as np

from GC_formation_model.get_tid import get_tid, get_tid_unit

__all__ = ['combine_gc', 'combine_gc_seed', 'assign_eig', 'assign_eig_seed', 
    'get_tid_i', 'check_independent_status', 'combine_independent', 'get_tid_parallel']

# combine all gcid files into one big file to save time
def combine_gc(params):
    for idx_p2 in range(len(params['p2_arr'])):
        for idx_p3 in range(len(params['p3_arr'])):
            if params['verbose']:
                print('p2:', params['p2_arr'][idx_p2], 'p3:', params['p3_arr'][idx_p3])

            allcat_name = params['allcat_base'] + '_s-%d_p2-%g_p3-%g.txt'%(
                params['seed'], params['p2_arr'][idx_p2], params['p3_arr'][idx_p3])
            fin_name = params['resultspath'] + allcat_name
            gcid_name = fin_name[:-4] + '_gcid.txt'
            root_name = fin_name[:-4] + '_offset_root.txt'

            # load GC id
            gcid, quality = np.loadtxt(gcid_name, ndmin=2, unpack=True, dtype='int64')

            # load root offset
            hid_root, idx_beg, idx_end, idx_beg_in_off, idx_end_in_off = np.loadtxt(
                root_name, unpack = True, dtype='int64')

            # create the combined catalog at first
            if idx_p2 == 0 and idx_p3 == 0:
                hid_root_combine = hid_root
                gcid_combine = [[] for x in range(len(hid_root))]

            for i in range(len(hid_root)):
                j = np.where(hid_root_combine == hid_root[i])[0]
                if not len(j):
                    continue
                j = j[0]

                gcid_combine[j] = np.union1d(gcid_combine[j], gcid[idx_beg[i]:idx_end[i]])

    # make offset
    pointer = 0
    idx_beg = []
    idx_end = []
    gcid_faltten = []
    for i in range(len(hid_root_combine)):
        idx_beg.append(pointer)
        pointer += len(gcid_combine[i])
        idx_end.append(pointer)
        gcid_faltten.extend(gcid_combine[i])

    output_root = np.array([hid_root_combine, idx_beg, idx_end], dtype=int).T
    header = 'SubfindID(z=0) | BeginIdx | EndIdx+1'
    np.savetxt(params['resultspath']+params['file_prefix']+'_offset_root.txt', 
        output_root, fmt='%d ', header=header)

    gcid_faltten = np.array(gcid_faltten, dtype=int)
    header = 'GC ID'
    np.savetxt(params['resultspath']+params['file_prefix']+'_gcid.txt', 
        gcid_faltten, fmt='%d ', header=header)

    return 0

# combine all gcid files into one big file to save time
def combine_gc_seed(params):
    for k, seed in enumerate(params['seed_list']):
        if params['verbose']:
            print('seed:', seed)

        allcat_name = params['allcat_base'] + '_s-%d_p2-%g_p3-%g.txt'%(seed,params['p2'],params['p3'])
        fin_name = params['resultspath'] + allcat_name
        gcid_name = fin_name[:-4] + '_gcid.txt'
        root_name = fin_name[:-4] + '_offset_root.txt'

        # load GC id
        gcid, quality = np.loadtxt(gcid_name, ndmin=2, unpack=True, dtype='int64')

        # load root offset
        hid_root, idx_beg, idx_end, idx_beg_in_off, idx_end_in_off = np.loadtxt(
            root_name, unpack = True, dtype='int64')

        # create the combined catalog at first
        if k == 0:
            hid_root_combine = hid_root
            gcid_combine = [[] for x in range(len(hid_root))]

        for i in range(len(hid_root)):
            j = np.where(hid_root_combine == hid_root[i])[0]
            if not len(j):
                continue
            j = j[0]

            gcid_combine[j] = np.union1d(gcid_combine[j], gcid[idx_beg[i]:idx_end[i]])

    # make offset
    pointer = 0
    idx_beg = []
    idx_end = []
    gcid_faltten = []
    for i in range(len(hid_root_combine)):
        idx_beg.append(pointer)
        pointer += len(gcid_combine[i])
        idx_end.append(pointer)
        gcid_faltten.extend(gcid_combine[i])

    output_root = np.array([hid_root_combine, idx_beg, idx_end], dtype=int).T
    header = 'SubfindID(z=0) | BeginIdx | EndIdx+1'
    np.savetxt(params['resultspath']+params['file_prefix']+'_offset_root.txt', 
        output_root, fmt='%d ', header=header)

    gcid_faltten = np.array(gcid_faltten, dtype=int)
    header = 'GC ID'
    np.savetxt(params['resultspath']+params['file_prefix']+'_gcid.txt', 
        gcid_faltten, fmt='%d ', header=header)

    return 0

def assign_eig(params):
    mpb_only = params['mpb_only']
    d_tid = params['d_tid'] * params['h100'] # in kpc/h
    z_list = params['redshift_snap']
    base = params['base']
    file_prefix = params['file_prefix']
    gcid_name = params['resultspath'] + file_prefix + '_gcid.txt'
    root_name = params['resultspath'] + file_prefix + '_offset_root.txt'

    # load GC id
    gcid_c = np.loadtxt(gcid_name, unpack=True, dtype='int64')

    # load root offset
    hid_root_c, idx_beg_c, idx_end_c = np.loadtxt(
        root_name, unpack=True, dtype='int64')

    eig1 = np.loadtxt(params['resultspath']+file_prefix+'_tideig1.txt')
    eig2 = np.loadtxt(params['resultspath']+file_prefix+'_tideig2.txt')
    eig3 = np.loadtxt(params['resultspath']+file_prefix+'_tideig3.txt')
    tag = np.loadtxt(params['resultspath']+file_prefix+'_tidtag.txt')

    for idx_p2 in range(len(params['p2_arr'])):
        for idx_p3 in range(len(params['p3_arr'])):
            if params['verbose']:
                print('p2:', params['p2_arr'][idx_p2], 'p3:', params['p3_arr'][idx_p3])

            allcat_name = params['allcat_base'] + '_s-%d_p2-%g_p3-%g.txt'%(
                params['seed'], params['p2_arr'][idx_p2], params['p3_arr'][idx_p3])
            fin_name = params['resultspath'] + allcat_name
            gcid_name = fin_name[:-4] + '_gcid.txt'
            root_name = fin_name[:-4] + '_offset_root.txt'

            # load GC id
            gcid, quality = np.loadtxt(gcid_name, ndmin=2, unpack=True, dtype='int64')

            # load root offset
            hid_root, idx_beg, idx_end, idx_beg_in_off, idx_end_in_off = np.loadtxt(
                root_name, unpack = True, dtype='int64')

            eig_1 = np.zeros([len(gcid), len(full_snap)])
            eig_2 = np.zeros([len(gcid), len(full_snap)])
            eig_3 = np.zeros([len(gcid), len(full_snap)])
            tidtag = np.zeros([len(gcid), len(full_snap)])

            for i in range(len(hid_root)):

                j = np.where(hid_root_c == hid_root[i])[0]
                if not len(j):
                    continue
                j = j[0]
                
                xy, idx_1, idx_2 = np.intersect1d(gcid[idx_beg[i]:idx_end[i]], 
                    gcid_c[idx_beg_c[j]:idx_end_c[j]], return_indices=True,
                    assume_unique=True)

                idx_1 = idx_1 + idx_beg[i]
                idx_2 = idx_2 + idx_beg_c[j]

                eig_1[idx_1] = eig1[idx_2]
                eig_2[idx_1] = eig2[idx_2]
                eig_3[idx_1] = eig3[idx_2]
                tidtag[idx_1] = tag[idx_2]

            np.savetxt(fin_name[:-4]+'_tidtag.txt', tidtag, fmt='%d')
            np.savetxt(fin_name[:-4]+'_tideig1.txt', eig_1, fmt='%.3e')
            np.savetxt(fin_name[:-4]+'_tideig2.txt', eig_2, fmt='%.3e')
            np.savetxt(fin_name[:-4]+'_tideig3.txt', eig_3, fmt='%.3e')

def assign_eig_seed(params):
    mpb_only = params['mpb_only']
    d_tid = params['d_tid'] * params['h100'] # in kpc/h
    z_list = params['redshift_snap']
    base = params['base']
    file_prefix = params['file_prefix']
    gcid_name = params['resultspath'] + file_prefix + '_gcid.txt'
    root_name = params['resultspath'] + file_prefix + '_offset_root.txt'

    # load GC id
    gcid_c = np.loadtxt(gcid_name, unpack=True, dtype='int64')

    # load root offset
    hid_root_c, idx_beg_c, idx_end_c = np.loadtxt(
        root_name, unpack=True, dtype='int64')

    eig1 = np.loadtxt(params['resultspath']+file_prefix+'_tideig1.txt')
    eig2 = np.loadtxt(params['resultspath']+file_prefix+'_tideig2.txt')
    eig3 = np.loadtxt(params['resultspath']+file_prefix+'_tideig3.txt')
    tag = np.loadtxt(params['resultspath']+file_prefix+'_tidtag.txt')

    for k, seed in enumerate(params['seed_list']):
        if params['verbose']:
            print('seed:', seed)

        allcat_name = params['allcat_base'] + '_s-%d_p2-%g_p3-%g.txt'%(seed,params['p2'],params['p3'])
        fin_name = params['resultspath'] + allcat_name
        gcid_name = fin_name[:-4] + '_gcid.txt'
        root_name = fin_name[:-4] + '_offset_root.txt'

        # load GC id
        gcid, quality = np.loadtxt(gcid_name, ndmin=2, unpack=True, dtype='int64')

        # load root offset
        hid_root, idx_beg, idx_end, idx_beg_in_off, idx_end_in_off = np.loadtxt(
            root_name, unpack = True, dtype='int64')

        eig_1 = np.zeros([len(gcid), len(full_snap)])
        eig_2 = np.zeros([len(gcid), len(full_snap)])
        eig_3 = np.zeros([len(gcid), len(full_snap)])
        tidtag = np.zeros([len(gcid), len(full_snap)])

        for i in range(len(hid_root)):

            j = np.where(hid_root_c == hid_root[i])[0]
            if not len(j):
                continue
            j = j[0]
            
            xy, idx_1, idx_2 = np.intersect1d(gcid[idx_beg[i]:idx_end[i]], 
                gcid_c[idx_beg_c[j]:idx_end_c[j]], return_indices=True,
                assume_unique=True)

            idx_1 = idx_1 + idx_beg[i]
            idx_2 = idx_2 + idx_beg_c[j]

            eig_1[idx_1] = eig1[idx_2]
            eig_2[idx_1] = eig2[idx_2]
            eig_3[idx_1] = eig3[idx_2]
            tidtag[idx_1] = tag[idx_2]

        np.savetxt(fin_name[:-4]+'_tidtag.txt', tidtag, fmt='%d')

        np.savetxt(fin_name[:-4]+'_tideig1.txt', eig_1, fmt='%.3e')
        np.savetxt(fin_name[:-4]+'_tideig2.txt', eig_2, fmt='%.3e')
        np.savetxt(fin_name[:-4]+'_tideig3.txt', eig_3, fmt='%.3e')


# get tidal tensor for one galaxy
def get_tid_i(i, gcid, hid_root, idx_beg, idx_end, params):
    basepath = params['resultspath'] + 'independent_tidal_outputs/'
    file_prefix = params['file_prefix']

    isExist = os.path.exists(basepath)
    if not isExist:
       os.makedirs(basepath)

    file_exist = os.path.isfile(basepath+file_prefix+'_tidtag_i%d.txt'%(i))

    if file_exist:
        return 0

    tag_i, eig_1_i, eig_2_i, eig_3_i = get_tid_unit(i, gcid, hid_root, idx_beg, idx_end, params)

    np.savetxt(basepath+file_prefix+'_tidtag_i%d.txt'%(i), tag_i, fmt='%d')
    np.savetxt(basepath+file_prefix+'_tideig1_i%d.txt'%(i), eig_1_i, fmt='%.3e')
    np.savetxt(basepath+file_prefix+'_tideig2_i%d.txt'%(i), eig_2_i, fmt='%.3e')
    np.savetxt(basepath+file_prefix+'_tideig3_i%d.txt'%(i), eig_3_i, fmt='%.3e')

def check_independent_status(params, irange=None):
    basepath = params['resultspath'] + 'independent_tidal_outputs/'
    file_prefix = params['file_prefix']
    
    if irange is None:
        irange = range(len(params['subs']))

    for i in irange:
        file_exist = os.path.isfile(basepath+file_prefix+'_tidtag_i%d.txt'%(i))
        if params['verbose']:
            print('i: %d'%(i), file_exist)

def combine_independent(params, irange=None):
    mpb_only = params['mpb_only']
    z_list = params['redshift_snap']
    base = params['base']
    file_prefix = params['file_prefix']
    gcid_name = params['resultspath'] + file_prefix + '_gcid.txt'
    root_name = params['resultspath'] + file_prefix + '_offset_root.txt'

    # load GC id
    gcid = np.loadtxt(gcid_name, unpack=True, dtype='int64')

    tag = np.zeros([len(gcid), len(full_snap)], dtype=int)
    eig1 = np.zeros([len(gcid), len(full_snap)])
    eig2 = np.zeros([len(gcid), len(full_snap)])
    eig3 = np.zeros([len(gcid), len(full_snap)])

    basepath = params['resultspath'] + 'independent_tidal_outputs/'

    if irange is None:
        irange = range(len(params['subs']))

    for i in irange:
        if params['verbose']:
            print('NO. %d'%i)

        tag_now = np.loadtxt(basepath+file_prefix+'_tidtag_i%d.txt'%(i))
        eig1_now = np.loadtxt(basepath+file_prefix+'_tideig1_i%d.txt'%(i))
        eig2_now = np.loadtxt(basepath+file_prefix+'_tideig2_i%d.txt'%(i))
        eig3_now = np.loadtxt(basepath+file_prefix+'_tideig3_i%d.txt'%(i))

        tag[idx_beg[i]:idx_end[i]] = tag_now
        eig1[idx_beg[i]:idx_end[i]] = eig1_now
        eig2[idx_beg[i]:idx_end[i]] = eig2_now
        eig3[idx_beg[i]:idx_end[i]] = eig3_now

        np.savetxt(params['resultspath']+file_prefix+'_tidtag.txt', tag, fmt='%d')
        np.savetxt(params['resultspath']+file_prefix+'_tideig1.txt', eig1, fmt='%.3e')
        np.savetxt(params['resultspath']+file_prefix+'_tideig2.txt', eig2, fmt='%.3e')
        np.savetxt(params['resultspath']+file_prefix+'_tideig3.txt', eig3, fmt='%.3e')

def get_tid_parallel(params, Np=32, file_prefix='combine', seed_based=False):
    run_params = copy(params)

    run_params['file_prefix'] = file_prefix

    # if seed_based:
    #     combine_gc_seed(run_params)
    # else:
    #     combine_gc(run_params)

    # load data
    gcid_name = run_params['resultspath'] + file_prefix + '_gcid.txt'
    root_name = run_params['resultspath'] + file_prefix + '_offset_root.txt'

    # load GC id
    gcid = np.loadtxt(gcid_name, unpack=True, dtype='int64')

    # load root offset
    hid_root, idx_beg, idx_end = np.loadtxt(
        root_name, ndmin=2, unpack=True, dtype='int64')[:3]
    
    para_list = []
    for i in range(len(run_params['subs'])):
        para_list.append((i, gcid, hid_root, idx_beg, idx_end, run_params))

    with Pool(Np) as p:
        p.starmap(get_tid_i, para_list)

    # # not parrelization
    # get_tid_i(7, gcid, hid_root, idx_beg, idx_end, params)

    check_independent_status(run_params)
    combine_independent(run_params)

    if seed_based:
        assign_eig_seed(run_params)
    else:
        assign_eig(run_params)

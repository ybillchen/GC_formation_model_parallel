# Licensed under BSD-3-Clause License - see LICENSE

# Fast drop-in replacement for GC_formation_model.get_tid.get_tid_unit.
#
# The original calc_eig builds one scipy LinearNDInterpolator (i.e., one Qhull
# Delaunay triangulation) per GC per grid point inside a Python loop, which
# dominates the runtime. Here the same interpolation is done in a numba kernel:
# for the k=8 nearest neighbors, the Delaunay tetrahedron containing the query
# point is found by brute force over all C(8,4)=70 tetrahedra (a containing
# tetrahedron whose circumsphere is empty of the other neighbors), followed by
# barycentric interpolation. Queries are parallelized over threads with prange.

import time
import warnings
from itertools import combinations

import numpy as np
import scipy.spatial as sp
import h5py
import numba
from numba import njit, prange

from GC_formation_model import loader

__all__ = ['calc_eig_fast', 'get_tid_unit_fast']

_K = 8 # number of neighbors used for interpolation, as in calc_eig
_TETS = np.array(list(combinations(range(_K), 4)), dtype=np.int64)

# grid offsets used by calc_eig (corners of the 3x3x3 cube are not needed)
_OFFSETS = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1)
    for c in (-1, 0, 1)], dtype=float)
_USED = np.array([i for i in range(27) if (i != 13) and
    (i not in (0, 2, 6, 8, 18, 20, 24, 26))], dtype=np.int64)

@njit(cache=True, fastmath=False)
def _interp_one(p, v, tets):
    # p: (K,3) neighbor positions relative to the query point (query at origin)
    # v: (K,) values at the neighbors
    # returns the linear Delaunay interpolation at the origin, or nan if the
    # origin is outside the convex hull of the neighbors
    K = p.shape[0]

    scale2 = 0.0
    for m in range(K):
        s = p[m,0]**2 + p[m,1]**2 + p[m,2]**2
        if s > scale2:
            scale2 = s
    if scale2 == 0.0:
        return v[0]
    eps_vol = 1e-12 * scale2**1.5
    eps_bary = 1e-10

    best_val = np.nan
    best_viol = np.inf

    for t in range(tets.shape[0]):
        ia = tets[t,0]; ib = tets[t,1]; ic = tets[t,2]; id_ = tets[t,3]
        ax = p[ia,0]; ay = p[ia,1]; az = p[ia,2]
        # edge vectors
        bx = p[ib,0]-ax; by = p[ib,1]-ay; bz = p[ib,2]-az
        cx = p[ic,0]-ax; cy = p[ic,1]-ay; cz = p[ic,2]-az
        dx = p[id_,0]-ax; dy = p[id_,1]-ay; dz = p[id_,2]-az

        det = bx*(cy*dz-cz*dy) - by*(cx*dz-cz*dx) + bz*(cx*dy-cy*dx)
        if abs(det) <= eps_vol:
            continue

        # barycentric coordinates of the origin: [b c d] lam = -a
        rx = -ax; ry = -ay; rz = -az
        l1 = (rx*(cy*dz-cz*dy) - ry*(cx*dz-cz*dx) + rz*(cx*dy-cy*dx)) / det
        l2 = (bx*(ry*dz-rz*dy) - by*(rx*dz-rz*dx) + bz*(rx*dy-ry*dx)) / det
        l3 = (bx*(cy*rz-cz*ry) - by*(cx*rz-cz*rx) + bz*(cx*ry-cy*rx)) / det
        l0 = 1.0 - l1 - l2 - l3
        if (l0 < -eps_bary) or (l1 < -eps_bary) or (l2 < -eps_bary) or (l3 < -eps_bary):
            continue

        # circumcenter relative to a: [b c d]^T u = 0.5*(|b|^2,|c|^2,|d|^2)
        hb = 0.5*(bx*bx+by*by+bz*bz)
        hc = 0.5*(cx*cx+cy*cy+cz*cz)
        hd = 0.5*(dx*dx+dy*dy+dz*dz)
        # solve rows (b,c,d) . u = (hb,hc,hd) by Cramer's rule
        ux = (hb*(cy*dz-cz*dy) - by*(hc*dz-cz*hd) + bz*(hc*dy-cy*hd)) / det
        uy = (bx*(hc*dz-cz*hd) - hb*(cx*dz-cz*dx) + bz*(cx*hd-hc*dx)) / det
        uz = (bx*(cy*hd-hc*dy) - by*(cx*hd-hc*dx) + hb*(cx*dy-cy*dx)) / det
        R2 = ux*ux + uy*uy + uz*uz

        # largest relative intrusion of the other neighbors into the circumsphere
        viol = -np.inf
        for m in range(K):
            if (m == ia) or (m == ib) or (m == ic) or (m == id_):
                continue
            ex = p[m,0]-ax-ux; ey = p[m,1]-ay-uy; ez = p[m,2]-az-uz
            w = (R2 - (ex*ex+ey*ey+ez*ez)) / R2
            if w > viol:
                viol = w

        if viol < best_viol:
            best_viol = viol
            best_val = l0*v[ia] + l1*v[ib] + l2*v[ic] + l3*v[id_]
            if viol <= 1e-10: # Delaunay tetrahedron found
                break

    return best_val

@njit(parallel=True, cache=True)
def _interp_batch(query, nbr, pos, pot, tets):
    # query: (M,3); nbr: (M,K) neighbor indices into pos/pot
    M = query.shape[0]
    K = nbr.shape[1]
    out = np.empty(M)
    for q in prange(M):
        p = np.empty((K,3))
        v = np.empty(K)
        for m in range(K):
            idx = nbr[q,m]
            p[m,0] = pos[idx,0] - query[q,0]
            p[m,1] = pos[idx,1] - query[q,1]
            p[m,2] = pos[idx,2] - query[q,2]
            v[m] = pot[idx]
        out[q] = _interp_one(p, v, tets)
    return out

def calc_eig_fast(tree, pos_gc, pot_gc, pos, pot, d_tid, workers=-1):
    # same inputs and output as GC_formation_model.get_tid.calc_eig
    # phi in km^2/s^2
    # pos and d_tid in kpc/h
    N = len(pos_gc)

    # (18, N, 3) grid points around each GC
    grid = pos_gc[None,:,:] + d_tid * _OFFSETS[_USED][:,None,:]
    query = grid.reshape(-1, 3)
    nbr = tree.query(query, k=_K, workers=workers)[1].astype(np.int64)

    pot_grid = np.zeros([27, N])
    pot_grid[_USED] = _interp_batch(query, nbr, pos, pot, _TETS).reshape(len(_USED), N)
    pot_grid[13] = pot_gc

    for i in range(13):
        nan_1 = np.isnan(pot_grid[i])
        nan_2 = np.isnan(pot_grid[26-i])
        if not (nan_1.any() or nan_2.any()):
            continue
        idx_0 = nan_1 & nan_2
        idx_1 = nan_1 & ~nan_2
        idx_2 = nan_2 & ~nan_1

        pot_grid[i][idx_0] = pot_gc[idx_0]
        pot_grid[26-i][idx_0] = pot_gc[idx_0]
        pot_grid[i][idx_1] = 2*pot_gc[idx_1] - pot_grid[26-i][idx_1]
        pot_grid[26-i][idx_2] = 2*pot_gc[idx_2] - pot_grid[i][idx_2]

    # tidal tensor
    Txx = (pot_grid[4] + pot_grid[22] - 2*pot_grid[13]) / d_tid**2
    Tyy = (pot_grid[10] + pot_grid[16] - 2*pot_grid[13]) / d_tid**2
    Tzz = (pot_grid[12] + pot_grid[14] - 2*pot_grid[13]) / d_tid**2
    Txy = (pot_grid[1] + pot_grid[25] - pot_grid[7] - pot_grid[19]) / 4 / d_tid**2
    Txz = (pot_grid[3] + pot_grid[23] - pot_grid[5] - pot_grid[21]) / 4 / d_tid**2
    Tyz = (pot_grid[9] + pot_grid[17] - pot_grid[11] - pot_grid[15]) / 4 / d_tid**2

    T = np.array([
        [Txx, Txy, Txz],
        [Txy, Tyy, Tyz],
        [Txz, Tyz, Tzz]]).T

    return 0.48 * np.linalg.eigvalsh(T) # in Gyr^-2, ascending

def _load_part(f, snap, subid, parttype, fields):
    d = f['snap_%d_halo_%d'%(snap,subid)][parttype]
    res = {'count': d.attrs['count']}
    for field in fields:
        res[field] = d[field][:]
    return res

# get tidal tensor for one galaxy
# same interface and output as GC_formation_model.get_tid.get_tid_unit
def get_tid_unit_fast(i, gcid, hid_root, idx_beg, idx_end, params, k=-1, nthreads=None):
    if nthreads is not None:
        numba.set_num_threads(max(1, min(int(nthreads), numba.config.NUMBA_NUM_THREADS)))
    workers = -1 if nthreads is None else max(1, int(nthreads))

    d_tid = params['d_tid'] * params['h100'] # in kpc/h
    z_list = params['redshift_snap']
    full_snap = params['full_snap']

    t0 = time.time()

    # load merger tree
    fields = ['SnapNum', 'SubfindID', 'SubhaloMass']
    tree = loader.load_merger_tree(params['base_tree'], hid_root[i], fields)

    # existing GCs at this snapshot
    idx_exist_gc = np.arange(idx_beg[i], idx_end[i])
    gcid_i = gcid[idx_exist_gc]

    tag = np.zeros([len(idx_exist_gc), len(full_snap)], dtype=int)
    eig_1 = np.zeros([len(idx_exist_gc), len(full_snap)])
    eig_2 = np.zeros([len(idx_exist_gc), len(full_snap)])
    eig_3 = np.zeros([len(idx_exist_gc), len(full_snap)])

    if k < 0:
        iterlist = range(len(full_snap))
    else:
        iterlist = [k]

    skip = params.get('skip')

    f = h5py.File(params['base_halo'] + 'halo_%d.hdf5'%hid_root[i], 'r')
    try:
        for j in iterlist:

            if skip is not None:
                if skip[0] == i and skip[1] == j:
                    continue

            t1 = time.time()
            t2 = 0 # load halo
            t3 = 0 # build tree
            t4 = 0 # calc eig

            snap = full_snap[j]
            scale_a = 1 / (1 + z_list[snap])

            # all subhalos at this snapshot
            idx_sub = np.where((tree['SnapNum']==snap) &
                (tree['SubhaloMass'] > 10**(params['log_Mhmin']-10)*params['h100']))[0]
            subfindid = tree['SubfindID'][idx_sub]

            count = 0 # found GCs
            # loop over all subhalos and load densities
            for subid in subfindid:
                t22 = time.time()
                # first, consider all GCs represented by dm
                fields = ['Coordinates', 'ParticleIDs', 'Potential']
                cutout = _load_part(f, snap, subid, 'dm', fields)
                pos = cutout['Coordinates'] * scale_a # in kpc/h
                pid = cutout['ParticleIDs'].astype(int)
                pot = cutout['Potential'] / scale_a # in km^2/s^2

                # second, consider all GCs represented by stars
                cutout = _load_part(f, snap, subid, 'stars', fields)
                if cutout['count'] > 0:
                    pos = np.concatenate((pos, cutout['Coordinates'] * scale_a)) # in kpc/h
                    pid = np.concatenate((pid, cutout['ParticleIDs'].astype(int)))
                    pot = np.concatenate((pot, cutout['Potential'] / scale_a)) # in km^2/s^2

                # find intersections, xy is useless
                xy, idx_1, idx_2 = np.intersect1d(pid, gcid_i, return_indices=True)

                if len(xy) == 0: # if gc particles not found
                    continue

                pos_gc = pos[idx_1]
                pot_gc = pot[idx_1]

                count += len(xy)

                fields = ['Coordinates', 'Potential']
                cutout = _load_part(f, snap, subid, 'gas', fields)
                if cutout['count'] > 0:
                    pos = np.concatenate((pos, cutout['Coordinates'] * scale_a)) # in kpc
                    pot = np.concatenate((pot, cutout['Potential'] / scale_a)) # in km^2/s^2

                t33 = time.time()
                t2 += (t33 - t22) # load halo

                pos = np.ascontiguousarray(pos, dtype=float)
                pot = np.ascontiguousarray(pot, dtype=float)
                kdtree = sp.cKDTree(pos)

                t44 = time.time()
                t3 += (t44 - t33) # build tree

                eig = calc_eig_fast(kdtree, pos_gc.astype(float), pot_gc.astype(float),
                    pos, pot, d_tid, workers=workers) # in Gyr^-2, ascending

                if not np.all(np.isfinite(eig)):
                    # keep the original behavior of skipping a problematic subhalo
                    t4 += time.time() - t44 # calc eig
                    warnings.warn('Non-finite eigenvalues at NO. %d snap %d!'%(i,full_snap[j]))
                    continue

                # update the tag and eig matrices
                tag[idx_2,j] = 1

                eig_1[idx_2,j] = eig[:,2]
                eig_2[idx_2,j] = eig[:,1]
                eig_3[idx_2,j] = eig[:,0]

                t4 += time.time() - t44 # calc eig

            if params['verbose']:
                print('  * NO. %d, hid: %d, snap: %d, %d/%d found, time: %.1fs'%(
                    i, hid_root[i], snap, count, len(idx_exist_gc), time.time()-t1))
                print('   - load halo: %.1fs, build tree: %.1fs, calc eig: %.1fs'%(t2,t3,t4))
    finally:
        f.close()

    if params['verbose'] and k < 0:
        print(' NO. %d, halo id: %d completed, total time: %.1f s'%(i, hid_root[i], time.time()-t0))

    if k < 0:
        return tag, eig_1, eig_2, eig_3

    return tag[:,k], eig_1[:,k], eig_2[:,k], eig_3[:,k]

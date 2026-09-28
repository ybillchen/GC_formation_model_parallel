# GC_formation_model_parallel

[![version](https://img.shields.io/badge/version-0.1-blue.svg)](https://github.com/ybillchen/GC_formation_model_parallel)
[![license](https://img.shields.io/github/license/ybillchen/GC_formation_model_parallel)](LICENSE)

A parallel toolkit for [`GC_formation_model`](https://github.com/ybillchen/GC_formation_model).

## Reproducible model jobs

This toolkit imports formation and assignment from the locally installed
`GC_formation_model`; use its updated explicit-RNG implementation. There are
no independent random draws in this toolkit. Model jobs discard transient
`rng`, `rng_smhm`, `rng_feh`, and `cosmo` entries from caller parameters and
rebuild stage state using the configured seeds.

The main model retains its formation seed (`seed + halo_id`) and assignment
seed (`seed`, reset per galaxy). SMHM scatter uses the separate
`SeedSequence([seed, halo_id, 0x534D484D])` stream; metallicity regeneration
uses `seed_feh + halo_id + 1`. Do not add worker-index seeds or reseed NumPy's
global RNG. This keeps random streams independent of scheduling, worker count,
and the galaxy list, for fixed inputs and numerical software versions.

Direct SMHM calls with `scatter=True` must pass `rng`; deterministic calls
consume no randomness. Optional modes previously using global randomness
produce a new reproducible realization after updating the main model.

Tidal checkpoints currently include galaxy **list indices** in their names.
Use a different `resultspath` when changing the galaxy list, its order, or
model settings, to avoid reusing checkpoints from another run.

Run the regression suite with `python -m unittest discover -s tests -v`.

## Maintainers

- [@Yingtian (Bill) Chen](https://github.com/ybillchen)

## License

[BSD 3-Clause License](LICENSE) &copy; Bill Chen

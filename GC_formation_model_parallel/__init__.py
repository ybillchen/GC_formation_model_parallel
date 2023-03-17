# Licensed under BSD-3-Clause License - see LICENSE

from . import run_parallel
from . import get_tid_parallel
__all__ = run_parallel.__all__ + get_tid_parallel.__all__

from .run_parallel import *
from .get_tid_parallel import *
from .version import __version__

__name__ = 'GC_formation_model_parallel'
__author__ = 'Bill Chen'

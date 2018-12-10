import warnings
import os

from numpy import f2py

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
IS_DEBUG = 'IS_DEVELOPMENT' in os.environ.keys()

FLAGS_DEBUG = []
FLAGS_DEBUG += ['-O', '-Wall', '-Wline-truncation', '-Wsurprising', '-Waliasing']
FLAGS_DEBUG += ['-Wunused-parameter', '-fwhole-file', '-fcheck=all']
FLAGS_DEBUG += ['-fbacktrace', '-g', '-fmax-errors=1', '-ffree-line-length-0']
FLAGS_DEBUG += ['-cpp', '-Wcharacter-truncation', '-Wimplicit-interface']

FLAGS_PRODUCTION = ['-O3', '-ffree-line-length-0']


HUGE_INT = 1000000000
HUGE_FLOAT = 1e15

DEFAULT_BOUND = 1e8


# We need to manage our warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '\n ... %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line
if not IS_DEBUG:
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def compile_f2py(is_debug=False):
    """This function compiles the F2PY interface."""
    base_path = os.getcwd()
    os.chdir(PACKAGE_DIR)

    if not is_debug:
        FLAGS = FLAGS_PRODUCTION
    else:
        FLAGS = FLAGS_DEBUG

    os.chdir('src')

    import respy
    path = os.path.dirname(respy.__file__)

    args = ''
    args += '--f90exec=mpif90 --f90flags=' + '"' + ' '.join(FLAGS) + '" '
    args += ' -I' + path + '/.bld  -L' + path + '/.bld/fortran'
    args += ' -lresfort_library -llapack'

    src = open('smm_interface.f90', 'rb').read()
    f2py.compile(src, 'smm_interface', args, extension='.f90')

    os.chdir(base_path)


try:
    from respy_smm.src import smm_interface
except (ModuleNotFoundError, ImportError) as e:
    compile_f2py(IS_DEBUG)


from respy_smm.optimizers_interface import optimize
from respy_smm.weighing import get_weighing_matrix
from respy_smm.moments import get_moments

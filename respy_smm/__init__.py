import socket
import os

from numpy import f2py

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

FLAGS_DEBUG = []
FLAGS_DEBUG += ['-O', '-Wall', '-Wline-truncation', '-Wsurprising', '-Waliasing']
FLAGS_DEBUG += ['-Wunused-parameter', '-fwhole-file', '-fcheck=all']
FLAGS_DEBUG += ['-fbacktrace', '-g', '-fmax-errors=1', '-ffree-line-length-0']
FLAGS_DEBUG += ['-cpp', '-Wcharacter-truncation', '-Wimplicit-interface']

FLAGS_PRODUCTION = ['-O3', '-ffree-line-length-0']


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
    # This ensures that the debug version is compiled on the development machines but not in
    # production.
    is_debug = socket.gethostname() in ['pontos', 'heracles']
    compile_f2py(is_debug)


from respy_smm.interface_scipy import run_scipy


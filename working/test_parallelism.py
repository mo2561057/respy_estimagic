import subprocess as sp

while True:
    sp.check_call('mpiexec -n 1 python parallel_test.py', shell=True)
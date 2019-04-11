import subprocess as sp
import os
import glob


cwd = os.getcwd()
os.chdir('../submodules/respy/respy')
sp.check_call('./waf configure build --debug', shell=True)
os.chdir(cwd)


os.chdir('../')
sp.check_call('git clean -df', shell=True)


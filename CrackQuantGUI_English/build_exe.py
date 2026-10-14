"""Build the standalone CrackQuant executable with PyInstaller.

Run from the activated environment that has torch (CUDA), PyQt5, scikit-image,
scipy and pyinstaller installed:

    python build_exe.py

Output: dist/CrackQuant_English/CrackQuant_English.exe
"""
import glob
import os
import shutil
import sys

import PyInstaller.__main__

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
DIST = os.path.join(ROOT, 'dist', 'CrackQuant_English')

# ---- Locate the Python DLL of the current interpreter ----------------------
_PYHOME = os.path.dirname(sys.executable)
python_dll = None
for dll_name in ('python310.dll', 'python311.dll', 'python312.dll', 'python313.dll'):
    candidate = os.path.join(_PYHOME, dll_name)
    if os.path.exists(candidate):
        python_dll = candidate
        break
if not python_dll:
    raise FileNotFoundError('No python DLL found in {}'.format(_PYHOME))

# ---- Locate Tcl/Tk relative to the current environment ---------------------
_tcl_base = os.path.join(os.path.dirname(os.path.dirname(_PYHOME)), 'Library', 'lib')
tcl_dir = os.path.join(_tcl_base, 'tcl8.6')
tk_dir = os.path.join(_tcl_base, 'tk8.6')
override = os.environ.get('TCL_LIB_DIR')
if override:
    tcl_dir = os.path.join(override, 'tcl8.6')
    tk_dir = os.path.join(override, 'tk8.6')
if not os.path.exists(tcl_dir):
    print('WARNING: Tcl/Tk not found at {} - set TCL_LIB_DIR if the build fails.'
          .format(tcl_dir))

print('Python DLL: {}'.format(python_dll))
print('Tcl dir: {} (exists={})'.format(tcl_dir, os.path.exists(tcl_dir)))

# Checkpoint to bundle: the model selected in the project's config.py.
sys.path.insert(0, PROJECT_ROOT)
from config import Config  # noqa: E402

checkpoint = Config.BEST_MODEL_PATH
if not os.path.exists(checkpoint):
    raise FileNotFoundError(
        'Checkpoint not found: {}\n'
        'Train the model first (python train.py), or set EXP_NAME in config.py.'
        .format(checkpoint)
    )
print('Checkpoint: {}'.format(checkpoint))

# ---- Clean previous output ------------------------------------------------
for d in (os.path.join(ROOT, 'build'), DIST):
    shutil.rmtree(d, ignore_errors=True)

PyInstaller.__main__.run([
    os.path.join(ROOT, 'main.py'),
    '--name=CrackQuant_English', '--onedir', '--windowed', '--noconfirm',
    '--distpath={}'.format(os.path.join(ROOT, 'dist')),
    '--workpath={}'.format(os.path.join(ROOT, 'build')),
    '--specpath={}'.format(ROOT),
    '--add-data={};models'.format(os.path.join(PROJECT_ROOT, 'models')),
    '--add-data={};.'.format(checkpoint),
    '--add-binary={};.'.format(python_dll),
    '--add-data={};tcl8.6'.format(tcl_dir),
    '--add-data={};tk8.6'.format(tk_dir),
    '--hidden-import=torch.distributed',
    '--hidden-import=torch.utils.data',
    '--hidden-import=skimage.morphology',
    '--hidden-import=skimage.draw',
    '--hidden-import=skimage.filters',
    '--hidden-import=scipy.ndimage',
    '--exclude-module=torch.utils.tensorboard',
    '--exclude-module=torchvision',
    '--exclude-module=matplotlib',
    '--hidden-import=appdirs',
    '--hidden-import=pkg_resources',
    '--collect-all=appdirs',
])

# ---- Post-build adjustments -----------------------------------------------
internal = os.path.join(DIST, '_internal')

# 1. Drop the pkg_resources runtime hook (incompatible with Python 3.13).
for f in glob.glob(os.path.join(internal, 'pyi_rth_pkgres*')):
    os.remove(f)
    print('Removed {}'.format(os.path.basename(f)))

# 2. Drop pkg_resources / setuptools themselves (not needed at run time).
for d in ('pkg_resources', 'setuptools'):
    p = os.path.join(internal, d)
    if os.path.exists(p):
        shutil.rmtree(p, ignore_errors=True)
        print('Removed {}'.format(d))

# 3. Drop the tkinter hook (this application uses PyQt5).
for f in glob.glob(os.path.join(internal, '*tkinter*')):
    os.remove(f)
    print('Removed {}'.format(os.path.basename(f)))

# 4. Copy the CUDA DLLs next to the executable.
torch_lib = os.path.join(internal, 'torch', 'lib')
if os.path.exists(torch_lib):
    markers = ('cuda', 'cublas', 'cudart', 'cufft', 'curand', 'cusolver',
               'cusparse', 'nvrtc', 'nvjitlink', 'cudnn', 'c10_cuda')
    for f in os.listdir(torch_lib):
        if f.endswith('.dll') and any(x in f for x in markers):
            dst = os.path.join(internal, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(torch_lib, f), dst)
    print('Copied CUDA DLLs to _internal/')

# 5. Place the MSVC/UCRT runtime DLLs beside c10.dll. Some Windows
#    installations do not search the parent _internal directory when resolving
#    transitive DLL dependencies.
runtime_names = [
    'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'ucrtbase.dll',
    'api-ms-win-crt-conio-l1-1-0.dll', 'api-ms-win-crt-convert-l1-1-0.dll',
    'api-ms-win-crt-environment-l1-1-0.dll', 'api-ms-win-crt-filesystem-l1-1-0.dll',
    'api-ms-win-crt-heap-l1-1-0.dll', 'api-ms-win-crt-locale-l1-1-0.dll',
    'api-ms-win-crt-math-l1-1-0.dll', 'api-ms-win-crt-multibyte-l1-1-0.dll',
    'api-ms-win-crt-private-l1-1-0.dll', 'api-ms-win-crt-process-l1-1-0.dll',
    'api-ms-win-crt-runtime-l1-1-0.dll', 'api-ms-win-crt-stdio-l1-1-0.dll',
    'api-ms-win-crt-string-l1-1-0.dll', 'api-ms-win-crt-time-l1-1-0.dll',
    'api-ms-win-crt-utility-l1-1-0.dll',
]
copied_runtime = 0
for name in runtime_names:
    src = os.path.join(internal, name)
    dst = os.path.join(torch_lib, name)
    if os.path.exists(src) and os.path.exists(torch_lib) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        copied_runtime += 1
print('Copied {} MSVC/UCRT runtime DLLs beside c10.dll'.format(copied_runtime))

print('\nBUILD DONE: {}'.format(os.path.join(DIST, 'CrackQuant_English.exe')))

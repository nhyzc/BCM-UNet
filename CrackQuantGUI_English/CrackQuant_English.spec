# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for CrackQuant. Paths are resolved relative to this file, so
# the project can be built from any checkout location. Prefer build_exe.py,
# which regenerates this spec with the correct absolute paths for your machine.
import os
from PyInstaller.utils.hooks import collect_all

ROOT = os.path.dirname(os.path.abspath(SPEC))
PROJECT_ROOT = os.path.dirname(ROOT)

datas = [
    (os.path.join(PROJECT_ROOT, 'models'), 'models'),
    (os.path.join(PROJECT_ROOT, 'output_results', 'BoundaryUNet_SR111', 'best_model.pth'), '.'),
]
binaries = []
hiddenimports = [
    'torch.distributed', 'torch.utils.data',
    'skimage.morphology', 'skimage.draw', 'skimage.filters',
    'scipy.ndimage', 'appdirs', 'pkg_resources',
]
tmp_ret = collect_all('appdirs')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

a = Analysis(
    [os.path.join(ROOT, 'main.py')],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch.utils.tensorboard', 'torchvision', 'matplotlib'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CrackQuant_English',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CrackQuant_English',
)

# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)

block_cipher = None
SETUP_DIR = 'F:\\罗德岛之心PC桌面程序\\'


a = Analysis(['main.py','comb_box_data.py',
             'lgb_model.py','main_ui.py'],
             pathex=['F:\\罗德岛之心PC桌面程序'],
             binaries=[],
             datas=[(SETUP_DIR+'ID','ID'),(SETUP_DIR+'icon','icon'),(SETUP_DIR+'model','model'),
             (SETUP_DIR+'profile','profile')],
             hiddenimports=['pandas._libs','pandas._libs.tslibs.np_datetime','pandas._libs.tslibs.timedeltas',
'pandas._libs.tslibs.nattype', 'pandas._libs.skiplist','scipy._lib','scipy._lib.messagestream',],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='RhodeIslandHeart',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False , icon='logo.ico')

import os

ECELL3_LIBDIR = os.environ['ECELL3_LIBDIR']
EDITOR_PATH = os.environ['MEPATH']

# where to place .glade files for main (non-plugin) windows
GLADEFILE_PATH = EDITOR_PATH + os.sep + 'glade'

VERSION = os.environ['VERSION']
DM_PATH = ECELL3_LIBDIR + os.sep + 'dms'
PACKAGE = 'modeleditor'

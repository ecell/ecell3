import os

PREFIX = os.environ['ECELL3_PREFIX']
EDITOR_PATH = os.environ['MEPATH']

# where to place .glade files for main (non-plugin) windows
GLADEFILE_PATH = EDITOR_PATH + os.sep + 'glade'

VERSION = os.environ['VERSION']
DM_PATH = PREFIX + os.sep + 'lib' + os.sep + 'ecell' + os.sep + VERSION
PACKAGE = 'modeleditor'

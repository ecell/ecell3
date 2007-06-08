import os

SESSIONMONITORPATH = os.environ['SESSIONMONITORPATH']

# where to place .glade files for main (non-plugin) windows
GLADEFILE_PATH = SESSIONMONITORPATH

# where to look at when loading plug-ins
PLUGIN_PATH = [ SESSIONMONITORPATH + os.sep + 'plugins', '.' ]

import os

GUI_OSOGO_PATH = os.environ['OSOGOPATH']
aHomeDir = os.path.expanduser( '~' )
if aHomeDir not in ( '~', '%USERPROFILE%' ):
	GUI_HOMEDIR = aHomeDir
elif os.name == 'nt' and aHomeDir == '%USERPROFILE%':
	GUI_HOMEDIR = os.environ['USERPROFILE']


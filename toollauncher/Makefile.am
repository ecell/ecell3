## Process this file with automake to produce Makefile.in
AUTOMAKE_OPTIONS = foreign

SUBDIRS = misc

toollauncherdir=${pkglibdir}
toollauncher_PYTHON=\
ToolLauncher.glade\
ToolLauncherErrorMessage.glade\
ToolLauncherOkCancel.glade\
ToolLauncherPreferences.glade\
ToolLauncherVersion.glade\
launcher.ini\
config.py\
ParentWindow.py\
ToolLauncher.py\
ToolLauncherErrorMessage.py\
ToolLauncherOkCancel.py\
ToolLauncherPreferences.py\
ToolLauncherVersion.py\
Window.py

EXTRA_DIST=${toollauncher_PYTHON}

bin_SCRIPTS =\
toollauncher

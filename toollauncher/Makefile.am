## Process this file with automake to produce Makefile.in
AUTOMAKE_OPTIONS = foreign

SUBDIRS = misc

toollauncherdir=${pkglibdir}
toollauncher_PYTHON=\
ToolLauncher.glade\
ToolLauncherErrorMessage.glade\
ToolLauncherOkCancel.glade\
ToolLauncherPreferences.glade\
AboutToolLauncher.glade\
config.py\
ParentWindow.py\
ToolLauncher.py\
ToolLauncherErrorMessage.py\
ToolLauncherOkCancel.py\
ToolLauncherPreferences.py\
AboutToolLauncher.py\
Window.py\
toollauncher.png\
toollauncher_large.png

EXTRA_DIST=${toollauncher_PYTHON}

# Windows py2exe setup options, ignore for Unix based systems
PY2EXE_SETUP = ../ecell/pyecs/windows/py2exeSetup
WIN_ICON = toollauncher.ico
PATH_VARIABLES = ../ecell/pyecs/windows/pathVariables

BINS =\
toollauncher

EXE = ${BINS:=$(EXEEXT)}
bin_SCRIPTS = ${EXE}

%.exe : %
	python $(PY2EXE_SETUP) --source=$< --icon=$(WIN_ICON) --merge=$(PATH_VARIABLES) --gui py2exe

clean-local:
	${RM} -rf build *.exe *.zip *.dll *.pyd


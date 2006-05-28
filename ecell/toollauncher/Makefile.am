## Process this file with automake to produce Makefile.in
AUTOMAKE_OPTIONS = foreign

SUBDIRS = misc


ECELL_DIRNAME=@ECELL_DIRNAME@
ECELL_LIB_DIR=${libdir}/${ECELL_DIRNAME}


toollauncherdir=${ECELL_LIB_DIR}/toollauncher
toollauncher_PYTHON=\
ToolLauncher.glade\
MessageWindow.glade\
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
toollauncher32.png\
toollauncher_large.png

EXTRA_DIST=${toollauncher_PYTHON} ${BINS}

# Windows py2exe setup options, ignore for Unix based systems
PY2EXE_SETUP = ../ecell/pyecs/windows/py2exeSetup
WIN_ICON = ecell3-toollauncher.ico
PATH_VARIABLES = ../ecell/pyecs/windows/pathVariables

BINS =\
ecell3-toollauncher

EXE = ${BINS:=$(EXEEXT)}
bin_SCRIPTS = ${EXE}

%.exe : %
	python $(PY2EXE_SETUP) --source=$< --icon=$(WIN_ICON) --merge=$(PATH_VARIABLES) --gui py2exe

clean-local:
	${RM} -rf build *.exe *.zip *.dll *.pyd


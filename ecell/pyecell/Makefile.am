AUTOMAKE_OPTIONS = foreign # 1.5

SUBDIRS = ecell

# Windows py2exe setup options, ignore for Unix based systems
PY2EXE_SETUP = ../pyecs/windows/py2exeSetup
WIN_ICON = ../pyecs/windows/ecell3-console.ico
PATH_VARIABLES = ../pyecs/windows/pathVariables

BINS =\
ecell3-session\
ecell3-session-manager\
ecell3-session-gui\
ecell3-sbml2eml\
ecell3-eml2sbml

EXTRA_DIST = ${BINS}

EXE = ${BINS:=$(EXEEXT)}
bin_SCRIPTS = ${EXE}

%.exe : %
	python $(PY2EXE_SETUP) --source=$< --icon=$(WIN_ICON) --merge=$(PATH_VARIABLES) --console py2exe

clean-local:
	${RM} -rf build *.exe *.zip *.dll *.pyd


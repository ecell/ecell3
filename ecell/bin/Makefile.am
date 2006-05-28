AUTOMAKE_OPTIONS = foreign

# Windows py2exe setup options, ignore for Unix based systems
PY2EXE_SETUP = ../pyecs/windows/py2exeSetup
WIN_ICON = ../pyecs/windows/ecell3-console.ico
PATH_VARIABLES = ../pyecs/windows/pathVariables

BINS =\
ecell3-dmc\
ecell3-em2eml\
ecell3-eml2em

EXTRA_DIST=${BINS}


EXE = ${BINS:=$(EXEEXT)}
bin_SCRIPTS = ${EXE}

%.exe : %
	python $(PY2EXE_SETUP) --source=$< --icon=$(WIN_ICON) --merge=$(PATH_VARIABLES) --console py2exe

clean-local:
	${RM} -rf build *.exe *.zip *.dll *.pyd



AUTOMAKE_OPTIONS=foreign

pkginclude_HEADERS=DynamicModule.hpp ModuleMaker.hpp DMObject.hpp

# Windows py2exe setup options, ignore for Unix based systems
PY2EXE_SETUP = ../ecell/pyecs/windows/py2exeSetup
WIN_ICON = ../ecell/pyecs/windows/ecell3-console.ico
PATH_VARIABLES = ../ecell/pyecs/windows/pathVariables

BINS =\
dmgen\
dmcompile

EXE = ${BINS:=$(EXEEXT)}
bin_SCRIPTS = ${EXE}

%.exe : %
	python $(PY2EXE_SETUP) --source=$< --icon=$(WIN_ICON) --merge=$(PATH_VARIABLES) --console py2exe

clean-local:
	${RM} -rf build *.exe *.zip *.dll *.pyd



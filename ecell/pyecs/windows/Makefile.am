AUTOMAKE_OPTIONS = foreign

winrootdir = ${prefix}
winroot_SCRIPTS = \
ecell3-inno.iss\
../../../osogo/ecell.ico

PY2EXE_SETUP = py2exeSetup
WIN_ICON = ecell3-console.ico

BINS = \
ecell3-console

BINS_SHARED = \
shared.zip

EXE = ${BINS:=$(EXEEXT)}
bin_SCRIPTS = ${EXE} ${BINS_SHARED}

%.exe : %
	python $(PY2EXE_SETUP) --source=$< --icon=$(WIN_ICON) --gui py2exe

clean-local:
	${RM} -rf build *.exe *.zip *.dll *.pyd

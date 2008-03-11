abs_top_builddir = @abs_top_builddir@

LTDL_SHLIB_EXT := $(shell echo @LTDL_SHLIB_EXT@)

SUFFIXES = .cpp $(LTDL_SHLIB_EXT)

SHELL = @SHELL@

AM_CXXFLAGS = @CXXFLAGS@ -I/usr/include/libxml++-1.0 -I/usr/include/moleculizer-1.0/moleculizer -I../../../dmtool/
AM_LDFLAGS = -lmoleculizer_cpx -lmoleculizer_ftr -lmoleculizer_nmr -lmoleculizer_utl -lmoleculizer_dimer -lmoleculizer_mol -lmoleculizer_plex -lmoleculizer_fnd	 -lmoleculizer_mzr -lmoleculizer_stoch

if OS_WIN32
  ECELL3_DMC = ../../bin/ecell3-dmc.exe
else
  ECELL3_DMC = ../../pyecs/ecell3-python ../../bin/ecell3-dmc
endif

ECELL_DIRNAME = @ECELL_DIRNAME@

DM_SO_DIR = ${libdir}/${ECELL_DIRNAME}/dms
DM_SRC_DIR =  ${datadir}/${ECELL_DIRNAME}/dms

PYTHON_INCLUDES = @PYTHON_INCLUDES@
PYTHON_LIBS = @PYTHON_LIBS@


DMS = \
CallbackVariable.cpp \
MoleculizerProcess.cpp

DMH = CallbackVariable.hpp \
MoleculizerProcess.hpp \
RegisterClass.hpp

DESCS = 

EXTRA_DIST = ${DMS} ${DMH} ${DESCS}
dmdir = ${DM_SRC_DIR}
dm_DATA = ${DMS} ${DMH}

SOS = ${DMS:.cpp=$(LTDL_SHLIB_EXT)}

dmsodir = ${DM_SO_DIR}
dmso_DATA = ${SOS} ${DESCS}

.cpp$(LTDL_SHLIB_EXT):
	${ECELL3_DMC} --verbose --dmcompile="../../../dmtool" --no-stdinclude --no-stdlibdir --ldflags="-L$(abs_top_builddir)/tmp$(libdir) ${AM_LDFLAGS}" --cxxflags='${AM_CXXFLAGS} ${CXXFLAGS} -I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir} -I${top_srcdir}/../libltdl  ${PYTHON_INCLUDES} ' $< 

clean-local:
	${RM} -rf ${SOS}
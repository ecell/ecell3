abs_top_builddir = @abs_top_builddir@

LTDL_SHLIB_EXT := $(shell echo @LTDL_SHLIB_EXT@)

SUFFIXES = .cpp $(LTDL_SHLIB_EXT)

SHELL = @SHELL@

AM_CXXFLAGS = @CXXFLAGS@

ECELL3_DMC = ../pyecs/ecell3-python ../bin/ecell3-dmc

ECELL_DIRNAME = @ECELL_DIRNAME@

DM_SO_DIR = ${libdir}/${ECELL_DIRNAME}/dms
DM_SRC_DIR =  ${datadir}/${ECELL_DIRNAME}/dms

PYTHON_INCLUDES = @PYTHON_INCLUDES@
PYTHON_LIBS = @PYTHON_LIBS@

DMS = MassActionFluxProcess.cpp \
MichaelisUniUniFluxProcess.cpp \
ConstantFluxProcess.cpp \
DecayFluxProcess.cpp \
SSystemProcess.cpp \
GillespieProcess.cpp \
TauLeapProcess.cpp \
TauLeapStepper.cpp \
FixedODE1Stepper.cpp \
FixedDAE1Stepper.cpp \
ODE23Stepper.cpp \
ODE45Stepper.cpp \
DAEStepper.cpp \
ODEStepper.cpp \
ESSYNSStepper.cpp \
GMAProcess.cpp \
SSystemProcess.cpp \
FluxDistributionStepper.cpp \
QuasiDynamicFluxProcess.cpp \
PythonFluxProcess.cpp \
PythonProcess.cpp \
ExpressionFluxProcess.cpp \
ExpressionAlgebraicProcess.cpp \
ExpressionAssignmentProcess.cpp

DMH =  \
	ESSYNSProcess.hpp \
	PythonFluxProcess.hpp \
	PythonProcess.hpp \
	PythonProcessBase.hpp \
	ExpressionProcessBase.hpp \
	ExpressionCompiler.hpp \
	FixedODE1Stepper.hpp \
	FixedDAE1Stepper.hpp \
	GillespieProcess.hpp \
	ESSYNSStepper.hpp \
	ODE23Stepper.hpp \
	ODEStepper.hpp \
	DAEStepper.hpp \
	FluxDistributionStepper.hpp \
	QuasiDynamicFluxProcess.hpp \
	TauLeapProcess.hpp \
	TauLeapStepper.hpp \
	ODE45Stepper.hpp

DESCS =  \
	ConstantFluxProcess.desc \
	DAEStepper.desc \
	DecayFluxProcess.desc \
	ESSYNSStepper.desc \
	ExpressionAlgebraicProcess.desc \
	ExpressionFluxProcess.desc \
	FixedDAE1Stepper.desc \
	FixedODE1Stepper.desc \
	FluxDistributionStepper.desc \
	QuasiDynamicFluxProcess.desc \
	GillespieProcess.desc \
	GMAProcess.desc \
	MassActionFluxProcess.desc \
	MichaelisUniUniFluxProcess.desc \
	ODEStepper.desc \
	ODE23Stepper.desc \
	ODE45Stepper.desc \
	PythonFluxProcess.desc \
	PythonProcess.desc \
	SSystemProcess.desc \
	System.desc \
	Variable.desc

SOS = \
	MassActionFluxProcess$(LTDL_SHLIB_EXT) \
	MichaelisUniUniFluxProcess$(LTDL_SHLIB_EXT) \
	ConstantFluxProcess$(LTDL_SHLIB_EXT) \
	DecayFluxProcess$(LTDL_SHLIB_EXT) \
	SSystemProcess$(LTDL_SHLIB_EXT) \
	GillespieProcess$(LTDL_SHLIB_EXT) \
	TauLeapProcess$(LTDL_SHLIB_EXT) \
	TauLeapStepper$(LTDL_SHLIB_EXT) \
	FixedODE1Stepper$(LTDL_SHLIB_EXT) \
	FixedDAE1Stepper$(LTDL_SHLIB_EXT) \
	ODE23Stepper$(LTDL_SHLIB_EXT) \
	ODE45Stepper$(LTDL_SHLIB_EXT) \
	DAEStepper$(LTDL_SHLIB_EXT) \
	ODEStepper$(LTDL_SHLIB_EXT) \
	ESSYNSStepper$(LTDL_SHLIB_EXT) \
	GMAProcess$(LTDL_SHLIB_EXT) \
	SSystemProcess$(LTDL_SHLIB_EXT) \
	FluxDistributionStepper$(LTDL_SHLIB_EXT) \
	QuasiDynamicFluxProcess$(LTDL_SHLIB_EXT) \
	ExpressionFluxProcess$(LTDL_SHLIB_EXT) \
	ExpressionAlgebraicProcess$(LTDL_SHLIB_EXT) \
	ExpressionAssignmentProcess$(LTDL_SHLIB_EXT)

if HAVE_PYTHON_LIB
SOS += \
	PythonFluxProcess$(LTDL_SHLIB_EXT) \
	PythonProcess$(LTDL_SHLIB_EXT)
endif

EXTRA_DIST = ${DMS} ${DMH} ${DESCS}
dmdir = ${DM_SRC_DIR}
dm_DATA = ${DMS} ${DMH}

dmsodir = ${DM_SO_DIR}
dmso_DATA = ${SOS} ${DESCS}

PythonFluxProcess$(LTDL_SHLIB_EXT) : PythonFluxProcess.cpp
	${ECELL3_DMC} --verbose --dmcompile="../../dmtool" --no-stdinclude --no-stdlibdir --ldflags="-L$(abs_top_builddir)/tmp$(libdir) $(PYTHON_LIBS)" --cxxflags='${CPPFLAGS} ${CXXFLAGS} -I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir} -I${top_srcdir}/../libltdl  ${PYTHON_INCLUDES} ' $< 

PythonProcess$(LTDL_SHLIB_EXT) : PythonProcess.cpp
	${ECELL3_DMC} --verbose --dmcompile="../../dmtool" --no-stdinclude --no-stdlibdir --ldflags="-L$(abs_top_builddir)/tmp$(libdir) $(PYTHON_LIBS)" --cxxflags='${CPPFLAGS} ${CXXFLAGS} -I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir} -I${top_srcdir}/../libltdl  ${PYTHON_INCLUDES} ' $< 

.cpp$(LTDL_SHLIB_EXT):
	${ECELL3_DMC} --verbose --dmcompile="../../dmtool" --no-stdinclude --no-stdlibdir --ldflags="-L$(abs_top_builddir)/tmp$(libdir) " --cxxflags='${CPPFLAGS} ${CXXFLAGS} -I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir} -I${top_srcdir}/../libltdl  ${PYTHON_INCLUDES} ' $< 

clean-local:
	${RM} -rf ${SOS}



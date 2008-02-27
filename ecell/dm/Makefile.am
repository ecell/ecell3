ECELL_DIRNAME = @ECELL_DIRNAME@

AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/.. -I$(top_srcdir)/../libltdl
AM_LDFLAGS = -module -avoid-version

pkglibdir = $(libdir)/$(ECELL_DIRNAME)/modules
pkglib_LTLIBRARIES =  \
	ConstantFluxProcess.la \
	DAEStepper.la \
	DecayFluxProcess.la \
	ESSYNSStepper.la \
	ExpressionAlgebraicProcess.la \
	ExpressionAssignmentProcess.la \
	ExpressionFluxProcess.la \
	FixedDAE1Stepper.la \
	FixedODE1Stepper.la \
	FluxDistributionStepper.la \
	GMAProcess.la \
	GillespieProcess.la \
	MassActionFluxProcess.la \
	MichaelisUniUniFluxProcess.la \
	ODE23Stepper.la \
	ODE45Stepper.la \
	ODEStepper.la \
	PythonFluxProcess.la \
	PythonProcess.la \
	QuasiDynamicFluxProcess.la \
	SSystemProcess.la \
	TauLeapProcess.la \
	TauLeapStepper.la

ConstantFluxProcess_la_SOURCES =  \
	ConstantFluxProcess.cpp
ConstantFluxProcess_ladir = $(pkglibdir)
ConstantFluxProcess_la_DATA = ConstantFluxProcess.desc

DAEStepper_la_SOURCES = \
	DAEStepper.cpp
DAEStepper_ladir = $(pkglibdir)
DAEStepper_la_DATA = DAEStepper.desc

DecayFluxProcess_la_SOURCES =  \
	DecayFluxProcess.cpp
DecayFluxProcess_ladir = $(pkglibdir)
DecayFluxProcess_la_DATA = DecayFluxProcess.desc

ESSYNSStepper_la_SOURCES = \
	ESSYNSStepper.cpp
ESSYNSStepper_ladir = $(pkglibdir)
ESSYNSStepper_la_DATA = ESSYNSStepper.desc

ExpressionAlgebraicProcess_la_SOURCES = \
	ExpressionAlgebraicProcess.cpp \
	ExpressionProcessBase.hpp \
	ExpressionCompiler.hpp \
	ExpressionCompiler.cpp \
	VirtualMachine.hpp \
	VirtualMachine.cpp
ExpressionAlgebraicProcess_ladir = $(pkglibdir)
ExpressionAlgebraicProcess_la_DATA = ExpressionAlgebraicProcess.desc

ExpressionAssignmentProcess_la_SOURCES = \
	ExpressionAssignmentProcess.cpp \
	ExpressionProcessBase.hpp \
	ExpressionCompiler.hpp \
	ExpressionCompiler.cpp \
	VirtualMachine.hpp \
	VirtualMachine.cpp
#ExpressionAssignmentProcess_ladir = $(pkglibdir)
#ExpressionAssignmentProcess_la_DATA = ExpressionAssignmentProcess.desc

ExpressionFluxProcess_la_SOURCES =  \
	ExpressionFluxProcess.cpp \
	ExpressionProcessBase.hpp \
	ExpressionCompiler.hpp \
	ExpressionCompiler.cpp \
	VirtualMachine.hpp \
	VirtualMachine.cpp
ExpressionFluxProcess_ladir = $(pkglibdir)
ExpressionFluxProcess_la_DATA = ExpressionFluxProcess.desc

FixedDAE1Stepper_la_SOURCES = \
	FixedDAE1Stepper.cpp
FixedDAE1Stepper_ladir = $(pkglibdir)
FixedDAE1Stepper_la_DATA = FixedDAE1Stepper.desc

FixedODE1Stepper_la_SOURCES = \
	FixedODE1Stepper.cpp
FixedODE1Stepper_ladir = $(pkglibdir)
FixedODE1Stepper_la_DATA = FixedODE1Stepper.desc

FluxDistributionStepper_la_SOURCES = \
	FluxDistributionStepper.cpp
FluxDistributionStepper_ladir = $(pkglibdir)
FluxDistributionStepper_la_DATA = FluxDistributionStepper.desc

GMAProcess_la_SOURCES = \
	GMAProcess.cpp
GMAProcess_ladir = $(pkglibdir)
GMAProcess_la_DATA = GMAProcess.desc

GillespieProcess_la_SOURCES = \
	GillespieProcess.cpp \
	GillespieProcess.hpp
GillespieProcess_ladir = $(pkglibdir)
GillespieProcess_la_DATA = GillespieProcess.desc

MassActionFluxProcess_la_SOURCES = \
	MassActionFluxProcess.cpp
MassActionFluxProcess_ladir = $(pkglibdir)
MassActionFluxProcess_la_DATA = MassActionFluxProcess.desc

MichaelisUniUniFluxProcess_la_SOURCES = \
	MichaelisUniUniFluxProcess.cpp
MichaelisUniUniFluxProcess_ladir = $(pkglibdir)
MichaelisUniUniFluxProcess_la_DATA = MichaelisUniUniFluxProcess.desc

ODE23Stepper_la_SOURCES = \
	ODE23Stepper.cpp
ODE23Stepper_ladir = $(pkglibdir)
ODE23Stepper_la_DATA = ODE23Stepper.desc

ODE45Stepper_la_SOURCES = \
	ODE45Stepper.cpp
ODE45Stepper_ladir = $(pkglibdir)
ODE45Stepper_la_DATA = ODE45Stepper.desc

ODEStepper_la_SOURCES = \
	ODEStepper.cpp
ODEStepper_ladir = $(pkglibdir)
ODEStepper_la_DATA = ODEStepper.desc

if HAVE_PYTHON_LIB
PythonFluxProcess_la_SOURCES = 
	PythonFluxProcess.cpp \
	PythonProcessBase.hpp
PythonFluxProcess_la_CPPFLAGS = @PYTHON_INCLUDES@ $(AM_CPPFLAGS)
PythonFluxProcess_la_LIBS = @PYTHON_LIBS@ $(AM_LIBS)
PythonFluxProcess_ladir = $(pkglibdir)
PythonFluxProcess_la_DATA = PythonFluxProcess.desc

PythonProcess_la_SOURCES = \
	PythonProcess.cpp \
	PythonProcessBase.hpp
PythonProcess_la_CPPFLAGS = @PYTHON_INCLUDES@ $(AM_CPPFLAGS)
PythonProcess_la_LIBS = @PYTHON_LIBS@ $(AM_LIBS)
PythonProcess_ladir = $(pkglibdir)
PythonProcess_la_DATA = PythonProcess.desc
endif

QuasiDynamicFluxProcess_la_SOURCES = \
	QuasiDynamicFluxProcess.cpp \
	QuasiDynamicFluxProcess.hpp
QuasiDynamicFluxProcess_ladir = $(pkglibdir)
QuasiDynamicFluxProcess_la_DATA = QuasiDynamicFluxProcess.desc

SSystemProcess_la_SOURCES = \
	SSystemProcess.cpp \
	ESSYNSProcess.hpp
SSystemProcess_ladir = $(pkglibdir)
SSystemProcess_la_DATA = SSystemProcess.desc

TauLeapProcess_la_SOURCES = \
	TauLeapProcess.cpp
#TauLeapProcess_ladir = $(pkglibdir)
#TauLeapProcess_la_DATA = TauLeapProcess.desc

TauLeapStepper_la_SOURCES = \
	TauLeapStepper.cpp \
	GillespieProcess.hpp
#TauLeapStepper_ladir = $(pkglibdir)
#TauLeapStepper_la_DATA = TauLeapStepper.desc

BUILTIN_DM_DESCS =  \
	System.desc \
	Variable.desc

builtindm_descdir = $(pkglibdir)
builtindm_desc_DATA = ${BUILTIN_DM_DESCS}

EXTRA_DIST = ${BUILTIN_DM_DESCS}


SUFFIXES = .cpp @LTDL_SHLIB_EXT@

SHELL = @SHELL@

LTDL_SHLIB_EXT = @LTDL_SHLIB_EXT@

CXXFLAGS = @CXXFLAGS@

if OS_WIN32
  ECELL3_DMC = ../../bin/ecell3-dmc.exe
else
  ECELL3_DMC = ../pyecs/ecell3-python ../../bin/ecell3-dmc
endif

DM_SO_DIR = ${prefix}/lib/${PACKAGE}/${VERSION}
DM_SRC_DIR =  ${prefix}/share/${PACKAGE}/${VERSION}/DM

BOOST_INCLUDE=@BOOST_INCLUDE@
BOOST_LIB=@BOOST_LIB@
PYTHON_INCLUDES=@PYTHON_INCLUDES@

DMS = MassActionFluxProcess.cpp\
CatalyzedMassActionFluxProcess.cpp\
MichaelisUniUniFluxProcess.cpp\
RandomBiUniFluxProcess.cpp\
ConstantFluxProcess.cpp\
RandomUniBiFluxProcess.cpp\
RandomBiBiFluxProcess.cpp\
RapidEquilibriumProcess.cpp\
BisectionRapidEquilibriumProcess.cpp\
DecayFluxProcess.cpp\
IsoUniUniFluxProcess.cpp\
OrderedBiBiFluxProcess.cpp\
OrderedBiUniFluxProcess.cpp\
OrderedUniBiFluxProcess.cpp\
SSystemProcess.cpp\
GillespieProcess.cpp\
TauLeapProcess.cpp\
TauLeapStepper.cpp\
NRStepper.cpp\
FixedODE1Stepper.cpp\
FixedDAE1Stepper.cpp\
ODE23Stepper.cpp\
ODE45Stepper.cpp\
DAEStepper.cpp\
ODEStepper.cpp\
ESSYNSStepper.cpp\
GMAProcess.cpp\
SSystemProcess.cpp\
FluxDistributionStepper.cpp\
FluxDistributionProcess.cpp\
QuasiDynamicFluxProcess.cpp\
PythonFluxProcess.cpp\
PythonProcess.cpp\
ExpressionFluxProcess.cpp\
ExpressionAlgebraicProcess.cpp\
ExpressionAssignmentProcess.cpp

DMH = \
ESSYNSProcess.hpp\
PythonFluxProcess.hpp\
PythonProcess.hpp\
PythonProcessBase.hpp\
ExpressionProcessBase.hpp\
ExpressionCompiler.hpp\
FixedODE1Stepper.hpp\
FixedDAE1Stepper.hpp\
GillespieProcess.hpp\
NRStepper.hpp\
ESSYNSStepper.hpp\
ODE23Stepper.hpp\
ODEStepper.hpp\
DAEStepper.hpp\
FluxDistributionStepper.hpp\
QuasiDynamicFluxProcess.hpp\
TauLeapProcess.hpp\
TauLeapStepper.hpp\
ODE45Stepper.hpp

DESCS = \
BisectionRapidEquilibriumProcess.desc\
CashKarp45Stepper.desc\
CatalyzedMassActionFluxProcess.desc\
ConstantFluxProcess.desc\
DAEStepper.desc\
DecayFluxProcess.desc\
ESSYNSStepper.desc\
ExpressionAlgebraicProcess.desc\
ExpressionFluxProcess.desc\
FixedDAE1Stepper.desc\
FixedODE1Stepper.desc\
FluxDistributionProcess.desc\
FluxDistributionStepper.desc\
GillespieProcess.desc\
GMAProcess.desc\
IsoUniUniFluxProcess.desc\
MassActionFluxProcess.desc\
MichaelisUniUniFluxProcess.desc\
NRStepper.desc\
ODEStepper.desc\
ODE23Stepper.desc\
ODE45Stepper.desc\
OrderedBiBiFluxProcess.desc\
OrderedBiUniFluxProcess.desc\
OrderedUniBiFluxProcess.desc\
PythonFluxProcess.desc\
PythonProcess.desc\
QuasiDynamicFluxProcess.desc\
RandomBiBiFluxProcess.desc\
RandomBiUniFluxProcess.desc\
RandomUniBiFluxProcess.desc\
RapidEquilibriumProcess.desc\
SSystemProcess.desc\
System.desc\
CompartmentSystem.desc\
Variable.desc

EXTRA_DIST = ${DMS} ${DMH} ${DESCS}
dmdir = ${DM_SRC_DIR}
dm_DATA = ${DMS} ${DMH}

SOS = ${DMS:.cpp=@LTDL_SHLIB_EXT@}

dmsodir = ${DM_SO_DIR}
dmso_DATA = ${SOS} ${DESCS}

%$(LTDL_SHLIB_EXT) : %.cpp
	${ECELL3_DMC} --verbose --dmcompile="../../dmtool" --no-stdinclude --no-stdlibdir --ldflags="-L${top_srcdir}/libecs ${BOOST_LIB}" --cxxflags='${CXXFLAGS} -I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir} ${PYTHON_INCLUDES} ${BOOST_INCLUDE}' $< 

clean-local:
	${RM} -rf ${SOS}



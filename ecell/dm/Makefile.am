
SUFFIXES = .cpp @LTDL_SHLIB_EXT@

SHELL = @SHELL@

ECELL3_PYTHON = ../pyecs/ecell3-python
ECELL3_DMC = ../../bin/ecell3-dmc
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
NRStepper.cpp\
FixedODE1Stepper.cpp\
FixedDAE1Stepper.cpp\
ODE23Stepper.cpp\
CashKarp45Stepper.cpp\
ODE45Stepper.cpp\
ESSYNSStepper.cpp\
GMAProcess.cpp\
SSystemProcess.cpp\
FluxDistributionStepper.cpp\
FluxDistributionProcess.cpp\
QuasiDynamicFluxProcess.cpp\
PythonFluxProcess.cpp\
PythonProcess.cpp\
ExpressionFluxProcess.cpp

DMH = \
ESSYNSProcess.hpp\
CashKarp45Stepper.hpp\
PythonFluxProcess.hpp\
PythonProcess.hpp\
PythonProcessBase.hpp\
ExpressionProcessBase.hpp\
FixedODE1Stepper.hpp\
FixedDAE1Stepper.hpp\
GillespieProcess.hpp\
NRStepper.hpp\
ESSYNSStepper.hpp\
ODE23Stepper.hpp\
FluxDistributionStepper.hpp\
QuasiDynamicFluxProcess.hpp\
ODE45Stepper.hpp


EXTRA_DIST = ${DMS} ${DMH}
dmdir = ${DM_SRC_DIR}
dm_DATA = ${DMS} ${DMH}

SOS = ${DMS:.cpp=@LTDL_SHLIB_EXT@}

dmsodir = ${DM_SO_DIR}
dmso_DATA = ${SOS}

%@LTDL_SHLIB_EXT@ : %.cpp
	${ECELL3_PYTHON} ${ECELL3_DMC} -v --dmcompile=${top_srcdir}/../dmtool --no-stdinclude --no-stdlibdir --ldflags="-L${top_srcdir}/libecs ${BOOST_LIB}" --cxxflags='-I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir} ${PYTHON_INCLUDES} ${BOOST_INCLUDE}' $< 

clean-local:
	${RM} -rf ${dmso_DATA}


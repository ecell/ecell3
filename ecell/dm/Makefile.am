SUFFIXES = .cpp .so

SHELL = @SHELL@

ECELL3_DMC = ../../bin/ecell3-dmc
DM_SO_DIR = ${prefix}/lib/${PACKAGE}/${VERSION}
DM_SRC_DIR =  ${prefix}/share/${PACKAGE}/${VERSION}/DM

BOOST_DIR=@BOOST_DIR@
PYTHON_INCLUDES=@PYTHON_INCLUDES@

DMS = MassActionProcess.cpp\
CatalyzedMassActionFluxProcess.cpp\
MichaelisUniUniProcess.cpp\
MichaelisUniUniReversibleProcess.cpp\
RandomBiUniFluxProcess.cpp\
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
FixedRungeKutta4Stepper.cpp\
ODE23Stepper.cpp\
CashKarp45Stepper.cpp\
ODE45Stepper.cpp\
PythonFluxProcess.cpp\
PythonProcess.cpp

DMH = CashKarp45Stepper.hpp\
PythonFluxProcess.hpp\
PythonProcess.hpp\
PythonProcessBase.hpp\
FixedODE1Stepper.hpp\
FixedRungeKutta4Stepper.hpp\
GillespieProcess.hpp\
NRStepper.hpp\
ODE23Stepper.hpp\
ODE45Stepper.hpp


EXTRA_DIST = ${DMS} ${DMH}
dmdir = ${DM_SRC_DIR}
dm_DATA = ${DMS} ${DMH}

SOS = ${DMS:.cpp=.so}

dmsodir = ${DM_SO_DIR}
dmso_DATA = ${SOS}

%.so : %.cpp
	${SHELL} ${ECELL3_DMC} -v --dmcompile=${top_srcdir}/../dmtool --no-stdinclude --no-stdlibdir --ldflags=-L${top_srcdir}/libecs --cxxflags='-I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir} ${PYTHON_INCLUDES} -I${BOOST_DIR}' $< 

clean-local:
	${RM} -rf ${dmso_DATA}


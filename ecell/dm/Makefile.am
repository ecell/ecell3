SUFFIXES = .cpp .so

SHELL = @SHELL@

ECELL3_DMC = ../../bin/ecell3-dmc
DM_SO_DIR = ${prefix}/lib/${PACKAGE}/${VERSION}
DM_SRC_DIR =  ${prefix}/share/${PACKAGE}/${VERSION}/DM

DMS = MassActionProcess.cpp\
MichaelisUniUniProcess.cpp\
RapidEquilibriumProcess.cpp\
MichaelisUniUniReversibleProcess.cpp\
CatalyzedMassActionFluxProcess.cpp\
DecayFluxProcess.cpp\
IsoUniUniFluxProcess.cpp\
SSystemProcess.cpp\
NRProcess.cpp\
NRStepper.cpp\
FixedODE1Stepper.cpp\
FixedRungeKutta4Stepper.cpp\
ODE23Stepper.cpp\
CashKarp45Stepper.cpp\
ODE45Stepper.cpp

EXTRA_DIST = ${DMS}
dmdir = ${DM_SRC_DIR}
dm_DATA = ${DMS}

SOS = ${DMS:.cpp=.so}

dmsodir = ${DM_SO_DIR}
dmso_DATA = ${SOS}

%.so : %.cpp
	${SHELL} ${ECELL3_DMC} -v --dmcompile=${top_srcdir}/../dmtool --no-stdinclude --no-stdlibdir --ldflags=-L${top_srcdir}/libecs $< -I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir}

clean-local:
	${RM} -rf ${dmso_DATA}


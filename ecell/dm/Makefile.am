SUFFIXES = .cpp .so

SHELL = @SHELL@

ECELL3_DMC = ../../bin/ecell3-dmc
DM_SO_DIR = ${prefix}/lib/${PACKAGE}/${VERSION}
DM_DIR =  ${prefix}/share/${PACKAGE}/${VERSION}/DM

DMS = MassActionProcess.cpp\
MichaelisUniUniProcess.cpp\
RapidEquilibriumProcess.cpp\
SSystemProcess.cpp

EXTRA_DIST = ${DMS}
dmdir = ${DM_DIR}
dm_DATA = ${DMS}

SOS = ${DMS:.cpp=.so}

dmsodir = ${DM_SO_DIR}
dmso_DATA = ${SOS}

%.so : %.cpp
	${SHELL} ${ECELL3_DMC} $< -I${top_srcdir}/libecs -I${top_srcdir} 

clean-local:
	${RM} -rf ${DM_DATA}


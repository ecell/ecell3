
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE@
KORANDOM_INCLUDE=-I@KORANDOM_DIR@

INCLUDES = -I..  -I../libecs ${KORANDOM_INCLUDE} ${DMTOOL_INCLUDE}
noinst_LTLIBRARIES = libemc.la

CXXFLAGS= @CXXFLAGS@

libemc_la_LDFLAGS=-static
libemc_la_SOURCES =\
Simulator.cpp\
SimulatorImplementation.cpp\
LocalSimulatorImplementation.cpp

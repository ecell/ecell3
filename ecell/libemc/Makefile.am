
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE@
KORANDOM_INCLUDE=-I@KORANDOM_INCLUDE@

INCLUDES = -I..  -I../include ${KORANDOM_INCLUDE} ${DMTOOL_INCLUDE}
noinst_LIBRARIES = libemc.a

CXXFLAGS=-D_GNU_SOURCE -D_REENTRANT -D_THREAD_SAFE

libemc_a_SOURCES =\
Simulator.cpp\
SimulatorImplementation.cpp\
LocalSimulatorImplementation.cpp

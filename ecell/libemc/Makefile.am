
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
KORANDOM_INCLUDE=-I@KORANDOM_DIR@
INCLTDL=@INCLTDL@
libdir=@libdir@

INCLUDES = -I..  -I../libecs ${KORANDOM_INCLUDE} ${DMTOOL_INCLUDE} ${INCLTDL}
noinst_LTLIBRARIES = libemc.la

CXXFLAGS= @CXXFLAGS@

libemc_la_LDFLAGS= -rpath ${libdir}
libemc_la_LIBADD= -lstdc++
libemc_la_SOURCES=\
Simulator.cpp\
SimulatorImplementation.cpp\
LocalSimulatorImplementation.cpp

libemcdir=${pkginclude}/libemc
libemc_HEADERS=\
LocalSimulatorImplementation.hpp\
Simulator.hpp\
SimulatorImplementation.hpp\
SimulatorMaker.hpp

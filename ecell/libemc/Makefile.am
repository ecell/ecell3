
ECELLDIR=..

DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
KORANDOM_INCLUDE=-I@KORANDOM_DIR@
INCLTDL=@INCLTDL@
libdir=@libdir@

INCLUDES = -I..  ${DMTOOL_INCLUDE} ${INCLTDL}
noinst_LTLIBRARIES = libemc.la

CXXFLAGS= @CXXFLAGS@ #-D_REENTRANT

libemc_la_LDFLAGS= -rpath ${libdir}

#FIXME: specifing .libs is ugly
libemc_la_LIBADD= -lstdc++ -L$(ECELLDIR)/libecs/.libs/ -lecs
libemc_la_DEPENDENCIES=$(ECELLDIR)/libecs/libecs.la
libemc_la_SOURCES=\
Simulator.cpp\
SimulatorImplementation.cpp\
LocalSimulatorImplementation.cpp\
LocalLoggerImplementation.cpp\
LoggerImplementation.cpp\
EmcLogger.cpp\
EmcDataPoint.cpp

libemcdir=${pkgincludedir}/libemc
libemc_HEADERS=\
LocalSimulatorImplementation.hpp\
Simulator.hpp\
SimulatorImplementation.hpp\
SimulatorMaker.hpp\
EmcDataPoint.hpp\
EmcLogger.hpp\
LocalLoggerImplementation.hpp\
LoggerImplementation.hpp\
libemc.hpp


ECELLDIR=..

DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
KORANDOM_INCLUDE=-I@KORANDOM_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
libdir=@libdir@

KORANDOM_DIR=@KORANDOM_DIR@
LIBKORANDOM=$(KORANDOM_DIR)/korandom/libkorandom.la
LIBECS=$(ECELLDIR)/libecs/libecs.la

INCLUDES = -I.. ${KORANDOM_INCLUDE} ${DMTOOL_INCLUDE} ${INCLTDL}
lib_LTLIBRARIES = libemc.la

CXXFLAGS= @CXXFLAGS@ #-D_REENTRANT

libemc_la_LDFLAGS=

#FIXME: specifing .libs is ugly
libemc_la_LIBADD= $(LIBECS) $(LIBKORANDOM) -lstdc++ 
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

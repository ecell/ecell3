libdir=@libdir@

ECELLDIR=..

DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
BOOST_DIR=@BOOST_DIR@

LIBECS=$(ECELLDIR)/libecs/libecs.la

INCLUDES = -I.. -I${BOOST_DIR} ${DMTOOL_INCLUDE} ${INCLTDL}
lib_LTLIBRARIES = libemc.la

CXXFLAGS= @CXXFLAGS@ #-D_REENTRANT

libemc_la_LDFLAGS= -version-info 1:0:0

libemc_la_LIBADD= $(LIBECS) -lstdc++ 
libemc_la_DEPENDENCIES=$(ECELLDIR)/libecs/libecs.la
libemc_la_SOURCES=\
libemc.cpp\
Simulator.cpp\
SimulatorImplementation.cpp\
LocalSimulatorImplementation.cpp

libemcdir=${pkgincludedir}/libemc
libemc_HEADERS=\
libemc.hpp\
LocalSimulatorImplementation.hpp\
Simulator.hpp\
SimulatorImplementation.hpp\
SimulatorMaker.hpp

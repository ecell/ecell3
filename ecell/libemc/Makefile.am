libdir=@libdir@

if OS_WIN32
  no_undefined = -no-undefined -Wl,--enable-runtime-pseudo-reloc
endif

ECELLDIR=..

DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
BOOST_DIR=@BOOST_INCLUDE@

LIBECS=$(ECELLDIR)/libecs/libecs.la

INCLUDES = -I.. ${BOOST_INCLUDE} ${DMTOOL_INCLUDE} ${INCLTDL}
lib_LTLIBRARIES = libemc.la

CXXFLAGS= @CXXFLAGS@ #-D_REENTRANT

libemc_la_LDFLAGS= -version-info 2:0:0 $(no_undefined)

libemc_la_LIBADD=$(LIBECS) -lstdc++ 
libemc_la_DEPENDENCIES=$(LIBECS)
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

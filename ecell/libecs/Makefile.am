
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
KORANDOM_INCLUDE=-I@KORANDOM_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
libdir=@libdir@

INCLUDES = -I..  -I../.. -I../include ${DMTOOL_INCLUDE} ${INCLTDL}
noinst_LTLIBRARIES = libecs.la

CXXFLAGS= @CXXFLAGS@ # -D_REENTRANT

libecs_la_LDFLAGS= -rpath ${libdir}
libecs_la_LIBADD= ${LIBLTDL} -lstdc++
libecs_la_SOURCES=\
Util.cpp\
UniversalVariable.cpp\
StringList.cpp\
Message.cpp\
MessageInterface.cpp\
AccumulatorMaker.cpp\
FQPI.cpp\
ReactorMaker.cpp\
Substance.cpp\
Accumulators.cpp\
Integrators.cpp\
RootSystem.cpp\
SubstanceMaker.cpp\
Stepper.cpp\
PrimitiveType.cpp\
System.cpp\
Entity.cpp\
Reactor.cpp\
StepperMaker.cpp\
SystemMaker.cpp\
DataPoint.cpp\
StlDataPointVector.cpp\
Logger.cpp\
LoggerBroker.cpp

libecsdir=${pkginclude}/libecs
libecs_HEADERS=\
libecs.hpp\
Defs.hpp\
Util.hpp\
StringList.hpp\
Message.hpp\
MessageInterface.hpp\
AccumulatorMaker.hpp\
FQPI.hpp\
ReactorMaker.hpp\
Substance.hpp\
Accumulators.hpp\
Integrators.hpp\
RootSystem.hpp\
SubstanceMaker.hpp\
Stepper.hpp\
PrimitiveType.hpp\
System.hpp\
Entity.hpp\
Reactor.hpp\
StepperMaker.hpp\
SystemMaker.hpp\
DataPoint.hpp\
StlDataPointVector.hpp\
Logger.hpp\
LoggerBroker.hpp

SUBDIRS = dmtemplates
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
KORANDOM_DIR=@KORANDOM_DIR@
KORANDOM_INCLUDE=-I${KORANDOM_DIR}
LIBKORANDOM=${KORANDOM_DIR}/korandom/libkorandom.la
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
libdir=@libdir@

INCLUDES = -I..  -I../.. -I../include ${KORANDOM_INCLUDE} ${DMTOOL_INCLUDE} ${INCLTDL}
lib_LTLIBRARIES = libecs.la

CXXFLAGS= @CXXFLAGS@ # -D_REENTRANT

libecs_la_LDFLAGS= #-rpath ${libdir}
libecs_la_LIBADD= ${LIBLTDL} -L${KORANDOM_DIR}/korandom/ -lkorandom -lstdc++
libecs_la_SOURCES=\
Util.cpp\
UVariable.cpp\
FullID.cpp\
Message.cpp\
PropertySlot.cpp\
PropertyInterface.cpp\
AccumulatorMaker.cpp\
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
DataPointStlVector.cpp\
Logger.cpp\
LoggerBroker.cpp

libecsdir=${pkgincludedir}/libecs
libecs_HEADERS=\
libecs.hpp\
Defs.hpp\
UVariable.hpp\
FullID.hpp\
Message.hpp\
PropertySlot.hpp\
PropertyInterface.hpp\
AccumulatorMaker.hpp\
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
DataPointStlVector.hpp\
Logger.hpp\
LoggerBroker.hpp\
Exceptions.hpp\
Util.hpp\
CoreLinuxCompatibility.hpp\
RCPtr.hpp\
Reactant.hpp

bin_SCRIPTS =\
emtool


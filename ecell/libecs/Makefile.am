libdir=@libdir@

SUBDIRS = dmtemplates

DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
KORANDOM_DIR=@KORANDOM_DIR@
KORANDOM_INCLUDE=-I${KORANDOM_DIR}
LIBKORANDOM=${KORANDOM_DIR}/korandom/libkorandom.la
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
BOOST_DIR=@BOOST_DIR@


INCLUDES = -I..  -I../.. -I${BOOST_DIR} ${KORANDOM_INCLUDE} ${DMTOOL_INCLUDE} ${INCLTDL}

lib_LTLIBRARIES = libecs.la

CXXFLAGS= @CXXFLAGS@ # -D_REENTRANT

libecs_la_LDFLAGS= -version-info 0:1:0 #-rpath ${libdir}
libecs_la_LIBADD= ${LIBLTDL} -L${KORANDOM_DIR}/korandom/ -lkorandom -lstdc++
libecs_la_SOURCES=\
Util.cpp\
Polymorph.cpp\
FullID.cpp\
PropertySlot.cpp\
PropertyInterface.cpp\
AccumulatorMaker.cpp\
ReactorMaker.cpp\
Substance.cpp\
Accumulators.cpp\
Integrators.cpp\
SubstanceMaker.cpp\
Stepper.cpp\
Model.cpp\
Scheduler.cpp\
EntityType.cpp\
System.cpp\
Entity.cpp\
Reactor.cpp\
StepperMaker.cpp\
SystemMaker.cpp\
DataPoint.cpp\
DataPointVector.cpp\
VVector.cpp\
osif.cpp\
Logger.cpp\
PhysicalLogger.cpp\
LoggerBroker.cpp

libecsdir=${pkgincludedir}/libecs
libecs_HEADERS=\
libecs.hpp\
Defs.hpp\
convertTo.hpp\
Polymorph.hpp\
FullID.hpp\
PropertySlot.hpp\
PropertySlotMaker.hpp\
PropertyInterface.hpp\
AccumulatorMaker.hpp\
ReactorMaker.hpp\
Substance.hpp\
Accumulators.hpp\
Integrators.hpp\
SubstanceMaker.hpp\
Stepper.hpp\
Model.hpp\
Scheduler.hpp\
EntityType.hpp\
System.hpp\
Entity.hpp\
Reactor.hpp\
SRMReactor.hpp\
RuleSRMReactor.hpp\
StepperMaker.hpp\
SystemMaker.hpp\
DataPoint.hpp\
DataPointVector.hpp\
VVector.h\
osif.h\
PhysicalLogger.hpp\
Logger.hpp\
LoggerBroker.hpp\
Exceptions.hpp\
Util.hpp\
CoreLinuxCompatibility.hpp\
RCPtr.hpp\
Reactant.hpp\
DynamicPriorityQueue.hpp\
AssocVector.h


bin_SCRIPTS =\
emtool


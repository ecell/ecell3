libdir=@libdir@

SUBDIRS = dmtemplates

GSL_LIBADD = -lgsl -lgslcblas
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
libecs_la_LIBADD= ${LIBLTDL} -L${KORANDOM_DIR}/korandom/ -lkorandom ${GSL_LIBADD}
libecs_la_SOURCES=\
Util.cpp\
Polymorph.cpp\
FullID.cpp\
PropertySlot.cpp\
PropertyInterface.cpp\
ReactorMaker.cpp\
Substance.cpp\
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

#Accumulators.cpp\
#AccumulatorMaker.cpp\


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
ReactorMaker.hpp\
Substance.hpp\
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

#AccumulatorMaker.hpp\
#Accumulators.hpp\



bin_SCRIPTS =\
emtool


libdir=@libdir@

SUBDIRS = dmtemplates

GSL_LIBADD = -lgsl -lgslcblas
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
BOOST_DIR=@BOOST_DIR@


INCLUDES = -I..  -I../.. -I${BOOST_DIR} ${DMTOOL_INCLUDE} ${INCLTDL}

lib_LTLIBRARIES = libecs.la

CXXFLAGS= @CXXFLAGS@ # -D_REENTRANT

libecs_la_LDFLAGS= -version-info 1:0:0 #-rpath ${libdir}
libecs_la_LIBADD= ${LIBLTDL} ${GSL_LIBADD}
libecs_la_SOURCES=\
Util.cpp\
Polymorph.cpp\
FullID.cpp\
PropertySlot.cpp\
PropertyInterface.cpp\
ProcessMaker.cpp\
Variable.cpp\
VariableMaker.cpp\
Stepper.cpp\
Steppers.cpp\
VariableProxy.cpp\
Model.cpp\
Scheduler.cpp\
EntityType.cpp\
System.cpp\
Entity.cpp\
Process.cpp\
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
ProcessMaker.hpp\
Variable.hpp\
VariableMaker.hpp\
Stepper.hpp\
Steppers.hpp\
VariableProxy.hpp\
Model.hpp\
Scheduler.hpp\
EntityType.hpp\
System.hpp\
Entity.hpp\
Process.hpp\
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
VariableReference.hpp\
DynamicPriorityQueue.hpp\
AssocVector.h\
FluxProcess.hpp

bin_SCRIPTS =\
emtool


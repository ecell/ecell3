libdir=@libdir@

SUBDIRS =

GSL_LIBADD = -lgsl -lgslcblas
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
BOOST_DIR=@BOOST_DIR@


INCLUDES = -I..  -I../.. -I${BOOST_DIR} ${DMTOOL_INCLUDE} ${INCLTDL}

lib_LTLIBRARIES = libecs.la

CXXFLAGS= @CXXFLAGS@ # -D_REENTRANT

libecs_la_LDFLAGS= -version-info 2:0:0 #-rpath ${libdir}
libecs_la_LIBADD=  -lstdc++ ${LIBLTDL} ${GSL_LIBADD}
libecs_la_SOURCES=\
Singleton.cpp\
SmallObj.cpp\
Util.cpp\
Polymorph.cpp\
FullID.cpp\
PropertySlot.cpp\
PropertySlotProxy.cpp\
PropertyInterface.cpp\
PropertiedClass.cpp\
ProcessMaker.cpp\
Variable.cpp\
VariableMaker.cpp\
Stepper.cpp\
DifferentialStepper.cpp\
DiscreteTimeStepper.cpp\
DiscreteEventStepper.cpp\
PassiveStepper.cpp\
VariableProxy.cpp\
Model.cpp\
Scheduler.cpp\
EntityType.cpp\
System.cpp\
Entity.cpp\
Process.cpp\
DiscreteEventProcess.cpp\
StepperMaker.cpp\
SystemMaker.cpp\
DataPoint.cpp\
DataPointVector.cpp\
VVector.cpp\
osif.cpp\
Logger.cpp\
PhysicalLogger.cpp\
LoggerBroker.cpp\
Exceptions.cpp


libecsdir=${pkgincludedir}/libecs
libecs_HEADERS=\
AssocVector.h\
Singleton.h\
SmallObj.h\
Threads.h\
libecs.hpp\
Defs.hpp\
convertTo.hpp\
Polymorph.hpp\
FullID.hpp\
PropertySlot.hpp\
PropertySlotProxy.hpp\
PropertyInterface.hpp\
PropertiedClass.hpp\
ProcessMaker.hpp\
Variable.hpp\
VariableMaker.hpp\
Stepper.hpp\
DifferentialStepper.hpp\
DiscreteTimeStepper.hpp\
DiscreteEventStepper.hpp\
PassiveStepper.hpp\
VariableProxy.hpp\
Model.hpp\
Scheduler.hpp\
EntityType.hpp\
System.hpp\
Entity.hpp\
Process.hpp\
DiscreteEventProcess.hpp\
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
FluxProcess.hpp


all-local: libecs.so

libecs.so: libecs.la
	ln -sf .libs/libecs.so ./libecs.so

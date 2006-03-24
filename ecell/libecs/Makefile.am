libdir=@libdir@
USE_COMPILER_CONVERSIONS=@USE_COMPILER_CONVERSIONS@

if OS_WIN32
  no_undefined = -no-undefined
  SHLEXT = .dll.a
else
  SHLEXT = @LTDL_SHLIB_EXT@
endif

SUBDIRS =

GSL_LIBADD = -lgsl -lgslcblas
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@

AM_CXXFLAGS=@CXXFLAGS@

INCLUDES = -I${top_srcdir}  -I${top_srcdir}/.. ${DMTOOL_INCLUDE} ${INCLTDL}

pkglib_LTLIBRARIES = libecs.la

libecs_la_LDFLAGS= -version-info 2:0:0 $(no_undefined) #-rpath ${libdir}
libecs_la_LIBADD=  -lstdc++ ${LIBLTDL} ${GSL_LIBADD}
libecs_la_SOURCES=\
libecs.cpp\
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
SystemStepper.cpp\
Interpolant.cpp\
Model.cpp\
EntityType.cpp\
System.cpp\
Entity.cpp\
Process.cpp\
ContinuousProcess.cpp\
StepperMaker.cpp\
SystemMaker.cpp\
DataPoint.cpp\
DataPointVector.cpp\
VVector.cpp\
osif.cpp\
Logger.cpp\
LoggerAdapter.cpp\
PhysicalLogger.cpp\
VariableReference.cpp\
LoggerBroker.cpp\
Exceptions.cpp


libecsdir=${pkgincludedir}/libecs
libecs_HEADERS=\
AssocVector.h\
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
ProcessEvent.hpp\
StepperEvent.hpp\
EventScheduler.hpp\
DifferentialStepper.hpp\
DiscreteTimeStepper.hpp\
DiscreteEventStepper.hpp\
PassiveStepper.hpp\
SystemStepper.hpp\
Interpolant.hpp\
Model.hpp\
EntityType.hpp\
System.hpp\
Entity.hpp\
Process.hpp\
ContinuousProcess.hpp\
StepperMaker.hpp\
SystemMaker.hpp\
DataPoint.hpp\
DataPointVector.hpp\
VVector.h\
osif.h\
PhysicalLogger.hpp\
Logger.hpp\
LoggerAdapter.hpp\
LoggerBroker.hpp\
Exceptions.hpp\
Util.hpp\
VariableReference.hpp\
DynamicPriorityQueue.hpp\
FluxProcess.hpp\
MethodProxy.hpp


all-local: libecs$(SHLEXT)

libecs$(SHLEXT): libecs.la
	ln -sf .libs/libecs$(SHLEXT) ./libecs$(SHLEXT)

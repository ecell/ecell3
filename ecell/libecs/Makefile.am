abs_top_builddir=@abs_top_builddir@

libdir=@libdir@

SHLEXT = @LTDL_SHLIB_EXT@

SUBDIRS = scripting . tests

ECELL_DIRNAME=@ECELL_DIRNAME@
ECELL_INCLUDE_DIR=$(includedir)/$(ECELL_DIRNAME)

GSL_LIBADD = -lgsl -lgslcblas
DMTOOL_INCLUDE = -I@DMTOOL_INCLUDE_DIR@
INCLTDL = @INCLTDL@
LIBLTDL = @LIBLTDL@

INCLUDES = -I$(top_srcdir) $(DMTOOL_INCLUDE) $(INCLTDL)
lib_LTLIBRARIES = libecs.la

AM_CXXFLAGS = @CXXFLAGS@

libecs_la_LDFLAGS = -version-info 2:0:0 -no-undefined
libecs_la_LIBADD =  -lstdc++ $(LIBLTDL) $(GSL_LIBADD) scripting/libecs-scripting.la
libecs_la_SOURCES = \
AdaptiveDifferentialStepper.cpp\
ContinuousProcess.cpp\
DifferentialStepper.cpp\
DiscreteEventStepper.cpp\
DiscreteTimeStepper.cpp\
Entity.cpp\
EntityType.cpp\
Exceptions.cpp\
FullID.cpp\
FullPN.cpp\
Logger.cpp\
LoggerAdapter.cpp\
LoggerManager.cpp\
Model.cpp\
PassiveStepper.cpp\
PhysicalLogger.cpp\
Polymorph.cpp\
Process.cpp\
ProcessMaker.cpp\
PropertiedClass.cpp\
PropertiedClassKind.cpp\
PropertiedObjectMaker.cpp\
PropertyInterface.cpp\
PropertySlotProxyLoggerAdapter.cpp\
PropertyType.cpp\
Stepper.cpp\
StepperMaker.cpp\
System.cpp\
SystemMaker.cpp\
SystemPath.cpp\
SystemStepper.cpp\
Util.cpp\
VVector.cpp\
Variable.cpp\
VariableMaker.cpp\
VariableReference.cpp\
VariableValueInterpolant.cpp\
libecs.cpp\
BlockIO.cpp

libecs_ladir=${ECELL_INCLUDE_DIR}/libecs
libecs_la_HEADERS=\
AdaptiveDifferentialStepper.hpp\
AssocVector.h\
CastUtils.hpp\
Converters.hpp\
ContinuousProcess.hpp\
DataPoint.hpp\
DataPointIntegrator.hpp\
Downsampler.hpp\
DataPointVector.hpp\
Defs.hpp\
DifferentialStepper.hpp\
DiscreteEventStepper.hpp\
DiscreteTimeStepper.hpp\
DynamicPriorityQueue.hpp\
Entity.hpp\
EntityType.hpp\
EventBase.hpp\
EventScheduler.hpp\
Exceptions.hpp\
FluxProcess.hpp\
FullID.hpp\
FullPN.hpp\
Happening.hpp\
Interpolant.hpp\
Integrator.hpp\
LocalID.hpp\
Logger.hpp\
LoggerAdapter.hpp\
LoggerManager.hpp\
MethodProxy.hpp\
Model.hpp\
PassiveStepper.hpp\
PhysicalLogger.hpp\
Polymorph.hpp\
Process.hpp\
ProcessEvent.hpp\
ProcessMaker.hpp\
PropertiedClass.hpp\
PropertiedClassKind.hpp\
PropertiedObjectMaker.hpp\
PropertyInterface.hpp\
PropertySlot.hpp\
PropertySlotProxy.hpp\
PropertySlotProxyLoggerAdapter.hpp\
PropertyType.hpp\
RealMath.hpp\
Stepper.hpp\
StepperEvent.hpp\
StepperMaker.hpp\
System.hpp\
SystemMaker.hpp\
StstemPath.hpp\
SystemStepper.hpp\
Util.hpp\
Variable.hpp\
VariableMaker.hpp\
VariableReference.hpp\
convertTo.hpp\
libecs.hpp

noinst_HEADERS=\
VVector.hpp\
BlockIO.hpp

ECELL_STAGING_HOME=$(abs_top_builddir)/tmp
OUTPUT=$(ECELL_STAGING_HOME)$(libdir)/libecs$(SHLEXT)
all-local: $(OUTPUT)

$(OUTPUT): libecs.la
	test -d $(ECELL_STAGING_HOME)$(libdir) || mkdir -p -- $(ECELL_STAGING_HOME)$(libdir)
	$(LIBTOOL) --mode=install cp $< $(ECELL_STAGING_HOME)$(libdir)

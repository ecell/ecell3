
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE@
KORANDOM_INCLUDE=-I@KORANDOM_DIR@

INCLUDES = -I..  -I../include ${KORANDOM_INCLUDE} ${DMTOOL_INCLUDE}
noinst_LIBRARIES = libecs.a

CXXFLAGS= @CXXFLAGS@ -D_GNU_SOURCE -D_REENTRANT -D_THREAD_SAFE 

libecs_a_SOURCES =\
Util.cpp\
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
SystemMaker.cpp


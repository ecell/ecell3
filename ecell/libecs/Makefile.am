
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE@
KORANDOM_INCLUDE=-I@KORANDOM_INCLUDE@

INCLUDES = -I..  -I../include ${KORANDOM_INCLUDE} ${DMTOOL_INCLUDE}
noinst_LIBRARIES = libecscore.a

//FIXME: temporary work around for a bug in Common C++(will be fixed in 1.2.1)
CXXFLAGS=-D_XOPEN_SOURCE=500

libecscore_a_SOURCES =\
AccumulatorMaker.C\
FQPN.C\
ReactorMaker.C\
Substance.C\
Accumulators.C\
Integrators.C\
RootSystem.C\
SubstanceMaker.C\
Primitive.C\
Stepper.C\
System.C\
Entity.C\
Reactor.C\
StepperMaker.C\
SystemMaker.C


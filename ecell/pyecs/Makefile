ECELLDIR=../../../ecell3
CXX=/usr/include/python1.6/CXX
CXXO=/usr/etc/CXX
PYLIB=/usr/include/python1.6
ECELL_INCLUDE=-I$(ECELLDIR) -I$(ECELLDIR)/ecell -I$(ECELLDIR)/ecell/include -I$(ECELLDIR)/dmtool/include
THREAD_OPTION=-D_REENTRANT -D_THREAD_SAFE -pthread  -g -O2 -D_GNU_SOURCE
COMMON_CPP=/usr/local/lib
LIBUTIL=$(ECELLDIR)/ecell/util/libutil.a
LIBECSCORE=$(ECELLDIR)/ecell/Koyurugi/libecscore.a
LIBKORANDOM=$(ECELLDIR)/korandom/libkorandom.a

G++ = /usr/local/bin/g++
#G++ = /usr/bin/g++

#all:	PyEcs.o	PySimulator.o SimulatorImplementation.o  Command.o StepCommand.o MakeSystemCommand.o Simulator.o PyEcs.so
all:	PyEcs.o	PySimulator.o SimulatorImplementation.o  Command.o StepCommand.o Simulator.o PyEcs.so

PyEcs.o:	PyEcs.h PyEcs.C 
	$(G++) -I$(CXX) -I$(PYLIB) $(ECELL_INCLUDE) -c PyEcs.C

PySimulator.o:	PySimulator.h PySimulator.C
	$(G++) -I$(CXX) -I$(PYLIB) $(ECELL_INCLUDE) -c PySimulator.C

SimulatorImplementation.o: SimulatorImplementation.h SimulatorImplementation.C
	$(G++) $(ECELL_INCLUDE) $(THREAD_OPTION) -c SimulatorImplementation.C

Simulator.o:	Simulator.h	Simulator.C
	$(G++) $(ECELL_INCLUDE) $(THREAD_OPTION) -c Simulator.C

Command.o:	Command.C Command.h
	$(G++) $(ECELL_INCLUDE) $(THREAD_OPTION) -c Command.C

StepCommand.o:	StepCommand.C	StepCommand.h
	$(G++) $(ECELL_INCLUDE) $(THREAD_OPTION) -c StepCommand.C

MakeSystemCommand.o:	MakeSystemCommand.C	MakeSystemCommand.h
	$(G++) $(ECELL_INCLUDE) $(THREAD_OPTION) -c MakeSystemCommand.C

#PyEcs.so: PyEcs.o PySimulator.o Simulator.o SimulatorImplementation.o Command.o StepCommand.o MakeSystemCommand.o  Makefile
PyEcs.so: PyEcs.o PySimulator.o Simulator.o SimulatorImplementation.o Command.o StepCommand.o Makefile
	$(G++) --shared $(CXXO)/*.o StepCommand.o Command.o Simulator.o SimulatorImplementation.o PySimulator.o PyEcs.o $(LIBECSCORE) $(LIBKORANDOM) $(LIBUTIL) $(COMMON_CPP)/*.a -o ecs.so

clean:
	rm -f *.o *~ *.so core








ECELLDIR=../../../ecell3
CXX=/usr/include/python1.6/CXX
CXXO=/usr/etc/CXX
PYLIB=/usr/include/python1.6
ECELL_INCLUDE=-I$(ECELLDIR) -I$(ECELLDIR)/ecell -I$(ECELLDIR)/ecell/include -I$(ECELLDIR)/dmtool/include
THREAD_OPTION=-D_REENTRANT -D_THREAD_SAFE -pthread  -g -O2 -D_GNU_SOURCE
LIBUTIL=$(ECELLDIR)/ecell/util/libutil.a
LIBECSCORE=$(ECELLDIR)/ecell/libecs/libecs.a
LIBKORANDOM=$(ECELLDIR)/korandom/libkorandom.a

G++ = /usr/local/bin/g++
#G++ = /usr/bin/g++

all:	LocalSimulatorImplementation.o SimulatorImplementation.o Simulator.o PyEcs.o PySimulator.o PyEcs.so

PyEcs.o:	PyEcs.hpp PyEcs.cpp 
	$(G++) -I$(CXX) -I$(PYLIB) $(ECELL_INCLUDE) -c PyEcs.cpp

PySimulator.o:	PySimulator.hpp PySimulator.cpp
	$(G++) -I$(CXX) -I$(PYLIB) $(ECELL_INCLUDE) -c PySimulator.cpp

LocalSimulatorImplementation.o: LocalSimulatorImplementation.hpp LocalSimulatorImplementation.cpp
	$(G++) $(ECELL_INCLUDE) $(THREAD_OPTION) -c LocalSimulatorImplementation.cpp

SimulatorImplementation.o: SimulatorImplementation.hpp SimulatorImplementation.cpp
	$(G++) $(ECELL_INCLUDE) $(THREAD_OPTION) -c SimulatorImplementation.cpp

Simulator.o:	Simulator.hpp	Simulator.cpp
	$(G++) $(ECELL_INCLUDE) $(THREAD_OPTION) -c Simulator.cpp

PyEcs.so: PyEcs.o PySimulator.o Simulator.o SimulatorImplementation.o LocalSimulatorImplementation.o Makefile
	$(G++) --shared $(CXXO)/*.o Simulator.o SimulatorImplementation.o LocalSimulatorImplementation.o PySimulator.o PyEcs.o $(LIBECSCORE) $(LIBUTIL) $(LIBKORANDOM) -o ecs.so

clean:
	rm -f *.o *~ *.so core








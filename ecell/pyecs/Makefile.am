

ECELLDIR=..
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@

#CXXSRCDIR=/usr/etc/CXX/
#CXXSRCDIR=/home/shafi/wrk/CXX/Src
#CXXINCLUDE=-I/home/shafi/wrk/CXX
CXXSRCDIR=../../PyCXX/Src
CXXINCLUDE=-I../..//PyCXX
PYTHONINCLUDE=-I/usr/include/python2.1 #-I@PYTHONINCLUDEDIR@
#PYTHONINCLUDE=-I/usr/include/python1.5
ECELLINCLUDES=-I${ECELLDIR} ${DMTOOL_INCLUDE} ${INCLTDL} ${PYTHONINCLUDE}
CXXFLAGS=@CXXFLAGS@ ${CXXINCLUDE} ${ECELLINCLUDES} #-D_REENTRANT
LIBLTDL=@LIBLTDL@

LIBKORANDOM=../../korandom/libkorandom.la

CFLAGS=${CXXFLAGS}

#CXXOBJS=${CXXSRCDIR}/*.o

CXXSRC=\
IndirectPythonInterface.cxx\
cxx_extensions.cxx\
cxxextensions.c\
cxxsupport.cxx

$(CXXSRC):
	cp -f ${CXXSRCDIR}/$@ .

clean-local:
	rm -f ${CXXSRC}

noinst_LTLIBRARIES=ecs.la
ecs_la_SOURCES=\
PyUVariable.cpp\
PyEcs.cpp\
PySimulator.cpp\
PyLogger.cpp\
$(CXXSRC) 

pyecs_HEADERS=\
PyUVariable.hpp\
PyEcs.hpp\
PySimulator.hpp\
PyLogger.hpp

ecs_la_LDFLAGS=-module -export-dynamic -avoid-version -rpath /usr/local/lib

#FIXME: specifing .libs is ugly
#ecs_la_LIBADD= /usr/local/lib/libstdc++.a.2.10.0 -L$(ECELLDIR)/libemc/.libs/ -lemc -L$(ECELLDIR)/libecs/.libs/ -lecs
ecs_la_LIBADD= -lstdc++ -L$(ECELLDIR)/libemc/.libs/ -lemc -L$(ECELLDIR)/libecs/.libs/ -lecs
ecs_la_DEPENDENCIES= $(ECELLDIR)/libemc/libemc.la $(ECELLDIR)/libecs/libecs.la $(LIBKORANDOM) ${LIBLTDL} 

all-local: ecs.so

ecs.so: ecs.la 
	ln -sf .libs/ecs.so .

clean-local:
	rm -f ecs.so
	rm -f $(CXXSRC)

pyecsdir=${pkgincludedir}/pyecs

EXTRA_DIST= \
testrun\
test.py




ECELLDIR=..
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
PYTHON_INCLUDES=@PYTHON_INCLUDES@

CXXSRCDIR=../../PyCXX/Src
CXXINCLUDE=-I../..//PyCXX

ECELLINCLUDES=-I${ECELLDIR} ${DMTOOL_INCLUDE} ${INCLTDL} ${PYTHON_INCLUDES}
CXXFLAGS=@CXXFLAGS@ ${CXXINCLUDE} ${ECELLINCLUDES} #-D_REENTRANT

KORANDOM_DIR=@KORANDOM_DIR@
LIBKORANDOM=$(KORANDOM_DIR)/korandom/libkorandom.la
LIBEMC=$(ECELLDIR)/libemc/libemc.la 
LIBECS=$(ECELLDIR)/libecs/libecs.la


CFLAGS=${CXXFLAGS}

CXXSRC=\
IndirectPythonInterface.cxx\
cxx_extensions.cxx\
cxxextensions.c\
cxxsupport.cxx

$(CXXSRC):
	cp -f ${CXXSRCDIR}/$@ .

clean-local:
	rm -f ${CXXSRC}

pyexec_LTLIBRARIES=ecs.la
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

ecs_la_LIBADD= $(LIBEMC) $(LIBECS) $(LIBKORANDOM) -lstdc++ $(LIBLTDL)

all-local: ecs.so

ecs.so: ecs.la 
	ln -sf .libs/ecs.so .

clean-local:
	rm -f ecs.so
	rm -f $(CXXSRC)

pyecsdir=${pkgincludedir}/pyecs

bin_SCRIPTS =\
ecell3


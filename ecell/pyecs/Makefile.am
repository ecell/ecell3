

ECELLDIR=..
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
PYTHON_INCLUDES=@PYTHON_INCLUDES@

ecellpyexecdir=$(pyexecdir)/ecell

CXXSRCDIR=../../PyCXX/Src
CXXINCLUDE=-I../../PyCXX -I../../boost

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


#FIXME: _ecs.la cannot be used with automake-1.4.

ecellpyexec_LTLIBRARIES = ecs.la

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

ecs_la_LDFLAGS=-module -export-dynamic -avoid-version

ecs_la_LIBADD= $(LIBEMC) $(LIBECS) $(LIBKORANDOM) -lstdc++ $(LIBLTDL)

all-local: ecs.so

#FIXME: should be removed after ecs.la is renamed to _ecs.la
install-exec-local:
	mv -f $(ecellpyexecdir)/ecs.so $(ecellpyexecdir)/_ecs.so
	mv -f $(ecellpyexecdir)/ecs.la $(ecellpyexecdir)/_ecs.la


ecs.so: ecs.la 
	ln -sf .libs/ecs.so ./ecs.so

clean-local:
	rm -f ecs.so
	rm -f $(CXXSRC)

pyecsdir=${pkgincludedir}/pyecs

bin_SCRIPTS =\
ecell3


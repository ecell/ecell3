

ECELLDIR=..
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
KORANDOM_INCLUDE=-I@KORANDOM_DIR@
INCLTDL=@INCLTDL@

CXXSRCDIR=/usr/etc/CXX/
PYTHONINCLUDE=-I/usr/include/python2.0 #-I@PYTHONINCLUDEDIR@
ECELLINCLUDES=-I${ECELLDIR} -I${ECELLDIR}/libecs -I${ECELLDIR}/emc ${KORANDOM_INCLUDE} ${DMTOOL_INCLUDE} ${INCLTDL} ${PYTHONINCLUDE}
CXXFLAGS=@CXXFLAGS@ ${CXXINCLUDE} ${ECELLINCLUDES}
LIBLTDL=@LIBLTDL@

LIBKORANDOM=../../korandom/libkorandom.la

CFLAGS=${CXXFLAGS}

#CXXOBJS=${CXXSRCDIR}/*.o

CXXSRC=\
cxx_extensions.cxx\
cxxextensions.c\
cxxsupport.cxx

$(CXXSRC):
	cp -f ${CXXSRCDIR}/$@ .

clean-local:
	rm -f ${CXXSRC}

noinst_LTLIBRARIES=ecs.la
ecs_la_SOURCES=PyEcs.cpp PySimulator.cpp $(CXXSRC) 
ecs_la_LDFLAGS=-module -export-dynamic -avoid-version -rpath /usr/local/lib
ecs_la_LIBADD= -lstdc++
#$(ECELLDIR)/emc/libemc.la $(ECELLDIR)/libecs/libecs.la $(LIBKORANDOM) ${LIBLTDL} 

all-local: ecs.so

ecs.so: ecs.la 
	ln -sf .libs/ecs.so .

clean-local:
	rm -f ecs.so

pyecsdir=${pkginclude}/pyecs
pyecs_HEADERS=\
PyEcs.hpp\
PySimulator.hpp
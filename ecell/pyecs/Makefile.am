

ECELLDIR=..
CXXFLAGS=@CXXFLAGS@
CXXSRCDIR=/usr/etc/CXX/
PYTHONINCLUDE=-I/usr/include/python2.0 #-I@PYTHONINCLUDEDIR@
ECELLINCLUDES=-I${ECELLDIR} -I${ECELLDIR}/libecs -I${ECELLDIR}/emc -I${ECELLDIR}/../dmtool/include ${PYTHONINCLUDE}
CXXFLAGS=${CXXINCLUDE} ${ECELLINCLUDES} #@CXXFLAGS@
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

noinst_LTLIBRARIES=ecs.la
ecs_la_SOURCES=PyEcs.cpp PySimulator.cpp $(CXXSRC)
ecs_la_LDFLAGS=-module -avoid-version -rpath /usr/local/lib
#ecs_la_LIBADD=$(CXXOBJS) $(ECELLDIR)/libecs/libecs.la $(ECELLDIR)/emc/libemc.la $(LIBKORANDOM) -lltdl
ecs_la_LIBADD=$(ECELLDIR)/libecs/libecs.la $(ECELLDIR)/emc/libemc.la $(LIBKORANDOM) -lltdl



